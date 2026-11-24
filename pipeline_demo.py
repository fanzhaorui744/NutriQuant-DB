"""
Unified 6-stage NutriQuant + DKDF inference engine shared by `main.py` (CLI)
and `demo_app.py` (local HTML Web UI).
"""
from __future__ import annotations

import json
import os
import time
import cv2
import joblib
import numpy as np
import torch

from depth import DepthEstimator
from nutrition import get_nutrition_info
from perspective import (
    _is_reasonable_pixel_per_cm,
    correct_perspective,
    estimate_plate_pixel_per_cm,
)
from recognition import FoodPredictor
from segmentation import FoodSegmentor
from volume import (
    calculate_adaptive_volume,
    enhance_low_contrast,
)


PALETTE_BGR = [
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 5),
    (234, 67, 53),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (124, 179, 66),
    (92, 107, 192),
    (236, 64, 122),
]


def _postprocess_mask(mask_bool: np.ndarray) -> np.ndarray | None:
    if mask_bool is None:
        return None
    m = np.asarray(mask_bool)
    if m.ndim != 2:
        return None
    mu8 = m.astype(np.uint8) * 255
    if int(np.sum(mu8 > 0)) < 25:
        return None

    kernel = np.ones((3, 3), np.uint8)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, kernel, iterations=1)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mu8 > 0).astype(np.uint8), connectivity=8)
    if num is None or int(num) <= 1:
        return mu8 > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mu8 > 0
    keep = int(np.argmax(areas)) + 1
    return (labels == keep).astype(bool)


def _mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    inter = np.logical_and(m1, m2)
    union = np.logical_or(m1, m2)
    u = int(np.sum(union))
    return float(np.sum(inter) / u) if u > 0 else 0.0


def _detect_plate_circle(image_bgr: np.ndarray):
    """Detect the primary circular reference plate (cx, cy, r) if clearly visible."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    h, w = gray.shape[:2]
    min_r = int(max(40, min(h, w) * 0.22))
    max_r = int(max(min_r + 1, min(h, w) * 0.52))
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(h, w) * 0.25,
        param1=120,
        param2=38,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None
    circles = np.asarray(circles, dtype=np.float32).reshape(-1, 3)
    if circles.size == 0:
        return None
    best = max(circles, key=lambda c: c[2])
    return float(best[0]), float(best[1]), float(best[2])


def _deduplicate_masks(raw_masks, img_shape, plate_circle=None, min_area_frac=0.003, max_area_frac=0.35, iou_thresh=0.55):
    h, w = img_shape[:2]
    total_px = float(h * w)
    min_px = max(600, int(total_px * min_area_frac))
    max_px = int(total_px * max_area_frac)

    candidates = []
    for m in raw_masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        area = int(np.sum(seg))
        if area < min_px or area > max_px:
            continue
        refined = _postprocess_mask(seg)
        if refined is None:
            continue
        refined_area = int(np.sum(refined))
        if refined_area < min_px or refined_area > max_px:
            continue

        ys, xs = np.where(refined)
        bbox_h = int(ys.max() - ys.min() + 1)
        bbox_w = int(xs.max() - xs.min() + 1)
        if bbox_w > 0.90 * w and bbox_h > 0.90 * h:
            continue

        # Skip stray border fragments (e.g., partial neighboring plate at image corner)
        if ys.min() <= 1 and ys.max() < 0.15 * h:
            continue
        if plate_circle is not None:
            pcx, pcy, pr = plate_circle
            mcx = float(xs.mean())
            mcy = float(ys.mean())
            dist = float(np.hypot(mcx - pcx, mcy - pcy))
            # Food resides inside the inner 90% of the plate rim
            if dist > 0.90 * pr:
                continue

        score = float(m.get("stability_score", 0.9)) * 0.5 + float(m.get("predicted_iou", 0.8)) * 0.5
        candidates.append({
            "mask": refined,
            "area_px": refined_area,
            "score": score,
            "predicted_iou": float(m.get("predicted_iou", 0.0)),
            "stability_score": float(m.get("stability_score", 0.0)),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    kept = []
    for c in candidates:
        if any(_mask_iou(c["mask"], k["mask"]) >= iou_thresh for k in kept):
            continue
        kept.append(c)

    kept.sort(key=lambda c: (int(np.where(c["mask"])[0].mean()), int(np.where(c["mask"])[1].mean())))
    return kept


def _shape_features(mask_bool: np.ndarray, scale_px_per_cm: float) -> dict:
    mask_u8 = mask_bool.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "circularity": 0.0,
            "aspect_ratio": 1.0,
            "solidity": 0.0,
            "extent": 0.0,
            "bbox_long_cm": 0.0,
            "bbox_short_cm": 0.0,
            "component_count": 1.0,
        }
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    circularity = float(4.0 * np.pi * area / (peri * peri + 1e-6)) if area > 0 and peri > 0 else 0.0
    _, (w, h), _ = cv2.minAreaRect(c)
    w, h = float(w), float(h)
    aspect_ratio = float(max(w, h) / (min(w, h) + 1e-6)) if w > 1e-6 and h > 1e-6 else 1.0
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area / (hull_area + 1e-6)) if hull_area > 0 else 0.0
    _, _, bw, bh = cv2.boundingRect(c)
    extent = float(area / max(1.0, float(bw) * float(bh))) if area > 0 else 0.0
    s = max(float(scale_px_per_cm), 1e-6)
    num, _ = cv2.connectedComponents(mask_u8)
    n_cc = max(1, int(num) - 1) if num is not None else 1
    return {
        "circularity": float(np.clip(circularity, 0.0, 1.5)),
        "aspect_ratio": float(np.clip(aspect_ratio, 1.0, 50.0)),
        "solidity": float(np.clip(solidity, 0.0, 1.0)),
        "extent": float(np.clip(extent, 0.0, 1.0)),
        "bbox_long_cm": float(np.clip(max(w, h) / s, 0.0, 200.0)),
        "bbox_short_cm": float(np.clip(min(w, h) / s, 0.0, 200.0)),
        "component_count": float(np.clip(n_cc, 1, 50)),
    }


def _colorize_depth(depth_map: np.ndarray) -> np.ndarray:
    d = np.asarray(depth_map, dtype=np.float32)
    finite = np.isfinite(d)
    if not np.any(finite):
        return np.zeros((*d.shape[:2], 3), dtype=np.uint8)
    lo = float(np.percentile(d[finite], 2))
    hi = float(np.percentile(d[finite], 98))
    if hi - lo < 1e-6:
        norm = np.zeros_like(d, dtype=np.uint8)
    else:
        norm = np.clip((d - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


class NutriQuantPipeline:
    """Lazy-loading, reusable 6-stage NutriQuant inference pipeline."""

    def __init__(self, bundle_root: str | None = None, encoder: str = "vits"):
        self.bundle_root = bundle_root or os.path.dirname(os.path.abspath(__file__))
        self.encoder = encoder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.sam_checkpoint = os.path.join(self.bundle_root, "checkpoints", "sam2_t.pt")
        self.sam_config = os.path.join(self.bundle_root, "vendor", "sam2", "configs", "sam2", "sam2_hiera_t.yaml")
        self.classifier_path = os.path.join(self.bundle_root, "result", "food_classifier.pth")
        self.calibration_path = os.path.join(self.bundle_root, "calibration_params.json")
        self.rf_dir = os.path.join(self.bundle_root, "models", "rf_regressors")

        self._segmentor = None
        self._classifier = None
        self._depth_estimator = None
        self._rf_cache = {}
        self.calibration_params = self._load_calibration()

    def _load_calibration(self) -> dict:
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _load_rf(self, food_code: str):
        if food_code in self._rf_cache:
            return self._rf_cache[food_code]
        path = os.path.join(self.rf_dir, f"{food_code}.pkl")
        if os.path.exists(path):
            try:
                self._rf_cache[food_code] = joblib.load(path)
            except Exception:
                self._rf_cache[food_code] = None
        else:
            self._rf_cache[food_code] = None
        return self._rf_cache[food_code]

    def ensure_loaded(self):
        if self._segmentor is None:
            if not os.path.exists(self.sam_checkpoint):
                raise FileNotFoundError(
                    f"SAM 2 checkpoint not found at {self.sam_checkpoint}. "
                    "Run `python download_models.py` or `git lfs pull` first."
                )
            self._segmentor = FoodSegmentor(self.sam_checkpoint, self.sam_config, points_per_side=24)

        if self._classifier is None:
            if not os.path.exists(self.classifier_path):
                raise FileNotFoundError(
                    f"Classifier checkpoint not found at {self.classifier_path}. "
                    "Run `git lfs pull` or `python recognition.py` first."
                )
            self._classifier = FoodPredictor(self.classifier_path)

        if self._depth_estimator is None:
            self._depth_estimator = DepthEstimator(encoder=self.encoder)

    def get_runtime_info(self) -> dict:
        return {
            "device": self.device,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "sam2_checkpoint": os.path.relpath(self.sam_checkpoint, self.bundle_root),
            "depth_checkpoint": f"checkpoints/depth_anything_v2_{self.encoder}.pth",
            "classifier_checkpoint": os.path.relpath(self.classifier_path, self.bundle_root),
        }

    def run(
        self,
        image_bgr: np.ndarray,
        scale_override_px_per_cm: float | None = None,
        scale_source_hint: str | None = None,
        max_items: int = 12,
        min_confidence: float = 0.50,
        return_visuals: bool = True,
    ) -> dict:
        t0 = time.perf_counter()
        self.ensure_loaded()

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Input image is empty or unreadable.")

        orig_h, orig_w = image_bgr.shape[:2]

        # ------------------------------------------------------------------
        # Stage 1: Perspective rectification & planar metric scale recovery
        # ------------------------------------------------------------------
        t1 = time.perf_counter()
        pc_out = correct_perspective(image_bgr)
        if isinstance(pc_out, tuple) and len(pc_out) >= 3:
            warped, _, detected_scale = pc_out[0], pc_out[1], float(pc_out[2])
        elif isinstance(pc_out, tuple) and len(pc_out) == 2:
            warped, detected_scale = pc_out[0], float(pc_out[1])
        else:
            warped, detected_scale = image_bgr.copy(), 1.0

        if warped is None or warped.shape[0] < 120 or warped.shape[1] < 120:
            warped = image_bgr.copy()
            detected_scale = 1.0

        if scale_override_px_per_cm is not None and float(scale_override_px_per_cm) > 1.0:
            warped = image_bgr.copy()
            scale_px_per_cm = float(scale_override_px_per_cm)
            scale_source = scale_source_hint or "user_override"
        elif _is_reasonable_pixel_per_cm(detected_scale):
            scale_px_per_cm = float(detected_scale)
            scale_source = "fiducial_grid_homography"
        else:
            plate_scale = estimate_plate_pixel_per_cm(warped)
            if _is_reasonable_pixel_per_cm(plate_scale):
                scale_px_per_cm = float(plate_scale)
                scale_source = "plate_circle_fallback"
            else:
                scale_px_per_cm = round(max(20.0, min(warped.shape[:2]) / 24.0), 2)
                scale_source = "heuristic_fallback"

        wh, ww = warped.shape[:2]
        max_side = max(wh, ww)
        if max_side > 1024:
            resize_ratio = 1024.0 / float(max_side)
            warped = cv2.resize(
                warped,
                (int(round(ww * resize_ratio)), int(round(wh * resize_ratio))),
                interpolation=cv2.INTER_AREA,
            )
            scale_px_per_cm = scale_px_per_cm * resize_ratio
        t_stage1 = time.perf_counter() - t1

        # ------------------------------------------------------------------
        # Stage 2: Low-contrast CLAHE + SAM 2 Instance Segmentation
        # ------------------------------------------------------------------
        t2 = time.perf_counter()
        plate_circle = _detect_plate_circle(warped)
        enhanced_for_seg = enhance_low_contrast(warped, stddev_threshold=30.0, clip_limit=3.0)
        clahe_applied = not np.array_equal(enhanced_for_seg, warped)
        raw_masks = self._segmentor.segment(enhanced_for_seg)
        kept_masks = _deduplicate_masks(raw_masks, warped.shape, plate_circle=plate_circle)
        t_stage2 = time.perf_counter() - t2

        # ------------------------------------------------------------------
        # Stage 4: Depth Anything V2 relative inverse depth inference
        # ------------------------------------------------------------------
        t4_infer = time.perf_counter()
        depth_map = self._depth_estimator.estimate(warped)
        t_depth_infer = time.perf_counter() - t4_infer

        # ------------------------------------------------------------------
        # Stage 3, Stage 4 (calibration), Stage 5 (DKDF), Stage 6 (Mass/Energy)
        # ------------------------------------------------------------------
        t356 = time.perf_counter()
        items = []
        seg_overlay = warped.copy()
        def_cal = self.calibration_params.get("default", {})

        for m_info in kept_masks:
            mask = m_info["mask"]
            ys, xs = np.where(mask)
            if ys.size == 0 or xs.size == 0:
                continue

            # 10-pixel padded crop clipped to image boundary
            y_min = max(0, int(ys.min()) - 10)
            y_max = min(warped.shape[0], int(ys.max()) + 10)
            x_min = max(0, int(xs.min()) - 10)
            x_max = min(warped.shape[1], int(xs.max()) + 10)
            crop = warped[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            # Stage 3: ResNet50 classification & confidence gating
            food_code, conf = self._classifier.predict(crop)
            conf = float(conf)
            if conf < float(min_confidence):
                continue

            # Stage 4 + Stage 5: Depth calibration + Ellipsoid proxy + DKDF fusion
            cls_cal = self.calibration_params.get(food_code, {})
            depth_scale_factor = float(cls_cal.get("depth_scale_factor", def_cal.get("depth_scale_factor", 1.0)))

            vol_details = calculate_adaptive_volume(
                mask=mask,
                depth_map=depth_map,
                pixel_per_cm=scale_px_per_cm,
                depth_scale_factor=depth_scale_factor,
                class_name=food_code,
                confidence=conf,
                return_details=True,
            )

            v_geo_raw = float(vol_details.get("rule_volume_cm3", 0.0))
            v_depth = float(vol_details.get("dl_volume_cm3", 0.0))
            alpha = float(vol_details.get("alpha", 0.5))
            q_contrast = float(vol_details.get("depth_contrast", 0.0))
            s_depth = float(vol_details.get("s_depth", 0.0))
            delta_c = float(vol_details.get("delta_c", 0.0))
            delta_gated = float(vol_details.get("delta_gated", 0.0))
            gate_active = bool(vol_details.get("gate_active", False))
            prior_group = str(vol_details.get("prior_group", "neutral"))
            area_cm2 = float(vol_details.get("area_cm2", 0.0))
            height_cm = float(vol_details.get("mean_height_cm", vol_details.get("height_cm", 0.0)))

            # Stage 6: Density & USDA SR28 coefficients + shape-calibrated geometry branch
            nutri = get_nutrition_info(food_code)
            density = float(nutri["density"])
            kcal_per_100g = float(nutri["calories_per_100g"])
            energy_coeff = float(nutri["energy_kcal_per_g"])

            m_depth = v_depth * density
            prior_mean = float(cls_cal.get("mean_area_density", def_cal.get("mean_area_density", 0.35)))
            w_prior = max(0.0, prior_mean * max(0.0, area_cm2))

            rf_model = self._load_rf(food_code)
            reg_rf_meta = cls_cal.get("regression_rf", {}) if isinstance(cls_cal, dict) else {}
            feats = reg_rf_meta.get("features") if isinstance(reg_rf_meta, dict) else None

            if rf_model is not None and isinstance(feats, list) and len(feats) > 0:
                sh = _shape_features(mask, scale_px_per_cm)
                feat_map = {
                    "weight_depth_g": float(m_depth),
                    "weight_prior_g": float(w_prior),
                    "area_cm2": float(area_cm2),
                    "height_cm": float(height_cm),
                    "depth_contrast": float(q_contrast),
                    **sh,
                }
                x_in = np.asarray([[float(feat_map.get(k, 0.0)) for k in feats]], dtype=np.float64)
                try:
                    m_geo = max(0.0, float(rf_model.predict(x_in)[0]))
                    v_geo = m_geo / max(density, 1e-6)
                except Exception:
                    v_geo = v_geo_raw
                    m_geo = v_geo * density
            else:
                v_geo = v_geo_raw
                m_geo = v_geo * density

            # DKDF weighted fusion in physical units
            v_fused = alpha * v_geo + (1.0 - alpha) * v_depth
            mass_g = v_fused * density
            energy_kcal = mass_g * energy_coeff

            item_id = len(items) + 1
            color = PALETTE_BGR[(item_id - 1) % len(PALETTE_BGR)]

            items.append({
                "id": item_id,
                "class_code": food_code,
                "display_name": nutri["display_name"],
                "confidence": round(conf, 4),
                "gate_active": gate_active,
                "bbox": [x_min, y_min, x_max, y_max],
                "area_px": int(m_info["area_px"]),
                "area_cm2": round(area_cm2, 2),
                "mean_height_cm": round(height_cm, 2),
                "v_geo_cm3": round(v_geo, 2),
                "v_depth_cm3": round(v_depth, 2),
                "v_fused_cm3": round(v_fused, 2),
                "depth_contrast_q": round(q_contrast, 3),
                "s_depth": round(s_depth, 3),
                "prior_group": prior_group,
                "delta_c": round(delta_c, 2),
                "delta_gated": round(delta_gated, 2),
                "alpha": round(alpha, 3),
                "density_g_cm3": round(density, 2),
                "usda_ndb_no": nutri["usda_ndb_no"],
                "energy_kcal_per_100g": round(kcal_per_100g, 1),
                "mass_g": round(mass_g, 2),
                "energy_kcal": round(energy_kcal, 2),
                "color_hex": "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0]),
            })

            if return_visuals:
                colored = np.zeros_like(seg_overlay, dtype=np.uint8)
                colored[mask] = color
                seg_overlay = cv2.addWeighted(seg_overlay, 1.0, colored, 0.38, 0)
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(seg_overlay, contours, -1, color, 2)
                cx = int(xs.mean())
                cy = int(ys.mean())
                label_txt = f"#{item_id} {nutri['display_name']} ({conf*100:.0f}%)"
                (tw, th), _ = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
                lx = max(4, min(seg_overlay.shape[1] - tw - 10, cx - tw // 2))
                ly = max(th + 10, min(seg_overlay.shape[0] - 6, cy))
                cv2.rectangle(seg_overlay, (lx - 4, ly - th - 6), (lx + tw + 4, ly + 4), color, -1)
                cv2.putText(seg_overlay, label_txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

            if len(items) >= int(max_items):
                break

        t_stage356 = time.perf_counter() - t356
        total_mass_g = sum(it["mass_g"] for it in items)
        total_energy_kcal = sum(it["energy_kcal"] for it in items)
        total_volume_cm3 = sum(it["v_fused_cm3"] for it in items)
        total_elapsed = time.perf_counter() - t0

        result = {
            "runtime": self.get_runtime_info(),
            "image_size": {"orig_w": orig_w, "orig_h": orig_h, "work_w": warped.shape[1], "work_h": warped.shape[0]},
            "stages": {
                "stage1_scale_recovery": {
                    "scale_px_per_cm": round(scale_px_per_cm, 3),
                    "scale_source": scale_source,
                    "elapsed_s": round(t_stage1, 3),
                },
                "stage2_sam2_segmentation": {
                    "clahe_applied": clahe_applied,
                    "raw_proposals": len(raw_masks),
                    "retained_instances": len(kept_masks),
                    "elapsed_s": round(t_stage2, 3),
                },
                "stage3_recognition": {
                    "backbone": "ResNet50 (10 NutriQuant-DB classes)",
                    "confidence_gate": 0.70,
                    "detected_items": len(items),
                },
                "stage4_depth_anything_v2": {
                    "encoder": self.encoder,
                    "rim_height_ref_cm": 2.0,
                    "elapsed_s": round(t_depth_infer, 3),
                },
                "stage5_dkdf_fusion": {
                    "contrast_lambda": 5.5,
                    "formula": "alpha = clip(s_depth + delta_tilde, 0, 1); V_hat = alpha*V_geo + (1-alpha)*V_depth",
                },
                "stage6_nutrition_totals": {
                    "total_volume_cm3": round(total_volume_cm3, 2),
                    "total_mass_g": round(total_mass_g, 2),
                    "total_energy_kcal": round(total_energy_kcal, 2),
                    "post_depth_elapsed_s": round(t_stage356, 3),
                },
            },
            "items": items,
            "total_elapsed_s": round(total_elapsed, 3),
        }

        if return_visuals:
            depth_color = _colorize_depth(depth_map)
            result["visuals"] = {
                "rectified_bgr": warped,
                "segmentation_bgr": seg_overlay,
                "depth_bgr": depth_color,
            }

        return result
