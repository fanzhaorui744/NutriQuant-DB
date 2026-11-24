import numpy as np

# DKDF category-level shape priors (Table 3 in NutriQuant manuscript)
GEOMETRY_PRIOR_CLASSES = {"tofu", "qiezi", "jirou"}   # delta_c = +0.3
DEPTH_PRIOR_CLASSES = {"lajiao", "qingcai", "xilanhua"}  # delta_c = -0.2
NEUTRAL_PRIOR_CLASSES = {"rice", "tudou", "egg", "tudousi"}  # delta_c = 0.0

DEFAULT_CONFIDENCE_GATE = 0.70
DEFAULT_CONTRAST_LAMBDA = 5.5


def get_category_shape_prior(class_name):
    """Return raw category shape prior delta_c and group description."""
    c = str(class_name or "").lower().strip()
    if c in GEOMETRY_PRIOR_CLASSES:
        return 0.3, "geometry-friendly"
    if c in DEPTH_PRIOR_CLASSES:
        return -0.2, "depth-favored"
    return 0.0, "neutral"


def calculate_image_stddev(image):
    import cv2
    if image is None:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.meanStdDev(gray)[1][0][0])


def enhance_low_contrast(image, stddev_threshold=30.0, clip_limit=3.0):
    """Apply CLAHE on luminance channel when image contrast stddev < 30.0."""
    import cv2
    stddev = calculate_image_stddev(image)
    if not np.isfinite(stddev) or stddev >= float(stddev_threshold):
        return image

    if len(image.shape) == 3:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(image)


def estimate_ellipsoid_volume(mask, pixel_per_cm, return_details: bool = True):
    """
    Stage 5 (Geometry branch): Compact ellipsoidal proxy V_geo = (4/3)*pi*a*b^2.
    """
    import cv2

    if mask is None:
        return None
    try:
        s = float(pixel_per_cm)
    except Exception:
        return None
    if not np.isfinite(s) or s <= 1e-6:
        return None

    mask_u8 = (np.asarray(mask).astype(np.uint8) * 255)
    if mask_u8.ndim != 2:
        return None

    area_px = float(np.sum(mask_u8 > 0))
    if area_px < 25:
        return None

    contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if c is None or len(c) < 5:
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        major_px = float(max(w, h))
        minor_px = float(min(w, h))
    else:
        try:
            (_, _), (ma, mi), _ = cv2.fitEllipse(c)
            major_px = float(max(ma, mi))
            minor_px = float(min(ma, mi))
        except Exception:
            rect = cv2.minAreaRect(c)
            (w, h) = rect[1]
            major_px = float(max(w, h))
            minor_px = float(min(w, h))

    if not np.isfinite(major_px) or not np.isfinite(minor_px) or major_px <= 1.0 or minor_px <= 1.0:
        return None

    major_cm = major_px / s
    minor_cm = minor_px / s

    a = 0.5 * major_cm
    b = 0.5 * minor_cm
    c_axis = float(b)

    if a <= 0 or b <= 0 or c_axis <= 0:
        return None

    volume_cm3 = float((4.0 / 3.0) * np.pi * a * b * c_axis)
    area_cm2 = float(area_px / (s * s))
    height_cm = float(2.0 * c_axis)

    if not return_details:
        return volume_cm3

    return {
        "volume_cm3": float(max(0.0, volume_cm3)),
        "height_cm": float(max(0.0, height_cm)),
        "height_rel": 0.0,
        "area_cm2": float(max(0.0, area_cm2)),
        "depth_contrast": 2.0,
        "ellipsoid_a_cm": float(a),
        "ellipsoid_b_cm": float(b),
        "ellipsoid_c_cm": float(c_axis),
    }


def _estimate_height_rel(mask, depth_map, bg_exclusion_px: int = 40):
    """
    Stage 4: Sample local background response outside a boundary exclusion margin
    (up to 40 px band) and compute relative food-to-background height and contrast q.
    """
    import cv2

    mask_u8 = mask.astype(np.uint8)
    if mask_u8.ndim != 2:
        return None

    # Local plate-background ring outside mask (avoiding boundary spill)
    kernel = np.ones((5, 5), np.uint8)
    dilated_outer = cv2.dilate(mask_u8, kernel, iterations=8)
    dilated_inner = cv2.dilate(mask_u8, kernel, iterations=2)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    ring = (dilated_outer - dilated_inner).astype(np.uint8)

    bg_depths = depth_map[ring == 1]
    if bg_depths.size < 10:
        # Fallback to outside dilated mask
        bg_depths = depth_map[dilated_inner == 0]
    if bg_depths.size < 10:
        bg_depths = depth_map[mask_u8 == 0]
    if bg_depths.size < 10:
        return None

    bg_depths = bg_depths[np.isfinite(bg_depths)]
    if bg_depths.size < 10:
        return None

    bg_depth = float(np.percentile(bg_depths, 10))
    std_bg = float(np.std(bg_depths))

    if eroded.sum() > 10:
        food_depths = depth_map[eroded.astype(bool)]
    else:
        food_depths = depth_map[mask.astype(bool)]

    food_depths = food_depths[np.isfinite(food_depths)]
    if food_depths.size < 10:
        return None

    mean_food = float(np.mean(food_depths))

    delta = food_depths - bg_depth
    heights_a = np.clip(delta, 0, None)
    heights_b = np.clip(-delta, 0, None)

    heights = heights_b if np.mean(heights_b) > np.mean(heights_a) else heights_a
    heights = heights[np.isfinite(heights)]
    heights = heights[heights >= 0]
    if heights.size < 10:
        return None

    p80 = float(np.percentile(heights, 80))
    top = heights[heights >= p80]
    if top.size < 10:
        top = heights

    height_rel = float(np.mean(top))
    mean_all_height = float(np.mean(heights))
    contrast = float(abs(mean_food - bg_depth) / (std_bg + 1e-6))
    return {
        "height_rel": height_rel,
        "mean_height": mean_all_height,
        "bg_depth": bg_depth,
        "mean_food_depth": mean_food,
        "std_bg_depth": std_bg,
        "depth_contrast": contrast,
    }


def calculate_volume(mask, depth_map, pixel_per_cm, depth_scale_factor=1.0, return_details=False):
    """
    Stage 4 (Depth branch): V_depth = sum_{(i,j) in mask} h_abs(i,j) * s^{-2}.
    """
    area_pixels = np.sum(mask)
    area_cm2 = area_pixels / (pixel_per_cm ** 2)

    hinfo = _estimate_height_rel(mask, depth_map)
    if hinfo is None:
        if return_details:
            return {
                "volume_cm3": 0.0,
                "height_cm": 0.0,
                "mean_height_cm": 0.0,
                "height_rel": 0.0,
                "area_cm2": float(area_cm2),
                "depth_contrast": 0.0,
            }
        return 0.0, 0.0

    mean_height = float(hinfo.get("mean_height", hinfo["height_rel"]))
    height_rel = float(hinfo["height_rel"])

    height_cm = mean_height * float(depth_scale_factor)
    volume_cm3 = float(area_cm2) * float(height_cm)
    peak_height_cm = height_rel * float(depth_scale_factor)

    if return_details:
        out = {
            "volume_cm3": float(volume_cm3),
            "height_cm": float(peak_height_cm),
            "mean_height_cm": float(height_cm),
            "height_rel": float(height_rel),
            "area_cm2": float(area_cm2),
        }
        out.update(hinfo)
        return out

    return float(volume_cm3), float(peak_height_cm)


def calculate_adaptive_volume(
    mask,
    depth_map,
    pixel_per_cm,
    depth_scale_factor=1.0,
    class_name=None,
    confidence=1.0,
    return_details=False,
    ablation_mode=None,
    contrast_lambda: float = DEFAULT_CONTRAST_LAMBDA,
    confidence_gate: float = DEFAULT_CONFIDENCE_GATE,
):
    """
    Stage 5: Data-Knowledge-Driven Adaptive Fusion (DKDF).
      q = |mean(D_food) - D_bg| / (sigma_bg + eps)
      s_depth = clip(q, 0, lambda) / lambda           (lambda = 5.5)
      delta_tilde = delta_c if p_c >= 0.70 else 0.0
      alpha = clip(s_depth + delta_tilde, 0, 1)
      V_hat = alpha * V_geo + (1 - alpha) * V_depth
    """
    dl_res = calculate_volume(mask, depth_map, pixel_per_cm, depth_scale_factor, return_details=True)
    dl_vol = float(dl_res.get("volume_cm3", 0.0))
    depth_contrast = float(dl_res.get("depth_contrast", 0.0))

    rule_res = estimate_ellipsoid_volume(mask, pixel_per_cm, return_details=True)
    rule_vol = float(rule_res.get("volume_cm3", 0.0)) if rule_res else dl_vol

    lam = max(float(contrast_lambda), 1e-6)
    s_depth = float(np.clip(depth_contrast, 0.0, lam) / lam)

    raw_delta, prior_group = get_category_shape_prior(class_name)
    conf_val = float(confidence) if confidence is not None and np.isfinite(float(confidence)) else 0.0
    gate_active = conf_val >= float(confidence_gate)
    delta_gated = raw_delta if gate_active else 0.0

    forced_mode = False
    if ablation_mode in ("pure_dl", "alpha_0"):
        alpha = 0.0
        forced_mode = True
    elif ablation_mode in ("pure_geo", "alpha_1"):
        alpha = 1.0
        forced_mode = True
    elif ablation_mode in ("fixed_0.5", "linear_0.5"):
        alpha = 0.5
        forced_mode = True
    elif ablation_mode == "ignore_class":
        alpha = s_depth
        delta_gated = 0.0
    elif ablation_mode == "ignore_contrast":
        alpha = float(np.clip(0.5 + delta_gated, 0.0, 1.0))
    else:
        alpha = float(np.clip(s_depth + delta_gated, 0.0, 1.0))

    if forced_mode:
        delta_gated = 0.0

    final_vol = alpha * rule_vol + (1.0 - alpha) * dl_vol

    if return_details:
        out = dl_res.copy()
        out["volume_cm3"] = float(final_vol)
        out["dl_volume_cm3"] = float(dl_vol)
        out["rule_volume_cm3"] = float(rule_vol)
        out["uncertainty_weight"] = float(alpha)
        out["alpha"] = float(alpha)
        out["s_depth"] = float(s_depth)
        out["lambda"] = float(lam)
        out["delta_c"] = float(raw_delta)
        out["delta_gated"] = float(delta_gated)
        out["gate_active"] = bool(gate_active)
        out["prior_group"] = prior_group
        out["method"] = "dkdf"
        out["ablation_mode"] = ablation_mode
        return out

    return float(final_vol), float(dl_res.get("height_cm", 0.0))
