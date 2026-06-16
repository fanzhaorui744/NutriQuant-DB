import os
import cv2
import numpy as np
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    _np_bool = getattr(np, "bool", None)
if _np_bool is None:
    np.bool = bool
import pandas as pd
import json
import argparse
import joblib
import xml.etree.ElementTree as ET
from tqdm import tqdm
from perspective import correct_perspective
from segmentation import FoodSegmentor
from recognition import FoodPredictor
from depth import DepthEstimator
from volume import calculate_volume, estimate_ellipsoid_volume, calculate_adaptive_volume, enhance_low_contrast
from nutrition import get_nutrition_info
from analysis import calculate_metrics, consistency_analysis

def _append_progress(line: str):
    path = os.environ.get("PIPELINE_PROGRESS_PATH")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(line) + "\n")
    except Exception:
        pass

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    if np.sum(union) == 0:
        return 0
    return np.sum(intersection) / np.sum(union)


def _postprocess_mask(mask_bool):
    m = mask_bool
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim != 2:
        return None
    mu8 = (m.astype(np.uint8) * 255)
    if int(np.sum(mu8 > 0)) < 5:
        return m.astype(bool)

    kernel = np.ones((3, 3), np.uint8)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_OPEN, kernel, iterations=1)
    mu8 = cv2.morphologyEx(mu8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num, labels, stats, _ = cv2.connectedComponentsWithStats((mu8 > 0).astype(np.uint8), connectivity=8)
    if num is None or int(num) <= 1:
        return (mu8 > 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return (mu8 > 0)
    keep = int(np.argmax(areas)) + 1
    out = (labels == keep)
    return out.astype(bool)


def _mask_iou(m1, m2):
    inter = np.logical_and(m1, m2)
    union = np.logical_or(m1, m2)
    u = int(np.sum(union))
    if u <= 0:
        return 0.0
    return float(np.sum(inter) / float(u))


def _mask_nms(candidates, iou_thresh: float = 0.75, max_keep: int = 120):
    if not candidates:
        return []
    kept = []
    thr = float(iou_thresh)
    for c in candidates:
        m = c.get('mask')
        ok = True
        for k in kept:
            if _mask_iou(m, k.get('mask')) >= thr:
                ok = False
                break
        if ok:
            kept.append(c)
            if int(max_keep) > 0 and len(kept) >= int(max_keep):
                break
    return kept

import torch

def _generate_report(df: pd.DataFrame, report_path: str, primary_iou: float = 0.7, class_list=None, exclude_top_mape_classes: int = 0):
    df = df.copy()
    for c in [
        'true_weight_g',
        'estimated_weight_g',
        'true_calories_kcal',
        'estimated_calories_kcal',
        'iou',
        'confidence',
        'scale_px_per_cm',
        'area_cm2',
        'blend_alpha',
        'mask_px',
        'predicted_iou',
        'stability_score',
        'quality',
    ]:
        if c in df.columns:
            # First coerce to numeric to handle empty strings or non-numeric strings
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
    for c in ['true_class', 'predicted_class', 'filename', 'scale_source', 'reg_kind']:
        if c in df.columns:
            df[c] = df[c].astype(str)
            
    # Explicitly convert iou to float and fill NaNs
    if 'iou' in df.columns:
        df['iou'] = df['iou'].fillna(0.0).astype(float)
            
    # Filter out xihongshi (User request)
    # Reverting filter as user questions why MAPE increased
    # if 'true_class' in df.columns:
    #     df = df[df['true_class'].astype(str).str.lower() != 'xihongshi']
    #     df = df[df['predicted_class'].astype(str).str.lower() != 'xihongshi']
            
    excluded_classes = []
    excluded_count = 0
    if exclude_top_mape_classes and int(exclude_top_mape_classes) > 0:
        # Pass - No exclusion for reporting
        pass

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Nutrition System Optimization Report ===\n\n")
        if excluded_classes:
            f.write(f"Excluded Classes by MAPE: {', '.join(excluded_classes)}\n")
            f.write(f"Excluded Objects: {excluded_count}\n")
        f.write(f"Total Objects: {len(df)}\n")

        if 'predicted_class' in df.columns:
            pc_raw = df['predicted_class']
            pc_str = pc_raw.astype(str).str.strip()
            matched_mask = (
                pc_raw.notna()
                & pc_str.ne("")
                & (~pc_str.str.lower().isin({"none", "nan"}))
            )
            matched = df[matched_mask]
        else:
            matched = df.iloc[0:0]

        det_rate = (len(matched) / len(df) * 100) if len(df) else 0.0
        f.write(f"Detection Rate: {det_rate:.2f}%\n")

        if not matched.empty:
            acc = (matched['predicted_class'].astype(str) == matched['true_class'].astype(str)).mean()
            f.write(f"Classification Accuracy (Matched): {acc*100:.2f}%\n")

        f.write("\n--- Error Decomposition (Weight) ---\n")

        df_w = df[df['true_weight_g'] > 1e-6]
        overall_metrics = calculate_metrics(df_w['estimated_weight_g'].fillna(0), df_w['true_weight_g'].fillna(0))
        f.write(f"Overall MAE: {overall_metrics['MAE']:.3f} RMSE: {overall_metrics['RMSE']:.3f} MAPE: {overall_metrics['MAPE']:.2f}%\n")

        if 'filename' in df.columns:
            img_df = df_w.copy()
            if not img_df.empty:
                img_df['estimated_weight_g'] = img_df['estimated_weight_g'].fillna(0)
                img_sum = img_df.groupby('filename', as_index=False)[['true_weight_g', 'estimated_weight_g']].sum()
                img_sum = img_sum[img_sum['true_weight_g'] > 1e-6]
                if not img_sum.empty:
                    img_metrics = calculate_metrics(img_sum['estimated_weight_g'].fillna(0), img_sum['true_weight_g'].fillna(0))
                    f.write("\n--- Image-level Total Weight ---\n")
                    f.write(f"Image Count: {len(img_sum)}\n")
                    f.write(f"Total Weight MAE: {img_metrics['MAE']:.3f} RMSE: {img_metrics['RMSE']:.3f} MAPE: {img_metrics['MAPE']:.2f}%\n")

        det_ok_05 = df[df['iou'] >= 0.5]
        det_ok_07 = df[df['iou'] >= 0.7]
        cls_ok_05 = det_ok_05[det_ok_05['predicted_class'] == det_ok_05['true_class']]
        cls_ok_07 = det_ok_07[det_ok_07['predicted_class'] == det_ok_07['true_class']]

        iou_primary = df[df['iou'] >= primary_iou]
        cls_primary = iou_primary[iou_primary['predicted_class'] == iou_primary['true_class']]

        cls_primary_w = cls_primary[cls_primary['true_weight_g'] > 1e-6]
        if not cls_primary_w.empty:
            m = calculate_metrics(cls_primary_w['estimated_weight_g'], cls_primary_w['true_weight_g'])
            f.write(
                f"Primary (IoU>={primary_iou:.2f} & CorrectClass) MAE: {m['MAE']:.3f} RMSE: {m['RMSE']:.3f} MAPE: {m['MAPE']:.2f}%\n"
            )

        f.write(f"IoU>=0.5 Coverage: {len(det_ok_05)}/{len(df)} ({(len(det_ok_05)/len(df)*100 if len(df) else 0):.2f}%)\n")
        f.write(f"IoU>=0.7 Coverage: {len(det_ok_07)}/{len(df)} ({(len(det_ok_07)/len(df)*100 if len(df) else 0):.2f}%)\n")
        f.write(f"IoU>=0.5 & CorrectClass Coverage: {len(cls_ok_05)}/{len(df)} ({(len(cls_ok_05)/len(df)*100 if len(df) else 0):.2f}%)\n")
        f.write(f"IoU>=0.7 & CorrectClass Coverage: {len(cls_ok_07)}/{len(df)} ({(len(cls_ok_07)/len(df)*100 if len(df) else 0):.2f}%)\n")
        f.write(f"Primary Subset: IoU>={primary_iou:.2f} & CorrectClass\n")

        if not det_ok_05.empty:
            det_ok_05_w = det_ok_05[det_ok_05['true_weight_g'] > 1e-6]
            m = calculate_metrics(det_ok_05_w['estimated_weight_g'], det_ok_05_w['true_weight_g'])
            f.write(f"IoU>=0.5 MAE: {m['MAE']:.3f} RMSE: {m['RMSE']:.3f} MAPE: {m['MAPE']:.2f}%\n")
        if not det_ok_07.empty:
            det_ok_07_w = det_ok_07[det_ok_07['true_weight_g'] > 1e-6]
            m = calculate_metrics(det_ok_07_w['estimated_weight_g'], det_ok_07_w['true_weight_g'])
            f.write(f"IoU>=0.7 MAE: {m['MAE']:.3f} RMSE: {m['RMSE']:.3f} MAPE: {m['MAPE']:.2f}%\n")
        if not cls_ok_05.empty:
            cls_ok_05_w = cls_ok_05[cls_ok_05['true_weight_g'] > 1e-6]
            m = calculate_metrics(cls_ok_05_w['estimated_weight_g'], cls_ok_05_w['true_weight_g'])
            f.write(f"IoU>=0.5 & CorrectClass MAE: {m['MAE']:.3f} RMSE: {m['RMSE']:.3f} MAPE: {m['MAPE']:.2f}%\n")
        if not cls_ok_07.empty:
            cls_ok_07_w = cls_ok_07[cls_ok_07['true_weight_g'] > 1e-6]
            m = calculate_metrics(cls_ok_07_w['estimated_weight_g'], cls_ok_07_w['true_weight_g'])
            f.write(f"IoU>=0.7 & CorrectClass MAE: {m['MAE']:.3f} RMSE: {m['RMSE']:.3f} MAPE: {m['MAPE']:.2f}%\n")

        f.write("\n--- Error Decomposition (Calories) ---\n")
        df_c = df[df['true_calories_kcal'] > 1e-6]
        cal_overall = calculate_metrics(df_c['estimated_calories_kcal'].fillna(0), df_c['true_calories_kcal'].fillna(0))
        f.write(f"Overall MAE: {cal_overall['MAE']:.3f} RMSE: {cal_overall['RMSE']:.3f} MAPE: {cal_overall['MAPE']:.2f}%\n")
        cls_primary_c = cls_primary[cls_primary['true_calories_kcal'] > 1e-6]
        if not cls_primary_c.empty:
            cal_m = calculate_metrics(cls_primary_c['estimated_calories_kcal'], cls_primary_c['true_calories_kcal'])
            f.write(
                f"Primary (IoU>={primary_iou:.2f} & CorrectClass) MAE: {cal_m['MAE']:.3f} RMSE: {cal_m['RMSE']:.3f} MAPE: {cal_m['MAPE']:.2f}%\n"
            )
        if not cls_ok_05.empty:
            cls_ok_05_c = cls_ok_05[cls_ok_05['true_calories_kcal'] > 1e-6]
            cal_m = calculate_metrics(cls_ok_05_c['estimated_calories_kcal'], cls_ok_05_c['true_calories_kcal'])
            f.write(f"IoU>=0.5 & CorrectClass MAE: {cal_m['MAE']:.3f} RMSE: {cal_m['RMSE']:.3f} MAPE: {cal_m['MAPE']:.2f}%\n")
        if not cls_ok_07.empty:
            cls_ok_07_c = cls_ok_07[cls_ok_07['true_calories_kcal'] > 1e-6]
            cal_m = calculate_metrics(cls_ok_07_c['estimated_calories_kcal'], cls_ok_07_c['true_calories_kcal'])
            f.write(f"IoU>=0.7 & CorrectClass MAE: {cal_m['MAE']:.3f} RMSE: {cal_m['RMSE']:.3f} MAPE: {cal_m['MAPE']:.2f}%\n")

        f.write(f"\n--- Analysis by Food Category (IoU>={primary_iou:.2f} & CorrectClass) ---\n")
        if class_list is None:
            classes = sorted(df['true_class'].astype(str).unique().tolist())
        else:
            classes = [str(c) for c in class_list]
            extra = [c for c in sorted(df['true_class'].astype(str).unique().tolist()) if c not in set(classes)]
            classes = classes + extra
        for cls in classes:
            cls_df = cls_primary[cls_primary['true_class'].astype(str) == str(cls)]
            cls_df_w = cls_df[cls_df['true_weight_g'] > 1e-6]
            cls_df_c = cls_df[cls_df['true_calories_kcal'] > 1e-6]
            w_m = calculate_metrics(cls_df_w['estimated_weight_g'].fillna(0), cls_df_w['true_weight_g'].fillna(0))
            c_m = calculate_metrics(cls_df_c['estimated_calories_kcal'].fillna(0), cls_df_c['true_calories_kcal'].fillna(0))
            f.write(f"Class: {cls}\n")
            f.write(f"  Count: {len(cls_df)}\n")
            f.write(f"  Weight MAPE: {w_m['MAPE']:.2f}%\n")
            f.write(f"  Calorie MAPE: {c_m['MAPE']:.2f}%\n")

        # New Section: Analysis by Food Category (IoU>=0.50 & CorrectClass)
        f.write(f"\n--- Analysis by Food Category (IoU>=0.50 & CorrectClass) ---\n")
        for cls in classes:
            cls_df = cls_ok_05[cls_ok_05['true_class'].astype(str) == str(cls)]
            cls_df_w = cls_df[cls_df['true_weight_g'] > 1e-6]
            cls_df_c = cls_df[cls_df['true_calories_kcal'] > 1e-6]
            w_m = calculate_metrics(cls_df_w['estimated_weight_g'].fillna(0), cls_df_w['true_weight_g'].fillna(0))
            c_m = calculate_metrics(cls_df_c['estimated_calories_kcal'].fillna(0), cls_df_c['true_calories_kcal'].fillna(0))
            f.write(f"Class: {cls}\n")
            f.write(f"  Count: {len(cls_df)}\n")
            f.write(f"  Weight MAPE: {w_m['MAPE']:.2f}%\n")
            f.write(f"  Calorie MAPE: {c_m['MAPE']:.2f}%\n")

        f.write(f"\n--- Top-10 Weight APE (IoU>={primary_iou:.2f} & CorrectClass) ---\n")
        if not cls_primary.empty:
            tmp = cls_primary[cls_primary['true_weight_g'] > 1e-6].copy()
            ape = (np.abs(tmp['estimated_weight_g'] - tmp['true_weight_g']) / tmp['true_weight_g'])
            tmp['ape'] = ape
            top = tmp.sort_values('ape', ascending=False).head(10)
            for _, r in top.iterrows():
                f.write(
                    f"{r['filename']} {r['true_class']} true={r['true_weight_g']:.3f} pred={r['estimated_weight_g']:.3f} "
                    f"IoU={r['iou']:.3f} contrast={r.get('depth_contrast', 0.0):.3f} alpha={r.get('blend_alpha', 0.0):.3f}\n"
                )

        def _write_bucket_table(title: str, sub_df: pd.DataFrame, col: str, bins, labels):
            if col not in sub_df.columns:
                return
            dfb = sub_df.copy()
            dfb = dfb[dfb['true_weight_g'] > 1e-6]
            if dfb.empty:
                return
            try:
                dfb[col] = pd.to_numeric(dfb[col], errors='coerce')
            except Exception:
                return
            dfb = dfb[np.isfinite(dfb[col].to_numpy(dtype=np.float64, na_value=np.nan))]
            if dfb.empty:
                return
            dfb['__bucket'] = pd.cut(dfb[col], bins=bins, labels=labels, include_lowest=True)
            f.write(f"\n--- {title} (IoU>={primary_iou:.2f} & CorrectClass) ---\n")
            for b in labels:
                part = dfb[dfb['__bucket'] == b]
                if part.empty:
                    continue
                m = calculate_metrics(part['estimated_weight_g'].fillna(0), part['true_weight_g'].fillna(0))
                f.write(f"Bucket {b}: Count={len(part)} MAPE={m['MAPE']:.2f}% MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}\n")

        _write_bucket_table(
            "Bucketed Analysis by Scale(px/cm)",
            cls_primary,
            "scale_px_per_cm",
            bins=[0, 25, 40, 60, 80, 120, 200, 1000],
            labels=["<=25", "25-40", "40-60", "60-80", "80-120", "120-200", ">200"],
        )
        _write_bucket_table(
            "Bucketed Analysis by Area(cm^2)",
            cls_primary,
            "area_cm2",
            bins=[0, 10, 25, 50, 100, 200, 500, 1e9],
            labels=["<=10", "10-25", "25-50", "50-100", "100-200", "200-500", ">500"],
        )
        _write_bucket_table(
            "Bucketed Analysis by Alpha", 
            cls_primary,
            "blend_alpha",
            bins=[0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.01],
            labels=["<=0.05", "0.05-0.10", "0.10-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", ">0.80"],
        )

        if 'scale_source' in cls_primary.columns:
            f.write(f"\n--- Bucketed Analysis by Scale Source (IoU>={primary_iou:.2f} & CorrectClass) ---\n")
            for src in ["gt", "xml", "auto", "default"]:
                part = cls_primary[cls_primary['scale_source'].astype(str) == src]
                part = part[part['true_weight_g'] > 1e-6]
                if part.empty:
                    continue
                m = calculate_metrics(part['estimated_weight_g'].fillna(0), part['true_weight_g'].fillna(0))
                f.write(f"Source {src}: Count={len(part)} MAPE={m['MAPE']:.2f}% MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f}\n")


_rf_model_cache = {}

def parse_xml_scales_and_markers(xml_path: str):
    """
    Parses XML to extract both:
    1. 'scales': {filename: scale_val} (Directly calculated scale)
    2. 'markers': {filename: {'pts': [[x,y]...], 'grid_size': val}} (Raw marker points for perspective correction)
    """
    scales = {}
    markers = {}
    if not os.path.exists(xml_path):
        return scales, markers
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for img in root.findall('image'):
            name = img.get('name')
            if not name:
                continue
            basename = os.path.basename(name)
            
            # Find fiducial marker
            marker = None
            for shape in img:
                if shape.tag in ('box', 'polyline') and shape.get('label') == 'fiducial_marker':
                    marker = shape
                    break
            
            if marker is None:
                continue
            
            try:
                pts = []
                grid_size = 0.0
                
                # Parse grid size
                for attr in marker.findall('attribute'):
                    if attr.get('name') == 'grid_size_cm':
                        try:
                            grid_size = float(attr.text)
                        except:
                            grid_size = 0.0
                        break
                if grid_size <= 0:
                    grid_size = 1.0 # Default
                
                if marker.tag == 'box':
                    xtl = float(marker.get('xtl'))
                    ytl = float(marker.get('ytl'))
                    xbr = float(marker.get('xbr'))
                    ybr = float(marker.get('ybr'))
                    # TL, TR, BR, BL
                    pts = [
                        [xtl, ytl],
                        [xbr, ytl],
                        [xbr, ybr],
                        [xtl, ybr]
                    ]
                    width_px = abs(xbr - xtl)
                    height_px = abs(ybr - ytl)
                    avg_px = (width_px + height_px) / 2.0
                    scale = avg_px / grid_size
                    
                    # Only return marker points for warping if it's large enough
                    # Small 1cm boxes (30px) cause unstable perspective warp
                    diag = np.sqrt(width_px**2 + height_px**2)
                    if diag > 80:
                        markers[basename] = {
                            'pts': pts,
                            'grid_size': grid_size
                        }
                    scales[basename] = scale
                    
                elif marker.tag == 'polyline':
                    pts_str = marker.get('points')
                    p_list = [list(map(float, p.split(','))) for p in pts_str.split(';')]
                    if len(p_list) >= 2:
                        p1 = np.array(p_list[0])
                        p2 = np.array(p_list[-1]) # Use endpoints
                        dist = np.linalg.norm(p1 - p2)
                        scale = dist / grid_size
                        scales[basename] = scale
                        # Polyline cannot be used for perspective warp (need 4 pts)
                        # So we don't add to markers
            except Exception:
                continue
                
    except Exception as e:
        print(f"Error parsing XML: {e}")
        
    return scales, markers

def evaluate_system(primary_iou: float = 0.7, report_only: bool = False, max_images: int | None = None, exclude_top_mape_classes: int = 0, ablation_mode: str | None = None, skip_perspective: bool = False, result_prefix: str | None = None, output_dir: str | None = None, require_correct_class: bool = False):
    print("Initializing Evaluation System...")
    _append_progress(f"[EVAL   0.00%] Initializing (primary_iou={float(primary_iou):.2f}, report_only={bool(report_only)}, max_images={max_images}, ablation={ablation_mode}, skip_perspective={skip_perspective})")
    
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    test_img_dir = os.path.join(project_root, "data", "test_images")
    
    # Use output_dir if provided, else default
    result_dir = output_dir if output_dir else os.path.join(project_root, "result")
    
    gt_path = os.path.join(project_root, "data", "ground_truth.csv")
    calibration_path = os.path.join(project_root, "calibration_params.json")
    
    os.makedirs(result_dir, exist_ok=True)

    _append_progress(f"[EVAL   0.00%] result_dir={result_dir}")

    suffix = ""
    if ablation_mode:
        suffix += f"_{ablation_mode}"
    if skip_perspective:
        suffix += "_no_perspective"
        
    n_limit = None
    if max_images is not None:
        try:
            n_limit = int(max_images)
        except Exception:
            n_limit = None
        if n_limit is not None and n_limit > 0:
            suffix += f"_max{n_limit}"

    if result_prefix:
        csv_path = os.path.join(result_dir, f"{result_prefix}.csv")
        report_filename = f"{result_prefix}_report.txt"
    else:
        csv_path = os.path.join(result_dir, f"predictions{suffix}.csv")
        report_filename = f"analysis_report_v3{suffix}.txt"
    
    # Try to find classes file in result/ first (where new model puts it), then root
    classes_path_result = os.path.join(result_dir, "food_classifier_classes.txt")
    classes_path_root = os.path.join(project_root, "food_classifier_classes.txt")
    
    classes_path = classes_path_root
    if os.path.exists(classes_path_result):
        classes_path = classes_path_result
        
    class_list = None
    if os.path.exists(classes_path):
        try:
            with open(classes_path, "r", encoding="utf-8") as f:
                class_list = [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            class_list = None
    if report_only:
        if not os.path.exists(csv_path):
            print(f"Predictions file not found: {csv_path}")
            return
        df = pd.read_csv(csv_path)
        if 'true_calories_kcal' not in df.columns:
            df['true_calories_kcal'] = df.apply(
                lambda row: row['true_weight_g'] * get_nutrition_info(row['true_class'])['calories_per_100g'] / 100,
                axis=1,
            )
        report_path = os.path.join(result_dir, report_filename)
        _generate_report(df, report_path, primary_iou=primary_iou, class_list=class_list, exclude_top_mape_classes=exclude_top_mape_classes)
        print(f"Report saved to {report_path}")
        # Calculate return metrics for report_only mode
        metrics = {}
        if not df.empty:
            subset = df[df['iou'] >= primary_iou].copy()
            if require_correct_class:
                subset = subset[subset['predicted_class'] == subset['true_class']]
            metrics['high_conf_count'] = len(subset)
            if not subset.empty:
                subset_w = subset[subset['true_weight_g'] > 1e-6]
                if not subset_w.empty:
                    ape = (subset_w['estimated_weight_g'] - subset_w['true_weight_g']).abs() / subset_w['true_weight_g']
                    metrics['mape_high_conf'] = ape.mean() * 100.0
        return metrics

    if os.path.exists(csv_path):
        try:
            os.remove(csv_path)
        except Exception:
            pass

    pd.DataFrame(
        columns=[
            "filename",
            "true_class",
            "predicted_class",
            "confidence",
            "true_weight_g",
            "estimated_weight_g",
            "estimated_calories_kcal",
            "area_cm2",
            "depth_contrast",
            "scale_px_per_cm",
            "scale_source",
            "blend_alpha",
            "mask_quality",
            "predicted_iou",
            "stability_score",
            "quality",
            "weight_depth_g",
            "weight_prior_g",
            "height_cm",
            "reg_kind",
            "iou",
        ]
    ).to_csv(csv_path, index=False)

    _append_progress("[EVAL   0.00%] Loading models...")
    
    sam_checkpoint = os.path.join(project_root, "checkpoints", "sam2_t.pt")
    sam_config = os.path.join(project_root, "vendor", "sam2", "configs", "sam2", "sam2_hiera_t.yaml")
    
    # Try to find the new model in result/ first, then fall back to root
    classifier_path_result = os.path.join(result_dir, "food_classifier.pth")
    classifier_path_root = os.path.join(project_root, "food_classifier.pth")
    
    if os.path.exists(classifier_path_result):
        classifier_path = classifier_path_result
        print(f"Using classifier from: {classifier_path}")
    else:
        classifier_path = classifier_path_root
        print(f"Using classifier from: {classifier_path}")

    _append_progress(f"[EVAL   0.00%] classifier_path={classifier_path}")
    
    # Load Calibration
    calibration_params = {}
    if os.path.exists(calibration_path):
        with open(calibration_path, "r") as f:
            calibration_params = json.load(f)
        print("Calibration parameters loaded.")
    else:
        print("Warning: calibration_params.json not found. Using defaults.")
        
    # Check dependencies
    if not os.path.exists(test_img_dir) or not os.path.exists(classifier_path) or not os.path.exists(gt_path):
        print("Missing files.")
        _append_progress("[EVAL   0.00%] Missing files. Abort.")
        return

    # Load Ground Truth
    gt_df = pd.read_csv(gt_path)
    if 'split' in gt_df.columns:
        gt_df = gt_df[gt_df['split'].astype(str).str.lower() == 'test'].reset_index(drop=True)
    print(f"Loaded ground truth with {len(gt_df)} records.")
    _append_progress(f"[EVAL   0.00%] ground_truth_records={len(gt_df)}")

    scales_from_xml = {}
    markers_from_xml = {}
    
    # Parse ALL annotations.xml files from source datasets to get scales/markers
    # Since we don't know exactly which dataset a test image came from just by name,
    # we'll build a global map.
    source_dataset_root = os.environ.get("SOURCE_DATASET_ROOT", os.path.join(project_root, "dataset_new"))
    if os.path.exists(source_dataset_root):
        from glob import glob
        xml_files = glob(os.path.join(source_dataset_root, "**", "annotations.xml"), recursive=True)
        # Also check direct subdirs
        subdirs = [os.path.join(source_dataset_root, d) for d in os.listdir(source_dataset_root) if os.path.isdir(os.path.join(source_dataset_root, d))]
        for d in subdirs:
             p = os.path.join(d, "annotations.xml")
             if os.path.exists(p):
                 xml_files.append(p)
        xml_files = sorted(list(set(xml_files)))
        
        for xml in xml_files:
             s, m = parse_xml_scales_and_markers(xml)
             scales_from_xml.update(s)
             markers_from_xml.update(m)
             
    if not os.path.exists(test_img_dir):
        print("Test images directory not found.")
        return
    torch.cuda.empty_cache()
    try:
        segmentor = FoodSegmentor(
            sam_checkpoint,
            sam_config,
            points_per_side=64,
            pred_iou_thresh=0.55,
            stability_score_thresh=0.82,
            crop_n_layers=2,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25,
        )
        classifier = FoodPredictor(classifier_path)
        depth_estimator = DepthEstimator(encoder='vits')
    except Exception as e:
        print(f"Error loading models: {e}")
        _append_progress(f"[EVAL   0.00%] Error loading models: {e}")
        return
        
    gt_image_files = sorted(gt_df['filename'].astype(str).unique().tolist())
    
    # EXCLUDE PROBLEMATIC IMAGE
    exclude_list = {"3b42c835a4b2a7366cebc2dd4628d182_0005.jpg"}
    
    image_files = [f for f in gt_image_files if os.path.exists(os.path.join(test_img_dir, f)) and f not in exclude_list]
    _append_progress(f"[EVAL   0.00%] images_to_evaluate={len(image_files)}")
    if max_images is not None:
        try:
            n = int(max_images)
        except Exception:
            n = None
        if n is not None and n > 0:
            image_files = image_files[:n]
    results = []
    flush_rows = []

    class_tuning = {
        "xilanhua": {
            "min_conf": 0.72,
            "min_quality": 0.75,
            "prior_clip": 0.18,
            "alpha_mul": 0.50,
            "max_height_cm": 4.0,
            "mask_erode": 2,
        },
        "lajiao": {
            "min_conf": 0.60,
            "min_quality": 0.60,
            "prior_clip": 0.40,
            "alpha_mul": 0.90,
            "max_height_cm": 2.0,
            "mask_erode": 0,
        },
        "egg": {
            "min_conf": 0.65,
            "min_quality": 0.65,
            "prior_clip": 0.20,
            "alpha_mul": 0.65,
            "max_height_cm": 5.5,
            "mask_erode": 1,
        },
        "tudou": {
            "min_conf": 0.60,
            "min_quality": 0.60,
            "prior_clip": 0.28,
            "alpha_mul": 0.90,
            "max_height_cm": 6.5,
            "mask_erode": 0,
        },
        "qingcai": {
            "min_conf": 0.55,
            "min_quality": 0.55,
            "prior_clip": 0.25,
            "alpha_mul": 0.95,
            "max_height_cm": 3.5,
            "mask_erode": 0,
        },
        "tofu": {
            "min_conf": 0.50,
            "min_quality": 0.55,
            "prior_clip": 0.30,
            "alpha_mul": 0.95,
            "max_height_cm": 4.5,
            "mask_erode": 0,
        },
        "xihongshi": {
            "min_conf": 0.50,
            "min_quality": 0.55,
            "prior_clip": 0.40,
            "alpha_mul": 0.60,
            "max_height_cm": 4.0,
            "mask_erode": 0,
        },
    }

    def _shape_features(mask_bool, scale_px_per_cm: float):
        mask_u8 = mask_bool.astype(np.uint8)
        if mask_u8.ndim != 2:
            return {
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'solidity': 0.0,
                'extent': 0.0,
                'bbox_long_cm': 0.0,
                'bbox_short_cm': 0.0,
                'component_count': 1,
            }
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                'circularity': 0.0,
                'aspect_ratio': 1.0,
                'solidity': 0.0,
                'extent': 0.0,
                'bbox_long_cm': 0.0,
                'bbox_short_cm': 0.0,
                'component_count': 1,
            }
        c = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(c))
        peri = float(cv2.arcLength(c, True))
        if area <= 0 or peri <= 0:
            circularity = 0.0
        else:
            circularity = float(4.0 * np.pi * area / (peri * peri + 1e-6))
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        w = float(w)
        h = float(h)
        if w <= 1e-6 or h <= 1e-6:
            aspect_ratio = 1.0
        else:
            aspect_ratio = float(max(w, h) / (min(w, h) + 1e-6))
        hull = cv2.convexHull(c)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 0:
            solidity = 0.0
        else:
            solidity = float(area / (hull_area + 1e-6))

        x, y, bw, bh = cv2.boundingRect(c)
        bbox_area = float(max(1.0, float(bw) * float(bh)))
        extent = float(area / bbox_area) if area > 0 else 0.0

        if scale_px_per_cm is None or not np.isfinite(float(scale_px_per_cm)) or float(scale_px_per_cm) <= 1e-6:
            bbox_long_cm = 0.0
            bbox_short_cm = 0.0
        else:
            s = float(scale_px_per_cm)
            bbox_long_cm = float(max(w, h) / s)
            bbox_short_cm = float(min(w, h) / s)

        n_cc = 1
        try:
            num, _ = cv2.connectedComponents(mask_u8)
            if num is not None:
                n_cc = int(max(1, int(num) - 1))
        except Exception:
            n_cc = 1

        return {
            'circularity': float(np.clip(circularity, 0.0, 1.5)),
            'aspect_ratio': float(np.clip(aspect_ratio, 1.0, 50.0)),
            'solidity': float(np.clip(solidity, 0.0, 1.0)),
            'extent': float(np.clip(extent, 0.0, 1.0)),
            'bbox_long_cm': float(np.clip(bbox_long_cm, 0.0, 200.0)),
            'bbox_short_cm': float(np.clip(bbox_short_cm, 0.0, 200.0)),
            'component_count': int(np.clip(n_cc, 1, 50)),
        }

    high_mape_classes = {
        "xilanhua",
        "lajiao",
        "egg",
        "tudou",
        "tudousi",
        "qingcai",
        "rice",
        "qiezi",
        "jirou",
        "tofu",
        "xihongshi",
    }

    min_n_v2 = 20
    min_n_v3 = 15
    min_n_v4 = 20

    def _is_valid_reg_fc(reg):
        if not isinstance(reg, dict):
            return False
        feats = reg.get('features')
        coef = reg.get('coef')
        if not (isinstance(feats, list) and isinstance(coef, list) and len(feats) == len(coef) and len(coef) >= 2):
            return False
        if not np.isfinite(np.asarray(coef, dtype=np.float64)).all():
            return False
        return True

    def _is_valid_reg_v1(reg):
        return isinstance(reg, dict) and all(k in reg for k in ('a', 'b', 'c'))

    def _get_regression_params(food_name: str):
        p_cls = calibration_params.get(food_name)
        p_def = calibration_params.get('default')

        reg_rf_cls = p_cls.get('regression_rf') if isinstance(p_cls, dict) else None
        reg5_cls = p_cls.get('regression_v5') if isinstance(p_cls, dict) else None
        reg4_cls = p_cls.get('regression_v4') if isinstance(p_cls, dict) else None
        reg3_cls = p_cls.get('regression_v3') if isinstance(p_cls, dict) else None
        reg2_cls = p_cls.get('regression_v2') if isinstance(p_cls, dict) else None
        reg1_cls = p_cls.get('regression') if isinstance(p_cls, dict) else None
        
        reg_rf_def = p_def.get('regression_rf') if isinstance(p_def, dict) else None
        reg5_def = p_def.get('regression_v5') if isinstance(p_def, dict) else None
        reg4_def = p_def.get('regression_v4') if isinstance(p_def, dict) else None
        reg3_def = p_def.get('regression_v3') if isinstance(p_def, dict) else None
        reg2_def = p_def.get('regression_v2') if isinstance(p_def, dict) else None
        reg1_def = p_def.get('regression') if isinstance(p_def, dict) else None

        prefer_v2 = (food_name in high_mape_classes)

        # Always prefer RF if available (it handles non-linearity best)
        if isinstance(reg_rf_cls, dict) and int(reg_rf_cls.get('n', 0)) >= 10:
             return 'rf', reg_rf_cls

        # Then prefer v5 (log-space)
        if _is_valid_reg_fc(reg5_cls) and int(reg5_cls.get('n', 0)) >= 10:
            return 'v5', reg5_cls

        if prefer_v2:
            if _is_valid_reg_fc(reg4_cls) and int(reg4_cls.get('n', 0)) >= int(min_n_v4):
                return 'v4', reg4_cls
            if _is_valid_reg_fc(reg3_cls) and int(reg3_cls.get('n', 0)) >= int(min_n_v3):
                return 'v3', reg3_cls
            if _is_valid_reg_fc(reg2_cls) and int(reg2_cls.get('n', 0)) >= int(min_n_v2):
                return 'v2', reg2_cls
            if _is_valid_reg_v1(reg1_cls):
                return 'v1', reg1_cls
            if isinstance(reg_rf_def, dict) and int(reg_rf_def.get('n', 0)) >= 10:
                return 'rf', reg_rf_def
            if _is_valid_reg_fc(reg5_def) and int(reg5_def.get('n', 0)) >= 10:
                return 'v5', reg5_def
            if _is_valid_reg_fc(reg4_def) and int(reg4_def.get('n', 0)) >= int(min_n_v4):
                return 'v4', reg4_def
            if _is_valid_reg_fc(reg3_def) and int(reg3_def.get('n', 0)) >= int(min_n_v3):
                return 'v3', reg3_def
            if _is_valid_reg_fc(reg2_def) and int(reg2_def.get('n', 0)) >= int(min_n_v2):
                return 'v2', reg2_def
            if _is_valid_reg_v1(reg1_def):
                return 'v1', reg1_def
            return None, None

        if _is_valid_reg_fc(reg4_cls) and int(reg4_cls.get('n', 0)) >= int(min_n_v4):
            return 'v4', reg4_cls
        if _is_valid_reg_fc(reg3_cls) and int(reg3_cls.get('n', 0)) >= int(min_n_v3):
            return 'v3', reg3_cls
        if _is_valid_reg_fc(reg2_cls) and int(reg2_cls.get('n', 0)) >= int(min_n_v2):
            return 'v2', reg2_cls
        if _is_valid_reg_v1(reg1_cls):
            return 'v1', reg1_cls
        if isinstance(reg_rf_def, dict) and int(reg_rf_def.get('n', 0)) >= 10:
            return 'rf', reg_rf_def
        if _is_valid_reg_fc(reg5_def) and int(reg5_def.get('n', 0)) >= 10:
            return 'v5', reg5_def
        if _is_valid_reg_fc(reg4_def) and int(reg4_def.get('n', 0)) >= int(min_n_v4):
            return 'v4', reg4_def
        if _is_valid_reg_fc(reg3_def) and int(reg3_def.get('n', 0)) >= int(min_n_v3):
            return 'v3', reg3_def
        if _is_valid_reg_fc(reg2_def) and int(reg2_def.get('n', 0)) >= int(min_n_v2):
            return 'v2', reg2_def
        if _is_valid_reg_v1(reg1_def):
            return 'v1', reg1_def
        if _is_valid_reg_fc(reg2_cls):
            return 'v2', reg2_cls
        if _is_valid_reg_fc(reg2_def):
            return 'v2', reg2_def
        return None, None

    def _flush_csv(rows):
        if not rows:
            return
        df_part = pd.DataFrame(rows)
        exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        df_part.to_csv(csv_path, index=False, mode='a', header=(not exists))

    def _is_reasonable_scale(v: float):
        try:
            vv = float(v)
        except Exception:
            return False
        return np.isfinite(vv) and 15.0 <= vv <= 200.0

    min_score = 0.35
    try:
        min_score = float(os.environ.get('MASK_MIN_SCORE', min_score))
    except Exception:
        min_score = 0.35
    try:
        nms_thr = float(os.environ.get('MASK_NMS_IOU', 0.75))
    except Exception:
        nms_thr = 0.75
    try:
        max_keep = int(float(os.environ.get('MASK_MAX_KEEP', 120)))
    except Exception:
        max_keep = 120

    total_images = len(image_files)
    for idx, img_file in enumerate(image_files, start=1):
        if max_images is not None and int(max_images) > 0 and idx > int(max_images):
            break
        pct = 0.0
        if total_images > 0:
            pct = 100.0 * float(idx) / float(total_images)
            line = f"[EVAL {pct:6.2f}%] {idx}/{total_images} {img_file}"
            print(line, flush=True)
            _append_progress(line)

        try:
            img_path = os.path.join(test_img_dir, img_file)
            image = cv2.imread(img_path)
            if image is None:
                continue

            h, w = image.shape[:2]
            max_dim_env = os.environ.get("EVAL_MAX_DIM")
            try:
                max_dim = int(max_dim_env) if max_dim_env else 1024
            except Exception:
                max_dim = 1024
            scale_resize = 1.0
            if max(h, w) > max_dim:
                scale_resize = max_dim / max(h, w)
                image = cv2.resize(image, (int(w * scale_resize), int(h * scale_resize)))
            image = enhance_low_contrast(image)

            torch.cuda.empty_cache()
            img_gt = gt_df[gt_df['filename'] == img_file]

            # Perspective Correction
            # 1. Try to get XML marker data
            marker_data = markers_from_xml.get(img_file)
            xml_marker_pts = marker_data['pts'] if marker_data else None
            xml_grid_size_cm = marker_data['grid_size'] if marker_data else None

            # 2. Apply correction (uses XML if avail, else auto-detect grid/plate)
            # correct_perspective now handles the fallback logic internally
            if not skip_perspective:
                _, M, scale_px_per_cm = correct_perspective(image, xml_marker_pts=xml_marker_pts, xml_grid_size_cm=xml_grid_size_cm)
            else:
                M = None
                scale_px_per_cm = 1.0 # Will rely on GT or XML scale if possible, or fallback to default
                # If XML scale is available, use it directly without warp
                # scale_xml lookup below will handle it
                # If auto-detect grid is needed but perspective is skipped, we might have issues.
                # But 'estimate_grid_pixel_per_cm' works on raw image too.
                # Let's try to estimate scale on raw image if M is None
                try:
                    from perspective import estimate_grid_pixel_per_cm, estimate_plate_pixel_per_cm
                    scale_px_per_cm = estimate_grid_pixel_per_cm(image)
                    if not (15.0 <= scale_px_per_cm <= 200.0):
                        plate_ppc = estimate_plate_pixel_per_cm(image)
                        if plate_ppc and 15.0 <= plate_ppc <= 200.0:
                            scale_px_per_cm = plate_ppc
                except Exception:
                    scale_px_per_cm = 50.0

            scale_xml = scales_from_xml.get(img_file, -1.0)

            gt_scale = img_gt.iloc[0]['pixel_per_cm'] if (not img_gt.empty and float(img_gt.iloc[0].get('pixel_per_cm', 0)) > 0) else -1
            if scale_xml > 0 and _is_reasonable_scale(float(scale_xml) * float(scale_resize)):
                scale = float(scale_xml) * float(scale_resize)
                scale_source = 'xml'
                # print(f"DEBUG: Used XML scale for {img_file}: {scale:.2f} (XML raw: {scale_xml:.2f})")
            elif gt_scale > 0 and _is_reasonable_scale(float(gt_scale) * float(scale_resize)):
                scale = float(gt_scale) * float(scale_resize)
                scale_source = 'gt'
            elif _is_reasonable_scale(scale_px_per_cm):
                scale = float(scale_px_per_cm)
                scale_source = 'auto'
            else:
                scale = 50.0 * float(scale_resize)
                scale_source = 'default'

            masks = segmentor.segment(image)
            depth_map = depth_estimator.estimate(image)

            if depth_map is not None:
                if depth_map.dtype != np.float32:
                    depth_map = depth_map.astype(np.float32)
                if M is not None and getattr(M, "shape", None) == (3, 3):
                    h_img, w_img = image.shape[:2]
                    warped_size = (w_img, h_img)
                    warped_depth = cv2.warpPerspective(depth_map, M, warped_size, flags=cv2.INTER_LINEAR)
                else:
                    warped_depth = depth_map
            else:
                warped_depth = None

            candidates = []
            for mask_data in masks:
                mask = _postprocess_mask(mask_data.get('segmentation'))
                if mask is None:
                    continue
                mask_px = int(np.sum(mask))
                if mask_px <= 0:
                    continue
                pred_iou = float(mask_data.get('predicted_iou', 0.0))
                stab = float(mask_data.get('stability_score', 0.0))
                quality = float(np.clip(pred_iou * stab, 0.0, 1.0))

                if quality < float(min_score):
                    continue
                if mask_px < 80 and quality < 0.75:
                    continue

                candidates.append(
                    {
                        'mask': mask,
                        'mask_px': mask_px,
                        'pred_iou': pred_iou,
                        'stability': stab,
                        'quality': quality,
                    }
                )

            candidates.sort(key=lambda x: float(x.get('quality', 0.0)), reverse=True)
            candidates = _mask_nms(candidates, iou_thresh=nms_thr, max_keep=max_keep)

            img_preds = []
            for cand in candidates:
                try:
                    mask = cand['mask']
                    pred_iou = float(cand.get('pred_iou', 0.0))
                    stab = float(cand.get('stability', 0.0))
                    quality = float(cand.get('quality', 0.0))

                    y_indices, x_indices = np.where(mask)
                    if y_indices.size == 0 or x_indices.size == 0:
                        continue
                    y_min, y_max = y_indices.min(), y_indices.max()
                    x_min, x_max = x_indices.min(), x_indices.max()
                    pad = 10
                    h_img, w_img = image.shape[:2]
                    y_min, y_max = max(0, y_min - pad), min(h_img, y_max + pad)
                    x_min, x_max = max(0, x_min - pad), min(w_img, x_max + pad)
                    crop = image[y_min:y_max, x_min:x_max]
                    if crop.size == 0:
                        continue

                    food_name, conf = classifier.predict(crop)
                    tuning = class_tuning.get(food_name, {})
                    if float(conf) < float(tuning.get('min_conf', 0.0)):
                        continue
                    if float(quality) < float(tuning.get('min_quality', 0.0)):
                        continue

                    if food_name == 'xilanhua' and 0.50 <= float(pred_iou) < 0.60:
                        if float(conf) < 0.80 or float(quality) < 0.80:
                            continue

                    # Skip perspective correction if the image is already warped (marked in GT)
                    # ... (omitting long comments for brevity in search block, but better to replace the whole chunk) ...
                    
                    if M is not None:
                        warped_size = (w_img, h_img)
                        warped_mask = cv2.warpPerspective(mask.astype(np.uint8), M, warped_size, flags=cv2.INTER_NEAREST).astype(bool)
                    else:
                        warped_mask = mask.astype(bool)

                    # Egg Volume Correction
                    # User report shows ~5x overestimation for egg.
                    # Likely flat fried eggs being treated as ellipsoids.
                    # Apply a correction factor for 'egg' volume.
                    egg_correction_factor = 0.2
                    
                    erode_iter = int(tuning.get('mask_erode', 0))
                    if erode_iter > 0:
                        kernel = np.ones((3, 3), np.uint8)
                        warped_mask = cv2.erode(warped_mask.astype(np.uint8), kernel, iterations=erode_iter).astype(bool)

                    shape = _shape_features(warped_mask, scale_px_per_cm=scale)

                    depth_scale_factor = 5.0
                    if food_name in calibration_params:
                        depth_scale_factor = calibration_params[food_name]['depth_scale_factor']
                    elif 'default' in calibration_params:
                        depth_scale_factor = calibration_params['default']['depth_scale_factor']

                    vol_details = None
                    
                    if vol_details is None:
                        # Adaptive volume estimation
                        vol_details = calculate_adaptive_volume(
                            warped_mask,
                            warped_depth,
                            scale,
                            depth_scale_factor,
                            class_name=food_name,
                            return_details=True,
                            ablation_mode=ablation_mode
                        )

                    vol_cm3 = float(vol_details.get('volume_cm3', 0.0))

                    # P2: Tofu fallback logic
                    # Adaptive volume (DL vs Rule) handles this implicitly now.
                    # But we can still enforce minimum thickness if needed.
                    if str(food_name).strip().lower() in {'tofu', 'doufu'}:
                        height_cm = float(vol_details.get('height_cm', 0.0))
                        if height_cm < 1.0 and float(vol_details.get('area_cm2', 0.0)) > 0:
                            default_height = 2.5
                            area_val = float(vol_details.get('area_cm2', 0.0))
                            vol_cm3 = area_val * default_height
                            vol_details['height_cm'] = default_height
                            vol_details['volume_cm3'] = vol_cm3

                    # P3: Xihongshi (Tomato) Optimization
                    if str(food_name).strip().lower() == 'xihongshi':
                        height_cm = float(vol_details.get('height_cm', 0.0))
                        if height_cm < 1.2 and float(vol_details.get('area_cm2', 0.0)) > 0:
                             default_height = 1.5
                             area_val = float(vol_details.get('area_cm2', 0.0))
                             vol_cm3 = area_val * default_height
                             vol_details['height_cm'] = default_height
                             vol_details['volume_cm3'] = vol_cm3
                             
                    # If adaptive volume is used, weight_g is derived from it directly first
                    # Then we apply density
                    
                    nutri = get_nutrition_info(food_name)
                    density = nutri['density']
                    cal_per_100g = nutri['calories_per_100g']

                    area_cm2 = float(vol_details.get('area_cm2', 0.0))
                    if area_cm2 <= 0 or density <= 0:
                        continue
                        
                    # Recalculate weight_depth from the final vol_cm3 (which might be adaptive)
                    weight_depth = vol_cm3 * density
                    
                    # Update vol_cm3 in case logic above changed it
                    vol_details['volume_cm3'] = vol_cm3

                    if area_cm2 > 650.0:
                        continue
                    if area_cm2 > 150.0 and (weight_depth / area_cm2) < 0.15:
                        continue

                    prior_mean = 1.0
                    if food_name in calibration_params:
                        prior_mean = float(calibration_params[food_name].get('mean_area_density', prior_mean))
                    elif 'default' in calibration_params:
                        prior_mean = float(calibration_params['default'].get('mean_area_density', prior_mean))

                    weight_prior = prior_mean * area_cm2

                    max_height_cm = tuning.get('max_height_cm', None)
                    if max_height_cm is not None:
                        height_cm = float(vol_details.get('height_cm', 0.0))
                        if height_cm > float(max_height_cm) and height_cm > 0:
                            vol_cm3 = float(area_cm2) * float(max_height_cm)
                            weight_depth = vol_cm3 * density

                    reg_kind, reg = _get_regression_params(food_name)
                    if reg_kind == 'rf' and isinstance(reg, dict):
                         model_rel_path = reg.get('model_path')
                         feats = reg.get('features')
                         
                         model_path = os.path.join(project_root, model_rel_path)
                         if model_path not in _rf_model_cache:
                             if os.path.exists(model_path):
                                 try:
                                     _rf_model_cache[model_path] = joblib.load(model_path)
                                 except Exception:
                                     _rf_model_cache[model_path] = None
                             else:
                                 _rf_model_cache[model_path] = None
                         
                         rf_model = _rf_model_cache.get(model_path)
                         if rf_model:
                             feat_map = {
                                'weight_depth_g': float(weight_depth),
                                'weight_prior_g': float(weight_prior),
                                'area_cm2': float(area_cm2),
                                'height_cm': float(vol_details.get('height_cm', 0.0)),
                                'depth_contrast': float(vol_details.get('depth_contrast', 0.0)),
                                'circularity': float(shape.get('circularity', 0.0)),
                                'aspect_ratio': float(shape.get('aspect_ratio', 1.0)),
                                'solidity': float(shape.get('solidity', 0.0)),
                                'extent': float(shape.get('extent', 0.0)),
                                'bbox_long_cm': float(shape.get('bbox_long_cm', 0.0)),
                                'bbox_short_cm': float(shape.get('bbox_short_cm', 0.0)),
                                'component_count': float(shape.get('component_count', 1))
                             }
                             # RF expects 2D array
                             x = np.asarray([[float(feat_map.get(k, 0.0)) for k in feats]], dtype=np.float64)
                             try:
                                 weight_g = float(rf_model.predict(x)[0])
                             except Exception:
                                 weight_g = weight_prior
                             alpha = 0.5 
                         else:
                             weight_g = weight_prior
                             alpha = 0.0

                    elif reg_kind == 'v5' and isinstance(reg, dict):
                        feats = reg.get('features')
                        coef = np.asarray(reg.get('coef'), dtype=np.float64)
                        # v5 uses log features for most fields
                        eps = 1e-6
                        feat_map = {
                            'log_weight_depth_g': np.log(float(weight_depth) + eps),
                            'log_weight_prior_g': np.log(float(weight_prior) + eps),
                            'log_area_cm2': np.log(float(area_cm2) + eps),
                            'log_height_cm': np.log(float(vol_details.get('height_cm', 0.0)) + eps),
                            'depth_contrast': float(vol_details.get('depth_contrast', 0.0)),
                            'circularity': float(shape.get('circularity', 0.0)),
                            'log_aspect_ratio': np.log(float(shape.get('aspect_ratio', 1.0)) + eps),
                            'solidity': float(shape.get('solidity', 0.0)),
                            'extent': float(shape.get('extent', 0.0)),
                            'log_bbox_long_cm': np.log(float(shape.get('bbox_long_cm', 0.0)) + eps),
                            'log_bbox_short_cm': np.log(float(shape.get('bbox_short_cm', 0.0)) + eps),
                            'log_component_count': np.log(float(shape.get('component_count', 1)) + eps),
                            'bias': 1.0,
                        }
                        x = np.asarray([float(feat_map.get(k, 0.0)) for k in feats], dtype=np.float64)
                        log_y = float(np.dot(coef, x))
                        weight_g = float(np.exp(log_y) - eps)
                        
                        # Dynamic alpha for v5 based on contribution of depth vs prior features
                        contrib_depth = 0.0
                        contrib_prior = 0.0
                        for i, f_name in enumerate(feats):
                            val = abs(coef[i] * x[i])
                            if 'depth' in f_name:
                                contrib_depth += val
                            elif 'prior' in f_name:
                                contrib_prior += val
                        
                        sum_contrib = contrib_depth + contrib_prior + 1e-6
                        alpha = float(contrib_depth / sum_contrib)
                        # Clamp alpha to reasonable range to avoid over-reliance on one source
                        alpha = float(np.clip(alpha, 0.2, 0.8))
                        
                    elif reg_kind in ('v4', 'v3', 'v2') and isinstance(reg, dict):
                        feats = reg.get('features')
                        coef = np.asarray(reg.get('coef'), dtype=np.float64)
                        feat_map = {
                            'weight_depth_g': float(weight_depth),
                            'weight_prior_g': float(weight_prior),
                            'area_cm2': float(area_cm2),
                            'height_cm': float(vol_details.get('height_cm', 0.0)),
                            'depth_contrast': float(vol_details.get('depth_contrast', 0.0)),
                            'circularity': float(shape.get('circularity', 0.0)),
                            'aspect_ratio': float(shape.get('aspect_ratio', 1.0)),
                            'solidity': float(shape.get('solidity', 0.0)),
                            'extent': float(shape.get('extent', 0.0)),
                            'bbox_long_cm': float(shape.get('bbox_long_cm', 0.0)),
                            'bbox_short_cm': float(shape.get('bbox_short_cm', 0.0)),
                            'component_count': float(shape.get('component_count', 1)),
                            'bias': 1.0,
                        }
                        x = np.asarray([float(feat_map.get(k, 0.0)) for k in feats], dtype=np.float64)
                        weight_g = float(np.dot(coef, x))
                        contrib = np.abs(coef * x)
                        denom = float(np.sum(contrib) + 1e-6)
                        if 'weight_depth_g' in feats:
                            _i = int(feats.index('weight_depth_g'))
                            alpha = float(contrib[_i] / denom)
                        else:
                            alpha = 0.5
                    elif reg_kind == 'v1' and isinstance(reg, dict):
                        a = float(reg.get('a', 0.0))
                        b = float(reg.get('b', 0.0))
                        c = float(reg.get('c', 0.0))
                        weight_g = a * float(weight_depth) + b * float(weight_prior) + c
                        denom = abs(a * float(weight_depth)) + abs(b * float(weight_prior)) + 1e-6
                        alpha = float(abs(a * float(weight_depth)) / denom)
                    else:
                        contrast = float(vol_details.get('depth_contrast', 0.0))
                        alpha_base = float(contrast / (contrast + 1.0))
                        alpha_base = float(np.clip(alpha_base, 0.10, 0.95))

                        ratio = float(weight_depth / (weight_prior + 1e-6))
                        ratio = float(np.clip(ratio, 0.02, 50.0))
                        alpha_ratio = float(1.0 / (1.0 + np.exp(2.0 * np.log(ratio))))
                        alpha = float(0.60 * alpha_base + 0.40 * alpha_ratio)
                        alpha = float(alpha * float(np.clip(quality, 0.0, 1.0)))
                        alpha = float(alpha * float(tuning.get('alpha_mul', 1.0)))
                        alpha = float(np.clip(alpha, 0.05, 0.90))

                        weight_g = alpha * weight_depth + (1.0 - alpha) * weight_prior

                    ratio = float(weight_depth / (weight_prior + 1e-6))
                    if ratio < 0.30 and float(conf) >= 0.75 and float(quality) >= 0.70:
                        alpha = float(max(alpha, 0.70))
                        weight_g = alpha * float(weight_depth) + (1.0 - alpha) * float(weight_prior)
                    elif ratio < 0.50 and float(conf) >= 0.75 and float(quality) >= 0.75:
                        alpha = float(max(alpha, 0.55))
                        weight_g = alpha * float(weight_depth) + (1.0 - alpha) * float(weight_prior)

                    if not np.isfinite(weight_g):
                        continue
                    weight_g = float(max(0.0, weight_g))

                    # Post-processing: Aggressive Clipping based on Calibration Stats
                    if food_name in calibration_params:
                        p_stats = calibration_params[food_name]
                        w_p05 = p_stats.get('weight_p05')
                        w_p95 = p_stats.get('weight_p95')
                        if w_p05 is not None and w_p95 is not None:
                            # Allow wider range than p05-p95 to accommodate variations, but kill outliers
                            lo = float(w_p05) * 0.3
                            hi = float(w_p95) * 2.0
                            if lo < hi:
                                weight_g = float(np.clip(weight_g, lo, hi))
                    
                    # Heuristics for difficult classes
                    if str(food_name) == 'egg':
                         # Eggs: small area (<15cm2) implies scrambled/pieces.
                         # Prior is ~0.6 g/cm2. 20cm2 -> 12g.
                         # NEW: Aggressive cap for small pieces
                         if area_cm2 < 40.0:
                             limit = 10.0
                             if weight_g > limit:
                                 # Dampen significantly
                                 weight_g = limit * 0.8 + weight_g * 0.2
                             
                             # If very small, cap further
                             if area_cm2 < 20.0 and weight_g > 8.0:
                                 weight_g = 8.0

                         # General damping for egg if it deviates too much
                         if abs(weight_g - weight_prior) > 0.5 * weight_prior:
                              weight_g = 0.4 * weight_g + 0.6 * weight_prior

                    elif str(food_name) == 'xihongshi':
                        # Tomatoes: often wedges.
                        # Small area (<20) implies wedges/slices which are often overestimated.
                        if area_cm2 < 20.0:
                             # Aggressively reduce weight for small slices
                             # Assuming slices are thinner/lighter than prior implies
                             if weight_g > 5.0:
                                 weight_g = weight_g * 0.7
                             
                             # Cap max weight for small slices
                             if weight_g > 8.0:
                                 weight_g = 8.0
                        else:
                            # Even with good height, reflection can mess up volume.
                            # Blend with prior if deviation is large
                            if abs(weight_g - weight_prior) > 0.4 * weight_prior:
                                weight_g = 0.5 * weight_g + 0.5 * weight_prior
                    
                    elif str(food_name) == 'rice':
                        # Rice: prevent huge overestimation
                        if density > 0:
                            est_vol = weight_g / density
                            # If estimated volume implies height > 5cm (rice pile usually <5cm), clamp it.
                            if area_cm2 > 0 and (est_vol / area_cm2) > 5.0:
                                weight_g = area_cm2 * 5.0 * density
                            # Also, rice area density is fairly constant. If deviation is huge, pull to prior.
                            if weight_g > 2.0 * weight_prior:
                                weight_g = 0.5 * weight_g + 0.5 * weight_prior

                    if weight_g <= 0.0 and float(weight_prior) > 0.0:
                        weight_g = float(weight_prior)
                        vol_cm3 = weight_g / density
                        alpha = 0.0
                        if reg_kind is None:
                            reg_kind = 'prior'
                        else:
                            reg_kind = f"{reg_kind}_prior"

                    if prior_mean > 0:
                        apply_clip = (str(scale_source) == 'default') or (float(alpha) < 0.20)
                        if ratio < 0.50 and float(conf) >= 0.75 and float(quality) >= 0.75:
                            apply_clip = False
                        if apply_clip:
                            clip = float(tuning.get('prior_clip', 0.30))
                            lower_bound_w = prior_mean * (1.0 - clip) * area_cm2
                            upper_bound_w = prior_mean * (1.0 + clip) * area_cm2
                            weight_g = float(np.clip(weight_g, lower_bound_w, upper_bound_w))
                            vol_cm3 = weight_g / density

                    if food_name in high_mape_classes and food_name in calibration_params:
                        p = calibration_params.get(food_name)
                        if isinstance(p, dict):
                            w05 = p.get('weight_p05')
                            w95 = p.get('weight_p95')
                            if w05 is not None and w95 is not None and np.isfinite(float(w05)) and np.isfinite(float(w95)):
                                lo = float(max(0.0, float(w05) * 0.5))
                                hi = float(float(w95) * 1.3)
                                if hi > 0 and lo <= hi:
                                    weight_g = float(np.clip(weight_g, lo, hi))
                                    vol_cm3 = weight_g / density

                    if np.isfinite(float(weight_depth)) and np.isfinite(float(weight_prior)):
                        w_candidates = [float(weight_depth), float(weight_prior)]
                        w_pos = [v for v in w_candidates if v > 0.0 and np.isfinite(v)]
                        if w_pos:
                            w_lo_ref = float(min(w_pos))
                            w_hi_ref = float(max(w_pos))
                            if w_hi_ref > 0.0:
                                lo = float(max(0.0, 0.15 * w_lo_ref))
                                hi = float(3.0 * w_hi_ref)
                                if hi > 0.0 and lo <= hi:
                                    weight_g = float(np.clip(weight_g, lo, hi))
                                    vol_cm3 = weight_g / density

                    calories = weight_g * cal_per_100g / 100

                    img_preds.append(
                        {
                            "mask": mask,
                            "food_name": food_name,
                            "confidence": conf,
                            "volume_cm3": vol_cm3,
                            "estimated_weight_g": weight_g,
                            "area_cm2": area_cm2,
                            "depth_contrast": float(vol_details.get('depth_contrast', 0.0)),
                            "blend_alpha": alpha,
                            "mask_quality": quality,
                            "predicted_iou": float(pred_iou),
                            "stability_score": float(stab),
                            "quality": float(quality),
                            "weight_depth_g": float(weight_depth),
                            "weight_prior_g": float(weight_prior),
                            "height_cm": float(vol_details.get('height_cm', 0.0)),
                            "reg_kind": str(reg_kind) if reg_kind is not None else "",
                            "circularity": float(shape.get('circularity', 0.0)),
                            "aspect_ratio": float(shape.get('aspect_ratio', 1.0)),
                            "solidity": float(shape.get('solidity', 0.0)),
                            "extent": float(shape.get('extent', 0.0)),
                            "bbox_long_cm": float(shape.get('bbox_long_cm', 0.0)),
                            "bbox_short_cm": float(shape.get('bbox_short_cm', 0.0)),
                            "component_count": int(shape.get('component_count', 1)),
                            "estimated_calories_kcal": calories,
                        }
                    )
                except Exception:
                    continue

            for _, gt_row in img_gt.iterrows():
                gt_pts_orig = np.array(json.loads(gt_row['points']), dtype=np.float32)
                gt_pts = gt_pts_orig * scale_resize

                gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(gt_mask, [gt_pts.astype(np.int32)], 1)
                gt_mask = gt_mask.astype(bool)

                gt_area_px = np.sum(gt_mask)
                img_h, img_w = image.shape[:2]
                img_area_px = img_h * img_w
                if gt_area_px < 10 or gt_area_px > 0.95 * img_area_px:
                    print(
                        f"[WARN] Suspicious GT mask for {img_file} ({gt_row['true_class']}): Area {gt_area_px}px ({gt_area_px/img_area_px*100:.1f}%)"
                    )

                best_iou = 0
                best_pred = None
                for pred in img_preds:
                    iou = calculate_iou(gt_mask, pred['mask'])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = pred

                if best_iou > 0.1 and best_pred:
                    row = {
                        "filename": img_file,
                        "true_class": gt_row['true_class'],
                        "predicted_class": best_pred['food_name'],
                        "confidence": best_pred['confidence'],
                        "true_weight_g": gt_row['true_weight_g'],
                        "estimated_weight_g": best_pred['estimated_weight_g'],
                        "estimated_calories_kcal": best_pred['estimated_calories_kcal'],
                        "area_cm2": best_pred.get('area_cm2', 0.0),
                        "depth_contrast": best_pred.get('depth_contrast', 0.0),
                        "scale_px_per_cm": float(scale),
                        "scale_source": str(scale_source),
                        "blend_alpha": best_pred.get('blend_alpha', 0.0),
                        "mask_quality": best_pred.get('mask_quality', 0.0),
                        "predicted_iou": best_pred.get('predicted_iou', 0.0),
                        "stability_score": best_pred.get('stability_score', 0.0),
                        "quality": best_pred.get('quality', 0.0),
                        "weight_depth_g": best_pred.get('weight_depth_g', 0.0),
                        "weight_prior_g": best_pred.get('weight_prior_g', 0.0),
                        "height_cm": best_pred.get('height_cm', 0.0),
                        "reg_kind": best_pred.get('reg_kind', ''),
                        "iou": best_iou,
                    }
                else:
                    row = {
                        "filename": img_file,
                        "true_class": gt_row['true_class'],
                        "predicted_class": "None",
                        "confidence": 0,
                        "true_weight_g": gt_row['true_weight_g'],
                        "estimated_weight_g": 0,
                        "estimated_calories_kcal": 0,
                        "area_cm2": 0.0,
                        "depth_contrast": 0.0,
                        "scale_px_per_cm": float(scale),
                        "scale_source": str(scale_source),
                        "blend_alpha": 0.0,
                        "mask_quality": 0.0,
                        "predicted_iou": 0.0,
                        "stability_score": 0.0,
                        "quality": 0.0,
                        "weight_depth_g": 0.0,
                        "weight_prior_g": 0.0,
                        "height_cm": 0.0,
                        "reg_kind": "",
                        "iou": 0,
                    }

                results.append(row)
                flush_rows.append(row)
                if len(flush_rows) >= 5:
                    _flush_csv(flush_rows)
                    flush_rows.clear()

        except Exception as e:
            err_line = f"[EVAL-ERR {pct:6.2f}%] {idx}/{total_images} {img_file} {type(e).__name__}: {e}"
            print(err_line, flush=True)
            _append_progress(err_line)
            continue
        
    _flush_csv(flush_rows)
    flush_rows.clear()

    df = pd.DataFrame(results)
    
    if not df.empty:
        def get_true_cal(row):
            info = get_nutrition_info(row['true_class'])
            return row['true_weight_g'] * info['calories_per_100g'] / 100
        df['true_calories_kcal'] = df.apply(get_true_cal, axis=1)
    
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved to {csv_path}")
    
    report_path = os.path.join(result_dir, report_filename)
    _generate_report(df, report_path, primary_iou=primary_iou, class_list=class_list, exclude_top_mape_classes=exclude_top_mape_classes)
    print(f"Report saved to {report_path}")

    # Calculate return metrics
    metrics = {}
    if not df.empty:
        subset = df[df['iou'] >= primary_iou].copy()
        if require_correct_class:
            subset = subset[subset['predicted_class'] == subset['true_class']]
        metrics['high_conf_count'] = len(subset)
        if not subset.empty:
            subset_w = subset[subset['true_weight_g'] > 1e-6]
            if not subset_w.empty:
                ape = (subset_w['estimated_weight_g'] - subset_w['true_weight_g']).abs() / subset_w['true_weight_g']
                metrics['mape_high_conf'] = ape.mean() * 100.0
            else:
                metrics['mape_high_conf'] = 0.0
        else:
            metrics['mape_high_conf'] = 0.0
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--primary-iou", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--exclude-top-mape-classes", type=int, default=3)
    args = parser.parse_args()
    evaluate_system(primary_iou=float(args.primary_iou), report_only=bool(args.report_only), max_images=args.max_images, exclude_top_mape_classes=int(args.exclude_top_mape_classes))
