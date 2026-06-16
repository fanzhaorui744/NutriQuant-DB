import numpy as np

def calculate_image_stddev(image):
    import cv2
    if image is None:
        return 0.0
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(cv2.meanStdDev(gray)[1][0][0])

def enhance_low_contrast(image, stddev_threshold=30.0):
    import cv2
    stddev = calculate_image_stddev(image)
    if not np.isfinite(stddev) or stddev >= float(stddev_threshold):
        return image
    
    if len(image.shape) == 3:
        # Convert to YUV for CLAHE on Luminance channel
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return enhanced
    
    # Grayscale CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)


def estimate_ellipsoid_volume(mask, pixel_per_cm, return_details: bool = True):
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
            (cx, cy), (ma, mi), angle = cv2.fitEllipse(c)
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

def _estimate_height_rel(mask, depth_map):
    import cv2

    mask_u8 = mask.astype(np.uint8)
    if mask_u8.ndim != 2:
        return None

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=4)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    ring = (dilated - mask_u8).astype(np.uint8)

    bg_depths = depth_map[ring == 1]
    if bg_depths.size < 10:
        bg_depths = depth_map[mask_u8 == 0]
    if bg_depths.size < 10:
        return None

    bg_depths = bg_depths[np.isfinite(bg_depths)]
    if bg_depths.size < 10:
        return None

    # Use lower percentile for background depth to avoid picking up far background (floor)
    # Background (table) should be the closest thing behind the food.
    # Metric Depth: Smaller Z = Closer. Larger Z = Farther.
    # The table is closer than the floor. So we want smaller Z values from the ring.
    # But wait, food is ON the table. Table is BEHIND food.
    # Z_food < Z_table < Z_floor.
    # So we want the SMALLEST Z in the background ring (closest to camera), which is the table.
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

    if np.mean(heights_b) > np.mean(heights_a):
        heights = heights_b
    else:
        heights = heights_a

    heights = heights[np.isfinite(heights)]
    heights = heights[heights >= 0]
    if heights.size < 10:
        return None

    p80 = float(np.percentile(heights, 80))
    top = heights[heights >= p80]
    if top.size < 10:
        top = heights

    height_rel = float(np.mean(top))
    mean_all_height = float(np.mean(heights)) # New: Mean height of all pixels
    contrast = float(abs(mean_food - bg_depth) / (std_bg + 1e-6))
    return {
        "height_rel": height_rel,
        "mean_height": mean_all_height,
        "bg_depth": bg_depth,
        "mean_food_depth": mean_food,
        "std_bg_depth": std_bg,
        "depth_contrast": contrast,
    }

def calculate_adaptive_volume(mask, depth_map, pixel_per_cm, depth_scale_factor=1.0, class_name=None, confidence=1.0, return_details=False, ablation_mode=None):
    """
    Adaptive volume estimation combining Deep Learning (depth map) and Rule-based (ellipsoid) approaches.
    Uses uncertainty weighting based on depth contrast and model confidence.
    """
    # 1. Calculate DL-based volume (from depth map)
    dl_res = calculate_volume(mask, depth_map, pixel_per_cm, depth_scale_factor, return_details=True)
    dl_vol = dl_res.get("volume_cm3", 0.0)
    depth_contrast = dl_res.get("depth_contrast", 0.0)
    
    # 2. Calculate Rule-based volume (ellipsoid approximation)
    rule_res = estimate_ellipsoid_volume(mask, pixel_per_cm, return_details=True)
    if rule_res:
        rule_vol = rule_res.get("volume_cm3", 0.0)
    else:
        rule_vol = dl_vol # Fallback
        
    # 3. Calculate Uncertainty Weighting
    
    # Default logic
    contrast_val = min(max(depth_contrast, 0.0), 5.0)
    unc_norm = 1.0 - (contrast_val / 5.0)
    
    # Ablation overrides for unc_norm
    forced_mode = False
    if ablation_mode == 'pure_dl' or ablation_mode == 'alpha_0':
        unc_norm = 0.0
        forced_mode = True
    elif ablation_mode == 'pure_geo' or ablation_mode == 'alpha_1':
        unc_norm = 1.0
        forced_mode = True
    elif ablation_mode == 'fixed_0.5' or ablation_mode == 'linear_0.5':
        unc_norm = 0.5
    elif ablation_mode == 'ignore_class':
        # B4: Alpha depends only on depth contrast (ignore class prior adjustment below)
        pass 
    elif ablation_mode == 'ignore_contrast':
        # B5: Alpha depends only on class prior (ignore contrast, set base to neutral 0.5)
        unc_norm = 0.5
    
    # Adjust by class if known (Prior knowledge)
    # Only apply if NOT in ignore_class mode AND not in forced mode
    if ablation_mode != 'ignore_class' and not forced_mode:
        # 强几何体优先 (Strongly Trust Geo: qiezi, jirou, xilanhua, tofu)
        if class_name in ['qiezi', 'jirou', 'xilanhua', 'tofu']:
            unc_norm = min(unc_norm + 0.5, 1.0)
        
        # 强深度优先 (Strongly Trust DL: lajiao, egg, qingcai, tudou)
        elif class_name in ['lajiao', 'egg', 'qingcai', 'tudou', 'suancaiyu', 'rice']:
            unc_norm = max(unc_norm - 0.5, 0.0)

        # 弱几何体优先 (Weakly Trust Geo)
        elif class_name in ['xihongshi']:
            unc_norm = min(unc_norm + 0.2, 1.0)

        # 土豆丝特殊处理: 它的不规则条状结构导致深度图噪声大，但几何近似也不准。
        # 之前的消融实验显示，Adapt (34.60%) 略优于 Geo (34.77%) 和 DL (35.35%)。
        # 维持默认的 unc_norm 计算 (基于对比度) 似乎是目前最佳策略。
        # 但考虑到它也是细长物体，稍微偏向 DL 可能更合理？
        # 让我们尝试给它一个轻微的 DL 偏置，或者保持中立。
        elif class_name in ['tudousi']:
            # 保持中立，或者微调。既然 analysis_report_v3 显示它 MAPE=12.81% (可能因为样本不同或 calibration 更好?)
            # 这里的 12.81% 远低于之前的 34.60%。这说明当前的策略 (默认对比度驱动) 在某些数据集上表现很好。
            # 让我们显式地将它设为“中立偏DL”，利用它细碎的特点。
            # 对比度通常较高（因为杂乱），所以 unc_norm 会自然偏低 (Trust DL)。
            # 我们不强制干预，让对比度算法自己决定，或者给一个很小的 DL 偏置。
            unc_norm = max(unc_norm - 0.1, 0.0)
            
        # Default: unc_norm remains as determined by depth contrast
        
    # Final Volume
    final_vol = (1.0 - unc_norm) * dl_vol + unc_norm * rule_vol
    
    if return_details:
        out = dl_res.copy()
        out['volume_cm3'] = final_vol
        out['dl_volume_cm3'] = dl_vol
        out['rule_volume_cm3'] = rule_vol
        out['uncertainty_weight'] = unc_norm
        out['method'] = 'adaptive'
        out['ablation_mode'] = ablation_mode
        return out
        
    return final_vol, dl_res.get('height_cm', 0.0)

def calculate_volume(mask, depth_map, pixel_per_cm, depth_scale_factor=1.0, return_details=False):
    """
    Calculates volume of the object in the mask.
    mask: boolean array
    depth_map: relative depth map
    pixel_per_cm: pixels per cm (from perspective calibration)
    depth_scale_factor: conversion from relative depth to cm
    """
    area_pixels = np.sum(mask)
    
    # Area in cm2
    # pixel_per_cm = pixels / cm
    # cm = pixels / pixel_per_cm
    # cm2 = pixels^2 / pixel_per_cm^2
    area_cm2 = area_pixels / (pixel_per_cm ** 2)
    
    hinfo = _estimate_height_rel(mask, depth_map)
    if hinfo is None:
        if return_details:
            return {
                "volume_cm3": 0.0,
                "height_cm": 0.0,
                "height_rel": 0.0,
                "area_cm2": float(area_cm2),
                "depth_contrast": 0.0,
            }
        return 0.0, 0.0

    # Use mean height for volume integration instead of peak height (height_rel)
    # height_rel is usually the top 20% mean, which overestimates volume for non-cylinders.
    mean_height = float(hinfo.get("mean_height", hinfo["height_rel"]))
    height_rel = float(hinfo["height_rel"])
    
    # Volume = Base Area * Mean Height
    height_cm = mean_height * float(depth_scale_factor)
    volume_cm3 = float(area_cm2) * float(height_cm)

    # For return details, we might still want to report the 'peak' height in cm
    peak_height_cm = height_rel * float(depth_scale_factor)

    if return_details:
        out = {
            "volume_cm3": float(volume_cm3),
            "height_cm": float(peak_height_cm), # Keep reporting peak height for compatibility
            "mean_height_cm": float(height_cm),
            "height_rel": float(height_rel),
            "area_cm2": float(area_cm2),
        }
        out.update(hinfo)
        return out

    return float(volume_cm3), float(peak_height_cm)
