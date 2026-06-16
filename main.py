import cv2
import os
import json
import random
import numpy as np
from perspective import correct_perspective
from segmentation import FoodSegmentor
from recognition import FoodPredictor
from depth import DepthEstimator
from volume import calculate_adaptive_volume
from nutrition import get_nutrition_info

def main():
    print("Initializing system...")
    bundle_root = os.path.dirname(os.path.abspath(__file__))
    
    # Paths
    sam_checkpoint = os.path.join(bundle_root, "checkpoints", "sam2_t.pt")
    # Config path inside bundled sam2 package
    sam_config = os.path.join(bundle_root, "vendor", "sam2", "configs", "sam2", "sam2_hiera_t.yaml")
    classifier_path = os.path.join(bundle_root, "result", "food_classifier.pth")
    calibration_path = os.path.join(bundle_root, "calibration_params.json")
    
    # Initialize models
    segmentor = FoodSegmentor(sam_checkpoint, sam_config)
    
    # Check if classifier exists
    if not os.path.exists(classifier_path):
        print("Classifier not found. Please run recognition.py to train.")
        return
    classifier = FoodPredictor(classifier_path)
    
    depth_estimator = DepthEstimator(encoder='vits')

    calibration_params = {}
    if os.path.exists(calibration_path):
        try:
            with open(calibration_path, "r", encoding="utf-8") as f:
                calibration_params = json.load(f)
            print("Calibration parameters loaded.")
        except Exception:
            calibration_params = {}
    
    # Test images
    test_dir = os.path.join(bundle_root, "data", "test_images")
    if not os.path.exists(test_dir):
        print(f"Test image directory not found: {test_dir}")
        return
        
    all_image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    sample_size = min(5, len(all_image_files))
    image_files = random.sample(all_image_files, sample_size) if sample_size > 0 else []
    
    for img_file in image_files:
        print(f"\nProcessing {img_file}...")
        img_path = os.path.join(test_dir, img_file)
        image = cv2.imread(img_path)
        
        # 1. Perspective Correction
        pc_out = correct_perspective(image)
        # Compatible with both old (warped, scale) and new (warped, M, scale) signatures
        if isinstance(pc_out, tuple) and len(pc_out) >= 3:
            warped, _, scale = pc_out[0], pc_out[1], pc_out[2]
        elif isinstance(pc_out, tuple) and len(pc_out) == 2:
            warped, scale = pc_out
        else:
            warped, scale = image, 1.0
        # scale is pixels per cm (if grid found)
        # If no grid, scale is 1.0 (invalid)
        # Let's assume a dummy scale if not found for demo
        if scale == 1.0:
            # Fallback scale: use a larger px/cm to avoid systematic volume overestimation.
            scale = 50.0
        
        # 2. Segmentation
        masks = segmentor.segment(warped)
        print(f"Found {len(masks)} segments.")
        
        # 3. Depth
        depth_map = depth_estimator.estimate(warped)
        
        total_calories = 0
        
        for i, mask_data in enumerate(masks):
            mask = mask_data['segmentation'] # boolean array
            
            # Filter by size
            if np.sum(mask) < 1000:
                continue

            # Shrink mask slightly to reduce edge spillover from segmentation.
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
            if np.sum(mask) < 800:
                continue
                
            # Crop object for classification
            y_indices, x_indices = np.where(mask)
            y_min, y_max = y_indices.min(), y_indices.max()
            x_min, x_max = x_indices.min(), x_indices.max()
            
            crop = warped[y_min:y_max, x_min:x_max]
            if crop.size == 0: continue
            
            # 4. Recognition
            food_name, conf = classifier.predict(crop)
            print(f"  Object {i}: {food_name} ({conf:.2f})")
            
            # 5. Volume
            cls_cal = calibration_params.get(food_name, {})
            def_cal = calibration_params.get("default", {})
            depth_scale_factor = float(cls_cal.get("depth_scale_factor", def_cal.get("depth_scale_factor", 1.0)))

            vol_details = calculate_adaptive_volume(
                mask,
                depth_map,
                scale,
                depth_scale_factor=depth_scale_factor,
                class_name=food_name,
                confidence=conf,
                return_details=True,
            )

            vol_cm3 = float(vol_details.get("volume_cm3", 0.0))
            height_cm = float(vol_details.get("height_cm", 0.0))
            area_cm2 = float(vol_details.get("area_cm2", 0.0))
            depth_contrast = float(vol_details.get("depth_contrast", 0.0))
            print(f"    Volume: {vol_cm3:.2f} cm3, Height: {height_cm:.2f} cm")
            
            # 6. Nutrition
            nutri = get_nutrition_info(food_name)
            density = nutri['density']
            cal_per_100g = nutri['calories_per_100g']
            
            # 7. Weight & Calories (lightweight calibration-aware fusion)
            weight_depth_g = vol_cm3 * density
            prior_mean = float(cls_cal.get("mean_area_density", def_cal.get("mean_area_density", 1.0)))
            weight_prior_g = float(max(0.0, prior_mean * max(0.0, area_cm2)))

            # Dynamic blend based on depth contrast: higher contrast => trust depth more.
            alpha = float(np.clip(depth_contrast / (depth_contrast + 1.0 + 1e-6), 0.20, 0.85))
            if weight_prior_g <= 0:
                alpha = 1.0
            weight_g = alpha * weight_depth_g + (1.0 - alpha) * weight_prior_g

            # Class-specific sanity clipping from calibration percentiles.
            w05 = cls_cal.get("weight_p05")
            w95 = cls_cal.get("weight_p95")
            if w05 is not None and w95 is not None:
                lo = max(0.0, float(w05) * 0.5)
                hi = float(w95) * 1.5
                if hi > lo:
                    weight_g = float(np.clip(weight_g, lo, hi))

            calories = weight_g * cal_per_100g / 100
            
            print(f"    Weight: {weight_g:.2f} g")
            print(f"    Calories: {calories:.2f} kcal")
            
            total_calories += calories
            
        print(f"Total Calories: {total_calories:.2f} kcal")

if __name__ == "__main__":
    main()
