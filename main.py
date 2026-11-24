"""
Command-line 6-stage NutriQuant + DKDF inference demo on real sample images.
For the interactive HTML interface, run `python demo_app.py`.
"""
import argparse
import os
import random
import cv2
import pandas as pd

from pipeline_demo import NutriQuantPipeline


def main():
    parser = argparse.ArgumentParser(description="Run NutriQuant 6-stage CLI demo on sample cafeteria images.")
    parser.add_argument("--image", type=str, default=None, help="Optional path to a specific input image.")
    parser.add_argument("--max-images", type=int, default=3, help="Number of sample images to process.")
    parser.add_argument("--scale", type=float, default=None, help="Optional planar scale override (px/cm).")
    args = parser.parse_args()

    bundle_root = os.path.dirname(os.path.abspath(__file__))
    print("Initializing NutriQuant 6-stage pipeline...")
    pipeline = NutriQuantPipeline(bundle_root=bundle_root, encoder="vits")
    pipeline.ensure_loaded()
    info = pipeline.get_runtime_info()
    print(f"Compute device: {info['device']} ({info['gpu_name']})")

    gt_path = os.path.join(bundle_root, "data", "ground_truth.csv")
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

    if args.image:
        image_paths = [args.image]
    else:
        test_dir = os.path.join(bundle_root, "data", "test_images")
        all_files = sorted(f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        chosen = random.sample(all_files, min(args.max_images, len(all_files))) if all_files else []
        image_paths = [os.path.join(test_dir, f) for f in chosen]

    if not image_paths:
        print("No input images found.")
        return

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[SKIP] Cannot read {img_path}")
            continue

        scale_override = args.scale
        scale_hint = "user_override" if scale_override else None
        if scale_override is None and gt_df is not None:
            rows = gt_df[gt_df["filename"] == fname]
            if not rows.empty and pd.notna(rows.iloc[0].get("pixel_per_cm")):
                scale_override = float(rows.iloc[0]["pixel_per_cm"])
                scale_hint = "dataset_calibrated_scale"

        res = pipeline.run(
            image,
            scale_override_px_per_cm=scale_override,
            scale_source_hint=scale_hint,
            return_visuals=False,
        )
        s1 = res["stages"]["stage1_scale_recovery"]
        s2 = res["stages"]["stage2_sam2_segmentation"]
        s6 = res["stages"]["stage6_nutrition_totals"]

        print(f"\n=== {fname} ({res['total_elapsed_s']}s) ===")
        print(f"  Stage 1 Scale : {s1['scale_px_per_cm']} px/cm ({s1['scale_source']})")
        print(f"  Stage 2 SAM 2 : {s2['retained_instances']} retained masks (raw {s2['raw_proposals']})")
        for it in res["items"]:
            gate = "ON" if it["gate_active"] else "OFF"
            print(
                f"  #{it['id']} {it['display_name']:<17} conf={it['confidence']:.2f} [gate {gate}] | "
                f"V_geo={it['v_geo_cm3']:6.2f}, V_depth={it['v_depth_cm3']:6.2f}, "
                f"alpha={it['alpha']:.2f} -> V_hat={it['v_fused_cm3']:6.2f} cm3 | "
                f"Mass={it['mass_g']:6.2f} g | Energy={it['energy_kcal']:6.2f} kcal"
            )
        print(f"  TOTAL -> Mass: {s6['total_mass_g']:.2f} g | Estimated Energy: {s6['total_energy_kcal']:.2f} kcal")


if __name__ == "__main__":
    main()
