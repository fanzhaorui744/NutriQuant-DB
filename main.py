"""Minimal CLI entrypoint for the 6-stage NutriQuant + DKDF demo."""
import argparse
import os
import cv2
import pandas as pd

from pipeline_demo import NutriQuantPipeline


def main():
    parser = argparse.ArgumentParser(description="Run NutriQuant 6-stage CLI demo.")
    parser.add_argument("--image", type=str, default=None, help="Path to a single meal photo.")
    parser.add_argument("--scale", type=float, default=None, help="Optional planar scale override (px/cm).")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    pipeline = NutriQuantPipeline(bundle_root=root, encoder="vits")
    pipeline.ensure_loaded()
    print(f"Compute: {pipeline.device} ({pipeline.get_runtime_info()['gpu_name']})")

    gt_path = os.path.join(root, "data", "ground_truth.csv")
    gt_df = pd.read_csv(gt_path) if os.path.exists(gt_path) else None

    if args.image:
        paths = [args.image]
    else:
        test_dir = os.path.join(root, "data", "test_images")
        paths = [os.path.join(test_dir, f) for f in sorted(os.listdir(test_dir)) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for img_path in paths:
        image = cv2.imread(img_path)
        if image is None:
            continue
        fname = os.path.basename(img_path)
        scale = args.scale
        if scale is None and gt_df is not None:
            rows = gt_df[gt_df["filename"] == fname]
            if not rows.empty and pd.notna(rows.iloc[0].get("pixel_per_cm")):
                scale = float(rows.iloc[0]["pixel_per_cm"])

        res = pipeline.run(image, scale_override_px_per_cm=scale, return_visuals=False)
        s1 = res["stages"]["stage1_scale_recovery"]
        s6 = res["stages"]["stage6_nutrition_totals"]
        print(f"\n=== {fname} ({res['total_elapsed_s']}s | scale={s1['scale_px_per_cm']} px/cm) ===")
        for it in res["items"]:
            print(
                f"  #{it['id']} {it['display_name']:<17} conf={it['confidence']:.2f} | "
                f"alpha={it['alpha']:.2f} -> V={it['v_fused_cm3']:5.2f} cm3 | "
                f"{it['mass_g']:5.2f} g | {it['energy_kcal']:5.2f} kcal"
            )
        print(f"  TOTAL: {s6['total_mass_g']:.2f} g | {s6['total_energy_kcal']:.2f} kcal")


if __name__ == "__main__":
    main()
