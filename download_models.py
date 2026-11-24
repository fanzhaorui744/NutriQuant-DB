"""
Download public foundation model checkpoints required by NutriQuant.
Users running on their own local compute can run this script if checkpoints
were not fetched via Git LFS or if they want to verify/re-download weights.
"""
import argparse
import os
import sys
import requests


MODELS = {
    "sam2_t.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "dest": os.path.join("checkpoints", "sam2_t.pt"),
        "source": "Meta AI Segment Anything 2 (Hiera-Tiny)",
    },
    "depth_anything_v2_vits.pth": {
        "url": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "dest": os.path.join("checkpoints", "depth_anything_v2_vits.pth"),
        "source": "Depth Anything V2 (ViT-Small)",
    },
}


def download_file(url: str, dest: str, overwrite: bool = False):
    if os.path.exists(dest) and os.path.getsize(dest) > 1024 * 1024 and not overwrite:
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"[SKIP] {dest} already exists ({size_mb:.1f} MB)")
        return

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[DOWNLOAD] {url} -> {dest}")
    with requests.get(url, stream=True, timeout=180) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        tmp_path = dest + ".part"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded * 100.0 / total
                    sys.stdout.write(f"\r  {downloaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MB ({pct:.1f}%)")
                    sys.stdout.flush()
        os.replace(tmp_path, dest)
    print("\n[OK] Saved:", dest)


def main():
    parser = argparse.ArgumentParser(description="Download SAM 2 and Depth Anything V2 checkpoints for NutriQuant.")
    parser.add_argument("--force", action="store_true", help="Re-download even if checkpoint exists.")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    for name, info in MODELS.items():
        print(f"\n=== {name} ({info['source']}) ===")
        download_file(info["url"], os.path.join(root, info["dest"]), overwrite=args.force)

    cls_path = os.path.join(root, "result", "food_classifier.pth")
    if os.path.exists(cls_path) and os.path.getsize(cls_path) > 1024 * 1024:
        print(f"\n[OK] Fine-tuned ResNet50 classifier found at {cls_path}")
    else:
        print(
            "\n[WARNING] result/food_classifier.pth is missing or an unpulled Git LFS pointer.\n"
            "Run `git lfs pull` to fetch the 10-category ResNet50 classifier weights, or train your own with `python recognition.py`."
        )


if __name__ == "__main__":
    main()
