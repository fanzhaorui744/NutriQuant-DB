"""
Local HTML Web Demo Server for NutriQuant (6-stage monocular food portion & energy estimation).
Run:
    python demo_app.py
Then open http://127.0.0.1:7860 in your browser.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import pandas as pd

from nutrition import get_nutrition_info
from pipeline_demo import NutriQuantPipeline


BUNDLE_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BUNDLE_ROOT, "web")
TEST_IMAGES_DIR = os.path.join(BUNDLE_ROOT, "data", "test_images")
GT_CSV_PATH = os.path.join(BUNDLE_ROOT, "data", "ground_truth.csv")

_PIPELINE_LOCK = threading.Lock()
_PIPELINE: NutriQuantPipeline | None = None
_GT_DF: pd.DataFrame | None = None


def _get_gt_df() -> pd.DataFrame | None:
    global _GT_DF
    if _GT_DF is None and os.path.exists(GT_CSV_PATH):
        _GT_DF = pd.read_csv(GT_CSV_PATH)
    return _GT_DF


def _get_pipeline(encoder: str = "vits") -> NutriQuantPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = NutriQuantPipeline(bundle_root=BUNDLE_ROOT, encoder=encoder)
        _PIPELINE.ensure_loaded()
    return _PIPELINE


def _encode_png_data_url(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _list_curated_samples(limit: int = 24) -> list[dict]:
    if not os.path.exists(TEST_IMAGES_DIR):
        return []
    files = sorted(f for f in os.listdir(TEST_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    gt = _get_gt_df()
    if gt is None or gt.empty:
        return [{"filename": f, "classes": [], "item_count": 0, "total_mass_g": None, "pixel_per_cm": None} for f in files[:limit]]

    valid_classes = {
        "rice", "tudou", "tudousi", "lajiao", "qingcai",
        "xilanhua", "tofu", "qiezi", "jirou", "egg",
    }
    grouped = []
    file_set = set(files)
    for fname, grp in gt.groupby("filename"):
        if fname not in file_set:
            continue
        cls_list = [str(c) for c in grp["true_class"].tolist()]
        if not all(c in valid_classes for c in cls_list):
            continue
        total_mass = float(grp["true_weight_g"].sum())
        ppc = float(grp["pixel_per_cm"].iloc[0]) if pd.notna(grp["pixel_per_cm"].iloc[0]) else None
        grouped.append({
            "filename": fname,
            "classes": sorted(set(cls_list)),
            "item_count": len(grp),
            "total_mass_g": round(total_mass, 2),
            "pixel_per_cm": round(ppc, 3) if ppc else None,
        })
    featured = [
        "44c6a7c9f6dd9b56071367810e3fa724_00004.jpg",
        "3a8a3b515872301bfd01f0cbd6e90f44_00001.jpg",
        "459ce29c0aae876edebe1c7b788d0485_00003.jpg",
        "0efec2951ccf2c9cfab5364ed5d1cc11_0006.jpg",
        "3e61d1b3d2e1fc335ea09d9117dc3de1_00002.jpg",
        "3526ce9f6fdd316823d6704c86dbe66c_00007.jpg",
    ]
    rank = {fn: i for i, fn in enumerate(featured)}
    grouped.sort(key=lambda x: (rank.get(x["filename"], 999), -x["item_count"], x["filename"]))
    return grouped[:limit]


def _ground_truth_summary(filename: str) -> dict | None:
    gt = _get_gt_df()
    if gt is None or not filename:
        return None
    sub = gt[gt["filename"] == filename]
    if sub.empty:
        return None
    rows = []
    total_mass = 0.0
    total_kcal = 0.0
    for _, r in sub.iterrows():
        cls = str(r["true_class"])
        w_g = float(r["true_weight_g"])
        nutri = get_nutrition_info(cls)
        kcal = w_g * float(nutri["energy_kcal_per_g"])
        total_mass += w_g
        total_kcal += kcal
        rows.append({
            "class_code": cls,
            "display_name": nutri["display_name"],
            "true_mass_g": round(w_g, 2),
            "true_energy_kcal": round(kcal, 2),
        })
    ppc = float(sub["pixel_per_cm"].iloc[0]) if pd.notna(sub["pixel_per_cm"].iloc[0]) else None
    return {
        "filename": filename,
        "pixel_per_cm": round(ppc, 3) if ppc else None,
        "items": rows,
        "total_mass_g": round(total_mass, 2),
        "total_energy_kcal": round(total_kcal, 2),
    }


class DemoHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[HTTP] {self.address_string()} - {format % args}")

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK):
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_file(self, path: str, content_type: str):
        if not os.path.exists(path):
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            self._send_file(os.path.join(WEB_DIR, "index.html"), "text/html; charset=utf-8")
            return

        if parsed.path == "/api/health":
            with _PIPELINE_LOCK:
                pipe = _get_pipeline()
                info = pipe.get_runtime_info()
            self._send_json({"status": "ok", "runtime": info})
            return

        if parsed.path == "/api/samples":
            self._send_json({"samples": _list_curated_samples(limit=24)})
            return

        if parsed.path == "/api/sample_image":
            qs = parse_qs(parsed.query)
            name = os.path.basename((qs.get("name") or [""])[0])
            target = os.path.join(TEST_IMAGES_DIR, name)
            if not name or not os.path.isfile(target):
                self.send_error(HTTPStatus.NOT_FOUND, "Sample image not found")
                return
            ctype = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            self._send_file(target, ctype)
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))

            sample_name = body.get("sample_name")
            image_b64 = body.get("image_base64")
            scale_val = body.get("scale_px_per_cm")
            max_items = int(body.get("max_items", 10))
            min_conf = float(body.get("min_confidence", 0.15))

            gt_info = None
            if sample_name:
                safe_name = os.path.basename(str(sample_name))
                img_path = os.path.join(TEST_IMAGES_DIR, safe_name)
                if not os.path.isfile(img_path):
                    self._send_json({"error": f"Sample not found: {safe_name}"}, status=HTTPStatus.BAD_REQUEST)
                    return
                img_bgr = cv2.imread(img_path)
                gt_info = _ground_truth_summary(safe_name)
            elif image_b64:
                if "," in image_b64:
                    image_b64 = image_b64.split(",", 1)[1]
                raw_bytes = base64.b64decode(image_b64)
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            else:
                self._send_json({"error": "Provide either image_base64 or sample_name."}, status=HTTPStatus.BAD_REQUEST)
                return

            if img_bgr is None:
                self._send_json({"error": "Failed to decode uploaded image."}, status=HTTPStatus.BAD_REQUEST)
                return

            scale_override = float(scale_val) if scale_val not in (None, "", 0, "0") else None
            scale_hint = "user_override" if scale_override else None
            if scale_override is None and gt_info and gt_info.get("pixel_per_cm"):
                scale_override = float(gt_info["pixel_per_cm"])
                scale_hint = "dataset_calibrated_scale"

            with _PIPELINE_LOCK:
                pipe = _get_pipeline()
                res = pipe.run(
                    img_bgr,
                    scale_override_px_per_cm=scale_override,
                    scale_source_hint=scale_hint,
                    max_items=max_items,
                    min_confidence=min_conf,
                    return_visuals=True,
                )

            vis = res.pop("visuals", {})
            res["visual_data_urls"] = {
                "rectified": _encode_png_data_url(vis["rectified_bgr"]) if "rectified_bgr" in vis else "",
                "sam2_masks": _encode_png_data_url(vis["sam2_masks_bgr"]) if "sam2_masks_bgr" in vis else "",
                "segmentation": _encode_png_data_url(vis["segmentation_bgr"]) if "segmentation_bgr" in vis else "",
                "depth": _encode_png_data_url(vis["depth_bgr"]) if "depth_bgr" in vis else "",
            }
            if gt_info:
                res["ground_truth"] = gt_info

            self._send_json(res)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def main():
    parser = argparse.ArgumentParser(description="Start the NutriQuant local HTML demo server.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser.")
    parser.add_argument("--preload", action="store_true", help="Load SAM 2, ResNet50, and Depth Anything V2 at startup.")
    args = parser.parse_args()

    if args.preload:
        print("Preloading models onto local compute device...")
        with _PIPELINE_LOCK:
            pipe = _get_pipeline()
            print("Loaded:", pipe.get_runtime_info())

    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"\nNutriQuant 6-Stage Local Demo running at: {url}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down NutriQuant demo server...")
        server.server_close()


if __name__ == "__main__":
    main()
