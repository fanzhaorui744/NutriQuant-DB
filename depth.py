import os
import sys
import cv2
import requests
import torch

# Add bundled Depth Anything V2 package parent to path (prefer local vendor copy)
bundle_root = os.path.dirname(os.path.abspath(__file__))
vendor_root = os.path.join(bundle_root, "vendor")
if vendor_root not in sys.path:
    sys.path.insert(0, vendor_root)

from depth_anything_v2.dpt import DepthAnythingV2


class DepthEstimator:
    CHECKPOINT_URLS = {
        "vits": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "vitb": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth",
        "vitl": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth",
    }

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    def __init__(self, encoder="vits", device=None, checkpoint_path=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if encoder not in self.MODEL_CONFIGS:
            raise ValueError(f"Unsupported encoder: {encoder}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        self.encoder = encoder
        self.model = DepthAnythingV2(**self.MODEL_CONFIGS[encoder])

        ckpt_name = f"depth_anything_v2_{encoder}.pth"
        ckpt_path = checkpoint_path or os.path.join(bundle_root, "checkpoints", ckpt_name)

        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found. Downloading from Hugging Face...")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            self.download_checkpoint(encoder, ckpt_path)

        self.model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    def download_checkpoint(self, encoder, path):
        url = self.CHECKPOINT_URLS.get(encoder)
        if not url:
            raise ValueError(f"No download URL configured for encoder {encoder}")

        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded Depth Anything V2 ({encoder}) to {path}")

    def estimate(self, image):
        """
        Stage 4: Predict affine-invariant inverse relative depth map D_rel (H, W).
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            if gray.std() < 30.0:
                if len(image.shape) == 3:
                    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
                    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    image = clahe.apply(image)
        except Exception:
            pass

        with torch.no_grad():
            if self.device == "cuda" or (hasattr(self.device, "type") and self.device.type == "cuda"):
                with torch.cuda.amp.autocast():
                    return self.model.infer_image(image)
            return self.model.infer_image(image)


if __name__ == "__main__":
    estimator = DepthEstimator(encoder="vits")
    print("Depth Estimator initialized.")
