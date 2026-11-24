import torch
import numpy as np
import cv2
import os
import sys

# Add bundled SAM2 package parent to path (prefer local vendor copy)
bundle_root = os.path.dirname(os.path.abspath(__file__))
vendor_root = os.path.join(bundle_root, "vendor")
if vendor_root not in sys.path:
    sys.path.insert(0, vendor_root)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

class FoodSegmentor:
    def __init__(
        self,
        checkpoint_path,
        config_path,
        device=None,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.75,
        stability_score_thresh: float = 0.88,
        crop_n_layers: int = 0,
        crop_n_points_downscale_factor: int = 2,
        min_mask_region_area: int = 50,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        print(f"Loading SAM2 from {checkpoint_path} with config {config_path} on {self.device}...", flush=True)
        self.sam2_model = build_sam2(config_path, checkpoint_path, device=self.device, apply_postprocessing=False)
        
        # We use automatic mask generator
        # You can tune points_per_side, pred_iou_thresh, etc. for better food segmentation
        def _env_int(name: str, default: int) -> int:
            try:
                v = os.environ.get(name, "")
                if v is None or str(v).strip() == "":
                    return int(default)
                return int(float(v))
            except Exception:
                return int(default)

        def _env_float(name: str, default: float) -> float:
            try:
                v = os.environ.get(name, "")
                if v is None or str(v).strip() == "":
                    return float(default)
                return float(v)
            except Exception:
                return float(default)

        pps = _env_int("SAM_POINTS_PER_SIDE", int(points_per_side))
        piou = _env_float("SAM_PRED_IOU_THRESH", float(pred_iou_thresh))
        stab = _env_float("SAM_STABILITY_THRESH", float(stability_score_thresh))
        cnl = _env_int("SAM_CROP_N_LAYERS", int(crop_n_layers))
        cnd = _env_int("SAM_CROP_N_PTS_DOWNSCALE", int(crop_n_points_downscale_factor))
        min_area = _env_int("SAM_MIN_MASK_AREA", int(min_mask_region_area))

        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            points_per_side=pps,
            pred_iou_thresh=piou,
            stability_score_thresh=stab,
            crop_n_layers=cnl,
            crop_n_points_downscale_factor=cnd,
            min_mask_region_area=min_area,
        )
        
    def segment(self, image):
        """
        Segments the image and returns a list of masks.
        Image: numpy array (H, W, 3) in RGB or BGR.
        Returns: list of dicts (segmentation result from SAM2)
        """
        # SAM2 expects RGB
        # If input is BGR (OpenCV default), convert it
        # Assuming input is BGR from cv2.imread
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use autocast for GPU memory saving and speedup (if on CUDA)
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                masks = self.mask_generator.generate(image_rgb)
        else:
            masks = self.mask_generator.generate(image_rgb)
            
        return masks

if __name__ == "__main__":
    # Test
    checkpoint = os.path.join(bundle_root, "checkpoints", "sam2_t.pt")
    # Config path needs to be relative to sam2 package or absolute?
    # build_sam2 uses hydra or relative path.
    # It seems build_sam2 expects the config name if it is in the package.
    config = "sam2_hiera_t.yaml" 
    
    # We need to make sure sam2 can find the config.
    # Usually it looks in sam2/configs/sam2
    
    segmentor = FoodSegmentor(checkpoint, config)
    print("Segmentor initialized.")
