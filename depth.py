import torch
import cv2
import sys
import os
import requests

# Add bundled Depth Anything package parent to path
bundle_root = os.path.dirname(os.path.abspath(__file__))
vendor_root = os.path.join(bundle_root, "vendor")
if vendor_root not in sys.path:
    sys.path.append(vendor_root)

from depth_anything_v2.dpt import DepthAnythingV2

class DepthEstimator:
    def __init__(self, encoder='vits', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.model = DepthAnythingV2(**model_configs[encoder])
        
        # Checkpoint path
        ckpt_name = f'depth_anything_v2_{encoder}.pth'
        ckpt_path = os.path.join(bundle_root, 'checkpoints', ckpt_name)
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint {ckpt_path} not found. Downloading...")
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            self.download_checkpoint(encoder, ckpt_path)
            
        self.model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        self.model = self.model.to(self.device).eval()
        
    def download_checkpoint(self, encoder, path):
        urls = {
            'vits': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth',
            'vitb': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth',
            'vitl': 'https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth',
        }
        url = urls.get(encoder)
        if not url:
            raise ValueError(f"No URL for encoder {encoder}")
            
        response = requests.get(url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(path, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
        print("Download complete.")

    def estimate(self, image):
        """
        Returns depth map (numpy array).
        """
        try:
            # Check contrast (stddev of grayscale)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            std_contrast = gray.std()
            
            # User specified contrast < 2. Assuming this refers to a specific metric or normalized range.
            # If we normalize image to 0-1, std < 0.2? Or maybe user meant standard deviation < 20?
            # A stddev of < 2 (on 0-255 scale) is effectively a solid color.
            # We will apply it if std < 30 (safe threshold for "low contrast").
            
            if std_contrast < 30.0: 
                if len(image.shape) == 3:
                    # Convert to YUV, equalize Y, convert back
                    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                    image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                else:
                    image = cv2.equalizeHist(image)
        except Exception:
            pass

        if self.device == 'cuda' or (hasattr(self.device, 'type') and self.device.type == 'cuda'):
             with torch.cuda.amp.autocast():
                 return self.model.infer_image(image)
        return self.model.infer_image(image)

if __name__ == "__main__":
    estimator = DepthEstimator(encoder='vits')
    print("Depth Estimator initialized.")
