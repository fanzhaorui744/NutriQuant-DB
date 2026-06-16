# NutriQuant-DB: Image-Based Food Volume & Calorie Estimation (Demo Bundle)

An end-to-end pipeline that recognizes food items from a single RGB image and
estimates their **volume, weight and calorie content** by fusing segmentation,
depth estimation, geometric reasoning and class-specific calibration.

```
Image -> Perspective Correction -> Segmentation -> Recognition
      -> Depth Estimation -> Volume / Weight -> Nutrition -> Report
```

> Large model weights (`.pt` / `.pth` / `.pkl`) are tracked with **Git LFS**.
> Make sure Git LFS is installed before cloning so the weights are fetched.

---

## 1. Repository Contents

| Path | Description |
|------|-------------|
| `main.py` | Single end-to-end inference demo (processes 5 sample images). |
| `evaluate_and_report.py` | Batch evaluation: produces prediction CSV + metrics report. |
| `perspective.py` | Table / reference-grid perspective correction and scale (px/cm) estimation. |
| `segmentation.py` | SAM2-based automatic instance segmentation (multi-food). |
| `recognition.py` | Food category classifier (ResNet-based). |
| `depth.py` | Depth Anything V2 monocular depth estimation. |
| `volume.py` | Mask + depth + scale -> volume / height estimation. |
| `nutrition.py` | Per-class density and kcal-per-100g lookup. |
| `analysis.py` | MAE / RMSE / MAPE and consistency statistics. |
| `calibration_params.json` | Per-class calibration: depth correction + regression coefficients. |
| `checkpoints/` | `sam2_t.pt`, `depth_anything_v2_vits.pth` (Git LFS). |
| `result/` | `food_classifier.pth` (Git LFS) + `food_classifier_classes.txt`. |
| `models/rf_regressors/*.pkl` | Per-class Random-Forest weight regressors (Git LFS). |
| `data/test_images/` | Demo images. |
| `data/ground_truth.csv` | Ground-truth weights for evaluation. |
| `vendor/sam2`, `vendor/depth_anything_v2` | Localized third-party source. |
| `requirements_demo.txt` | Python dependencies. |

## 2. Quick Start

### 2.1 Clone (with Git LFS)

```bash
git lfs install          # one-time
git clone https://github.com/fanzhaorui744/NutriQuant-DB.git
```

### 2.2 Environment

Python 3.10 / 3.11 recommended.

```bash
pip install -r requirements_demo.txt
```

For NVIDIA GPU users: install a `torch`/`torchvision` build matching your CUDA.
CPU-only also works but is slower.

> If you see `ModuleNotFoundError: hydra`, run:
> `pip install hydra-core omegaconf iopath`

### 2.3 Run the end-to-end demo

```bash
python main.py
```

Randomly picks 5 images from `data/test_images` and runs the full pipeline:
perspective -> segmentation -> recognition -> depth -> volume -> calories.
Per-target and total calories are printed to the terminal.

### 2.4 Run evaluation & report

```bash
python evaluate_and_report.py --max-images 5
```

Writes prediction CSV and a metrics report (MAE / RMSE / MAPE, IoU coverage,
classification accuracy) into `result/`.

## 3. Notes

- Paths are relative; the bundle runs from any directory after extraction/clone.
- Reading external annotation XML is supported via the `SOURCE_DATASET_ROOT`
  environment variable (point it at your dataset root).
- To speed up inference, use a GPU interpreter or lower `--max-images`.

## 4. Architecture (execution chain)

- **Entry layer**: `main.py`, `evaluate_and_report.py`
- **Perception (CV)**: `perspective.py`, `segmentation.py`, `recognition.py`, `depth.py`
- **Estimation (geometry + nutrition)**: `volume.py`, `calibration_params.json`, `models/rf_regressors/*.pkl`, `nutrition.py`
- **Evaluation**: `analysis.py`, `evaluate_and_report.py`
- **Assets**: `checkpoints/`, `result/`, `data/`, `vendor/`

Key design points:
- **Multi-model collaboration**: SAM2 + classifier + depth model + regression
  calibration jointly estimate weight and calories.
- **Calibration enhancement**: `calibration_params.json` + `rf_regressors`
  correct per-class bias.
- **Closed-loop evaluation**: `evaluate_and_report.py` produces reproducible,
  quantified reports.
