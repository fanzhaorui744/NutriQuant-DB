# 食物营养识别系统（演示包）运行说明

## 1. 演示包内容
- 核心脚本：`main.py`、`evaluate_and_report.py`、`segmentation.py`、`recognition.py`、`depth.py`、`volume.py`、`nutrition.py`、`perspective.py`、`analysis.py`
- 模型文件：`checkpoints/sam2_t.pt`、`checkpoints/depth_anything_v2_vits.pth`、`result/food_classifier.pth`
- 分类标签：`result/food_classifier_classes.txt`
- 标定与回归参数：`calibration_params.json`、`models/rf_regressors/*.pkl`
- 测试数据：`data/test_images/*`、`data/ground_truth.csv`
- 第三方源码（本地化）：`vendor/sam2`、`vendor/depth_anything_v2`

## 2. PyCharm 配置步骤
1. 打开 PyCharm，选择 `Open`，加载本演示包根目录（解压后的 `demo_runtime_bundle`）。
2. 进入 `Settings -> Project -> Python Interpreter`，创建或选择 Python 3.10/3.11 环境。
3. 在 PyCharm Terminal 执行：

```bash
pip install -r requirements_demo.txt
```

4. 若使用 NVIDIA GPU，建议安装与你 CUDA 匹配的 `torch/torchvision` 版本；CPU 也可运行（速度较慢）。

## 3. 演示一：端到端识别（推荐课堂现场演示）
在 PyCharm Terminal 运行：

```bash
python main.py
```

- 程序会对 `data/test_images` 前 5 张图执行：
- 透视矫正 -> 分割 -> 识别 -> 深度估计 -> 体积/重量 -> 热量计算
- 结果在终端打印每个目标和总热量。

## 4. 演示二：评估与报告
在 PyCharm Terminal 运行：

```bash
python evaluate_and_report.py --max-images 5
```

- 输出预测 CSV 和评估报告到 `result/` 目录。
- 课堂上可打开报告文本讲解 MAE/RMSE/MAPE 与 IoU 覆盖率。

## 5. 常见问题与处理
- 报错 `ModuleNotFoundError: hydra`：执行 `pip install hydra-core omegaconf iopath`。
- 报错模型文件缺失：确认 `checkpoints/` 和 `result/food_classifier.pth` 是否存在。
- 运行慢：使用 GPU 解释器，或先把 `--max-images` 设小（如 3~5）。
- 如果需要读取外部标注 XML，可设置环境变量 `SOURCE_DATASET_ROOT` 指向你的数据根目录。

## 6. 说明
- 本演示包不会删除或修改你原始项目文件。
- 路径已改为相对路径，解压到任意目录都可运行。
