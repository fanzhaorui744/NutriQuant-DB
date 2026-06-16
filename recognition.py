import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
import time
import cv2
import copy
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        weight=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = float(gamma)
        if weight is None:
            self.register_buffer("weight", None)
        else:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.reduction = str(reduction)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        try:
            ce = torch.nn.functional.cross_entropy(
                logits,
                target,
                weight=self.weight,
                reduction="none",
                label_smoothing=float(self.label_smoothing),
            )
        except TypeError:
            ce = torch.nn.functional.cross_entropy(
                logits,
                target,
                weight=self.weight,
                reduction="none",
            )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def _infer_input_size(backbone: str) -> int:
    b = str(backbone).strip().lower()
    if b in {"resnet50", "resnet-50"}:
        return 224

    if b in {"efficientnet_b3", "efficientnet-b3"}:
        return 300
    if b in {"efficientnet_b4", "efficientnet-b4"}:
        return 380
    return 224


def _build_model(backbone: str, num_classes: int, pretrained: bool = True):
    b = str(backbone).strip().lower()
    if b in {"resnet50", "resnet-50"}:
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            try:
                model = models.resnet50(pretrained=bool(pretrained))
            except Exception:
                model = models.resnet50(pretrained=False)
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"resnet18", "resnet-18"}:
        try:
            model = models.resnet18(pretrained=bool(pretrained))
        except Exception:
            model = models.resnet18(pretrained=False)
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"efficientnet_b3", "efficientnet-b3"}:
        try:
            model = models.efficientnet_b3(pretrained=bool(pretrained))
        except Exception:
            model = models.efficientnet_b3(pretrained=False)
        in_features = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"efficientnet_b4", "efficientnet-b4"}:
        try:
            model = models.efficientnet_b4(pretrained=bool(pretrained))
        except Exception:
            model = models.efficientnet_b4(pretrained=False)
        in_features = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"convnext_tiny", "convnext-tiny"}:
        try:
            model = models.convnext_tiny(pretrained=bool(pretrained))
        except Exception:
            model = models.convnext_tiny(pretrained=False)
        in_features = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"swin_t", "swin-t", "swint", "swin_tiny"}:
        try:
            model = models.swin_t(pretrained=bool(pretrained))
        except Exception:
            model = models.swin_t(pretrained=False)
        in_features = int(model.head.in_features)
        model.head = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    if b in {"resnet50", "resnet-50"}:
        try:
            from torchvision.models import ResNet50_Weights
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        except Exception:
            try:
                model = models.resnet50(pretrained=bool(pretrained))
            except Exception:
                model = models.resnet50(pretrained=False)
        in_features = int(model.fc.in_features)
        model.fc = nn.Linear(in_features, max(2, int(num_classes)))
        return model

    return _build_model("resnet50", num_classes=num_classes, pretrained=bool(pretrained))

def _append_progress(line: str):
    path = os.environ.get("PIPELINE_PROGRESS_PATH")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(line) + "\n")
    except Exception:
        pass

def train_model(
    data_dir,
    num_epochs=10,
    batch_size=48,
    output_model_path="food_classifier.pth",
    class_weight_multiplier=None,
    resume_from_path=None,
    learning_rate=0.001,
    save_best=True,
    loss_name: str = "cross_entropy",
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    optimizer_name: str = "sgd",
    use_class_weights_in_loss: bool = False,
    backbone: str = "resnet18",
    pretrained: bool = True,
):
    print(f"Training model on data from {data_dir}...")
    
    # Create a unique output directory
    best_model_path = os.path.join(data_dir, '..', '..', 'result', 'food_classifier.pth')
    if not os.path.exists(os.path.dirname(best_model_path)):
        os.makedirs(os.path.dirname(best_model_path))
    
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "test")
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Train/Test split folders not found. Please run prepare_dataset.py first.")
        return

    input_size = _infer_input_size(backbone)

    # Data augmentation and normalization for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.02),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.25),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value='random'),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, data_transforms['train']),
        'val': datasets.ImageFolder(val_dir, data_transforms['val'])
    }
    
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")

    train_sampler = None
    loss_weight = None
    if class_weight_multiplier:
        targets = image_datasets['train'].targets
        counts = np.bincount(np.array(targets, dtype=np.int64), minlength=len(class_names)).astype(np.float32)
        counts[counts <= 0] = 1.0
        class_weights = 1.0 / counts
        for cls_name, mul in class_weight_multiplier.items():
            if cls_name in class_names:
                idx = class_names.index(cls_name)
                class_weights[idx] = class_weights[idx] * float(mul)
        sample_weights = class_weights[np.array(targets, dtype=np.int64)]
        train_sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        if bool(use_class_weights_in_loss):
            cw = class_weights.astype(np.float32)
            cw = cw / float(np.mean(cw))
            loss_weight = torch.as_tensor(cw, dtype=torch.float32)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    
    dataloaders = {
        'train': DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=False,
        ),
        'val': DataLoader(
            image_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=False,
        )
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(f"Dataset sizes: {dataset_sizes}")

    num_classes = len(class_names)
    
    if num_classes < 2:
        print("Warning: Only 1 class found. Training might be trivial.")
        
    backbone_used = str(backbone)
    try:
        model = _build_model(backbone=backbone, num_classes=num_classes, pretrained=bool(pretrained))
    except Exception:
        backbone_used = "resnet18"
        model = _build_model(backbone=backbone_used, num_classes=num_classes, pretrained=False)

    if resume_from_path is None and os.path.exists(output_model_path):
        try:
            existing = torch.load(output_model_path, map_location='cpu')
        except Exception:
            existing = None

        existing_arch = None
        existing_sd = None
        if isinstance(existing, dict) and 'state_dict' in existing:
            existing_arch = str(existing.get('arch', '')).strip().lower()
            existing_sd = existing.get('state_dict')

        can_resume = True
        if existing_arch and str(backbone_used).strip().lower() != existing_arch:
            can_resume = False
        if isinstance(existing_sd, dict):
            w = existing_sd.get('fc.weight')
            try:
                in_features = int(getattr(w, 'shape', [None, None])[1])
            except Exception:
                in_features = None
            if str(backbone_used).strip().lower() in {'resnet50', 'resnet-50'} and in_features not in {None, 2048}:
                can_resume = False
            if str(backbone_used).strip().lower() in {'resnet18', 'resnet-18'} and in_features not in {None, 512}:
                can_resume = False

        if can_resume:
            resume_from_path = output_model_path

    if resume_from_path is not None and os.path.exists(resume_from_path):
        try:
            ckpt = torch.load(resume_from_path, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                model.load_state_dict(ckpt)
            print(f"Resumed weights from {resume_from_path}")
        except Exception as e:
            print(f"Warning: Could not resume from {resume_from_path}: {e}")

    model = model.to(device)

    if loss_weight is not None:
        try:
            loss_weight = loss_weight.to(device)
        except Exception:
            pass

    ln = str(loss_name).strip().lower()
    if ln in {"focal", "focal_loss"}:
        criterion = FocalLoss(
            gamma=float(focal_gamma),
            weight=loss_weight,
            reduction="mean",
            label_smoothing=float(label_smoothing),
        )
    else:
        try:
            criterion = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=float(label_smoothing))
        except TypeError:
            criterion = nn.CrossEntropyLoss(weight=loss_weight)

    try:
        criterion = criterion.to(device)
    except Exception:
        pass

    on = str(optimizer_name).strip().lower()
    if on in {"adamw", "adam"}:
        optimizer = optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.01)
    else:
        optimizer = optim.SGD(model.parameters(), lr=float(learning_rate), momentum=0.9)
    
    # AMP Scaler
    use_amp = (device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Train loop
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        pct = 100.0 * float(epoch + 1) / float(max(1, num_epochs))
        line = f"[TRAIN {pct:6.2f}%] Epoch {epoch}/{num_epochs - 1}"
        print(line, flush=True)
        _append_progress(line)
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            total_batches = None
            try:
                total_batches = len(dataloaders[phase])
            except Exception:
                total_batches = None
            
            # Use a more stable loop structure, avoiding complex wrappers
            loader = dataloaders[phase]
            # if tqdm is not None and total_batches is not None:
            #     try:
            #         loader = tqdm(
            #             dataloaders[phase],
            #             total=int(total_batches),
            #             desc=f"{phase} E{epoch + 1}/{num_epochs}",
            #             leave=False,
            #         )
            #     except Exception:
            #         loader = dataloaders[phase]

            for batch_idx, (inputs, labels) in enumerate(loader, start=1):
                if phase == 'train' and total_batches is not None:
                    if batch_idx == 1 or batch_idx == total_batches or batch_idx % 5 == 0:
                        b_pct = 100.0 * float(batch_idx) / float(max(1, total_batches))
                        line_b = f"[TRAINB {b_pct:6.2f}%] Epoch {epoch + 1}/{num_epochs} Batch {batch_idx}/{total_batches}"
                        print(line_b, flush=True)
                        _append_progress(line_b)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # if tqdm is not None and hasattr(loader, "set_postfix"):
                #     try:
                #         loader.set_postfix(loss=float(loss.item()))
                #     except Exception:
                #         pass

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            msg = f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}'
            print(msg, flush=True)
            _append_progress(msg)
            
            # Deep copy the model
            if save_best and phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    done_msg = f'Training complete. Best val Acc: {best_acc:4f}'
    print(done_msg, flush=True)
    _append_progress(done_msg)

    if save_best:
        model.load_state_dict(best_model_wts)
    ckpt = {
        "arch": str(backbone_used),
        "input_size": int(input_size),
        "class_names": [str(c) for c in class_names],
        "state_dict": model.state_dict(),
    }
    torch.save(ckpt, output_model_path)
    print(f"Best model saved to {output_model_path}")
    
    # Save class names mapping
    with open(output_model_path.replace(".pth", "_classes.txt"), "w") as f:
        for c in class_names:
            f.write(f"{c}\n")

class FoodPredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        backbone = "resnet18"
        input_size = 224
        self.class_names = []

        ckpt = None
        try:
            ckpt = torch.load(model_path, map_location='cpu')
        except Exception:
            ckpt = None

        state_dict = None
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            backbone = str(ckpt.get('arch', backbone))
            input_size = int(ckpt.get('input_size', input_size))
            cn = ckpt.get('class_names')
            if isinstance(cn, list) and len(cn) > 0:
                self.class_names = [str(x) for x in cn]
            state_dict = ckpt.get('state_dict')

        if not self.class_names:
            class_file = model_path.replace(".pth", "_classes.txt")
            if os.path.exists(class_file):
                with open(class_file, "r") as f:
                    self.class_names = [line.strip() for line in f.readlines()]
            else:
                self.class_names = ["food_object"]

        def _infer_backbone_from_state_dict(sd, fallback: str):
            if not isinstance(sd, dict):
                return fallback
            fc_w = sd.get('fc.weight')
            try:
                in_features = int(getattr(fc_w, 'shape', [None, None])[1])
            except Exception:
                in_features = None
            if in_features == 512:
                return 'resnet18'
            if in_features == 2048:
                return 'resnet50'
            return fallback

        inferred_backbone = _infer_backbone_from_state_dict(state_dict, backbone)
        num_classes = len(self.class_names)
        self.model = _build_model(backbone=inferred_backbone, num_classes=num_classes, pretrained=False)

        loaded_ok = False
        load_err = None
        try:
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                sd = ckpt['state_dict']
                if isinstance(sd, dict):
                    sd = {k.replace('module.', ''): v for k, v in sd.items()}
                self.model.load_state_dict(sd, strict=True)
                loaded_ok = True
            else:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
                loaded_ok = True
        except Exception as e:
            load_err = e

        if not loaded_ok:
            raise RuntimeError(f"Failed to load classifier weights from {model_path}: {load_err}")

        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(int(input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image_bgr):
        # Convert BGR to RGB (PIL)
        from PIL import Image
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        idx = preds.item()
        if idx < len(self.class_names):
            label = self.class_names[idx]
        else:
            label = "unknown"
            
        return label, conf.item()

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data", "classification")
    if os.path.exists(data_path):
        output_path = os.path.join(project_root, "result", "food_classifier.pth")
        
        # Use ResNet50 by default now
        backbone_env = os.environ.get("CLS_BACKBONE", "resnet50")
        
        train_model(
            data_path, 
            num_epochs=36, 
            loss_name="focal", 
            optimizer_name="adamw",
            learning_rate=0.0003, # Lower LR slightly for larger pre-trained model
            output_model_path=output_path,
            backbone=backbone_env,
            pretrained=(str(os.environ.get("CLS_PRETRAINED", "1")).strip() not in {"0", "false", "no"}),
            label_smoothing=float(os.environ.get("CLS_LABEL_SMOOTHING", "0.0")),
        )
