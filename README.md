# ResNet50 + YOLO26 OBB (Oriented Bounding Box) Model

A hybrid PyTorch architecture combining an ImageNet-pretrained **ResNet-50 backbone**, PyTorch **Feature Pyramid Network (FPN)** neck, and Ultralytics **OBB26 detection head** for oriented object detection tasks (e.g., aerial imagery, rotated document detection, or industrial inspection).

---

## Architecture Overview

```
Input Image (3 x H x W)
       │
┌──────▼───────────────────────────┐
│  ResNet-50 Backbone              │
│  - Stem (Conv1 + BN + ReLU + Pool)│
│  - Layer1 (256 ch, stride 4)     │
│  - Layer2 (512 ch, stride 8)     │
│  - Layer3 (1024 ch, stride 16)   │
│  - Layer4 (2048 ch, stride 32)   │
└──────┬───────────────────────────┘
       │ [Layer1, Layer2, Layer3, Layer4]
┌──────▼───────────────────────────┐
│  Feature Pyramid Network (FPN)   │
│  - Out channels: 256             │
└──────┬───────────────────────────┘
       │ 4 x Feature Maps (256 ch each)
┌──────▼───────────────────────────┐
│  Ultralytics OBB26 Head          │
│  - Strides: [4, 8, 16, 32]       │
│  - Classes: 15                   │
└──────┬───────────────────────────┘
       │
Output: Bounding Boxes (x, y, w, h, angle) + Class Scores

```

---

## Features

* **Custom Backbone Support**: Replaces standard CSPDarknet backbones with a pre-trained ResNet-50.
* **Multi-Scale Feature Fusion**: Uses `torchvision.ops.FeaturePyramidNetwork` to generate multi-scale representations across 4 feature levels.
* **Oriented Object Detection**: Integrated with Ultralytics `OBB26` head to detect rotated boxes (`x, y, w, h, angle`).
* **Custom Loss Adapter**: Includes `YOLOLossMock` to bridge custom PyTorch network output directly with Ultralytics `v8OBBLoss` task-aligned assigners.

---

## Setup & Installation

### Dependencies

```bash
pip install torch torchvision ultralytics numpy pandas tqdm

```

---

## Implementation Details

### Model Definition (`model.py`)

```python
from collections import OrderedDict
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.ops import FeaturePyramidNetwork
from ultralytics.nn.modules.head import OBB26

class Resnet50Yolo26(nn.Module):
    def __init__(self, num_classes: int = 15):
        super(Resnet50Yolo26, self).__init__()

        # ResNet-50 Backbone
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # FPN Neck
        in_channels = [256, 512, 1024, 2048]
        self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels, out_channels=256)

        # Ultralytics OBB Head
        self.detectionHead = OBB26(
            nc=num_classes,
            end2end=False,
            ch=(256, 256, 256, 256)
        )
        
        # Downsampling strides matching stem + layers (P2, P3, P4, P5)
        self.detectionHead.stride = torch.tensor([4.0, 8.0, 16.0, 32.0])

    def forward(self, x: torch.Tensor):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        features = OrderedDict([
            ('layer1', l1),
            ('layer2', l2),
            ('layer3', l3),
            ('layer4', l4)
        ])

        fpn_outputs = list(self.fpn(features).values())
        return self.detectionHead(fpn_outputs)

```

---

## Usage

### Training Loop Example

```python
from types import SimpleNamespace
import torch
from torch.optim import Adam
from ultralytics.utils.loss import v8OBBLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Initialize Model
model = Resnet50Yolo26(num_classes=15).to(device)

# 2. Mock Loss Wrapper for Ultralytics Loss Integration
class YOLOLossMock(torch.nn.Module):
    def __init__(self, obb_head):
        super().__init__()
        self.model = torch.nn.ModuleList([obb_head])
        self.stride = obb_head.stride
        self.nc = obb_head.nc
        self.no = obb_head.no
        self.args = getattr(obb_head, 'args', {})

mock_yolo_model = YOLOLossMock(model.detectionHead)

# 3. Configure Loss Function & Hyperparameters
criterion = v8OBBLoss(model=mock_yolo_model, tal_topk=10)
criterion.hyp = SimpleNamespace(
    box=7.5,
    cls=0.5,
    dfl=1.5,
    angle=1.06
)

optimizer = Adam(model.parameters(), lr=1e-4)

# 4. Step Training Iteration
model.train()
for data, targets in trainDataLoader:
    data = data.to(device)
    
    # Forward Pass
    preds = model(data)

    # Format targets for v8OBBLoss: [batch_idx, cls, x, y, w, h, angle]
    batch_labels = []
    for batch_idx, target_dict in enumerate(targets):
        cls = target_dict["cls"].float().unsqueeze(1)
        bboxes = target_dict["bboxes"].float() # Shape: (N, 5) -> x, y, w, h, angle
        batch_idx_col = torch.full((len(cls), 1), batch_idx, device=device)
        batch_labels.append(torch.cat([batch_idx_col, cls, bboxes], dim=1))

    all_targets = torch.cat(batch_labels, dim=0)
    loss_target = {
        "cls": all_targets[:, 1],
        "bboxes": all_targets[:, 2:],
        "batch_idx": all_targets[:, 0],
        "train_targets": all_targets
    }

    # Backward Pass
    loss_tuple, _ = criterion(preds, loss_target)
    total_loss = loss_tuple.sum()

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

```

### Evaluation / Inference Example

```python
model.eval()
with torch.no_grad():
    for data, _ in testDataLoader:
        data = data.to(device)
        outputs = model(data)

        # In eval mode, outputs[0] contains formatted predictions
        if isinstance(outputs, tuple):
            predictions = outputs[0]
            scores = predictions[..., 4]
            print(f"Mean Confidence: {scores.mean().item():.4f}")
            print(f"Max Confidence:  {scores.max().item():.4f}")

```

---

## Expected Target Data Format

During training, dataset bounding boxes provided inside `targets` must be oriented bounding boxes with **5 parameters**:

$$\text{bbox} = [x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}, \text{angle}]$$

* **$x, y$**: Center coordinates normalized relative to image resolution.
* **$w, h$**: Width and height normalized relative to image resolution.
* **$\text{angle}$**: Angle in radians or degrees matching standard `v8OBBLoss` expectations.
