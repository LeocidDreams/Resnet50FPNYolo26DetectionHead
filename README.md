# ResNet50-YOLO OBB Object Detection Pipeline

A custom PyTorch-based oriented bounding box (OBB) object detection model integrating a **ResNet50** backbone with a **Feature Pyramid Network (FPN)** and an Ultralytics **OBB detection head**.

---

## 📌 Architecture Overview

* **Backbone:** ResNet50 (Pretrained `IMAGENET1K_V2`) extracts multi-scale feature maps from layers 1 through 4.
* **Neck:** Feature Pyramid Network (FPN) standardizes multi-scale input channels (`[256, 512, 1024, 2048]`) to a uniform 256-channel representation.
* **Head:** Ultralytics `OBB26` detection head processes multi-level feature maps to predict class scores, oriented bounding boxes, and angles.
* **Loss Function:** Integrates Ultralytics `v8OBBLoss` paired with Task-Aligned Assigner (TAL) for OBB target matching and loss computation.

---

## ⚙️ Model Setup & Configuration

### Prerequisites

* Python 3.10+
* PyTorch (CUDA-enabled recommended)
* `torchvision`
* `ultralytics`
* `numpy` / `pandas` / `tqdm`

### Environment Flags

```python
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

```

---

## 🚀 Usage Guide

### 1. Model Initialization & Forward Pass

```python
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize custom hybrid model
model = Resnet50Yolo26().to(device)

# Example input tensor: Batch of 2 images (3 channels, 640x640)
x = torch.randn(2, 3, 640, 640).to(device)
outputs = model(x)

```

### 2. Loss Criterion & Training Setup

```python
from torch.optim import Adam
from ultralytics.utils.loss import v8OBBLoss
from types import SimpleNamespace

# Wrap the OBB head to mirror expected Ultralytics model interface
mock_yolo_model = YOLOLossMock(model.detectionHead)

# Initialize loss and hyper-parameters
criterion = v8OBBLoss(model=mock_yolo_model, tal_topk=10)
criterion.hyp = SimpleNamespace(
    box=7.5,
    cls=0.5,
    dfl=1.5,
    angle=1.06
)

optimizer = Adam(model.parameters(), lr=1e-4)

```

### 3. Training Loop

```python
model.train()
for epoch in range(epochs):
    for data, targets in trainDataLoader:
        data = data.to(device)
        
        # Prepare target batch dict
        labels = targets[0]
        loss_target = {
            "cls": torch.stack([x["cls"] for x in labels]).float().to(device),
            "bboxes": torch.stack([x["bboxes"] for x in labels]).float().to(device),
            "batch_idx": torch.zeros(len(labels), dtype=torch.long, device=device)
        }

        # Forward pass & loss calculation
        output = model(data)
        lossItems, lossDetails = criterion.loss(output, loss_target)
        totalLoss = lossItems.sum()

        # Backward pass
        optimizer.zero_grad()
        totalLoss.backward()
        optimizer.step()

```

### 4. Evaluation / Inference

```python
model.eval()
with torch.no_grad():
    for data, targets in testDataLoader:
        data = data.to(device)
        output = model(data)
        
        # Extract predictions and class confidence
        scores = output[1]["scores"]
        probs = scores.sigmoid()
        confidence = probs.max(dim=1).values
        
        print(f"Mean confidence: {confidence.mean().item():.4f}")

```
