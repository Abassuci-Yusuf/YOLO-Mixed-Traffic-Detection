# Trained Model Weights

## Available Models

### YOLOv9 Models
- `yolov9_k9_baseline.pt` - 51.6% mAP, 27ms inference
- `yolov9_k9_lcn.pt` - 19.9% mAP, 28ms inference

### YOLOv10 Models  
- `yolov10_k9_baseline.pt` - 57.6% mAP, 24ms inference ⭐ Best speed/accuracy
- `yolov10_k9_lcn.pt` - 25.3% mAP, 24ms inference

### YOLOv11 Models
- `yolov11_k3_baseline.pt` - 3.3% mAP
- `yolov11_k3_lcn.pt` - 67.7% mAP
- `yolov11_k7_baseline.pt` - 2.9% mAP  
- `yolov11_k7_lcn.pt` - 70.7% mAP ⭐ Best accuracy
- `yolov11_k9_baseline.pt` - 50.6% mAP
- `yolov11_k9_lcn.pt` - 12.5% mAP

## Model Details

**Training Configuration:**
- Epochs: 100
- Batch size: 16
- Image size: 640x640
- Optimizer: AdamW
- Dataset: 2,000 images (Banda Aceh traffic)

**File Sizes:** ~50-100MB per model

## Download

⚠️ Model weights not included in repository due to size constraints.

**Available upon request via:**
- Email: abassuci21@email.com
- Google Drive: [Link available on request]

## Usage
```python
from ultralytics import YOLO

# Load best performing model
model = YOLO('yolov11_k7_lcn.pt')

# Run inference
results = model('traffic_image.jpg')
results.show()
```

## Performance Summary

See `/results/metrics.csv` for complete metrics.

**Recommended for Production:**
- Speed-critical: YOLOv10 k=9 baseline (57.6% mAP, 24ms)
- Accuracy-critical: YOLOv11 k=7 LCN (70.7% mAP, 39ms)
