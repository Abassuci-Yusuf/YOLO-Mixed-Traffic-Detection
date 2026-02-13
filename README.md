# YOLO-Mixed-Traffic-Detection
Evaluating YOLO v9-v11 architectures with Mixed Traffic condition detection - analyzing impact of preprocessing techniques on modern object detection model

> Systematic evaluation of YOLOv9, v10, and v11 performance with preprocessing techniques on mixed traffic conditions

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

> ⚠️ **Note**: Final results pending thesis completion (June 2026). 
> Preliminary findings shown below.
## Problem Statement

Standard YOLO models trained on Western datasets (COCO, ImageNet) struggle with Southeast Asian traffic due to:

- **Mixed vehicle types**: Cars, motorcycles, tuk-tuks, bicycles, informal vehicles in same frame
- **Informal traffic patterns**: Lane-less driving, motorcycles weaving between vehicles
- **Variable lighting**: Harsh tropical sun, sudden monsoon darkness, minimal street lighting
- **Different infrastructure**: Unique signage, road markings, and urban layouts

**Research Question**: Can classical preprocessing techniques like Local Contrast Normalization (LCN) improve modern YOLO architectures (v9-v11) for Southeast Asian traffic detection?

---

## Methodology

### Models Evaluated

| Model | Key Features | Parameters |
|-------|-------------|------------|
| **YOLOv9** | Generalized Efficient Layer Aggregation Network (GELAN) | Multi-layer feature extraction |
| **YOLOv10** | NMS-free architecture, dual assignments | Simplified post-processing |
| **YOLOv11** | Enhanced feature pyramid, improved small object detection | Additional processing layers |

### Preprocessing Approaches

1. **Baseline**: Standard YOLO preprocessing (resize, normalize)
2. **Local Contrast Normalization (LCN)**: Classical technique enhancing local feature contrast

**Hypothesis**: LCN would improve detection by enhancing edges/features in challenging lighting

### Dataset Characteristics

- **Size**: 4,000+ manually labeled images
- **Source**: Indonesian road traffic (Banda Aceh)
- **Vehicle classes**: Car, motorcycle, truck, bus, person, motorized-trishaw, traffic sign
- **Conditions**: 
  - Time: Dawn, evening, night
  - Weather: overcast, rain, heavy rain
  - Lighting: Natural, street lights, headlights, shadows

### Evaluation Metrics

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **Precision/Recall**: Per-class performance
- **Inference Speed**: FPS on standard hardware (RTX 3060) [soon]

---

## Key Findings

### Result: LCN Decreased Performance ⬇️

| Model   | Baseline mAP | With LCN |   Change   |
|---------|--------------|----------|------------|
| YOLOv9  |     83%      |    67%   |    -16%    |
| YOLOv10 |     75%      |    70%   |    -5%     |
| YOLOv11 |     79%      |    64%   |    -15%    |

### Why Did This Happen?

**Modern YOLO architectures have internalized preprocessing through learned features.**

1. **YOLOv9 & v11**: Multi-layer feature extraction (GELAN, enhanced FPN) conflicts with handcrafted normalization
   - Extra processing layers modify pixel distributions
   - LCN preprocessing gets "overwritten" by learned features
   - Network has to "unlearn" the LCN effect

2. **YOLOv10**: NMS-free design with dual assignments showed **30% more resilience** (-5% vs -16%/-15%)
   - Simpler architecture = less interference
   - Fewer post-processing layers = preprocessing signal preserved better

### Additional Insights

**Class-specific performance:**
- Motorcycles: Most affected by LCN (-12% precision) - likely due to smaller size and edge complexity
- Cars/Trucks: Moderate impact (-5-7%)
- Pedestrians: Minimal impact (-2%)

**Lighting conditions:**
- LCN helped slightly in extreme darkness (+3% night performance)
- Hurt performance in normal/good lighting (-8% clear)
- Net effect: Negative overall

## Implications 

### Key Takeaways

1. **❌ Stop using handcrafted preprocessing** on modern YOLO (v8+)
   - Models are designed to learn optimal preprocessing
   - Classical techniques are obsolete for deep architectures

2. **✅ Focus on data quality over preprocessing tricks**
   - Diverse training data > algorithmic tweaks
   - Better to collect 1,000 more labeled images than tune preprocessing

3. **✅ Consider architecture choice for production**
   - YOLOv10's simplicity = more robust to variations
   - Good for production where consistency > peak performance

4. **✅ Domain-specific training trumps general preprocessing**
   - Train on SEA data > Apply Western preprocessing to Western-trained models

### Recommended Approach for SEA Traffic Detection
```python
# GOOD: Just use baseline YOLO with domain-specific training
model = YOLO('yolov10n.pt')
model.train(data='sea_traffic.yaml', epochs=100)

# BAD: Don't add manual preprocessing
# preprocessed_images = apply_lcn(images)  # ❌ Don't do this
# model.train(data=preprocessed_images)
```

---

## 🛠️ Tech Stack

- **Python** 3.8+
- **PyTorch** 2.0+
- **Ultralytics** (YOLOv8-11 implementation)
- **OpenCV** (image processing)
- **NumPy**, **Pandas** (data manipulation)
- **Matplotlib**, **Seaborn** (visualization)
- **LabelImg** (annotation tool)
- **Google Colab** (training environment)

---

## 📁 Repository Structure
```
YOLO-Mixed-Traffic-Detection/
├── data/                   # Dataset (not included - too large)
│   ├── images/            # Raw traffic images
│   ├── labels/            # YOLO format annotations
│   └── README.md          # Dataset documentation
├── models/                # Trained model weights
│   ├── yolov9_baseline.pt
│   ├── yolov10_baseline.pt
│   └── yolov11_baseline.pt
├── notebooks/             # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_experiments.ipynb
│   └── 03_results_analysis.ipynb
├── results/               # Evaluation outputs
│   ├── confusion_matrices/
│   ├── performance_charts/
│   └── metrics.csv
├── docs/                  # Additional documentation
│   ├── METHODOLOGY.md
│   └── LESSONS_LEARNED.md
├── requirements.txt       # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 How to Use

### Installation
```bash
# Clone repository
git clone https://github.com/[your-username]/YOLO-Mixed-Traffic-Detection.git
cd YOLO-Mixed-Traffic-Detection

# Install dependencies
pip install -r requirements.txt
```

### Training (Example)
```bash
# Train YOLOv10 baseline
yolo detect train data=sea_traffic.yaml model=yolov10n.pt epochs=100 imgsz=640

# Evaluate model
yolo detect val model=models/yolov10_baseline.pt data=sea_traffic.yaml
```

### Inference
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/yolov10_baseline.pt')

# Run inference
results = model('path/to/traffic_image.jpg')
results[0].show()  # Display results
```

---

## 📈 Future Work

- [ ] Evaluate newer architectures (YOLOv12+)
- [ ] Test on other SEA countries (Malaysia, Thailand, Vietnam, Philippines)
- [ ] Explore learned preprocessing alternatives (attention mechanisms, adaptive normalization)
- [ ] Production deployment optimization (TensorRT, ONNX quantization)
- [ ] Real-time video stream inference
- [ ] Edge device deployment (Jetson Nano, Raspberry Pi)

---

## 🎓 Academic Context

This research was conducted as part of my final year thesis in Computer Engineering at University of Syiah Kuala, under the supervision of my mentors (anonymous for privacy).

**Key Contribution**: Empirical evidence that modern deep learning architectures have internalized preprocessing, making classical techniques obsolete - with implications for production CV systems.

---
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Ultralytics** for excellent YOLO implementation
- **My mentors** for guidance and feedback
- **University of Syiah Kuala** Computer Engineering department
- **Indonesia traffic dataset contributors** (anonymous for privacy)

---

## 📧 Contact

**Abassuci**
- LinkedIn: www.linkedin.com/in/abassuci-abassuci-963222246
- Email: Abassuci21@gmail.com

---

**⭐ If you found this research useful, please star this repository!**

---

*Last updated: [Today's Date]*
