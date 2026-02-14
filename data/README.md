# Dataset Information

This folder contains the traffic detection dataset used for thesis research.

## Dataset Overview
- **Total images**: 6,000+ labeled traffic images
- **Selected for experiments**: 2,000 systematically chosen images
- **Source**: Banda Aceh street traffic (collected Sep 2024 - Jan 2025)
- **Annotation tool**: LabelImg (YOLO format)

## Classes
- Car
- Motorcycle
- Truck
- Bus
- Bicycle
- Pedestrian

## Conditions Covered
- **Lighting**: Dawn, dusk, night
- **Weather**: Overcast, rain
- **Traffic density**: Light, moderate, heavy

## Data Structure
```
data/
├── images/          # Original traffic images
├── labels/          # YOLO format annotations (.txt)
└── splits/
    ├── train/       # Training set (80%)
    ├── val/         # Validation set (10%)
    └── test/        # Test set (10%)
```

## Sample Statistics
- Average image size: 640x640
- File format: JPG
- Total dataset size: ~12GB

⚠️ **Note**: Dataset not included in repository due to size constraints 
and privacy considerations. Sample images available upon request.

## Citation
If using this dataset structure:
```
Abassuci, A. (2026). Mixed Traffic Detection Dataset - Banda Aceh. 
Universitas Syiah Kuala.
```
