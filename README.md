# Visual-Complexity-Analyzer-Driving-Behavior-Prediction

This repository contains code for analyzing traffic scene complexity based on computer vision techniques and predicting driving behaviors using machine learning models.

## Repository Structure
```
├── 1_Calculation_methods/
│   ├── Colourfulness.py          - Image color richness calculation
│   ├── Edge_density.py           - Edge concentration analysis
│   ├── FC_SE.py                  - Feature Congestion and Subband Entropy analysis
│   ├── RMS_Contrast.py           - Root Mean Square contrast measurement
│   ├── Structual_Similarity.py   - Structural similarity index (SSIM)
│   └── Vehicle_detection_yolo/   - YOLOv8-based vehicle detection system
│       ├── car_number.py         - Vehicle counting from video streams
│       ├── data/
│       │   ├── examples/         - Sample test media
│       │   ├── datasets/         - Dataset configuration
│       │   └── scripts/          - Data conversion utilities
│       ├── models/               - Model training/export scripts
│       ├── inference/            - Detection implementation
│       ├── runs/                 - Output storage
│
├── 2_Prediction_of_complexity/
│   ├── Linear_regression/
│   │   ├── All_Subset_Regression_Model.py  - Exhaustive feature selection
│   │   ├── Regression_multiple.py          - All-feature regression model
│   │   └── Regression_single.py            - Single-feature regression analysis
│   └── Random_forests_regression/
│       ├── RF_all_features_allframes.py    - Full-frame RF analysis
│       └── RF_all_features_keyFrames.py    - Keyframe RF analysis
│
└── 3_Prediction_of_driving_behaviours/
    ├── DrivingBehavior_RF_Prediction.py    - Behavior classification
    └── PCA.py                              - Dimensionality reduction
│
│── requirements.txt      - Dependencies
```

## Features
- **Traffic Complexity Analysis:** Uses visual and statistical methods to assess the complexity of traffic scenes.
- **Vehicle Detection:** Utilizes YOLOv8 to detect and count vehicles in traffic videos.
- **Regression Models:** Implements multiple regression techniques to predict traffic complexity.
- **Random Forest Prediction:** Uses Random Forest models to predict traffic complexity and driving behaviors.
- **PCA for Feature Reduction:** Principal Component Analysis (PCA) is applied for dimensionality reduction.

## Installation
Clone the repository and install the dependencies:
```sh
git clone https://github.com/WillZhang115/Visual-Complexity-Analyzer-Driving-Behavior-Prediction.git
cd traffic-complexity-analysis
pip install -r requirements.txt
```

## Usage
### Running Vehicle Detection
```sh
python 1_Calculation_methods/Vehicle_detection_yolo/car_number.py --source path/to/video.mp4
```

### Running Traffic Complexity Prediction
```sh
python 2_Prediction_of_complexity/Random_forests_regression/RF_all_features_allframes.py
```

### Running Driving Behavior Prediction
```sh
python 3_Prediction_of_driving_behaviours/DrivingBehavior_RF_Prediction.py
```

## Dependencies
See `requirements.txt` for package dependencies.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
