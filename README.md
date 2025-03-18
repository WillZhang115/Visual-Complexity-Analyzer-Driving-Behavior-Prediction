# Visual-Complexity-Analyzer-Driving-Behavior-Prediction
A comprehensive toolkit for visual feature calculation, complexity prediction, and driving behavior analysis.

Project Structure
```
├── 1_Calculation_methods/
│   ├── Colourfulness.py          - Image color richness calculation
│   ├── Edge_density.py           - Edge concentration analysis
│   ├── FC_SE.py                  - Feature Congestion and Subband Entropy** analysis
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
│       └── requirements.txt      - Dependencies

├── 2_Prediction_of_complexity/
│   ├── Linear_regression/
│   │   ├── All_Subset_Regression_Model.py  - Exhaustive feature selection
│   │   ├── Regression_multiple.py          - All-feature regression model
│   │   └── Regression_single.py            - **Single-feature regression** analysis
│   └── Random_forests_regression/
│       ├── RF_all_features_allframes.py    - Full-frame RF analysis
│       └── RF_all_features_keyFrames.py    - Keyframe RF analysis

└── 3_Prediction_of_driving_behaviours/
    ├── DrivingBehavior_RF_Prediction.py    - Behavior classification
    └── PCA.py                              - Dimensionality reduction
```
