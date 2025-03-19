import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import os

# General settings
# Modify the file path, feature selection, and other parameters as needed
file_path = "path_to_your_file/totalRank.xlsx"  # Path to the Excel file
target_column = 'SubRank'  # The name of the target column
n_videos = 29  # The total number of videos, update this as per your dataset

# Feature columns for different subsets (generalized for various features based on all frames or keyframes)
a = [3, 4, 5, 7, 10, 11, 12, 13, 15, 17]  # All features based on all frames
b = [3, 4, 6, 8, 9, 11, 12, 14, 16, 18]  # All features based on keyframes
c = [12, 13]  # 2 features based on all frames
d = [12, 14]  # 2 features based on keyframes
e = [12]  # 1 feature based on all frames
f = [14]  # 1 feature based on keyframes

# Load data
if os.path.exists(file_path):
    rank = pd.read_excel(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

Sub = rank[target_column]  # Target column
Key = rank.iloc[:, b]  # Selecting keyframe features

# Pre-allocate arrays to store results
key_Pre_total = np.zeros(n_videos)  # Predictions for all samples

# RMSE function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Cross-validation for training set
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 10-fold cross-validation

# 1st-9th Cross-validation (training set)
for i, (train_idx, test_idx) in enumerate(kf.split(Key)):
    if i == 9:  # Skip the last fold for training
        break
    
    x_train, x_test = Key.iloc[train_idx], Key.iloc[test_idx]
    y_train, y_test = Sub.iloc[train_idx], Sub.iloc[test_idx]
    
    # Train RandomForestRegressor on training set
    rf_key = RandomForestRegressor(n_estimators=100)
    rf_key.fit(x_train, y_train)
    
    # Predict on the test set
    key_Pre = rf_key.predict(x_test)
    key_Pre_total[test_idx] = key_Pre

# 10th Cross-validation (final test on last fold)
x_train, x_test = Key.iloc[kf.split(Key).__next__()[0]], Key.iloc[kf.split(Key).__next__()[1]]
y_train, y_test = Sub.iloc[kf.split(Key).__next__()[0]], Sub.iloc[kf.split(Key).__next__()[1]]

rf_key = RandomForestRegressor(n_estimators=100)
rf_key.fit(x_train, y_train)
key_Pre = rf_key.predict(x_test)
key_Pre_total[test_idx] = key_Pre

# Compute evaluation metrics
RMSE_key = rmse(Sub, key_Pre_total)
MAE_key = mean_absolute_error(Sub, key_Pre_total)
r_P, _ = pearsonr(key_Pre_total, Sub)
rsq = r_P**2
r_S, _ = spearmanr(key_Pre_total, Sub)

# Output the results
print(f'RMSE (Keyframes): {RMSE_key}')
print(f'MAE (Keyframes): {MAE_key}')
print(f'Pearson R^2: {rsq}')
print(f'Spearman R: {r_S}')

# Cross-validation to test the model (on keyframes)
rsq = np.zeros(10)
rs = np.zeros(10)
ps = np.zeros(10)
RMSE = np.zeros(10)
MAE = np.zeros(10)

# Cross-validation (testing the model)
for i, (train_idx, test_idx) in enumerate(kf.split(Key)):
    if i == 9:  # Skip the last fold for testing
        break
    
    x_train, x_test = Key.iloc[train_idx], Key.iloc[test_idx]
    y_train, y_test = Sub.iloc[train_idx], Sub.iloc[test_idx]
    
    # Train RandomForestRegressor on training set
    rf_key_pre = RandomForestRegressor(n_estimators=300, min_samples_leaf=5)
    rf_key_pre.fit(x_train, y_train)
    
    # Predict on the test set
    Key_Pre = rf_key_pre.predict(x_test)
    
    # Evaluate performance using different metrics
    r_P, _ = pearsonr(Key_Pre, y_test)
    rsq[i] = r_P**2
    r_S, _ = spearmanr(Key_Pre, y_test)
    rs[i] = r_S
    ps[i] = _
    RMSE[i] = rmse(y_test, Key_Pre)
    MAE[i] = mean_absolute_error(y_test, Key_Pre)

# Final test on the last fold
x_train, x_test = Key.iloc[kf.split(Key).__next__()[0]], Key.iloc[kf.split(Key).__next__()[1]]
y_train, y_test = Sub.iloc[kf.split(Key).__next__()[0]], Sub.iloc[kf.split(Key).__next__()[1]]

rf_key_pre = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, oob_score=True)
rf_key_pre.fit(x_train, y_train)
Key_Pre = rf_key_pre.predict(x_test)

r_P, _ = pearsonr(Key_Pre, y_test)
rsq[9] = r_P**2
r_S, _ = spearmanr(Key_Pre, y_test)
rs[9] = r_S
ps[9] = _
RMSE[9] = rmse(y_test, Key_Pre)
MAE[9] = mean_absolute_error(y_test, Key_Pre)

# Mean performance across all folds
rsq_mean = np.mean(rsq)
rs_mean = np.mean(rs)
ps_mean = np.mean(ps)
RMSE_mean = np.mean(RMSE)
MAE_mean = np.mean(MAE)

print(f'Mean RSQ: {rsq_mean}')
print(f'Mean RS: {rs_mean}')
print(f'Mean Spearman: {ps_mean}')
print(f'Mean RMSE: {RMSE_mean}')
print(f'Mean MAE: {MAE_mean}')

# Feature Importance (from OOB prediction error)
imp = rf_key_pre.feature_importances_

# Plot feature importance
plt.figure()
plt.bar(range(len(imp)), imp)
plt.xlabel('Predictors')
plt.ylabel('Importance Estimates')
plt.xticks(range(len(imp)), rf_key_pre.feature_names_in_, rotation=45)
plt.show()

# Scatter plot of predicted vs true values (on training set)
plt.figure()
plt.scatter(Key_Pre, y_train)
plt.plot([1, n_videos], [1, n_videos], '--', color='r')  # Adjust this based on number of videos
plt.xlabel('Predicted Complexity')
plt.ylabel('Subjective Complexity')
plt.show()
