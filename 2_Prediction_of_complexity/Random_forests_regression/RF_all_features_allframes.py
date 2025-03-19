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
feature_columns = [3, 4, 5, 7, 10, 11, 12, 13, 15, 17]  # Columns for feature selection
target_column = 'SubRank'  # The name of the target column

# Adjustable parameters
n_estimators = 300  # Number of trees in the random forest
min_leaf_sizes = [5, 10, 20, 50, 100, 200, 500]  # Different minimum leaf sizes
kf_splits = 10  # Number of splits for cross-validation

# Read data from the file
if os.path.exists(file_path):
    rank = pd.read_excel(file_path)
else:
    raise FileNotFoundError(f"File not found: {file_path}")

# Extract the target and feature columns
Sub = rank[target_column]
all_features = rank.iloc[:, feature_columns]

# Pre-allocate arrays to store results
rsq = np.zeros(kf_splits)
rs = np.zeros(kf_splits)
ps = np.zeros(kf_splits)
RMSE = np.zeros(kf_splits)
MAE = np.zeros(kf_splits)

# Set colors for the plot
col = ['r', 'g', 'b', 'c', 'm']

# Plot the OOB error for different leaf sizes
plt.figure('RF Leaves and Trees')
for i in range(5):
    rf_model = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_leaf_sizes[i], oob_score=True)
    rf_model.fit(all_features, Sub)
    oob_error = np.mean((rf_model.oob_prediction_ - Sub)**2)
    plt.plot(range(1, len(oob_error)+1), oob_error, color=col[i])

plt.xlabel('Number of Grown Trees')
plt.ylabel('Mean Squared Error')
plt.legend([str(leaf) for leaf in min_leaf_sizes[:5]], loc='northeast')
plt.title('Number of Leaves')
plt.show()

# Function to calculate RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Cross-validation: training and testing
kf = KFold(n_splits=kf_splits, shuffle=True, random_state=42)
all_Pre_total = np.zeros(len(all_features))

# Loop through cross-validation splits
for i, (train_idx, test_idx) in enumerate(kf.split(all_features)):
    x_train, x_test = all_features.iloc[train_idx], all_features.iloc[test_idx]
    y_train, y_test = Sub.iloc[train_idx], Sub.iloc[test_idx]
    
    # Train a random forest regressor
    rf_all = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5)
    rf_all.fit(x_train, y_train)
    
    # Predict on the training set
    all_Pre = rf_all.predict(x_train)
    
    # Pearson and Spearman correlation
    r_p, _ = pearsonr(all_Pre, y_train)
    rsq[i] = r_p**2
    r_s, _ = spearmanr(all_Pre, y_train)
    rs[i] = r_s
    
    # Calculate RMSE and MAE
    RMSE[i] = rmse(y_train, all_Pre)
    MAE[i] = mean_absolute_error(y_train, all_Pre)

# Calculate average values
rsq_mean = np.mean(rsq)
rs_mean = np.mean(rs)
ps_mean = np.mean(ps)
RMSE_mean = np.mean(RMSE)
MAE_mean = np.mean(MAE)

print(f'RSQ Mean: {rsq_mean}, RS Mean: {rs_mean}, PS Mean: {ps_mean}')
print(f'RMSE Mean: {RMSE_mean}, MAE Mean: {MAE_mean}')

# Testing on the test set (out-of-sample performance)
for i, (train_idx, test_idx) in enumerate(kf.split(all_features)):
    x_train, x_test = all_features.iloc[train_idx], all_features.iloc[test_idx]
    y_train, y_test = Sub.iloc[train_idx], Sub.iloc[test_idx]
    
    # Train a random forest regressor
    rf_all = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=5)
    rf_all.fit(x_train, y_train)
    
    # Predict on the test set
    all_Pre_total[test_idx] = rf_all.predict(x_test)

# Calculate performance metrics on the entire dataset
RMSE_all = rmse(Sub, all_Pre_total)
MAE_all = mean_absolute_error(Sub, all_Pre_total)
r_p, _ = pearsonr(all_Pre_total, Sub)
rsq_final = r_p**2
r_s, _ = spearmanr(all_Pre_total, Sub)

print(f'Final RMSE: {RMSE_all}, MAE: {MAE_all}')
print(f'Final RSQ: {rsq_final}, Spearman RS: {r_s}')
