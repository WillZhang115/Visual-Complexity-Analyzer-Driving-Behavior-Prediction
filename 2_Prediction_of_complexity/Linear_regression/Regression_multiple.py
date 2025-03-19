import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# Load the data (Make sure to replace the file paths with your actual file paths)
all_frame_path = "your_file_path/totalRank.xlsx"  # <-- Replace with your actual file path

# Read data from Excel
AllFrame = pd.read_excel(all_frame_path)
SubRank = AllFrame['SubRank']
SizeMp4 = AllFrame['SizeMp4']
SizeZip = AllFrame['SizeZip']
fcAll = AllFrame['FCall']
fcKey = AllFrame['FCkey']
seAll = AllFrame['SEall']
seKey = AllFrame['SEkey']
edKey = AllFrame['EdgeDenKey']
edAll = AllFrame['EdgeDenAll']
edVideo = AllFrame['EdgeVideo']
carNum = AllFrame['CarNum']
ssimAll = AllFrame['SsimAll']
ssimKey = AllFrame['SsimKey']
colorAll = AllFrame['ColorAll']
colorKey = AllFrame['ColorKey']
contrastAll = AllFrame['ContrastAll']
contrastKey = AllFrame['ContrastKey']

# Create the linear regression models for each predictor
def linear_regression_predictor(X, y):
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)
    y_fit = model.predict(X.reshape(-1, 1))
    return y_fit

# Generate predictions for each predictor
yfit1 = linear_regression_predictor(SizeMp4, SubRank)
yfit2 = linear_regression_predictor(SizeZip, SubRank)
yfit3 = linear_regression_predictor(fcAll, SubRank)
yfit4 = linear_regression_predictor(fcKey, SubRank)
yfit5 = linear_regression_predictor(seAll, SubRank)
yfit6 = linear_regression_predictor(seKey, SubRank)
yfit7 = linear_regression_predictor(edKey, SubRank)
yfit8 = linear_regression_predictor(edAll, SubRank)
yfit9 = linear_regression_predictor(edVideo, SubRank)
yfit10 = linear_regression_predictor(carNum, SubRank)
yfit11 = linear_regression_predictor(ssimAll, SubRank)
yfit12 = linear_regression_predictor(ssimKey, SubRank)
yfit13 = linear_regression_predictor(colorAll, SubRank)
yfit14 = linear_regression_predictor(colorKey, SubRank)
yfit15 = linear_regression_predictor(contrastAll, SubRank)
yfit16 = linear_regression_predictor(contrastKey, SubRank)

# Combine all the predicted values into a matrix
yfit = np.column_stack([yfit1, yfit2, yfit3, yfit4, yfit5, yfit6,
                        yfit7, yfit8, yfit9, yfit10, yfit11, yfit12,
                        yfit13, yfit14, yfit15, yfit16])

# Calculate the Spearman correlation
rs, ps = spearmanr(yfit, SubRank)

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(SubRank, yfit))
mae = mean_absolute_error(SubRank, yfit)

# Print Spearman correlation, RMSE, and MAE
print(f"Spearman correlation: {rs}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# Plot the scatter plots and regression lines for each predictor
fig, axes = plt.subplots(4, 4, figsize=(15, 15))
axes = axes.flatten()

# List of the predictors for looping
predictors = [SizeMp4, SizeZip, fcAll, fcKey, seAll, seKey, edKey, edAll, 
              edVideo, carNum, ssimAll, ssimKey, colorAll, colorKey, contrastAll, contrastKey]
yfits = [yfit1, yfit2, yfit3, yfit4, yfit5, yfit6, yfit7, yfit8, 
         yfit9, yfit10, yfit11, yfit12, yfit13, yfit14, yfit15, yfit16]

labels = [
    'M1 File Size (MP4)', 'M2 File Size (Zip)', 'M3 Feature congestion (All frames)', 'M4 Feature congestion (Keyframes)',
    'M5 Subband entropy (All frames)', 'M6 Subband entropy (Keyframes)', 'M7 Edge density (All frames)', 'M8 Edge density (Keyframes)',
    'M9 File size (Canny video)', 'M10 Number of objects', 'M11 SSIM (All frames)', 'M12 SSIM (Keyframes)',
    'M13 Colourfulness (All frames)', 'M14 Colourfulness (Keyframes)', 'M15 Contrast (All frames)', 'M16 Contrast (Keyframes)'
]

# Loop to plot each scatter plot with regression line
for i, (predictor, yfit, label) in enumerate(zip(predictors, yfits, labels)):
    axes[i].scatter(predictor, SubRank, label='Actual', color='blue')
    axes[i].plot(predictor, yfit, label='Fitted', color='red')
    axes[i].set_xlabel(label)
    axes[i].set_ylabel('Subjective ranking')
    axes[i].legend()

# Show all the plots
plt.tight_layout()
plt.show()
