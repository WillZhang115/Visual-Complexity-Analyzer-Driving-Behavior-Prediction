import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data (Make sure to replace the file paths with your own)
all_frame_path = "your_file_path/totalRank.xlsx"  # <-- Replace with your actual file path
combine_all_path = "your_file_path/combine linear all.xlsx"
combine_key_path = "your_file_path/combine linear key.xlsx"

# Read the data
AllFrame = pd.read_excel(all_frame_path)
SubRank = AllFrame['SubRank']

# Single Predictor Linear Models
predictors = ['SizeMp4', 'SizeZip', 'FCall', 'FCkey', 'SEall', 'SEkey', 'EdgeDenKey', 'EdgeDenAll', 'EdgeVideo', 
              'CarNum', 'SsimAll', 'SsimKey', 'ColorAll', 'ColorKey', 'ContrastAll', 'ContrastKey']

models = {}
for predictor in predictors:
    models[predictor] = sm.OLS(SubRank, sm.add_constant(AllFrame[predictor])).fit()

# MAE of Single Linear Predictor
obs = AllFrame.iloc[:, 2:18].values
per = AllFrame.iloc[:, 1].values
mae1 = np.mean(np.abs(obs - per))
print("MAE of Single Linear Predictor:", mae1)

# Save MAE to file
mae_df = pd.DataFrame([mae1], columns=['MAE'])
mae_df.to_excel("MAE_Single_Linear.xlsx", index=False)

# Combine Linear with All Frames
combine_all = pd.read_excel(combine_all_path)
rank = pd.read_excel(all_frame_path)

# Dynamically calculate the number of rows and columns based on the size of `combine_all`
num_rows = combine_all.shape[0]
num_models = 5  # Number of models you want to generate (change if needed)

combine_allframe = np.zeros((num_rows, num_models))
for i in range(num_models):
    combine_allframe[:, i] = combine_all.iloc[i, 0] + combine_all.iloc[i, 1] * rank['CarNum'] + \
                             combine_all.iloc[i, 2] * rank['SsimAll'] + combine_all.iloc[i, 3] * rank['SizeMp4'] + \
                             combine_all.iloc[i, 4] * rank['ContrastAll'] + combine_all.iloc[i, 5] * rank['SizeZip'] + \
                             combine_all.iloc[i, 6] * rank['EdgeVideo']

# Save the combined frame results
combine_allframe_df = pd.DataFrame(combine_allframe, columns=['Car', 'Car_SSIM', 'Car_SSIM_SizeMp4', 'Car_SSIM_SizeMp4_Contrast', 'Car_SSIM_SizeZip_Contrast_EdgeVideo'])
combine_allframe_df.to_excel('combine_linear_all_rank.xlsx', index=False)

# F-test and accuracy based on all frames
def run_anova(model):
    return sm.stats.anova_lm(model)

# Model 1: CarNum
mdl_Car = sm.OLS(SubRank, sm.add_constant(AllFrame['CarNum'])).fit()
print("ANOVA for CarNum model:")
print(run_anova(mdl_Car))

# Model 2: CarNum + SsimAll
Car_SSIM = sm.OLS(SubRank, sm.add_constant(AllFrame[['CarNum', 'SsimAll']])).fit()
print("ANOVA for CarNum + SsimAll model:")
print(run_anova(Car_SSIM))

# Model 3: CarNum + SsimAll + SizeMp4
Car_SSIM_SizeMp4 = sm.OLS(SubRank, sm.add_constant(AllFrame[['CarNum', 'SsimAll', 'SizeMp4']])).fit()
print("ANOVA for CarNum + SsimAll + SizeMp4 model:")
print(run_anova(Car_SSIM_SizeMp4))

# Additional models and F-tests (you can add more combinations here like in the original code)

# Combine Linear with Key Frames
combine_key = pd.read_excel(combine_key_path)

# Dynamically calculate the number of rows and columns based on the size of `combine_key`
combine_keyframe = np.zeros((num_rows, num_models))  # Same dynamic row size as the "all" dataset
for i in range(num_models):
    combine_keyframe[:, i] = combine_key.iloc[i, 0] + combine_key.iloc[i, 1] * rank['CarNum'] + \
                             combine_key.iloc[i, 2] * rank['SsimAll'] + combine_key.iloc[i, 3] * rank['ColorKey'] + \
                             combine_key.iloc[i, 4] * rank['SizeZip'] + combine_key.iloc[i, 5] * rank['EdgeVideo'] + \
                             combine_key.iloc[i, 6] * rank['EdgeDenKey']

combine_keyframe_df = pd.DataFrame(combine_keyframe, columns=['SSIM', 'Car_SSIM', 'Car_ED_EV', 'Car_ED_EV_SizeZip', 'Car_ED_EV_SizeZip_Color'])
combine_keyframe_df.to_excel('combine_linear_key_rank.xlsx', index=False)

# Residual and QQ Plot for the models
# Residual plot
x1 = sm.add_constant(AllFrame['CarNum'])
y1 = SubRank
b, bint, r, rint, stats = sm.OLS(y1, x1).fit().params, None, None, None, None
plt.figure(figsize=(10, 6))
plt.scatter(r, rint)
plt.title("Residuals vs Fitted")
plt.xlabel("Residuals")
plt.ylabel("Fitted values")
plt.show()

# QQ Plot for Car_SSIM
sm.qqplot(Car_SSIM.fittedvalues, line ='45')
plt.show()

# Normal plot for Car_SSIM
sm.graphics.tsa.plot_acf(Car_SSIM.fittedvalues)
plt.show()
