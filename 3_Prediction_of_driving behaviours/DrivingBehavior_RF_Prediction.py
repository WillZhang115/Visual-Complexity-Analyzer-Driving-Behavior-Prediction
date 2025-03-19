import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# General path setup
def load_data(file_path):
    """
    Load data file, assuming it's either an Excel or CSV file.
    :param file_path: str, the path to the data file
    :return: pandas DataFrame, the loaded data
    """
    _, file_extension = os.path.splitext(file_path)
    
    if file_extension == '.csv':
        data = pd.read_csv(file_path)
    elif file_extension == '.xlsx':
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    return data

def preprocess_intersection_data():
    """
    Preprocess intersection data for video ID mapping.
    :return: pandas DataFrame, the updated intersection data
    """
    # Read intersection and rank data
    Inter = load_data("newIntersection.xlsx")
    Rank = load_data("totalRank.xlsx")

    # Update VideoID based on intersection and direction
    for i in range(1, 13):
        r = str(i)
        for direction, suffix in zip([1, 2, 3], ["L", "R", "B"]):
            mask = (Inter['intersection_ID'] == f"TJ_{r}") & (Inter['enter_direction_num'] == direction)
            Inter.loc[mask, 'VideoID'] = f"T0{r}{suffix}"

    # TDL, TDR, TDB mapping
    for direction, suffix in zip([1, 2, 3], ["L", "R", "B"]):
        mask = (Inter['intersection_ID'] == "TJ_DL_1") & (Inter['enter_direction_num'] == direction)
        Inter.loc[mask, 'VideoID'] = f"TD{suffix}"

    Inter['VideoID'] = Inter['VideoID'].astype(str)
    
    # Join with rank data
    InterRank = pd.merge(Inter, Rank, left_on='VideoID', right_on='VideoID', how='inner')

    return InterRank

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=5):
    """
    Train a Random Forest model and evaluate performance.
    :param X_train: pandas DataFrame, features for training
    :param y_train: pandas DataFrame, target for training
    :param X_test: pandas DataFrame, features for testing
    :param y_test: pandas DataFrame, target for testing
    :param n_estimators: int, number of trees in the forest
    :param max_depth: int, maximum depth of each tree
    :return: trained Random Forest model, predicted values
    """
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(((y_pred - y_test) ** 2).mean())
    mae = np.abs(y_pred - y_test).mean()

    return rf, y_pred, rmse, mae

def plot_results(y_test, y_pred, label):
    """
    Plot the predicted vs actual values.
    :param y_test: actual target values
    :param y_pred: predicted values
    :param label: label for the plot
    """
    plt.scatter(y_test, y_pred)
    plt.plot([0, 1], [0, 1], color='red', linewidth=2)  # line of equality
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"True vs Predicted {label}")
    plt.show()

def main():
    # Preprocess the data
    InterRank = preprocess_intersection_data()

    # Define feature columns
    Allfeatures = [5, 20, 23]  # Adjust based on the required features
    Keyfeatures = [5, 11, 12, 14, 16, 17, 19, 20, 22, 24, 26]  # Adjust based on key features

    # Split data into training and testing sets
    a = np.random.permutation(len(InterRank))
    b = a[:1056]
    c = a[1057:1508]

    # Train and evaluate the Random Forest for the full feature set
    X_All_train = InterRank.iloc[b, Allfeatures]
    Y_ComDis_train = InterRank['commit_distance'].iloc[b]
    Y_SpeMin_train = InterRank['speed_min'].iloc[b]
    Y_SpeDro_train = InterRank['relative_speed_drop'].iloc[b]

    X_All_test = InterRank.iloc[c, Allfeatures]
    Y_ComDis_test = InterRank['commit_distance'].iloc[c]
    Y_SpeMin_test = InterRank['speed_min'].iloc[c]
    Y_SpeDro_test = InterRank['relative_speed_drop'].iloc[c]

    rf_CoD, ComDis_pred, RMSE_CoD, MAE_CoD = train_and_evaluate_rf(X_All_train, Y_ComDis_train, X_All_test, Y_ComDis_test)
    rf_SpM, SpeMin_pred, RMSE_SpM, MAE_SpM = train_and_evaluate_rf(X_All_train, Y_SpeMin_train, X_All_test, Y_SpeMin_test)
    rf_SpD, SpeDro_pred, RMSE_SpD, MAE_SpD = train_and_evaluate_rf(X_All_train, Y_SpeDro_train, X_All_test, Y_SpeDro_test)

    # Plot results
    plot_results(Y_ComDis_test, ComDis_pred, "Commit Distance")
    plot_results(Y_SpeMin_test, SpeMin_pred, "Speed Min")
    plot_results(Y_SpeDro_test, SpeDro_pred, "Speed Drop")

    # Save results to file (e.g., 'driving_pre_all_select.xlsx')
    results = pd.DataFrame({
        'RMSE_CoD': [RMSE_CoD],
        'MAE_CoD': [MAE_CoD],
        'RMSE_SpM': [RMSE_SpM],
        'MAE_SpM': [MAE_SpM],
        'RMSE_SpD': [RMSE_SpD],
        'MAE_SpD': [MAE_SpD]
    })

    results.to_excel('driving_pre_all_select.xlsx', index=False)

if __name__ == "__main__":
    main()
