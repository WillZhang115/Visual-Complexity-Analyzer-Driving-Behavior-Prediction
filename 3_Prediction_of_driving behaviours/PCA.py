import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def perform_pca(data, n_components=2):
    """
    Perform PCA analysis.
    :param data: pandas DataFrame, the input data
    :param n_components: int, the number of principal components to keep
    :return: PCA object, transformed data
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create PCA model
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Return PCA results (principal components) and explained variance ratio
    return pca, principal_components

def plot_pca(principal_components, pca, explained_variance_ratio):
    """
    Plot PCA components and explained variance chart.
    :param principal_components: PCA transformed data
    :param pca: PCA model
    :param explained_variance_ratio: the explained variance ratio of each principal component
    """
    # Plot the PCA components chart
    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1])
    plt.title("PCA: First and Second Principal Components")
    plt.xlabel(f"Principal Component 1 ({explained_variance_ratio[0]:.2f} explained variance)")
    plt.ylabel(f"Principal Component 2 ({explained_variance_ratio[1]:.2f} explained variance)")
    plt.show()

    # Plot the explained variance ratio chart
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(explained_variance_ratio)+1), explained_variance_ratio, alpha=0.7, color='g')
    plt.title("Explained Variance Ratio by Principal Component")
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()

def main():
    # Input file path (this can be changed to user-specified path)
    file_path = "your_file_path/datafile.xlsx"  # <-- Replace with actual path
    
    # Load data
    data = load_data(file_path)
    
    # Select features for PCA (assuming data starts from the second column)
    features = data.iloc[:, 1:]  # From the second column onwards, assuming these are the features
    
    # Perform PCA
    pca, principal_components = perform_pca(features, n_components=2)
    
    # Get explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained Variance Ratio:", explained_variance_ratio)
    
    # Plot PCA results
    plot_pca(principal_components, pca, explained_variance_ratio)

    # Optionally: Save principal components to a file
    pca_df = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(principal_components.shape[1])])
    pca_df.to_excel("pca_result.xlsx", index=False)
    print("PCA results saved to pca_result.xlsx")

if __name__ == "__main__":
    main()
