# Clustering-and-Dimensionality-Reduction
This project aims to predict whether a couple is likely to get divorced or not based on various factors such as age, number of children, education, and religious values.

# Data
The data used for this project is the Divorce dataset from the UCI Machine Learning Repository. It consists of 170 records, with 9 features and a target variable ('Class') indicating whether the couple divorced (1) or not (0).

# Preprocessing
The data is preprocessed by dropping the 'Class' column and standardizing the features using StandardScaler.

# Clustering
Two clustering models are used to group the couples into two clusters:

    MiniBatchKMeans
    Birch

The clusters predicted by both models are visualized using a scatter plot. The silhouette score is also calculated for both models to evaluate the quality of the clusters.

# Visualization
A pairplot of the data is also created using seaborn to visualize the relationships between the different features. Principal Component Analysis (PCA) is performed on the original and standardized data to reduce the dimensionality of the data for visualization purposes. The explained variance ratio for the PCA is also calculated and printed.

# Requirements
To run this code, you will need the following libraries:

    pandas
    seaborn
    matplotlib
    sklearn

# Running the code
To run the code, simply run the divorce_prediction.py file. The preprocessing, clustering, and visualization steps will be performed automatically. The resulting plots and scores will be displayed.
