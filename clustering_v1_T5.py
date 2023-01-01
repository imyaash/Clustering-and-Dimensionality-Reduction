# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:58:04 2022

@author: imyaash-admin
"""

"""Import necessary libraries"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, Birch
from sklearn.metrics import silhouette_score

"""Load the dataset and drop the Class column"""
df = pd.read_csv("Datasets/divorce.csv", sep = ";")
df = df.drop("Class", axis = 1)

"""Create a pairplot of the data"""
sns.pairplot(df)

"""Scale the data using StandardScaler"""
scaler = StandardScaler()
dfScaled = scaler.fit_transform(df)

"""Perform PCA on the original data with 2 components for visualisation"""
pca = PCA(n_components = 2, random_state = 99)
dfDecayed = pca.fit_transform(df)
sum(pca.explained_variance_ratio_)  # Calculating and printing the explained variance ratio for the PCA

"""Perform PCA on the scaled data with 2 components for visualisation"""
pca = PCA(n_components = 2, random_state = 99)
dfDecayedScaled = pca.fit_transform(dfScaled)
sum(pca.explained_variance_ratio_)  # Calculating and printing the explained variance ratio for the PCA

"""Create a MiniBatchKMeans model with 2 clusters"""
mbClustering = MiniBatchKMeans(n_clusters = 2, init = "k-means++", verbose = 1, random_state = 99)
mbClusters = mbClustering.fit_predict(dfScaled) # Fit the model to the scaled data and predict the clusters
mbClustering.inertia_ # Calculate and print the inertia of the resulting clusters
# Calculate and print the silhouette score for the MiniBatchKMeans model
mbScore = silhouette_score(df, mbClusters)
print("Silhouette Score for Mini Batch K-Means Clusters:", mbScore)

# Plotting a scatter plot to visualise the clusters
colours = ["red", "blue"]
plt.figure(figsize = (8, 8))
for i in range(2):
    plt.scatter(dfDecayedScaled[mbClusters == i, 0], dfDecayedScaled[mbClusters == i, 1],
                c = colours[i], s = 200, label = "Cluster " + str(i + 1))
plt.title("Clusters as per Mini Batch K-Means Clustering Model")
plt.legend()
plt.show()

"""Create a Birch model with 2 clusters"""
birchClustering = Birch(n_clusters = 2)
birchClusters = birchClustering.fit_predict(df) # Fit the model to the original data and predict the clusters
# Calculate and print the silhouette score for the Birch model
birchScore = silhouette_score(df, birchClusters)
print("Silhouette Score for Birch Clusters:", birchScore)

# Plotting a scatter plot to visualise the clusters
colours = ["red", "blue"]
plt.figure(figsize = (8, 8))
for i in range(2):
    plt.scatter(dfDecayed[birchClusters == i, 0], dfDecayed[birchClusters == i, 1],
                c = colours[i], s = 200, label = "Cluster " + str(i + 1))
plt.title("Clusters as per Birch Clustering Model")
plt.legend()
plt.show()