# Learning goals:
## Expand on Distance and now apply Kmeans
## - Kmeans applications
## - Evaluate cluster solutions 
## - hands on with Kmeans and quick review of DBSCan

# resources
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py
# https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Scikit_Learn_Cheat_Sheet_Python.pdf
# https://scikit-learn.org/stable/modules/clustering.html#dbscan


# installs
# notebook/colab
# ! pip install scikit-plot

# local/server
# pip install scikit-plot



# imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import fcluster

from sklearn import metrics 
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt


# COLAB setup ------------------------------------------
# from google.colab import auth
# auth.authenticate_user()
# PROJECT = ''    # <------ change to your project
# SQL = "SELECT * from `questrom.datasets.judges`"
# judges = pd.read_gbq(SQL, PROJECT)


# dataset urls:
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/Election08.csv
# https://vincentarelbundock.github.io/Rdatasets/csv/Stat2Data/MedGPA.csv


# useful code snippets below ---------------------------------

# scale the data
# el_scaler = StandardScaler()
# el_scaler.fit(election)
# election_scaled = el_scaler.transform(election)

# kmeans
# k5 = KMeans(5,  n_init=100)
# judges['k5'] = k5.fit_predict(j)


# k5_centers = k5.cluster_centers_
# sns.scatterplot(data=judges, x="CONT", y="INTG", cmap="virdis", hue="k5")
# plt.scatter(k5_centers[:,0], k5_centers[:,1], c="g", s=100)


# KRANGE = range(2, 30)
# # containers
# ss = []
# for k in KRANGE:
#   km = KMeans(k)
#   lab = km.fit_predict(j)
#   ss.append(km.inertia_)


# skplt.metrics.plot_silhouette(j, k5.predict(j), figsize=(7,7))

