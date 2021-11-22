# -*- coding: utf-8 -*-
"""820 assignment 1 .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cSSPhOGYrqUTiQxBaO8s2gjCmICDBQfI
"""

# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

# scipy
from scipy.spatial.distance import pdist

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
# scikit
from sklearn.metrics import pairwise_distances 
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import MDS
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

forums = pd.read_pickle("/content/drive/MyDrive/820/BA820/forums.pkl")
forums.describe().T

from google.colab import drive
drive.mount('/content/drive')

forums

forums.shape

"""##Exploratory Data Analysis, Data Cleaning, and Preprocessing"""

! pip install umap-learn

#drop missing values
forums.dropna()
forums.shape

#drop duplicate values
forums.drop_duplicates(inplace=True)
forums.shape

#drop string col
forums.drop(columns='text',inplace=True)
forums.shape

"""##Hierarchical Clustering and K-Means Clustering

###Hierarchical Clustering:
"""

df_f=pd.DataFrame(forums)
# create distance matrix
fd = pdist(forums.values)

# squareform --- visualize get a sense thing
sns.heatmap(squareform(fd), cmap="Reds")
plt.show()

# first cluster -- ward
hc1 = linkage(fd,method='ward',metric='euclidean')
type(hc1)

hc1

# first dendrogram
plt.figure(figsize=(15,15))
dendrogram(hc1,
           color_threshold = 20
           )
plt.title('Dendrogram')
plt.axvline(x=20,c='gray',lw=1,linestyle='dashed')

plt.show()

dendrogram(hc1)
plt.show()

# second cluster -- complete
hc_com = linkage(fd,method='complete',metric='euclidean')
type(hc_com)

# second dendrogram
plt.figure(figsize=(15,15))
dendrogram(hc_com,
           color_threshold = 20
           )
plt.axvline(x=20,c='gray',lw=1,linestyle='dashed')

plt.show()

"""##PCA"""

cor_f=forums.corr()
sns.heatmap(cor_f,cmap='Reds',center=0)
plt.show()

pca=PCA(0.9)
pcs=pca.fit_transform(forums)
pcs.shape

varf = pca.explained_variance_ratio_
varf.shape

#plot
plt.title('explained variance ratio of forums.csv')
sns.lineplot(range(1,len(varf)+1),varf)  # +1 because the x-axis
plt.show()

plt.title('CUM explained variance ratio of forums.csv')
sns.lineplot(range(1,len(varf)+1),np.cumsum(varf))  # +1 because the x-axis
plt.axhline(.9)
plt.show()

"""##KMeans--First time

"""

# scale, because clearly these are not on the same scale, and I want to ensure each variable has equal weight
sc = StandardScaler()
xf = sc.fit_transform(pcs)
X = pd.DataFrame(xf, index=forums.index)

X.shape

from sklearn.metrics import silhouette_samples, silhouette_score

# PRE - Kmeans
KS = range(2, 15)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  km.fit(X)
  labs = km.predict(X)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(X, labs))

#Plot PRE-Kmeans
plt.figure(figsize=(15,5))


plt.subplot(1, 2, 1)
plt.title("Inertia")
sns.lineplot(KS, inertia)

plt.subplot(1, 2, 2)
plt.title("Silohouette Score")
sns.lineplot(KS, silo)

plt.show()

"""##T-SNE"""

np.sum(pcs.explained_variance_ratio_)

#put pcs to a new dataset
forums_pcs=pd.DataFrame(pcs,index=forums.index)

tsne = TSNE()
tsne.fit(pcs)

# get the embeddings
te = tsne.embedding_

# the shape
te.shape

#2d tsne dataframe
d2data=pd.DataFrame(te, columns=['e1','e2'])
d2data.head(3)

#tsne plot
PAL = sns.color_palette("bright", 10) 
plt.figure(figsize=(6, 4))
sns.scatterplot(x="e1", y="e2", data=d2data, legend="full", palette=PAL)

"""##K-Means ( after PCA and T-SNE)

"""

#K-means clustering
KS = range(2, 15)

# storage
inertia = []
silo = []

for k in KS:
  km = KMeans(k)
  labs = km.fit_predict(d2data)
  inertia.append(km.inertia_)
  silo.append(silhouette_score(d2data, labs))

#plot inertia
sns.lineplot(KS,inertia)
plt.title('interia')
plt.axvline(x = 7,color='blue')
plt.axvline(x = 6,color='red')
plt.show()

#plot silo-score
sns.lineplot(KS,silo)
plt.title('silhouette score')
plt.axvline(x = 7,color='blue')
plt.axvline(x = 6,color='red')
plt.show()

#k=6
k6=KMeans(6)
k6_labs=k6.fit_predict(d2data)

# metrics
k6_silo = silhouette_score(d2data, k6_labs)
k6_ssamps = silhouette_samples(d2data, k6_labs)
np.unique(k6_labs)

!pip install scikit-plot
import scikitplot as skplot

skplot.metrics.plot_silhouette(d2data, k6_labs, title="KMeans - 6", figsize=(5,5))
plt.show()

forums2=forums.copy()
forums2['k6_labs'] = k6_labs

k6profile = forums2.groupby('k6_labs').mean()

sc6 = StandardScaler()
k6profile_scaled = sc6.fit_transform(k6profile)

plt.figure(figsize=(15,5))
pal = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(k6profile_scaled, center=0, cmap=pal, xticklabels=k6profile.columns)

#k=7
k7=KMeans(7)
k7_labs=k7.fit_predict(d2data)

k7_silo = silhouette_score(d2data, k7_labs)
k7_ssamps = silhouette_samples(d2data, k7_labs)
np.unique(k7_labs)

skplot.metrics.plot_silhouette(d2data, k7_labs, title="KMeans - 7", figsize=(5,5))

forums['k3_labs'] = k3_labsplt.show()

forums3=forums.copy()
forums3['k7_labs'] = k7_labs

k7profile = forums3.groupby('k7_labs').mean()

sc7 = StandardScaler()
k7profile_scaled = sc7.fit_transform(k7profile)

plt.figure(figsize=(15,5))
pal = sns.color_palette("vlag", as_cmap=True)
sns.heatmap(k7profile_scaled, center=0, cmap=pal, xticklabels=k7profile.columns)