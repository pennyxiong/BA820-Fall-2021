# imports 

# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

# our dataset
SQL = "SELECT * from `questrom.datasets.mtcars`"
YOUR_BILLING_PROJECT = "questrom"
cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

# what do we have
type(cars)
cars.shape

# numpy ---- really simple dataset
x = np.array([1,3])
y = np.array([3,4])
z = np.array([2,4])
a = np.stack([x,y,z])
type(a)
a_df = pd.DataFrame(a)

# create our first distance matrix
d1 = pdist(a)
d1
type(d1)
a

# the condensed representation -> squareform
squareform(d1)

# cosine
cd = pdist(a, metric="cosine")
squareform(cd)

# cityblock
cb = pdist(a, metric="cityblock")
squareform(cb)

# sklearn
pairwise_distances(a_df, metric="euclidean")

# cars
cars.head(3)
cars.dtypes


# exercise
# model as the index
# make sure the model doesnt exist -- just a numeric dataframe
# exploration

cars.sample(3)
cars.index = cars.model
cars.sample(3)

# numeric matrix is desired
cars.drop(columns="model", inplace=True)
cars.head(3)

# quick look
cars.describe().T

# keep just the columns of interest
COLS = ['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qsec', 'vs', 'am', 'gear','carb']
cars2 = cars[COLS]
cars2.head(3)

# cars distance -- passing in the numpy array from the pandas frame via values
cdist = pdist(cars2.values)

# squareform --- visualize get a sense thing
sns.heatmap(squareform(cdist), cmap="Reds")
plt.show()

# our first cluster!
hc1 = linkage(cdist)
type(hc1)

# our first dendrogram
dendrogram(hc1, labels=cars.index)
plt.show()

help(linkage)

# our second plot
DIST = 10
plt.figure(figsize=(5,6))
dendrogram(hc1, 
           labels = cars.index,
           orientation = "left", 
           color_threshold = DIST)
plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()


# how identify the clusters
fcluster(hc1, 2, criterion="maxclust")
cars2['cluster1'] = fcluster(hc1, 2, criterion="maxclust")
cars2.head(3)

# "profiling"
cars2.cluster1.value_counts()


# how about distance assignment
c2 = fcluster(hc1, 80, criterion="distance")
c2
cars2['cluster2'] = c2
cars2.head(3)

# different linkage methods
linkage(cars2.values, method=)

# standardizing values
cars2.describe()