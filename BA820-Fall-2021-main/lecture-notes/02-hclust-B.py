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

SQL = "SELECT * from `questrom.datasets.mtcars`"
YOUR_BILLING_PROJECT = "questrom"
cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)
cars.shape

# numpy
x = np.array([1,2])
y = np.array([3,4])
z = np.array([2,4])
a = np.stack([x,y,z])
a
a_df = pd.DataFrame(a)
a_df


# start with our first
d1 = pdist(a)
d1

# squareform
squareform(d1)

# manhattan
cb = pdist(a, metric="cityblock")
squareform(cb)

# cosine
cs = pdist(a, metric="cosine")
squareform(cs)

# sklearn
pairwise_distances(a_df, metric="euclidean")

# cars
cars.shape
cars.sample(3)

# exercise
# move the model column to the index
# remove the model column
# a quick look at the dataset and see if anything with respect to the encoding of the data

cars.head(3)
cars.index = cars.model 
cars.drop(columns="model", inplace=True)

cars.describe().T

cars.cyl.value_counts()

cars.head(3)
cdist = pdist(cars.values)

cdist.shape

# visualize the dataset
sns.heatmap(squareform(cdist), cmap="Reds")
plt.show()


# lets build our first cluster solution!
# by default, single linkage
hc1 = linkage(cdist)

# what do we have
type(hc1)
hc1.shape
len(cars)
hc1

# lets create our first dendrogram
dendrogram(hc1, labels=cars.index)
plt.show()

# tweak 
DIST = 80
plt.figure(figsize=(5,6))
dendrogram(hc1, 
           labels = cars.index,
           orientation = "left", 
           color_threshold = DIST)
plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
plt.show()


# fcluster - identify the number of clusters
fcluster(hc1, 80, criterion="distance")
c1 = fcluster(hc1, 80, criterion="distance")
cars['cluster1'] = c1
cars.head(3)

# number of clusters we want
c2 = fcluster(hc1, 2, criterion="maxclust")
c2
cars['cluster2'] = c2
cars.head(3)

cars.describe().T