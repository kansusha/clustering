import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 
'exec(%matplotlib inline)'
filename = 'cars_clus.csv'

#Read csv
pdf = pd.read_csv(filename)
print ("Shape of dataset: ", pdf.shape)

pdf.head(5)

#Data Cleaning
#lets simply clear the dataset by dropping the rows that have null value:
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)
#Feature selection
#Lets select our feature set:
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length',
 'curb_wgt', 'fuel_cap', 'mpg']]

#Normalization
#Now we can normalize the feature set. MinMaxScaler transforms features 
#by scaling each feature to a given range. It is by default (0, 1). 
#That is, this estimator scales and translates 
#each feature individually such that it is between zero and one.
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]
#Clustering using Scipy
#In this part we use Scipy package to cluster the dataset:
#First, we calculate the distance matrix.
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
#In agglomerative clustering, at each iteration, the algorithm must 
#update the distance matrix to reflect the distance of the newly formed cluster 
#with the remaining clusters in the forest. The following methods are supported 
#in Scipy for calculating the distance between the newly formed cluster and 
#each: - single - complete - average - weighted - centroid
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')
#Essentially, Hierarchical clustering does not require a pre-specified number 
#of clusters. However, in some applications we want a partition of 
#disjoint clusters just as in flat clustering. So you can use a cutting line:
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id],
    int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, 
leaf_font_size =12, orientation = 'right')
#Clustering using scikit-learn
dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)
agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_
pdf['cluster_'] = agglom.labels_
pdf.head()
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))


