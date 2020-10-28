# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:54:00 2020

@author: Shraddha Bhat
"""


import pandas as pd 
import numpy as np
wine = pd.read_csv("E:\\Excelr\\PCA\\wine.csv")
wine.describe()
wine.head()
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#droping type column and storing it in y 
x = wine.drop('Type',1)
y=wine.iloc[:,0]
x.head()
y
# Standardizing the features
x = StandardScaler().fit_transform(x)
pca = PCA()
pca_values = pca.fit_transform(x)
pca_values.shape

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1
#array([ 36.2 ,  55.41,  66.53,  73.6 ,  80.16,  85.1 ,  89.34,  92.02,
   #     94.24,  96.17,  97.91,  99.21, 100.01])
# Variance plot for PCA components obtained 
plt.plot(var1,color="green")

# plot between PCA1 and PCA2 
x1 = np.array(pca_values[:,0])
y1 = np.array(pca_values[:,1])
z = np.array(pca_values[:,2])
plt.plot(x1,y1,"go")

###################  hierarchial clustering by using first three PCA components  ##########################
df_norm = pd.DataFrame(pca_values[:,0:3])
df_norm
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as shc # for creating dendrogram 

type(df_norm)
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(df_norm, method='complete'))
plt.axhline(y=7, color='r', linestyle='--')

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)

wine['clust']=cluster_labels # creating a  new column and assigning it to new column 
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
wine.head()
################## Correlation between clustering grouping and initial grouping##################
r = np.corrcoef(wine.Type, wine.clust)
r
plt.scatter(wine.Type, wine.clust)
plt.show()
##### a moderate positive correlation 0.5654798#################################################
############as it having a little difference ie., cluster 0 is similar to type 1################
############### but cluster 1 is similar to type 3 #############################################
############### and cluster 2 is similar to type 2 #############################################
#
######         getting aggregate mean of each cluster   #######################################       
###############################################################################################
wine.iloc[:,2:15].groupby(wine.clust).mean()
wine.iloc[:,2:15].groupby(wine.Type).mean()


