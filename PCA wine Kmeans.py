# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:40:02 2020

@author: acer
"""


import pandas as pd 
import numpy as np
wine = pd.read_csv("E:\\Excelr\\PCA\\wine.csv")
wine.describe()
wine.head()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
wine.data = wine.iloc[:,1:]
wine.data.head(4)
#droping type cloumn
x = wine.drop('Type',1)
y=wine.iloc[:,0]
x
y
# Normalizing the numerical data 
wine_normal = scale(x)
pca = PCA()
pca_values = pca.fit_transform(wine_normal)
pca_values.shape

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var
pca.components_[0]

# Cumulative variance 

var1 = np.cumsum(np.round(var,decimals = 4)*100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1,color="green")

# plot between PCA1 and PCA2 
x = np.array(pca_values[:,0])
y = np.array(pca_values[:,1])
z = np.array(pca_values[:,2])
plt.plot(x,y,"go")

################### performing clustering for the first three PCA components  ##########################
new_df = pd.DataFrame(pca_values[:,0:3])
new_df
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(new_df)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(new_df.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,new_df.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))


# Scree plot 
plt.plot(k,TWSS, 'go-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)
########from the elbow curve the optimum number of clusters is choosen as 3##########################
kmeans = KMeans(n_clusters = 3)
kmeans.fit(new_df)
kmeans.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(kmeans.labels_)  # converting numpy array into pandas series object 
wine['clust']=md # creating a  new column and assigning it to new column 
wine.head()
wine = wine.iloc[:,[14,0,1,2,3,4,5,6,7,8,9,10,11,12,13]]

################## Correlation between clustering grouping and initial grouping##################
r = np.corrcoef(wine.Type, wine.clust)
r

######         getting aggregate mean of each cluster   #######################################       
###############################################################################################
wine.iloc[:,2:15].groupby(wine.clust).mean()
wine.iloc[:,2:15].groupby(wine.Type).mean()

