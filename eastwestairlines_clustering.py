# -*- coding: utf-8 -*-
"""
Created on Fri May 29 17:55:58 2020

@author: acer
"""


import pandas as pd

import numpy as np
import matplotlib.pylab as plt
from  sklearn.cluster	import	KMeans
from scipy.spatial.distance import cdist 
from sklearn import preprocessing
# Kmeans on University Data set 
Airlines = pd.read_excel("E:\Excelr\clustering\EastWestAirlines.xlsx")

Airlines.head()
Airlines.isnull().sum()
# no null in the data
Airlines.dtypes
Airlines.columns
Airlines.apply(lambda col:pd.to_numeric(col, errors='coerce'))
Airlines.columns
def norm_func(i):
    x = (i-i.min())	/	(i.max()	-	i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airlines.iloc[:,0:])
df_norm.head(10)

###### screw plot or elbow curve ############
k = list(range(2,8))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(df_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,df_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

# Scree plot 
plt.plot(k,TWSS, 'go-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(df_norm)
model.labels_
model.labels_ # getting the labels of clusters assigned to each row 
md = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Airlines['value'] = model.labels_ # creating a  new column and assigning it to new column 
Airlines.head(10)
Airlines.value.value_counts()
Airlines.head(10)

Airlines.iloc[:,1:7].groupby(Airlines.value).mean()

##Hirerichial 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch

type(df_norm)

#p = np.array(df_norm) # converting into numpy array format 
help(linkage)
z = linkage(df_norm, method="complete",metric="euclidean")

plt.figure(figsize=(15, 5));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

help(linkage)

# Now applying AgglomerativeClustering choosing 3 as clusters from the dendrogram
from	sklearn.cluster	import	AgglomerativeClustering 
h_complete	=	AgglomerativeClustering(n_clusters=3,	linkage='complete',affinity = "euclidean").fit(df_norm) 

h_complete.labels_

cluster_labels=pd.Series(h_complete.labels_)

Airlines['value']=cluster_labels # creating a  new column and assigning it to new column 
Airlines.value.value_counts()
Airlines.groupby(Airlines.value).mean()
Airlines.head(10)
Airlines = Airlines.iloc[:,[7,0,1,2,3,4,5,6]]


# creating a csv file 
Airlines.to_csv("Airlines_csv") #,encoding="utf-8")


#DENDROGRAM OF AVERAGE LINKAGE METHOD
#p = np.array(df_norm) # converting into numpy array format 
Z = linkage(df_norm, method="average",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    Z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

#DENDROGRAM OF SINGLE LINKAGE METHOD
#p = np.array(df_norm) # converting into numpy array format 
Z = linkage(df_norm, method="single",metric="euclidean")
plt.figure(figsize=(15, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(
    Z,
    leaf_rotation=0.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the dendrogram(optional)
from sklearn.cluster import	AgglomerativeClustering 

h_complete = AgglomerativeClustering( n_clusters=5, linkage='complete', affinity = "euclidean").fit(df_norm) 


cluster_labels=pd.Series(h_complete.labels_) 

h_complete.labels_

# creating a  new column and assigning it to new column 

Airlines['value']=cluster_labels
#print(Airlines)
Airlines.head(10)

# getting aggregate mean of each cluster
Airlines.iloc[:,2:].groupby(Airlines.value).mean()
