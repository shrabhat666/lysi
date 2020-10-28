# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 18:08:31 2020

@author: acer
"""


# Importing Libraries 
import pandas as pd
import numpy as np
glass = pd.read_csv("E:\\Excelr\\KNN\\glass.csv")
glass.describe()
glass.head()
glass.groupby('Type').size()
# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2,random_state= 0) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)

# Fitting with training data 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) 
train_acc #  83.62%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) 
test_acc  #  55.81%

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])

acc
# FROM GRAPH WE CONSIDER  for 5 nearest neighbours
# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 5)

# Fitting with training data 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
y_pred = neigh.predict(test.iloc[:,0:9])
y_pred

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) # 94 %
train_acc #  77.19%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) # 100%
test_acc  #  58.13%
