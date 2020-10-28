# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 20:01:07 2020

@author: acer
"""


# Importing Libraries 
import pandas as pd 
import numpy as np
zoo = pd.read_csv("E:\\Excelr\\KNN\\Zoo.csv")
zoo.describe()
zoo.groupby('animal name').size()

# Training and Test data using 
from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2,random_state= 0) # 0.2 => 20 percent of entire data 

# KNN using sklearn 
# Importing Knn algorithm from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier as KNC

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,20,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0])
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot           and             # test accuracy plot

plt.plot(np.arange(3,20,2),[i[0] for i in acc],"bo-")
plt.plot(np.arange(3,20,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])

acc

# from the above graph, it is suggested to go for 3 nearest neighbours 
neigh = KNC(n_neighbors= 3)
# Fitting with training data 
neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
y_pred = neigh.predict(test.iloc[:,1:18])
y_pred
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0]) # 94 %
train_acc #27.5%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0]) # 100%
test_acc  #4.76%
