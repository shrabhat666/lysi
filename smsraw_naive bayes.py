# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 21:54:47 2020

@author: Shraddha Bhat
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

##importing the data####
sms_raw = pd.read_csv("E:\\Excelr\\naive bayes\\sms_raw_NB.csv",encoding = "ISO-8859-1")

# cleaning data 
import re
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>2:
            w.append(word)
    return (" ".join(w))

sms_raw.text = sms_raw.text.apply(cleaning_text)

# removing empty rows 
sms_raw.shape
sms_raw = sms_raw.loc[sms_raw.text != " ",:]

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return (i.split(" "))

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

sms_train,sms_test = train_test_split(sms_raw,test_size=0.3)

# Preparing email texts into word count matrix format 
sms_raws = CountVectorizer(analyzer=split_into_words).fit(sms_raw.text)

# For all messages
all_sms_matrix = sms_raws.transform(sms_raw.text)
all_sms_matrix.shape # (5559,7429)

# For training messages
train_sms_matrix = sms_raws.transform(sms_train.text)
train_sms_matrix.shape # (3891,7429)

# For testing messages
test_sms_matrix = sms_raws.transform(sms_test.text)
test_sms_matrix.shape # (1668,7429)

####### Without TFIDF matrices ########################
# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB


# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_sms_matrix,sms_train.type)
train_pred_m = classifier_mb.predict(train_sms_matrix)
accuracy_train_m = np.mean(train_pred_m==sms_train.type) # 98.86%

test_pred_m = classifier_mb.predict(test_sms_matrix)
accuracy_test_m = np.mean(test_pred_m==sms_test.type) # 97.78%

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_sms_matrix.toarray(),sms_train.type.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_sms_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==sms_train.type) # 93.85%

test_pred_g = classifier_gb.predict(test_sms_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==sms_test.type) # 87.23%



from sklearn.datasets import make_blobs
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
# x axis values 
x = [train_sms_matrix] 
# corresponding y axis values 
y = [test_sms_matrix] 

X,y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
fig, X,y= plt.subplots()

sms_raw.scatter(X[:,1], X[:, 2], c=y, s=50, cmap='twilight_shifted_r')
sms_raw.set_title('Naive Bayes Model', size=14)

xlim = (-8, 8)
ylim = (-15, 5)

xg = np.linspace(xlim[0], xlim[1], 60)
yg = np.linspace(ylim[0], ylim[1], 40)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

for label, color in enumerate(['red', 'blue']):
    mask = (y == label)
    mu, std = X[mask].mean(0), X[mask].std(0)
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    sms_raw.pcolorfast(xg, yg, Pm.reshape(xx.shape), alpha=0.5,
                  cmap=color.title() + 's')
    sms_raw.contour(xx, yy, P.reshape(xx.shape),
               levels=[0.01, 0.1, 0.5, 0.9],
               colors=color, alpha=0.2)
    
sms_raw.set(xlim=xlim, ylim=ylim)


