# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 17:10:13 2020

@author: Shraddha Bhat
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

#########Salary########
salary_test = pd.read_csv("E:\\Excelr\\naive bayes\\SalaryData_Test.csv")
salary_train = pd.read_csv("E:\\Excelr\\naive bayes\\SalaryData_Train.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
number = preprocessing.LabelEncoder()

for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]


sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 79.46%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+469))  # 77.49%

