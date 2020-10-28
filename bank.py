# -*- coding: utf-8 -*-
"""
Created on Mon May 18 06:02:59 2020

@author: Shraddha Bhat
"""

# =============================================================================
# Logistic regression for client has subscribed a term deposit or not
# =============================================================================

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
#Importing Data
bank = pd.read_csv("E:\\Excelr\\logistic regression\\bank-full.csv")
bank.head()
bank.isnull().sum()
# no null in the data
bank.dtypes
bank.columns

bank.describe()

bank['default'] = bank['default'].apply(lambda x: 0 if x=='no' else 1)
bank['housing'] = bank['housing'].apply(lambda x: 0 if x=='no' else 1)
bank['loan'] = bank['loan'].apply(lambda x: 0 if x=='no' else 1)
bank['y'] = bank['y'].apply(lambda x: 0 if x=='no' else 1)

# function for converting months variable to number
def month_converter(mnth):
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    return months.index(mnth) + 1
bank['month'] = bank['month'].apply(lambda x: month_converter(x))

bank.job.value_counts()
bank.job.unique() 
# Import label encoder 
from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
  
# Encode labels in column 'species'. 
bank.job = label_encoder.fit_transform(bank.job) 
bank.job.unique() 
bank.education = label_encoder.fit_transform(bank.education) 
bank.education.unique() 
bank.contact = label_encoder.fit_transform(bank.contact) 
bank.contact.unique() 
bank.poutcome = label_encoder.fit_transform(bank.poutcome) 
bank.poutcome.unique() 
bank.marital = label_encoder.fit_transform(bank.marital)
bank.marital.unique()

bank.education.value_counts()

import seaborn as sns
#plt.boxplot(card.age)

from scipy import stats
import scipy.stats as st
##st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome', data = bank).fit()

#summary
logit_model.summary()
y_pred = logit_model.predict(bank)
y_pred
bank["pred_prob"] = y_pred
# Creating new column for storing predicted class of y

# filling all the cells with zeroes
bank["Att_val"] = 0
# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
bank.loc[y_pred>=0.5,"Att_val"] = 1
bank.Att_val

from sklearn.metrics import classification_report
classification_report(bank.Att_val,bank.y)

# confusion matrix 
confusion_matrix = pd.crosstab(bank['y'],bank.Att_val)
confusion_matrix
accuracy = (39142+1155)/(39147+1155+4134+775) # 89.14%
accuracy

# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(bank.y, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc #0.87188

### Dividing data into train and test data sets
bank.drop("Att_val",axis=1,inplace=True)
bank.drop
from sklearn.model_selection import train_test_split

train,test = train_test_split(bank,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('y~age+job+marital+education+default+balance+housing+loan+contact+day+month+duration+campaign+pdays+previous+poutcome', data = bank).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train.iloc[:,0:])
print(train.iloc[:,0:])
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
train["train_pred"] = np.zeros

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1
train_pred

# confusion matrix 
confusion_matrix = pd.crosstab(train['y'],train.train_pred)

confusion_matrix
accuracy_train = (27442+795)/(27442+795+562+2848) # 89.22%
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)
test_pred
# Creating new column for storing predicted class of Attorney

# filling all the cells with zeroes
test["test_pred"] = np.zeros(13564)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1
test_pred
# confusion matrix 
confusion_matrix = pd.crosstab(test['y'],test.test_pred)

confusion_matrix
accuracy_test = (11705+360)/(11705+360+213+1286) # 88.95%
accuracy_test

