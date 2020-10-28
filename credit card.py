# -*- coding: utf-8 -*-
"""
Spyder Editor

Shraddha Bhat.
"""

# =============================================================================
# Logistic regression for the application for a credit card accepted or not
# =============================================================================

import pandas as pd 
import numpy  as np
import matplotlib.pyplot as plt
#Importing Data
ccard = pd.read_csv("E:\\Excelr\\logistic regression\\creditcard.csv")
ccard.head()
ccard.isnull().sum()
# no null in the data
ccard.dtypes
ccard.columns
ccard.share
ccard.reports

#'card', 'reports', 'age', 'income', 'share','expenditure', 'owner', 'selfemp', 'dependents', 'months', 'majorcards','active'
ccard.describe()
ccard.drop(["Unnamed: 0"],inplace=True,axis = 1)
ccard['selfemp'] = ccard['selfemp'].apply(lambda x: 0 if x=='no' else 1)
ccard['card'] = ccard['card'].apply(lambda x: 0 if x=='no' else 1)
ccard['owner'] = ccard['owner'].apply(lambda x: 0 if x=='no' else 1)

#correlation based feature selction
data = ccard
corr=ccard.corr()
corr
def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    print(dataset)
    print(dataset)
correlation(data, 0.8)
data.columns  #expenditure column is eliminated
#'card', 'reports', 'age', 'income', 'share', 'owner', 'selfemp', 'dependents', 'months', 'majorcards', 'active'
corr1=data.corr()
corr1
import seaborn as sns
sns.heatmap(corr)

# Feature Selection with Univariate Statistical Tests
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data.iloc[:,1:12]  # 10 independent columns
y = data.iloc[:,0]    #target column i.e card
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=4)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(4,'Score'))  #print 4 best features
# top 4 columns/ features selected by univariante statistics method are
# reports, active, share, owner
# hence final credit card dataset for modelling is crcard selecting top 3 features

crcard = ccard[['card', 'reports', 'active', 'share']]
crcard
from scipy import stats
import scipy.stats as st
##st.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

#Model building 
import statsmodels.formula.api as sm
logit_model = sm.logit('card~reports+active+share',data=crcard).fit(method='bfgs')
#p value for shares = 1.. hence we can discard shares variable in the model
#summary
corr2=crcard.corr()
corr2
#correlation for shares and expenditure is more hence we can discard shares
logit_model.summary()
y_pred = logit_model.predict(crcard)
y_pred
crcard["pred_prob"] = y_pred
# Creating new column for storing predicted class of card

# filling all the cells with zeroes
crcard["Att_val"] = 0

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
crcard.loc[y_pred>=0.5,"Att_val"] = 1
crcard.Att_val

from sklearn.metrics import classification_report
classification_report(crcard.Att_val,crcard.card)

# confusion matrix 
confusion_matrix = pd.crosstab(crcard['card'],crcard.Att_val)
confusion_matrix
#before feature selection
#accuracy = (290+999)/(290+999+24+6) # 97.72

# after feature selection by univariant analysis 
accuracy = (286+986)/(286+986+10+37) 
accuracy  # 96.43
# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(crcard.card, y_pred)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")
 # before feature selection 0.9953

roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 
roc_auc  # after feature selection 0.99.25


### Dividing data into train and test data sets
crcard.drop("Att_val",axis=1,inplace=True)
from sklearn.model_selection import train_test_split

train,test = train_test_split(crcard,test_size=0.3)

# checking na values 
train.isnull().sum();test.isnull().sum()

# Building a model on train data set 

train_model = sm.logit('card~reports+active+share',data = train).fit()

#summary
train_model.summary()
train_pred = train_model.predict(train.iloc[:,1:])
train

# Creating new column for storing predicted class of card

# filling all the cells with zeroes
train["train_pred"] = np.zeros(923)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[train_pred>0.5,"train_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(train['card'],train.train_pred)

confusion_matrix
# before feature selction 98.15
accuracy_train = (193+711)/(193+711+2+17) # after feature selction 97.94
accuracy_train

# Prediction on Test data set

test_pred = train_model.predict(test)

# Creating new column for storing predicted class of card

# filling all the cells with zeroes
test["test_pred"] = np.zeros(396)

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
test.loc[test_pred>0.5,"test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test['card'],test.test_pred)

confusion_matrix
# before feature selction 98.73
accuracy_test = (101+288)/(101+288+0+7) #  after feature selction 98.23
accuracy_test

