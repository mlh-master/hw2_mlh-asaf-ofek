#!/usr/bin/env python
# coding: utf-8

# Theoretical questions
# Q1: To evaluate how well our model performs at T1D classification, we need to have evaluation metrics that measures of its performances/accuracy. Which evaluation metric is more important to us: model accuracy or model performance? Give a simple example that illustrates your claim.
# 
# A1: Performance means how good our model is doing its job. Accuracy is the number of correct predictions made by the model by the total number of records. For us, performance is more important beacuse accuracy can be deceiving if the data is not representive enough.
# For instance, if we use a binary naive classifier to determine whether a person has cancer, However the data only includes 5 positive cases out of 100 patients, we get 95% accuracy.
# 
# Q2: T1D is often associated with other comorbidities such as a heart attack. You are asked to design a ML algorithm to predict which patients are going to suffer a heart attack. Relevant patient features for the algorithm may include blood pressure (BP), body-mass index (BMI), age (A), level of physical activity (P), and income (I). You should choose between two classifiers: the first uses only BP and BMI features and the other one uses all of the features available to you. Explain the pros and cons of each choice.
# 
# A2: 
# 
# Using BMI and BP:
#     
#     pros
#       - Weights will be easier to find.
#       - The calculations will be simpler.
#       - Data cleanning will only include two coloumns.
#     cons
#       - We can't be sure that these are the most important features. This can lead to diminished performance and crucial mistakes. 
#         For example, a false negative case; One may not be treated well which can lead to devastating consequences.
#     
# Using all features:
# 
#     pros
#       - The model will most likely perform better and will have better accuracy (under the assumption that our data is reliable enough)
#     cons
#       - Data exploration and cleanning will be difficult.
#       - We'll have a lot of data meanning extended computation time.
#     
#         
# 
# Q3: A histologist wants to use machine learning to tell the difference between pancreas biopsies that show signs of T1D and those that do not. She has already come up with dozens of measurements to take, such as color, size, uniformity and cell-count, but she isn’t sure which model to use. The biopsies are really similar, and it is difficult to distinguish them from the human eye, or by just looking at the features. Which of the following is better: logistic regression, linear SVM or nonlinear SVM? Explain your answer.
# 
# A3:
# 
# We can assume that the data won't be linearly seperable since the samples look alike and have similar properties.
# Thus, we will use nonlinear SVM to find the correct boundary line.
# 
# Q4: What are the differences between LR and linear SVM and what is the difference in the effect/concept of their hyper-parameters tuning?
# 
# A4: LR - returns the probabilities to be a part of certain class.
# 

# In[2]:


import pandas as pd 
import numpy as np
from pathlib import Path
import random
import distutils
# from Additional_functions import *
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
df = pd.read_csv('HW2_data.csv')
def nan2num(dataframe):
    """
    :param dataframe: Pandas series of features
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    # dataframe = pd.DataFrame(dataframe).drop(extra_feature, 1)
    c_cdf = {}
    c_cdf = dataframe
    for column in dataframe.columns:
        if column == 'Age':
            i=0
        else:
            replacing_value = np.random.choice(dataframe[column])
            c_cdf[column].replace(to_replace=np.nan, value=replacing_value, inplace=True)
    return pd.DataFrame(c_cdf)

def str_to_bool_series(s):
    for index, value in s.items():
        if value=='Yes':
            s[index] = True
        elif value=='No':
            s[index] = False
        elif value=='Male':
            s[index] = True
        elif value == 'Female':
            s[index] = False
        elif value=='Positive':
            s[index] = True
        elif value == 'Negative':
            s[index] = False
#         elif(value==1):
#             s[index] = True
#         elif(value==0):
#             s[index] = False
    return s
t1d_clean=nan2num(df)
for coloumn in t1d_clean.columns:
    t1d_clean[coloumn]=str_to_bool_series(t1d_clean[coloumn])
# print(df)
# In[ ]:
diagnosis = t1d_clean['Diagnosis']
X_train, X_test, Y_train, y_test = train_test_split(t1d_clean, np.ravel(diagnosis), test_size=0.2,
                                                    random_state=0, stratify=np.ravel(diagnosis))
# print(X_train)
# print(X_test)

def print_dist_features(df, x_train, x_test):
    print("Feature            | Train % | Test % | Difference")
    for column in df:
        if(column=='Age'):
            age_mean_train = round(x_train[column].mean(),2)
            print(column ,end =  " "*(21-len(column)))
            print(age_mean_train, end = " "*(5))
            age_mean_test = round(X_test[column].mean(),2)
            print(age_mean_test, end = " "*(5))
            difference = round(age_mean_train-age_mean_test,2)
            print(difference, end = " "*(5))
            print()
        elif(column=='Family History'):
            train_count = round(x_train[column].value_counts(ascending = True)/len(x_train[column])*100,2)
            test_count = round(x_test[column].value_counts(ascending = True)/len(x_test[column])*100,2)
            difference = round(train_count[1]-test_count[1],2)
            print(column, end=" "*(21-len(column)))
            print(train_count[1], end = " "*(10-len(str(train_count[1]))))
            print(test_count[1], end = " "*(10-len(str(test_count[1]))))
            print(difference)
        else:
            train_count = round(x_train[column].value_counts(ascending = True)/len(x_train[column])*100,2)
            test_count = round(x_test[column].value_counts(ascending = True)/len(x_test[column])*100,2)
            difference = round(train_count[True]-test_count[True],2)
            print(column, end=" "*(21-len(column)))
            print(train_count[True], end = " "*(10-len(str(train_count[True]))))
            print(test_count[True], end = " "*(10-len(str(test_count[True]))))
            print(difference)

print_dist_features(df, X_train, X_test)

import seaborn as sbn
import matplotlib.ticker as ticker

def feature_corr(t1d_df):
    fig, axes = plt.subplots(3, 6,figsize=(12, 6) )
    sbn.set_context("paper", font_scale=0.7)
    fig.suptitle("Relationship between features and label")
    i=0
    for column in t1d_df:
        if(column=='Gender'):
            feat_lab = sbn.countplot(ax = axes[i//6,i%6], x='Gender', hue = 'Diagnosis', data = t1d_df)
            feat_lab.set(xticklabels=['Male', 'Female'])
        elif(column=='Age'):
            feat_lab = sbn.countplot(ax = axes[i//6,i%6], x='Age', hue = 'Diagnosis', data = t1d_df)
            feat_lab.xaxis.set_major_locator(ticker.LinearLocator(10))
        elif(i>1):
            feat_lab = sbn.countplot(ax = axes[i//6,i%6], x=column, hue = 'Diagnosis', data = t1d_df)
            feat_lab.set(xticklabels=['No', 'Yes'])
        i+=1
    plt.show()
    return()

feature_corr(t1d_clean)

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
# transform data
onehot = encoder.fit_transform(t1d_clean.drop(columns = ['Age']))
print(onehot)

#Section 5
# x_train_orig, x_val_orig, y_train, y_val= train_test_split(X_train, Y_train, test_size = 0.20, random_state = 10, stratify=Y_train)
# scaler = StandardScaler()
# max_iter=2000
# x_train = scaler.fit_transform(x_train_orig)
# x_val = scaler.transform(x_val_orig)
# log_reg = LogisticRegression(random_state=5, penalty='none', max_iter=max_iter, solver='lbfgs')
# log_reg.fit(x_train, y_train)
# y_pred_train = log_reg.predict_proba(x_train)
# y_pred_val = log_reg.predict_proba(x_val)
# print("Train loss is {:.2f}".format(log_loss(y_train,y_pred_train)))
# print("Validation loss is {:.2f}".format(log_loss(y_val,y_pred_val)))


