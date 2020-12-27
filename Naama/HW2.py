#!/usr/bin/env python
# coding: utf-8

## HW2 - Daniel & Naama

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures

# section 1 - import and preprocessing:
file = Path.cwd().joinpath('HW2_data.csv')
T1D_dataset = pd.read_csv(file)

random.seed(10)

# Check maximum amount of missing values in one feature:
from functions import nancount as ncount
nan_count = ncount(T1D_dataset)
print(nan_count)
# The largest number of nan's is in the feature 'Increased Thirst' and is equal to 20 (out of 565)
# This is a relatively small number, so we don't have to through the feature
# In most of the features there are no nan's at all

# Plot a histogram of the diagnosis before and after we drop rows with missing values:
feat = 'Diagnosis_Positive'
T1D_dummy = pd.get_dummies(T1D_dataset, dummy_na=False, drop_first=True)  # turn all the columns to 0/1 (except 'Age')
T1D_diag = pd.DataFrame(T1D_dummy[feat])

T1D_no_missing = T1D_dataset.copy().dropna()
T1D_no_missing_dummy = pd.get_dummies(T1D_no_missing, dummy_na=False, drop_first=True)
T1D_diag_no_missing = pd.DataFrame(T1D_no_missing_dummy[feat])

fig, axs = plt.subplots(1, 2)
plt.setp(axs, xticks=[0, 1], xticklabels=['Negative', 'Positive'])
T1D_diag.hist(ax=axs[0], bins=10)
axs[0].set_title('Before dropping rows with nan')
axs[0].set_xlabel('Diagnosis')
axs[0].set_ylabel('Count')
T1D_diag_no_missing.hist(ax=axs[1], bins=10)
axs[1].set_title('After dropping rows with nan')
axs[1].set_xlabel('Diagnosis')
axs[1].set_ylabel('Count')
plt.show()

# Pie plot
T1D_dataset['Diagnosis'].value_counts().plot(kind="pie", labels=['Positive', 'Negative'], colors=['steelblue', 'salmon'], autopct='%1.1f%%', fontsize=16)
plt.show()
T1D_no_missing['Diagnosis'].value_counts().plot(kind="pie", labels=['Positive', 'Negative'], colors=['steelblue', 'salmon'], autopct='%1.1f%%', fontsize=16)
plt.show()

diag_cnt = T1D_dataset['Diagnosis'].value_counts().to_dict()
diag_cnt_no_missing = T1D_no_missing['Diagnosis'].value_counts().to_dict()
print("Before dropping rows with missing values we had:")
print(diag_cnt)
print("After dropping rows with missing values we have:")
print(diag_cnt_no_missing)

# We lost a similar number of patients from each diagnosis group
# We understand from it that by erasing full rows we did not change the balance dramatically but we did loose
# information. Since We have less negative than positive patients, we don't want to loose patients from the negative
# group. We will complete the missing values by random sampling of each series values.

# Replace nan's with random samples of each series values:
from functions import nan2samp
T1D_clean = nan2samp(T1D_dataset)

# section 2
lbl = np.ravel(T1D_clean['Diagnosis'])
X_train, X_test, y_train, y_test = train_test_split(T1D_clean, lbl, test_size=0.2, random_state=0, stratify=lbl)

# Section 3.a - show that the distribution of the features is similar between test and train
# Using a function to create a table of positive rates for every feature in the train/test groups:
from functions import dist_table as dist
X_test_dummy = pd.get_dummies(X_test, dummy_na=False, drop_first=True)
X_train_dummy = pd.get_dummies(X_train, dummy_na=False, drop_first=True)
d_table = dist(X_train_dummy, X_test_dummy)
print(d_table.transpose())

# Section 3.b - show the relationship between feature and label:
from functions import feat_lab_cor as fl_cor
fl_cor(T1D_dataset)

# Section 3.c - additional plots
#
