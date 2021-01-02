#!/usr/bin/env python
# coding: utf-8

## HW2 - Daniel & Naama

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
##
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold as SKFold
from sklearn import preprocessing
from sklearn.metrics import roc_curve
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
X_train_tmp, X_test_tmp, y_train, y_test = train_test_split(T1D_clean, lbl, test_size=0.2, random_state=10, stratify=lbl)
X_train = X_train_tmp.drop(columns=['Diagnosis'])
X_test = X_test_tmp.drop(columns=['Diagnosis'])

# lbl = np.ravel(T1D_clean['Diagnosis'])
# X_train, X_test, y_train, y_test = train_test_split(T1D_clean, lbl, test_size=0.2, random_state=10, stratify=lbl)
# X_train.drop(columns=['Diagnosis'], inplace=True)
# X_test.drop(columns=['Diagnosis'], inplace=True)

# Section 3.a - show that the distribution of the features is similar between test and train
# Using a function to create a table of positive rates for every feature in the train/test groups:
from functions import dist_table as dist
X_test_dummy = pd.get_dummies(X_test, dummy_na=False, drop_first=True)
X_train_dummy = pd.get_dummies(X_train, dummy_na=False, drop_first=True)
d_table = dist(X_train_dummy, X_test_dummy)
d_table.transpose()

# Section 3.b - show the relationship between feature and label:
from functions import feat_lab_cor as fl_cor
fl_cor(T1D_clean)

# Section 3.c - additional plots
#

# Section 4 - One hot vectors encoding:
# Encoding the data into one hot vectors while ignoring the 'Age' feature
ohe = OneHotEncoder(sparse=False)
# Encode X_train and X_test
X_train_ohe = ohe.fit_transform(X_train.iloc[:, 1:])
X_test_ohe = ohe.fit_transform(X_test.iloc[:, 1:])
# Encode y_train and y_test and
y_train_ohe = ohe.fit_transform(pd.DataFrame(y_train))
y_test_ohe = ohe.fit_transform(pd.DataFrame(y_test))
y_train_ohe_vec = y_train_ohe[:, 1]  # in the second column 0='Negative' and 1='Positive'
y_test_ohe_vec = y_test_ohe[:, 1]

# Section 5 - machine learning models
# Section 5.a.i - linear models
# Trying simple logistic regression:
from functions import pred_log
log_reg = LogisticRegression(solver='saga', multi_class='ovr', penalty='none', max_iter=10000, random_state=10)
y_pred, w = pred_log(log_reg, X_train_ohe, y_train_ohe_vec, X_test_ohe)
y_pred_p, w_p = pred_log(log_reg, X_train_ohe, y_train_ohe_vec, X_test_ohe, flag=True)
print('AUROC with logistic regression is {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p[:, 1])))

# 5K-cross fold - logistic regression
from functions import kcfold
C = np.array([0.01, 0.1, 1, 5, 10])  # regularization parameters
K = 5  # number of folds
penalty = ['l1', 'l2']  # types of penalties
val_dict = kcfold(X_train_ohe, y_train_ohe_vec, C=C, penalty=penalty, K=K)

# choosing the best parameters and predict:
c = 5
p = 'l2'
log_reg_best = LogisticRegression(solver='saga', multi_class='ovr', penalty=p, C=c, max_iter=10000, random_state=10)
y_pred_best, w_best = pred_log(log_reg_best, X_train_ohe, y_train_ohe_vec, X_test_ohe)
y_pred_p_best, w_p_best = pred_log(log_reg_best, X_train_ohe, y_train_ohe_vec, X_test_ohe, flag=True)

# display performance:
print("evaluation metrics for logistic regression with 5K-cross fold validation:")
print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred), average='macro'))) + "%")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(pd.DataFrame(y_test_ohe_vec), pd.DataFrame(y_pred))) + "%"))
print('AUROC is {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p_best[:, 1])))

cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_best)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

# Section 5.a.ii - non linear models
# 5K-cross fold - non linear SVM
n_splits = K
skf = StratifiedKFold(n_splits=n_splits, random_state=10, shuffle=True)
svc = SVC(probability=True)
C = np.array([1, 100, 1000])
pipe = Pipeline(steps=[('scale', StandardScaler()), ('svm', svc)])
svm_nonlin = GridSearchCV(estimator=pipe, param_grid={'svm__C': C, 'svm__kernel': ['rbf', 'poly'],
                        'svm__gamma': ['auto', 'scale']}, scoring=['roc_auc'],
                        cv=skf, refit='roc_auc', verbose=0, return_train_score=True)
svm_nonlin.fit(X_train_ohe, y_train_ohe_vec)

# Choose the best estimator and print
best_svm_nonlin = svm_nonlin.best_estimator_
print("Non-linear SVM best parameters are:")
print(svm_nonlin.best_params_)

# predict and calculate performance
y_pred_test = best_svm_nonlin.predict(X_test_ohe)
y_pred_proba_test = best_svm_nonlin.predict_proba(X_test_ohe)
TN = confusion_matrix(y_test_ohe_vec, y_pred_test)[0, 0]
FP = confusion_matrix(y_test_ohe_vec, y_pred_test)[0, 1]
FN = confusion_matrix(y_test_ohe_vec, y_pred_test)[1, 0]
TP = confusion_matrix(y_test_ohe_vec, y_pred_test)[1, 1]
Se = TP/(TP+FN)
Sp = TN/(TN+FP)
PPV = TP/(TP+FP)
NPV = TN/(TN+FN)
Acc = (TP+TN)/(TP+TN+FP+FN)
F1 = (2*PPV*Se)/(PPV+Se)

# display performance:
print("evaluation metrics for Non-linear SVM:")
print(f'F1 is {F1:.2f}')
print(f'Acc is {Acc:.2f}')
print('AUROC is {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_proba_test[:, 1])))

cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred_test)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

# Section 6 - feature selection
# Random forest:
rf_clf = rfc(random_state=10)
rf_clf.fit(X_train_ohe, y_train_ohe_vec)
y_pred = rf_clf.predict(X_test_ohe)
y_pred_p_rf = rf_clf.predict_proba(X_test_ohe)

print("evaluation metrics for random forest classifier:")
print("Acc is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test_ohe_vec, y_pred))) + "%")
print("F1 is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test_ohe_vec, y_pred, average='macro'))) + "%")
print('AUROC is {:.3f}'.format(roc_auc_score(y_test_ohe_vec, y_pred_p_rf[:, 1])))

cnf_matrix = metrics.confusion_matrix(y_test_ohe_vec, y_pred)
cnf_heat_map = sns.heatmap(cnf_matrix, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
cnf_heat_map.set(ylabel='True labels', xlabel='Predicted labels')
plt.show()

# Section 6.A.i - select 2 most important features according to the random forest:
# Make a df of feature and importance
features = rf_clf.feature_importances_
feat_imp_dict = []
i = 0
for column in X_train.iloc[:, 1:].columns:
    feat_imp_dict.append({"feature": column+" 0", "importance": features[i]})
    feat_imp_dict.append({"feature": column+" 1", "importance": features[i+1]})
    i += 2
feat_imp = pd.DataFrame(feat_imp_dict)
feature_names = np.array(feat_imp['feature'])

sorted_idx = features.argsort()
y_ticks = np.arange(0, len(features))
fig, ax = plt.subplots()
ax.barh(y_ticks, features[sorted_idx])
ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
ax.set_yticklabels(feature_names[sorted_idx])
ax.set_yticks(y_ticks)
ax.set_title("Random Forest Feature Importance")
fig.tight_layout()
plt.show()

p=1



#
# 5K-cross fold - logistic regression visualization:
# for d in val_dict:
#     x = np.linspace(0, d['mu'] + 3 * d['sigma'], 1000)
#     plt.plot(x, stats.norm.pdf(x, d['mu'], d['sigma']), label="p = " + d['penalty'] + ", C = " + str(d['C']))
#     plt.title('Gaussian distribution of the loss')
#     plt.xlabel('Average loss')
#     plt.ylabel('Probability density')
# plt.legend()
# plt.show()

# logistic regression with GreadSearchCV
# logistic regression with k-cross fold using GridSearchCV
# C_gs = np.array([0.0001, 0.001, 0.01, 1, 10, 100, 1000, 10000])
# k = 5
# SKF = SKFold(n_splits=k, random_state=10, shuffle=True)
# log_reg_gs = LogisticRegression(random_state=5, max_iter=10000, solver='liblinear')
# penalty = ['l1', 'l2']
# #pipe = pipeline(steps=[('logistic', log_reg_gs)])
# rf_clf = GridSearchCV(estimator=log_reg_gs, param_grid={'logistic__C': C_gs, 'logistic__penalty': penalty},
#                    scoring=['roc_auc'], cv=SKF, refit='roc_auc', verbose=3, return_train_score=True)
# rf_clf.fit(X_train_ohe, y_train_ohe_vec)

# T1D_clean_dummy = pd.get_dummies(T1D_clean, drop_first=True)
# label_encoder = LabelEncoder()
# T1D_clean_val = T1D_clean.values
# T1D_shape = T1D_clean_val.shape
# T1D_lbl_encoded = np.zeros(T1D_shape)
# for i in range(len(T1D_clean.columns)):
#     if i == 1:
#         T1D_lbl_encoded[:, i] = T1D_clean_val[:, i]
#     else:
#         T1D_lbl_encoded[:, i] = label_encoder.fit_transform(T1D_clean_val[:, i])

# plot the performances
# clf_type = ['rbf', 'scale']

# calc_TN = lambda y_true, y_pred_1: confusion_matrix(y_true, y_pred)[0, 0]
# calc_FP = lambda y_true, y_pred_1: confusion_matrix(y_true, y_pred)[0, 1]
# calc_FN = lambda y_true, y_pred_1: confusion_matrix(y_true, y_pred)[1, 0]
# calc_TP = lambda y_true, y_pred_1: confusion_matrix(y_true, y_pred)[1, 1]

# plot_confusion_matrix(svm_nonlin, X_test_ohe, y_test_ohe_vec, cmap=plt.cm.Blues)
# plt.show()


#J_train = np.zeros((len(penalty), len(C)))
#J_val = np.zeros((len(penalty), len(C)))

# y_pred_train = log_reg_func.predict_proba(x_train)
# J_train[penalty.index(p), np.where(C == c)] = log_loss(y_train, y_pred_train)
# J_val[penalty.index(p), np.where(C == c)] = log_loss(y_val, y_pred_val)

# plt.plot(1 / C, J_train[penalty.index(p), :])
# plt.plot(1 / C, J_val[penalty.index(p), :])
# plt.xlabel('C')
# plt.ylabel('Loss')
# plt.legend(['J_train (n = ' + str(x_train.shape[0]) + ')', 'J_val (n = ' + str(x_val.shape[0]) + ')'])
# plt.show()