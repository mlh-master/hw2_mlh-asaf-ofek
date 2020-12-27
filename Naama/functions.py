# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def nancount(T1D_dataset):
    """
    :param T1D_dataset: Pandas series of CTG features
    :return: A dictionary of number of nan's in each feature in T1D_dataset
    """
    T1D_dataset_temp = T1D_dataset.copy()
    c_T1D = {}  # initialize a dictionary
    for column in T1D_dataset_temp.columns:
        c_T1D[column] = len(T1D_dataset) - len(T1D_dataset_temp[column].dropna())
    return c_T1D


def nan2samp(T1D_dataset):
    """
    :param T1D_dataset: Pandas series of T1D_dataset
    :return: A pandas dataframe of the dictionary T1D_d containing the "clean" features
    """
    T1D_d = {}
    T1D_dataset_temp = T1D_dataset.copy()
    for column in T1D_dataset_temp.columns:
        col = T1D_dataset_temp[column]
        i = col.isnull()  # create a boolean vector with true values where col has nan values
        idx = np.zeros(np.sum(i))
        t = 0  # initialize a counter for the nan locations vector (idx)
        for j in range(1, len(i)):
            if i[j] == 1:
                idx[t] = j
                t += 1
        temp = np.random.choice(col.dropna(), size=len(idx))  # random sampling of len(idx) values from col
        col[idx] = temp
        T1D_d[column] = col
    return pd.DataFrame(T1D_d)


def dist_table(X_train,X_test):
    """
    :param x_train: train df of T1D features
    :param x_test: test df of T1D features
    :return: a table of the positive rates for every feature in the train/test groups
    """
    x_train = X_train.copy()
    x_test = X_test.copy()
    x_train.drop(columns=['Age'], inplace=True)
    x_test.drop(columns=['Age'], inplace=True)
    d_table = {}  # initialize a dictionary
    for column in x_train.columns:
        curr_train = x_train[column]
        curr_test = x_test[column]
        train_prc = 100*(curr_train.sum()/curr_train.size)
        test_prc = 100*(curr_test.sum()/curr_test.size)
        d_table[column] = {"train%": train_prc, "test%": test_prc, "delta%": train_prc - test_prc}
    return pd.DataFrame(d_table)


def feat_lab_cor(T1D_dataset):
    """
    :param T1D_dataset: test df of T1D features
    """
    fig, axes = plt.subplots(3, 6, figsize=(12, 6))
    sns.set_context("paper", font_scale=0.7)
    fig.suptitle('Relationships between features and labels')
    i = 0
    for column in T1D_dataset:
        if column == 'Gender':
            feat_diag = sns.countplot(ax=axes[0, 0], x='Gender', hue='Diagnosis', data=T1D_dataset)
            feat_diag.set(xticklabels=['Male', 'Female'])
        elif column == 'Age':
            feat_diag = sns.countplot(ax=axes[0, 1], x='Age', hue='Diagnosis', data=T1D_dataset)
            feat_diag.xaxis.set_major_locator(ticker.LinearLocator(10))
        else:
            if i < 18:
                feat_diag = sns.countplot(ax=axes[i//6, i % 6], x=column, hue='Diagnosis', data=T1D_dataset)
                feat_diag.set(xticklabels=['No', 'Yes'])
        i += 1
    plt.show()
    return()