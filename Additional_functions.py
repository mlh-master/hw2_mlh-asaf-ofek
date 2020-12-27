#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
import random
import distutils

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
            c_cdf[column].replace(to_replace= np.nan , value=replacing_value)
    return pd.DataFrame(c_cdf)

def str_to_bool_series(s):
    for index, value in s.items():
        if(value=='Yes'):
            s[index] = True
        elif(value=='No'):
            s[index] = False
        elif(value=='Male'):
            s[index] = True
        elif(value == 'Female'):
            s[index] = False
        elif(value=='Positive'):
            s[index] = True
        elif (value == 'Negative'):
            s[index] = False
        if(s=='Family History'):
            if(value==1):
                s[index] = True
            else:
                s[index] = False
    return s

