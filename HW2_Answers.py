#!/usr/bin/env python
# coding: utf-8

# ***HW2 - Detecting Type 1 Diabetes***

# # Theory Assignment

# **A1**
# 
# For our opinion, when it comes to healthcare ML algorithms, Accuracy is more important than performance.
# Because we're speaking on human lives, it is better to be more precise than, for example, when estimating house value.
# 
# האם התכוונו כאן לביצועיות של המודל? כלומר למהירות בה הוא מגיע לתוצאות?

# **A2**
# 
# Too many features is often a bad thing. It may lead to Overfitting, meaning that the fitting of your parameters is too tightly to the training data. This results in model discovering random noise in the finite training set instead of the wider relationship between the features and the output variable. Consequently, the model will often perform very well on the training data but perform quite poorly on the test data.
# In conclusion, overfitting may be done due to choosing all features, especially whenn taking irrelevant feature like income.
# 
# On the other hand, choosing only 2 features for such complex problem may cause under-fitting. meaning that both training accuracy and testing accuracy will be poor.
# 

# # Coding Assignment

# Import and load packages:

# In[31]:


import pandas as pd 
import numpy as np
from pathlib import Path
import random
import distutils
from Additional_functions import *


# Data loading and preprocessing:

# In[56]:





# In[65]:


df = pd.read_csv('HW2_data.csv')

#df.apply(bool(distutils.util.strtobool()))


# In[67]:


for column in df.columns:
    df[column] = str_to_bool_series(df[column])
df.tail()

d_f = nan2num(df)
df.fillna(value = True)
df.tail()

