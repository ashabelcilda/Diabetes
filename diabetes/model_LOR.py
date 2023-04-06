#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

df = pd.read_csv('diabetes.csv')
X = df.drop(['Outcome'],axis =1)
y = df[['Outcome']]

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size =0.25, random_state =42)

LOR = LogisticRegression()
LOR.fit(X_train,y_train)
LOR.predict(X_test)

pickle.dump((LOR), open('model_LOR.pkl','wb'))
model_LOR = pickle.load(open('model_LOR.pkl','rb'))


# In[ ]:




