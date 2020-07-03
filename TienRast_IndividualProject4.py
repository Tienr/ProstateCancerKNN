# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 21:04:04 2020

@author: tiera
"""


import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
df = pd.read_csv('C:/Users/tiera/Downloads/Prostate_Cancer.csv')
df.head(10)
df.info()
print (df)
df.drop(['id'],axis =1, inplace=True)
df.diagnosis_result = [1 if each == 'M' else 0 for each in df.diagnosis_result]
df.diagnosis_result.value_counts()
y= df.diagnosis_result.values
x1 = df.drop(['diagnosis_result'],axis=1)
print(y)
x1.head(10)
scaler = MinMaxScaler(feature_range = (0,1))
x = scaler.fit_transform(x1)
x
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 0, test_size = 0.3)
import math
math.sqrt(len(y_test))
classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric = 'euclidean')
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)

y_test
print(accuracy_score(y_test,y_pred))

