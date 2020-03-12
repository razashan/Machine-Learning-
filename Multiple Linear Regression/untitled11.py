# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:41:02 2019

@author: Microsoft
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

x = dataset.iloc[::-1]
y = dataset.iloc[:,4]


states = pd.get_dummies(x['State'],drop_first=True)

x = x.drop('State',axis=1)
x =pd.concat([x,states],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)

y_pred= reg.predict(x_test)


from sklearn.metrics import r2_square

score = r2_score(y_pred,y_test)