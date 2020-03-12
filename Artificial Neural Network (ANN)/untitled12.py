# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 13:47:25 2019

@author: Microsoft
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_csv('BankCustomers.csv')
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


states=pd.get_dummies(x['Geography'],drop_first=True)
gender=pd.get_dummies(x['Gender'],drop_first=True)

x =pd.concat([x,states,gender],axis=1)
x =x.drop(['Geography','Gender'],axis=1)


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

