# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 10:09:14 2020

@author: nishi
"""

import numpy as np
#from matplotlib import pyplot as mtp
import matplotlib.pyplot as mtp
import pandas as pd
#import dataset
data_set=pd.read_csv('Salary_Data.csv')
print(data_set.head(5))
#extract dependent and independent attribute
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
print('intercept is',regressor.intercept_)
print('coef is',regressor.coef_)
#the real values observation in blue dots
#pred values are red regression line
#reg line shows a correlartion bw dep. and indep. varaible
#good fit of the line observed by calculating the diff. bw actual and pred val
#most of the observations Are close to the regression line
#hence our model is good for training set
y1_pred=regressor.predict(x_test)
y2_pred=regressor.predict(x_train)
mtp.scatter(x_train,y_train,color="blue")
mtp.plot(x_train,y2_pred,color="red")
mtp.title("Salary vs Experience")
mtp.xlabel("Experience")
mtp.ylabel("Salary")
mtp.show()

mtp.scatter(x_test,y_test,color="green")
mtp.plot(x_train,y2_pred,color="red")
mtp.title("Salary vs Experience")
mtp.xlabel("Experience")
mtp.ylabel("Salary")
mtp.show()

from sklearn import metrics
print('mean absolute error',metrics.mean_absolute_error(y_test,y1_pred))


from sklearn import metrics
print('mean squared error',metrics.mean_squared_error(y_test,y1_pred))






