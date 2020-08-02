# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 13:18:56 2020

@author: nishi
"""

import pandas as pd
data=pd.read_csv('Social_Network_Ads.csv')
data.size
x=data.iloc[:,1:4].values
y=data.iloc[:,4].values
print(x)
print(y)
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])
print(x)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=0)
from sklearn.preprocessing import StandardScaler
st= StandardScaler()  
x_train=st.fit_transform(x_train)  
y_train=st.fit_transform(y_train)
from sklearn.svm import SVC
clf=SVC(kernel='linear')
print(x_train)
print(x_test)
clf.fit(x_train,y_train)
y_prod=clf.predict(x_test)

from sklearn.metrics import  classification_report,confusion_matrix
print(confusion_matrix(y_test,y_prod))
print(classification_report(y_test,y_prod))
print('accuracy',metrics.accuracy_score(y_test,y_prod))


from sklearn.svm import SVR 
clf=SVR(kernel='linear')
print(x_train)
print(x_test)
clf.fit(x_train,y_train)
y_prod=clf.predict(x_test)