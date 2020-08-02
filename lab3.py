# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 13:40:58 2020

@author: nishi
"""
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
data=pd.read_csv('Data.csv')
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
               

from sklearn.preprocessing import LabelEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])

print(x)
print(x[:,0])

from sklearn.preprocessing import Imputer  
imputer= Imputer(missing_values ='NaN', strategy='mean', axis = 0)  
#Fitting imputer object to the independent variables x.   
imputerimputer= imputer.fit(x[:, 1:3])  
#Replacing missing data with the calculated mean value  
x[:, 1:3]= imputer.transform(x[:, 1:3])  

from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])  
#Encoding for dummy variables  
onehot_encoder= OneHotEncoder(categorical_features= [0])    
x= onehot_encoder.fit_transform(x).toarray()  
print(x[:,0])

labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y)  
print(y)

from sklearn.preprocessing import StandardScaler  

st_x= StandardScaler()  
x= st_x.fit_transform(x)  
print(x[:,2])


























