#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('Data.csv')  


# In[3]:


print(data.shape)


# In[4]:


data.head(20)


# In[5]:



data.info()


# In[6]:


types = data.dtypes
print(types)


# In[7]:


x= data.iloc[:,:-1].values  


# In[5]:


print(x)


# In[8]:


y=data.iloc[:,3].values  
print(y)


# In[9]:


y= data.iloc[:,3].values 


# In[11]:


description = data.describe()
print(description)


# In[12]:


class_counts = data.groupby('Purchased').size()
print(class_counts)


# In[13]:


correlations = data.corr(method='pearson')
print(correlations)


# In[14]:


import matplotlib.pyplot as plt


# In[15]:


from matplotlib import pyplot


# In[18]:


data.hist()
pyplot.show()


# In[20]:


data.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()


# In[21]:


data.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()


# In[30]:


#handling missing data (Replacing missing data with the mean value)  
from sklearn.preprocessing import Imputer  
imputer= Imputer(missing_values ='NaN', strategy='mean', axis = 0)  
#Fitting imputer object to the independent variables x.   
imputerimputer= imputer.fit(x[:, 1:3])  
#Replacing missing data with the calculated mean value  
x[:, 1:3]= imputer.transform(x[:, 1:3])  


# In[31]:


print(x)


# In[24]:


#Catgorical data  
#for Country Variable  
from sklearn.preprocessing import LabelEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])
print(x)


# In[15]:


#for Country Variable  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder  
label_encoder_x= LabelEncoder()  
x[:, 0]= label_encoder_x.fit_transform(x[:, 0])  
#Encoding for dummy variables  
onehot_encoder= OneHotEncoder(categorical_features= [0])    
x= onehot_encoder.fit_transform(x).toarray()  
print(x)


# In[25]:


labelencoder_y= LabelEncoder()  
y= labelencoder_y.fit_transform(y)  
print(y)


# In[26]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)  


# In[27]:


from sklearn.preprocessing import StandardScaler  


# In[28]:


st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)  
print(x_train)


# In[29]:


x_test= st_x.transform(x_test)
print(x_test)


# In[ ]:




