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
print(data_set.head)
#extract dependent and independent attribute
x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,-1].values
print(x)
print(y)