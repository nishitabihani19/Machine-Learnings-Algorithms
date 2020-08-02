import pandas as pd
import matplotlib.pyplot as plt
skip=['marks']
df=pd.read_csv('student_list.csv',usecols=#lamda"doyourself")
#df=pd.read_csv('student_list.csv',usecols=['marks','age'])
#print(df)
print(df.head())
print('head',df.head(2))
print('tail',df.tail(2))
plt.plot.bar()
plt.show()
