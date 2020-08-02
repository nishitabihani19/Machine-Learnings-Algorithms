'''
logistic regression
sbm
random forest cl;assifier
'''
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import svm
feature=[[120,0],[140,0],[160,1],[170,1],[190,1]]
label=['apple','apple','orange','orange','orange']
clf1=tree.DecisionTreeClassifier()
clf2=LogisticRegression()
clf3=RandomForestClassifier()
clf4=GaussianNB()
clf5=MultinomialNB()
clf6=BernoulliNB()
clf7=svm.SVC(C=1,gamma=1)
train1=clf1.fit(feature,label)
train2=clf2.fit(feature,label)
train3=clf3.fit(feature,label)
train4=clf4.fit(feature,label)
train5=clf5.fit(feature,label)
train6=clf6.fit(feature,label)
train7=clf7.fit(feature,label)
testdata=[[150,1]]
result1=train1.predict(testdata)
result2=train2.predict(testdata)
result3=train3.predict(testdata)
result4=train4.predict(testdata)
result5=train5.predict(testdata)
result6=train6.predict(testdata)
result7=train7.predict(testdata)
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)
print(result6)
print(result7)
