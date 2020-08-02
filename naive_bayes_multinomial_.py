#from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
#from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
iris=load_iris()
from sklearn import tree
training=iris.data
label=iris.target
training=iris.data
label=iris.target
#clf=MultinomialNB()
#clf=GaussianNB()
clf=BernoulliNB()

decision=clf.fit(training,label)
result=decision.predict(training)
print(iris.target)
print((iris.target==result).sum())
