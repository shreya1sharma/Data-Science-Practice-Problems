# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 15:02:54 2018

Goal:
    1. Learn basic steps of machine learning problem
    2. Learn classification problem using sklearn
    3. Apply different machine learning techniques
    
Problem: Given 4 features of a iris flowers, classify them into three classes.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import datasets

#Loading data
iris = datasets.load_iris()
features = iris.data
labels = iris.target

df = pd.DataFrame([features[:,0], features[:,1], features[:,2], features[:,3], labels],
                  index = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'class'] )
df= df.T

#Summarize data
print(df.describe()) #statistical summary
print(df.groupby('class').size())  #class distribution

#Visualize data
#univariate plots
boxplot = df.boxplot(column=['Sepal Length', 'Sepal Width', 'Petal Length','Petal Width'])
df.hist()

#multivariate plots
scatter_matrix(df,figsize = (10,10))
plt.show()

#evaluate algorithms
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 7
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y, 
                                                                                         test_size = validation_size, 
                                                                                         random_state = seed)
scoring = 'accuracy'
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []
for name, model in models:
    kfold = cross_validation.KFold(n = X_train.shape[0],n_folds = 10, random_state = seed)
    cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


#compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#selecting best algorithm and making predictions
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
prediction = knn.predict(X_validation)
print(accuracy_score(Y_validation, prediction))
print(confusion_matrix(Y_validation, prediction))
print(classification_report(Y_validation, prediction))


 
 
    
    
    



