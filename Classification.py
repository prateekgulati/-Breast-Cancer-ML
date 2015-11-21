__author__ = 'Prateek'

import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets,metrics,svm,tree

def LogReg(s,e):
    f=open("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt")
    data= pandas.read_csv("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    Z = logreg.predict(X)

    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))

def SVM(s,e):
    f=open("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt")
    data= pandas.read_csv("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]

    logreg=svm.SVC()
    logreg.fit(X, Y)

    Z = logreg.predict(X)
    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))

def DTC(s,e):
    f=open("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt")
    data= pandas.read_csv("C:\Users\Prateek\PycharmProjects\BreastCancerML\data\wdbc_data.txt", header=None, sep=r"\s+")
    X=np.array(data)
    Y=X[:,1]
    Y= np.where(Y=='M', 1, 0)
    X=X[:,s:e]
    logreg = tree.DecisionTreeClassifier()
    logreg.fit(X, Y)

    Z = logreg.predict(X)

    print(metrics.classification_report(Y, Z))
    print(metrics.confusion_matrix(Y,Z))