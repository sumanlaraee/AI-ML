# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 10:19:32 2023

@author: Suman Laraee
"""

#pupose of this coding tutorial is classifing digit dataset which is sklearn library
# we will use different models svm etc and using kfold classification evaluate it 
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression 

digits= load_digits()

# we need to split dataset into training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(digits.data, digits.target, test_size=0.3)

#lets use logistic regression 
lg = LogisticRegression()
lg.fit(X_train, y_train)
acc=lg.score(X_test, y_test)
print("the accuracy of logistic regression using df is ::", acc)

#now lets use svm 
svm =SVC()
svm.fit(X_train, y_train)
acc2=svm.score(X_test, y_test)
print("the accuracy of svm using df is ::", acc2)


#lets try random forest 
rd = RandomForestClassifier(n_estimators=50)
rd.fit(X_train, y_train)
acc3=rd.score(X_test, y_test)
print("the accuracy of random forest using df is ::", acc3)


#problem with train test method is u hv to run it multiple times to be sure about accuracies bcz it keeps on changing random values 

#so tht's y lets try kfold classifier 

from sklearn.model_selection import KFold
kf=KFold(n_splits=3)
#n_split means how many folds you wanna create 

for train_index , test_index in kf.split([1, 2, 3, 4, 5, 6, 7,8  ,9]):
    print(train_index, test_index)



#we can make a generic function for model tarining and return  accuracy 
def get_score(model , X_train, X_test, y_train, y_test):
    model.fit(X_train,y_train)
    return model.score( X_test,y_test)


#stratifiedkfold is same as kfold but better than that bcz it divide each classification catogery in uniform way 
from sklearn.model_selection import StratifiedKFold
skf= StratifiedKFold()

'''
score_l=[]
score_s=[]
score_r=[]

for train_index, test_index in skf.split(digits.data, digits.target):
     X_train, X_test, y_train, y_test= digits.data[train_index], digits.data[test_index],digits.target[train_index], digits.target[test_index]
   # print("print acc of log:", get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
     print( 'hi',score_l.append( get_score(LogisticRegression(), X_train, X_test, y_train, y_test)))
     score_s.append( get_score(SVC(), X_train, X_test, y_train, y_test))
     score_r.append( get_score(RandomForestClassifier(n_estimators=50), X_train, X_test, y_train, y_test)    )
     
     '''
     
#insead of this entire code we can use core_val_score function 

from sklearn.model_selection import cross_val_predict
print("cross_val_predict",cross_val_predict(LogisticRegression(), digits.data, digits.target))