# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:25:49 2023

@author: Suman Laraee
"""

#random forest 

import pandas as pd 
from sklearn.datasets import load_digits
digits=load_digits()

d=dir(digits)
print(d)

import matplotlib.pyplot as plt 
p=plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])  
    #output 8 by 8 mutlidimensional array 
    
df=pd.DataFrame(digits.data)
print("first five datapoints",df.head())

df['target']=digits.target
print(df.head())


from sklearn.model_selection import train_test_split
x_train , x_test, y_train , y_test= train_test_split(df.drop(['target'], axis="columns"), digits.target, test_size=0.2)

#ensemble is the term used when there are multiple algo's are used to predict outcome

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50 )
#here we are training our model 
model.fit(x_train, y_train)

#estimators => means no of random tress we are gonna use 


#now lets check accuracy of our model 
acc=model.score(x_train, y_train)
print("accuracy of this model is ::",acc) #wohooo 100%

#to print confusion matxics i need y_predicted and orgional 
y_predicted= model.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)  
# here i get a matrix but it is not best way of visualizing iam using seaborn visualization libraray

import seaborn as sb
plt.figure(figsize=(10,7))
sb.heatmap(cm, annot=True)
plt.xlabel("predicted")
plt.ylabel("truth")