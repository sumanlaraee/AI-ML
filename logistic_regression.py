# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 08:53:53 2023

@author: Suman Laraee
"""

import pandas as pd 
import matplotlib.pyplot as plt 

df=pd.read_csv(r"C:\Users\Suman Laraee\Documents\GitHub\AI-ML\insurance_data.csv")
print(df)

plt.scatter(df.age, df.bought_insurance, color="red", marker="+")

sp=df.shape
print(sp)   #27 rows and 2 columns 
#now we need to split data into training and testing data
from sklearn.model_selection import train_test_split
x=df[["age"]]
y=df["bought_insurance"]
x_train,x_test,  y_train,y_test= train_test_split(x, y, test_size=0.1, random_state=2)
print(x_train) 

#lets now implement logistic regression 
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(x_train, y_train)
pr =reg.predict(x_test)
print("predicted values ::", pr)


acc=reg.score(x_test, y_test)
print("accuracy of model ::",acc)

#we can see probability also 
print(x_test)
prob=reg.predict_proba(x_test)


#kaggle is good to find datasets 


print("probability that person will buy insurance the with this age :",prob)