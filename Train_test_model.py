# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 04:37:46 2023

@author: Suman Laraee
"""

import pandas as pd 
df=pd.read_csv(r"C:\Users\Suman Laraee\Documents\GitHub\AI-ML\carprices.csv")
#it is important to use r before path to tell we are using  raw string by prefixing the string with 'r'

print(df)


import matplotlib.pyplot as plt 

plt.scatter(df[["Mileage"]], df["Sell Price($)"]) 

plt.scatter(df[["Age(yrs)"]], df["Sell Price($)"]) 

#plotted relationships of variables to see linear model is applicable or not 

x=df[['Mileage',"Age(yrs)"]]
y=df["Sell Price($)"]
print(x)
print(y)


from sklearn.model_selection import train_test_split
x_train , x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=10)
# i wanna sample shuld remain same so i used random state 
print("trainin data ",x_train) #it chooses data points rendomly which is good 
#20% data must be available for testing 
#80% data is training data 
tr_len=len(x_train)
ts_len=len(x_test)
print(tr_len,ts_len )


#now iam begning to train my model 
from sklearn import linear_model
reg = linear_model.LinearRegression()

reg.fit(x_train , y_train)

pr=reg.predict(x_test)
print("predicted values :",pr)

print(y_test)


acc=reg.score(x_test, y_test)
print(acc)