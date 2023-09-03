# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 06:01:45 2023

@author: Suman Laraee
"""
#importing libraries that i wanna use 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn import linear_model

#reading a csv file 
df=pd.read_csv("Home.csv")
print(df)

#defining labels and scattering points 
#%matplotlib inline
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("home price distribution ")
plt.scatter(df.area, df.price, marker="+",  color="red")

#here we are gonna get linear reg object from linear reg model 
reg=linear_model.LinearRegression()

#now we need to train linear reg model based on original data by using fit()
reg.fit(df[['area']], df.price)

#predicting price for an area 
pred_val=reg.predict(np.array([[5000]]))
print("predicted value:",pred_val)


#we know the equation for price = m*area +b 
print("value for corff:",reg.coef_ )  #this is m

print("value for intercept:",reg.intercept_) #this is b

import pickle 

#these line of code iam saving file on desktop
with open('model_pickle', 'wb') as f:
    pickle.dump(reg,f)

#these line of code iam loading file in memory 
with open("model_pickle ", "rb") as f:
    mp= pickle.load(f)
pr=mp.predict([[5000]])
print("my friend again predicted from model i trained :", pr)

#i can give fle named model_pickle to friend of mine and say ask question or more predictions to it
'''alternative approach of pickle is sklearn joblib ,
 advantage of joblib is when we use numpy in  bulk sklearn if favrble '''
 
import joblib
joblib.dump(reg, "model_joblib")
mj=joblib.load("model_joblib")
jb=mj.predict([[5000]])
print("using joblib:", jb)


#so finally got same results throgh joblib of sklearn and pickle 