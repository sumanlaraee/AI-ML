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
df=pd.read_csv("DESKTOP/Home.csv")
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
pred_val=reg.predict(np.array([[3300]]))
print("predicted value:",pred_val)


#we know the equation for price = m*area +b 
print("value for corff:",reg.coef_ )  #this is m

print("value for intercept:",reg.intercept_) #this is b


#we know the equation for price = m*area +b
#lets confirm we got right values 
price_p =135.78767123*3300+180616.43835616432
print("is it equal to predicted value ", price_p, "yes ")


#i have another csv with list of area 
df_2=pd.read_csv("DESKTOP/Home_2.csv")
print(df_2.head(3))

p=reg.predict(df_2)
df_2["prices"]=p
print("here is data frame 2",df_2)

#if you wanna export prdicted values into your csv file 
df_2.to_csv("DESKTOP/Home_2.csv")