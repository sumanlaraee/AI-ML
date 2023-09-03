# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 07:33:45 2023

@author: Suman Laraee
"""

#find home prices for 3000 sqr/ft , 3 bedroom , 40 years old 
#before using any model we have to analyze data point's relation with eachother 
#if there is linear relation btw data instances we can use linear regression()
#linear regression with multiple variables 
#price=m1*area+ m2*Bedroom+ m3*age+ b
#y=m1x1+m2x2+m3x3+b

#here we gonna understand datapreprocessing :Handling NA values 


import pandas as pd 
import numpy as np 
from sklearn import linear_model

df=pd.read_csv("Home.csv")
print(df)
# iwas gettina nan data value in bedroom so i wll fix it by taking median

median =df.bedroom.median()
print(median) #this gives float value i need integer 

 
'''Data preprocessing is a critical step in the data analysis 
and machine learning pipeline. It involves cleaning, transforming,
 and organizing raw data into a format that is suitable for 
 analysis or for training machine learning models. Proper data 
 preprocessing can have a significant impact on the quality and
 effectiveness of your analysis or models. Here are some common techniques 
 and steps involved in data preprocessing:

'''
#data preprocessing starts here
import math
median_bed= math.floor(df.bedroom.median())
print(median_bed)   #here how i get integer value 

df.bedroom=df.bedroom.fillna(median_bed)
print(df)
#so datapreprocessing step is over here finishes 

#before applying machine learning model you need to apply preprocessing techniques 

#now applying machine learning model 
reg=linear_model.LinearRegression()
#independent variables are area , bedroom, age, dependent is price 
reg.fit(df[["area", "bedroom", "age"]], df.price  
            #training the model 
        ) 
#coefficients 
print("coefficients::",reg.coef_)

#intercepts 
print("intercepts::",reg.intercept_ )

p=reg.predict([["3000", "3", "40"]])
print("predicted price for area 3000, 3 bedroom adn age is 40:", p)