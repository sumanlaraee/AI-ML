# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:42:16 2023

@author: Suman Laraee
"""

import pandas as pd 

df=pd.read_csv("homeprices.csv")
print(df)

d=pd.get_dummies(df.town)
print(d)

merged=pd.concat([df, d], axis="columns")
print(merged)

#now no more town attribute is needed 


'''dummy variable trap when there are multiple dummy variables it create problem for ml model 
whn one variable can be derived from multiple variales that concept is called collinearity 
so it creates dumy variable trap so we have to drop one of column from 4 0r 3 how many we have 
''' 

final=merged.drop(["town", "west windsor"], axis="columns")
    #i have droped west wintsor 
print(final) 

#if you dont drop it linear reg model will drop it itself but it is a good practice to drop by hand 

from sklearn import linear_model

reg = linear_model.LinearRegression()

x=final.drop("price", axis="columns")
print(x)

y=final.price
print(y)


reg.fit(x, y)

pred_val=reg.predict([[2800, 0, 1]])
print("predicted price :: ",pred_val)

'''to check accuracy we have  method called 
score it will calculate all predicted values for all rows in x 
then it will compare predicted prices /values with actual prices / values 
which is y 
'''

acc= reg.score(x, y)
print("accuracy of my model ",acc)  
#it gives output 0.95 means 95%accuracy is there .



#we can use sklearn one hot encoder which does actually same thing
print (df)


#i need encoder from sklearn to apply labels on town columns 

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_=df
#it takes labels columns input and create labels on that basis 
df_.town =le.fit_transform(df.town)
print(df_ )


x=df_[['town', 'area']].values 
print(x)

y=df_.price
print(y)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline








'''
    this was old piece of code now categorical_features is removed from sklearn

ohe= OneHotEncoder(Categorical_features=[0])
        #mean 0th column is my categorical featureor x not all x features 
x=ohe.fit_transform(x).toarray()
print(x)

'''
categorical_cols = [0]

# Create a ColumnTransformer to apply transformations to specific columns
# In this case, we apply OneHotEncoder to the categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'  # You can specify how to handle non-categorical columns
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    # Add other steps here
])

# Fit and transform your data using the pipeline
x = pipeline.fit_transform(x)


print(x)



#now i have four columns but i wanna three columns 

x=x[:, 1:]
print(x)

reg.fit(x, y)
reg.predict([[1, 0, 2800]])