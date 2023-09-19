# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:40:16 2023

@author: Suman Laraee
"""
import pandas as pd 
df= pd.read_csv("titanic.csv")
print(df)

df.drop(['PassengerId', "Name",'SibSp' , "Parch", 'Ticket', 'Cabin', "Embarked"], axis="columns" ,inplace=True)
print(df)

#seprating dependent and independent variables 
target=df.Survived
inputs=df.drop("Survived", axis="columns")

#creating dummies for sex variable bcz ml doesnt deal with text we need numbers
dumies= pd.get_dummies(inputs.Sex)
print(dumies.head())

#now we need to attach dumies to input 
inputs=pd.concat([inputs, dumies], axis="columns")
print(inputs.head())

#now i wanna drop sex column bcz i have dummies now
inputs.drop('Sex', axis="columns",inplace=True)
print(inputs.head())

#i wanna see if there are any NAN values in any column
print(inputs.columns[inputs.isna().any()]) 
#age column has some na values 

#printed first ten rows to see if nan values are there 
print(inputs.Age[:10])

#now i wanna fill those nan values with mean values
inputs.Age=inputs.Age.fillna(inputs.Age.mean())
print(inputs.Age[:10])
#woohoo done 

#now we will split our data into train test split 
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test=train_test_split(inputs, target, test_size=0.2)

print(len(x_train))


#now we are gonna train our model there are couple of naive based model 
#we are gonna use guassian 

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


model.fit(x_train, y_train)


acc=model.score(x_test, y_test)
print(acc)

#this is oriigional testing outputs
print(y_train[:10])

#these are predicted values 
print(model.predict(x_train[:10]))

#you can see many disimilarities in outputs so our model isnot 100% accurate 


