# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 09:45:29 2023

@author: Suman Laraee
"""

import pandas as pd 
df= pd.read_csv(r"C:\Users\Suman Laraee\Documents\GitHub\AI-ML\decision tree\salaries.csv")

print(df)

#i wanna divide my datset btw dependent(x variable) and independent variable(target)
x_inputs=df.drop(["salary_more_then_100k"], axis="columns") 
print(x_inputs)
y_target=df.salary_more_then_100k
print(y_target)

from sklearn.preprocessing import LabelEncoder
le_company=LabelEncoder()
le_jobs=LabelEncoder()
le_degree=LabelEncoder()


# here i need to create an extra column 
x_inputs["company_n"]=le_company.fit_transform(x_inputs['company'])
x_inputs["job_n"]=le_jobs.fit_transform(x_inputs['job'])
x_inputs["degree_n"]=le_degree.fit_transform(x_inputs['degree'])

ins=x_inputs.head()
print(ins)

input_n= x_inputs.drop(["company","job","degree"], axis="columns")
print(input_n)


from sklearn import tree
reg = tree.DecisionTreeClassifier()

#ideally we should split our data into traing and testing dataset but iam training entire data set here 

reg.fit(input_n, y_target)
#by default it use gini as impurity para but we can use entropy also

#lets find accuracy after trainin 
acc=reg.score(input_n, y_target)
print(acc)   #100% we got 

#lets do some prediction 
pr=reg.predict([[2,2, 1]])
print("prediction that person salary is more than  100k 0 or 1 :",pr)
