# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 12:06:52 2023

@author: Suman Laraee
"""
#we are working with svm using iris dataset which is contained by sklearn 

import pandas as pd 
from sklearn.datasets import load_iris
iris =load_iris()
c=dir(iris)
print(c)
#The line dir(iris) is a Python command used to list the attributes and methods of an object.

fn=iris.feature_names
print(fn)  #4 feature (petal:width height)(sepal:width height)
# so data in iris i will convert into dataframe its easy to see 

df=pd.DataFrame(iris.data, columns=iris.feature_names)
d=df.head()
print(d)

df['target']=iris.target
d1=df.head()
print(d1)

#possible values for target name is 0, 1, 2
nm=iris.target_names
print(nm)   #0 => setosa , 1=> versicolor, 2=>virginica

#i wanna see which data items have 1 (versicolor )
vr=df[df.target==1].head()
print(vr)

# adding another column known as flower name 
df['flower_name']=df.target.apply(lambda x:iris.target_names[x])
f=df.head()
print(f)    #now i have flower name column 
'''lambda is function used for transormation or mapping target index 0 => setosa , 1=> versicolor, 2=>virginica
x is considered as index if it is one means flower name is mapped as versicolor so on ..
'''

# now we need to visualize data , so usually i use matplotlib for data visualizaion

from matplotlib import pyplot as plt 


df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

#lets create an scatter plot  its a 2d plot
'''plt.xlabel('sepal length (cm)') 
plt.ylabel('sepal width (cm)') 
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'], marker="+", color="red"
            )
plt.scatter(df1['sepal length (cm)'], df1['sepal width (cm)'], marker="*", color="blue"
            )'''
# same as we  can plot for petals 


plt.xlabel('petal width (cm)') 
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'], marker="+", color="red"
            )
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'], marker="*", color="blue"
            )

#it seems extreamly beautiful 

#now its time to train our model before that we nedd to split our data set 
from sklearn.model_selection  import train_test_split

#why we split bcz the dataset we are using for training we cant use for testing bcz it might be baised
#i need four characteristic only for my data set training and testing droping others 

x=df.drop(['target', 'flower_name'], axis="columns")
print(x)

y=df.target
print("these are target values ", y)

# now ready to split 
x_train, x_test, y_train , y_test=train_test_split(x, y, test_size=0.2)

print('length of training data :', len(x_train))
print('length of testing data :', len(x_test))

#now we need to train our model using support vector classifier 

from sklearn.svm import SVC
reg = SVC()
tr=reg.fit(x_train, y_train)
acc=reg.score(x_test,y_test)
print(acc)

