# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 10:59:58 2023

@author: Suman Laraee
"""

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df= pd.read_csv("income.csv")
print(df.head())

#plt.scatter(df[['Age']], df['Income($)'])

#here i gonna define object of model 
km = KMeans(n_clusters=3)
print(km)

#now iam gonna train and predict together u can do indivisual also 
y_predicted=km.fit_predict(df[['Age']], df['Income($)'])
print(y_predicted)
'''it will return an array with labeling clusters 
from 0 on wards it gives 0 1 2 for diff data points 
means only three clusters are there 
'''

#visualization in array is not good so lets try another way 
df['cluster']=y_predicted
print(df.head())


#storing each cluster in diff variable
df1=df[df.cluster==0]

df2=df[df.cluster==1]

df3=df[df.cluster==2]
'''
#visulaizing clusters through scatter plot 
plt.scatter(df1.Age, df1['Income($)'], color="red", label="df1")
plt.scatter(df3.Age, df3['Income($)'], color="green", label="df2")
plt.scatter(df2.Age, df2['Income($)'], color="blue"  ,label="df3")
'''
#setting labels 
plt.xlabel("Age")
plt.ylabel("Income($)")

plt.legend()

''' all points of clusters are overlapping due to difference in range of x and y axis 
so lets do scaling for that iam using minmaxscale '''

scaler=MinMaxScaler()
scaler.fit(df[['Income($)']])
df['Income($)']=scaler.transform(df[['Income($)']])
print(df.head())
    #now you can see income is scaled in points 
    
#now lets scale age 
scaler.fit(df[['Age']])
df.Age=scaler.transform(df[['Age']])
print(df.head())
        #now you can see age  is  also scaled in points
#plt.scatter(df[['Age']], df['Income($)'])

#after scalling we need to train again our model using Kmeans algo

km = KMeans()
y_predicted=km.fit_predict(df[["Age", "Income($)"]])
print("predicted clusters ", y_predicted)


# to see values its better way to create a new column and replace old cluster column
df['clusterz']=y_predicted
df.drop('cluster', axis='columns', inplace=True)
print(df)
#you can see new clustering result 

#now lets plot scatter plot 

df0=df[df.clusterz==0]
df1=df[df.clusterz==1]
df2=df[df.clusterz==2]
'''
plt.scatter(df1[['Age']], df1["Income($)"], color="red")
plt.scatter(df2[['Age']], df2["Income($)"], color="blue")
plt.scatter(df3[['Age']], df3["Income($)"], color="green")
'''

print('cluster centre::',km.cluster_centers_)
# here in array there ix and y cordinates of cenrtroid


#we can draw scatter plot of centroids
#plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],marker="+", color="orange", label="center")
plt.legend() 


#when data is so scary use elbow method 
k_rng=range(1, 10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age']], df["Income($)"])
    sse.append(km.inertia_)
print(sse)

plt.xlabel("K")
plt.ylabel("SSE")
plt.plot(k_rng,sse)

    #here is an elbow plot 