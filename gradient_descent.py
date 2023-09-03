# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 12:42:19 2023

@author: Suman Laraee
"""

#using gradient descent 

import numpy as np 
'''we use gradient_descent algorithm to get best fit line in our model 
    some of formulas are given here of m and b derivatives '''
def gradient_descent(x, y):
    m_curr=b_curr=0
    iterations=1000
    learning_rate=0.0001
    n=len(x)
    
    #you can change iteration and learnng rate to reduce more and more cost /mse and get expected value
    
    for i in range(iterations):
       
        y_pred= m_curr*x+b_curr
        cost=((1/n)*sum([ val**2 for val in(y-y_pred)]))
        m_derivative=-(2/n)*sum(x*(y-y_pred))
        b_derivative=-(2/n)*sum(y-y_pred)
        m_curr=m_curr-learning_rate *m_derivative
        b_curr=b_curr-learning_rate*b_derivative
        print('m {} ,b{} ,cost{}, iteration{}'.format(m_curr, b_curr,cost,i))

x=np.array([1,2, 3, 4, 5])
y=np.array([5, 7, 9 ,11, 13])


gradient_descent(x, y)