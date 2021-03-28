# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 12:22:59 2021

@author: shubh
"""

import numpy as np
import numpy.random as npr

npr.seed(42)

#Single layered Deep Neural Network Class for predicting compressional slowness using density and gamma ray values in single layered Deep neural network


# Data
X=np.array([[44.7916, 2.432129],
            [48,4.6],
            [35.8,4.6],
            [102,2.1],
            [108,2.4]]) # Pair of Gr and Rho values for five instances X.shape =(5,2)
            
Y=np.array([53,81,98,58,66]) # Values of respective slowness for five instances Y.shape=(5,) 
#Need to be reshaped to 5,1 for consistency purposes
Y=Y.reshape(-1,1)

def normalise(A):
    
    mu=A.mean(axis=0)
    std=(np.var(A,axis=0))**(1/2)
    
    return [(A-mu)/std, mu, std]

# Normalize data for faster and accurate optimization
X=normalise(X)[0]
y=normalise(Y)[0]



#Non interactive neural network class
'''
W1.shape=3,2   X.shape=5,2

I1=X @ W1.T
o1=act(I1)
O1 shape = 5*3

W2.shape= 1,3
I2=O1 * W2.T
O2=I2 no activation used 
'''



class NN():
    
    def __init__(self):
        
        self.isize=2
        self.osize=1
        self.h1size=3
        
        self.w1=npr.normal(size=(self.h1size,self.isize))
        self.w2=npr.normal(size=(self.osize,self.h1size))
        
        
    @staticmethod
    def sigmoid(X):
        return 1/(1+ np.exp(-X))
        
    
    
    def sigmoid_delta(self,X):        
        return self.sigmoid(X)*(1-self.sigmoid(X))
    
    def prop_forward(self,X):
        
        ''' 
        X,W1-- I1-O1
        O1,W2--I2-O2
        '''
        self.I1=X @ self.w1.T
        self.O1= self.sigmoid(self.I1)
        #shape of O1 is =instances, 3
        
        self.I2=self.O1 @ self.w2.T
        self.O2=self.I2
        #Shape of O2 is instances,1
        
        return self.O2
    
    def prop_back(self,X,y):
        
        self.y_pred= self.prop_forward(X)
        #print(self.y_pred)
        
        self.err=self.y_pred-y
        
        self.W2_jac=(self.O1.T) @ (self.err*1)
        self.W2_jac=self.W2_jac.T
        print(self.W2_jac)
        
        
        self.delta=np.dot(self.err,self.w2)
        print(self.delta)
        self.delta=self.delta*self.sigmoid_delta(self.I1)
        print('delta = {}'.format(self.delta))# 5,3
        
        self.W1_jac=((self.delta).T) @ X 
        
        return self.W1_jac,self.W2_jac
    
    
    
    def train(self,num_iter=1000):
        
        for num in num_iter:
            pass
        return
    
    def loss(self,X,y):
        y_pred=self.prop_forward(X)
        return 0.5*np.sum((np.power((y_pred-y),2)))
            
            
    
class Trainer(NN):
    
    def train(self,X,y,num_iter=100,alpha=0.01):
        
        loss_lt=[]
        print(self.loss(X,y))
        
        for n in range(num_iter):
            
            
            temp_w1,temp_w2=self.prop_back(X,y)
            self.w1-= temp_w1*alpha
            self.w2-= temp_w2*alpha
            
            if n%10==0:
                print(n,self.loss(X,y))
                loss_lt.append(self.loss(X,y))
                
        return loss_lt
                
                
            
    
t=Trainer()
lt_loss=t.train(X,y,num_iter=1000)

y_final=t.prop_forward(X)


tt=normalise(Y)
Y_rev_transformed= (y_final*tt[-1]) + tt[1]
    
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
    
