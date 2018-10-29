#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

epsilon = sys.float_info.epsilon

class wining_pridictor():
    def __init__(self,d=0.1,max_iter=5000,eta=0.001,step=1,eta_decay=0.99):
        self.d = d
        self.max_iter = max_iter
        self.eta = eta
        self.step = step
        self.eta_decay = eta_decay
    
    # winning function type:tanh
    def w(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return np.tanh(b*self.d)

    def w_der(self, b):
        # b is bid price
        # d is hyperparameter need to be fit
        return self.d*(1-np.tanh(b*self.d)**2)
    
    def w_der_d(self, b):
        return b*(1-np.tanh(b*self.d)**2)
    
#    def w_der_d_kl(self,b,y):
#        return (self.w_der_d(b)).transpose().dot(
#                np.log(self.w(b)/y) + 1)
        
    def fit(self, x, y):
        print("Now start to fit winning function")
        record_num = np.shape(x)[0]
        
        for i in range(self.max_iter):
            # least squares loss
            error = self.w(x)-y
            self.d = self.d - self.eta * (1/record_num) * (error.transpose().dot(self.w_der_d(x)))
            loss = 1/2 * (1/record_num) * (error**2).sum()
            
            # kl divergence loss
#            self.d = self.d - self.eta * (1/record_num)*self.w_der_d_kl(x,y)
#            loss = entropy(self.w(x), y)

            if i % 50 == 0:
                print("epoch {}, d {}, loss {}".format(i, self.d, loss))
            
            self.eta = self.eta * self.eta_decay
            
        print("Fitting winning function finished")
        
        # do integrate to speed up predicting winning price
        bins = 2000
        step = self.step
        self.integrate_b = np.arange(0, bins+1, step)
        self.integrate_b_y = self.w_der(self.integrate_b)
        temp = self.integrate_b * self.integrate_b_y
        self.integrate = np.zeros_like(self.integrate_b, dtype=np.float)
        self.integrate[0] = temp[0]*step
        for i in range(bins):
            self.integrate[i+1] = self.integrate[i]*1.0 + temp[i+1]*step
        self.integrate[0] = self.integrate[1]  # incase of zero
                    
    def plot(self, x, y):
        y_prediction = self.w(x)
        
        plt.plot(y, label='truth')
        plt.plot(y_prediction, label='w=tanh({0:.5f}x)'.format(self.d))
        plt.legend()        

    def predict(self, x):
        return self.w(x)
         
    def predict_win_price(self, x):
        x = np.round(x)
        r = np.zeros_like(x)
        for i in range(np.shape(x)[0]):
            index = int(x[i])
            try:
                r[i] = self.integrate[index]
            except IndexError:
                r[i] = self.integrate[-1]
                
        return r
