#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import sys
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

np.set_printoptions(suppress=True)

class Bidder():
    def __init__(self, k_model, encoder, winner, rho=1, c=10, eta=1e4, labda=0, gamma=0.99, max_iter=100, learning=1e1, tol=1e-0, strategy='lin'):
        self.k_model = k_model
        self.encoder = encoder
        self.winner = winner
        self.name_col = encoder.name_col
        self.rho = rho
        self.c = c
        self.eta = eta
        self.gamma = gamma
        self.max_iter = max_iter
        self.learning = learning
        self.tol = tol
        self.labda = labda
        self.strategy = strategy
        
    def optimize(self, X):
        return self.optimize_k(self.k(X))

    def optimize_k(self, k):
        self.alpha = np.array([5e3, 1e0], dtype=np.float)
        r = minimize(self.loss,
                     self.alpha,
                     args=(k),
                     method='BFGS',
                     jac=self.loss_der,
                     tol=self.tol,
                     options={'maxiter': 30, 'disp': False})

        self.alpha = r.x
        self.r = r

        return self.r
    
    def evaluate(self, X_test, logs):
        assert(X_test.shape[0] == len(logs))
        bids = self.generate_bid(self.k(X_test))
        
        total_cost = 0.0
        
        imps = 0.1
        clicks = 0.1
        pays = 0.1
        for i in range(len(logs)):
            total_cost += logs[i][1]
            if bids[i] >= logs[i][1]:
                # if win
                imps += 1
                clicks += logs[i][0]
                pays += logs[i][1]
                
        roi = clicks*self.eta/pays
            
        print ("===evaluate done: dataset total cost:{}, value of click/conversion:{}, expect ROI:{}, regularization parameter:{}, omega:{}, omega jac:{},  bids: {}, cost: {}, imps: {}, clicks: {}, roi:{}, ctr: {}%, cpc: {}"\
               .format(total_cost, self.eta, self.rho,
                       self.c, self.alpha, self.r['jac'],
                       len(bids),
                       math.floor(pays), math.floor(imps),
                       math.floor(clicks), roi,
                       clicks/imps*100.0, pays/clicks))
        return "{},{},{},{},{},{},{},{},{},{},{},{}".format(self.strategy, self.c, self.eta, self.rho,
                       self.alpha,
                       len(bids),
                       math.floor(pays), math.floor(imps),
                       math.floor(clicks), roi,
                       clicks/imps*100.0, pays/clicks)

    def sigmoid(self, inX):
        return 1.0 / (1 + np.exp(-inX))   
        
    def hinge_loss(self, X):
        return np.max(np.vstack((np.zeros((len(X),)), 1-X)), axis=0)
        
    def generate_bid(self, Ks):
        return self.bid(Ks, self.alpha)
        
    def bid(self, Ks, alpha):
        bids = np.zeros_like(Ks)
        if self.strategy == 'lin':
            bids = Ks*alpha[0]
        elif self.strategy == 'lin+':
            bids = Ks*alpha[0]/self.k_model.theta_e
        elif self.strategy == 'const':
            bids = np.zeros_like(Ks)+alpha[0]
        elif self.strategy == 'sqrt':
            bids = np.sqrt(Ks)*alpha[0]
        elif self.strategy == 'sqrt2':
            bids = (np.sqrt(Ks+alpha[1]**4)-alpha[1]**2)*alpha[0]
        elif self.strategy == 'sigmod':
            bids = (self.sigmoid(Ks)-0.5)*alpha[0]  
        elif self.strategy == 'sigmod2':
            bids = (self.sigmoid(Ks+alpha[1])-self.sigmoid(alpha[1]))*alpha[0]
        elif self.strategy == 'tanh':
            bids = np.tanh(Ks)*alpha[0]
        elif self.strategy == 'tanh2':
            bids = (np.tanh(Ks+alpha[1])-np.tanh(alpha[1]))*alpha[0]
        else:
            raise Exception("Unsupport bid strategy {}".format(self.strategy))

        # if bid<0 then bid = 0
        r = np.max(np.vstack((np.zeros((len(bids),)), bids)), axis=0) + sys.float_info.epsilon
        return r
    
    def k(self, x):  # CVR prediction
        return self.k_model.predict_proba(x)[:, 1]
#        return self.k_model.predict(x)/self.k_model.beta
    
    def eCost(self, b):  # expected cost prediction
        return self.winner.predict_win_price(b)
    
    def w(self, bids):  # winning probability prediction
        return self.winner.predict(bids)
    
    def loss(self, alpha, Ks):  # loss function
        N = Ks.shape[0]
#        Ks += sys.float_info.epsilon # in case of log0
        bids = self.bid(Ks, alpha)
        wins = self.w(bids)
        eCost = self.eCost(bids)
        eValues = wins*Ks*self.eta
        eROI = eValues/eCost
        
        soft_margin = self.hinge_loss(
                eROI/self.rho
                )
    
        penalty = (soft_margin*eCost).sum()*self.c
        norm = 0.5*self.labda*np.sum(alpha**2)
        
        cost = eCost.sum()
        
        l = cost - penalty - norm
        return -l
    
    def loss_der(self, alpha, Ks):  # derivative of loss function
        # not finished
        a = np.zeros_like(alpha)
        
        #Ks is also the derivative of alpha in b
        bids = self.bid(Ks, alpha)
        wins = self.w(bids)
        w_ders = self.winner.w_der(bids)
        eCost = self.eCost(bids)
        eCost_der = w_ders*bids
        eValues = wins*Ks*self.eta
        eROI = eValues/eCost

        b_der_0 = np.zeros_like(Ks)  # derivative of first parameter
        b_der_1 = np.zeros_like(Ks)  # derivative of second parameter
        if self.strategy == 'lin':
            b_der_0 = Ks
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'lin+':
            b_der_0 = Ks/self.k_model.theta_e
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'const':
            b_der_0 = np.zeros_like(Ks) + 1.0
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'sqrt':
            b_der_0 = np.sqrt(Ks)
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'sqrt2':
            temp = np.sqrt(Ks+alpha[1]**4)
            b_der_0 = temp - alpha[1]**2
            b_der_1 = ((np.zeros_like(Ks)+alpha[1]**2)/temp - 1)*alpha[0]*alpha[1]*2
        elif self.strategy == 'sigmod':
            b_der_0 = self.sigmoid(Ks)-0.5
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'sigmod2':
            sig_2 = self.sigmoid(np.zeros_like(Ks)+alpha[1])
            sig_ks_2 = self.sigmoid(Ks+alpha[1])
            b_der_0 = sig_ks_2- sig_2
            b_der_1 = (sig_ks_2*(-sig_ks_2+1) - sig_2*(-sig_2+1))*alpha[0]
        elif self.strategy == 'tanh':
            b_der_0 = np.tanh(Ks)
            b_der_1 = np.zeros_like(Ks)
        elif self.strategy == 'tanh2':
            tanh_2 = np.tanh(np.zeros_like(Ks)+alpha[1])
            tanh_ks_2 = np.tanh(Ks+alpha[1])
            b_der_0 = tanh_ks_2 - tanh_2
            b_der_1 = (tanh_2**2-tanh_ks_2**2)*alpha[0]
        else:
            raise Exception("Unsupport bid strategy {}".format(self.strategy))

        soft_margin = self.hinge_loss(eROI/self.rho)
        
        I = np.zeros_like(Ks)
        I[np.nonzero(soft_margin)] = 1
         
        l_der_b_cost = eCost_der*(-I*self.c+1+I*self.c/self.rho*eValues/eCost)
        l_der_b_penalty = I*w_ders*Ks*(eCost-bids*wins)*self.eta/self.rho/eCost*self.c                   
        
        cost_der_0 = (l_der_b_cost*b_der_0).sum()
        cost_der_1 = (l_der_b_cost*b_der_1).sum()
        penalty_der_0 = (l_der_b_penalty*b_der_0).sum()
        penalty_der_1 = (l_der_b_penalty*b_der_1).sum()
        
        a[0] = cost_der_0 + penalty_der_0
        a[1] = cost_der_1 + penalty_der_1
        
        return -a
