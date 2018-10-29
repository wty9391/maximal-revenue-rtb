#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:31:33 2018

@author: wty
"""

import sys
import pickle

import encoder
import bidder



# ./result/1458 ../make-ipinyou-data/1458/test.log.txt
#
if len(sys.argv) < 3:
    print 'Usage: train_bidder.py result_root_path test_log_path'
    exit(-1)

f_encoder = open(sys.argv[1]+'/encoder', 'r')
f_imp_eval_model = open(sys.argv[1]+'/imp_eval_model', 'wr')
f_bidder = open(sys.argv[1]+'/bidder', 'wr')

my_encoder = pickle.load(f_encoder)
my_bidder=pickle.load(f_bidder)

X_test_raw = []
first = True
for line in open(sys.argv[2],'r'):
    if first:
        first = False
        continue
    X_test_raw.append(line)
    
X_test = my_encoder.encode(X_test_raw,0)
pickle.dump(X_test, open(sys.argv[1]+'/x_test', 'w'))  
 
#X_test = pickle.load(open(sys.argv[1]+'/x_test', 'r')) 

my_bidder.evaluate(X_test,X_test_raw)

















    
    
    