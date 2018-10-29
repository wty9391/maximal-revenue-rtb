#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import numpy as np
import random

import encoder
import bidder

if len(sys.argv) < 4:
    print ('Usage: train_bidder.py result_root_path trian_log_path test_log_path')
    exit(-1)

f_encoder = open(sys.argv[1]+'/encoder', 'rb')
f_x_train = open(sys.argv[1]+'/x_train', 'rb')
f_x_test = open(sys.argv[1]+'/x_test', 'rb')
f_imp_eval_model = open(sys.argv[1]+'/imp_eval_model', 'rb')
f_winner = open(sys.argv[1]+'/winning_model', 'rb')
f_train_log = open(sys.argv[2], 'r', encoding="utf-8")
f_test_log = open(sys.argv[3], 'r', encoding="utf-8")

f_bidder = open(sys.argv[1]+'/bidder', 'wb')

my_imp_eval_model = pickle.load(f_imp_eval_model)
my_encoder = pickle.load(f_encoder)
my_winner = pickle.load(f_winner)
X_train = pickle.load(f_x_train)
X_test = pickle.load(f_x_test) 

click_index = my_encoder.name_col['click']
pay_index = my_encoder.name_col['payprice']

simulate_logs_train = []
simulate_logs_test = []

print("start to load raw train file")
count = 0
first = True
for line in f_train_log:
    if first:
        first = False
        continue
    s = line.split("\t")
    simulate_logs_train.append([int(s[click_index]),float(s[pay_index])])
    
    count += 1
print("all [{}] train records has been loaded".format(count))    

print("start to load raw test file")
count = 0
first = True
for line in f_test_log:
    if first:
        first = False
        continue
    s = line.split("\t")
    simulate_logs_test.append([int(s[click_index]),float(s[pay_index])])
    
    count += 1
print("all [{}] test records has been loaded".format(count)) 

# regularization parameter
cs = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 2.0, 5.0, 10.0]
strategies = ['lin', 'sqrt2']

train_log_in = open(sys.argv[1]+"/train_log.csv", "w")
test_log_in = open(sys.argv[1]+"/test_log.csv", "w")

log_header = "strategy,C,V,R,omega,bids,cost,impressions,clicks,roi,ctr(%),cpc"
train_log_in.write(log_header + "\n")
test_log_in.write(log_header + "\n")

for s in strategies:
    print("=Current strategy is [{}]".format(s))
    for c in cs:
        print("==Current C is [{}]".format(c))
        my_bidder = bidder.Bidder(my_imp_eval_model,
                                  my_encoder,
                                  my_winner,
                                  c=c, strategy=s)
        r = my_bidder.optimize(X_train)
        print("strategy: {}, C: {}, optimize result:".format(s, c))
        print("For training data:")
        train_log = my_bidder.evaluate(X_train, simulate_logs_train)
        train_log_in.write(train_log + "\n")
        print("For test data:")
        test_log = my_bidder.evaluate(X_test, simulate_logs_test)
        test_log_in.write(test_log + "\n")
        print("")
    print("")

train_log_in.flush()
train_log_in.close()
test_log_in.flush()
test_log_in.close()
