#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pickle
import math

import numpy as np
from scipy.stats import norm
from scipy.sparse import csr_matrix, vstack
from scipy.stats import entropy, wasserstein_distance
import matplotlib.pyplot as plt

import sklearn.metrics
from sklearn import linear_model

import encoder
import winner
import pay


if len(sys.argv) < 5:
    print ('Usage: train_init.py trian_log_path test_log_path feat_path result_root_path')
    exit(-1)
    
    
# all constant
epsilon = sys.float_info.epsilon
batch_size = 1e6  # we load training data and test data in batch

f_train_log = open(sys.argv[1], 'r', encoding="utf-8")
f_test_log = open(sys.argv[2], 'r', encoding="utf-8")

# load payment logs
train_pay = pay.pay(f_train_log)
train_pay.load()
test_pay = pay.pay(f_test_log)
test_pay.load()

# init winning function
my_winner = winner.wining_pridictor(d=0.01, max_iter=500, eta=1e-3)
# fit winning function
my_winner.fit(np.arange(1, train_pay.max_pay+1, 1), train_pay.pay_cdf[1:])
print("Fit complete! Beta: ", my_winner.d)

# evaluate the winning function by wasserstein distance
pred_cdf = my_winner.w(np.arange(0,train_pay.max_pay+1,1))
pred_cdf[0] += epsilon
train_WD = wasserstein_distance(pred_cdf, train_pay.pay_cdf)
print("Training set: wasserstein distance between truth_cdf and pred_cdf: ", train_WD)
test_WD = wasserstein_distance(pred_cdf, test_pay.pay_cdf)
print("Test set: wasserstein distance between truth_cdf and pred_cdf: ", test_WD)
   

# init ipinyou encoder
ipinyou = encoder.Encoder_ipinyou(sys.argv[3], train_pay.name_col)

X_train_raw = []
X_train = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_train = np.zeros((0, 1), dtype=np.int8)
X_test_raw = []
X_test = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_test = np.zeros((0, 1), dtype=np.int8)

print("start to load raw training data")
count = 0
first = True
f_train_log.seek(0)
for line in f_train_log:
    if first:
        first = False
        continue
    X_train_raw.append(line)
    count += 1
    if count % batch_size == 0:
        X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
        Y_train = np.vstack((Y_train, ipinyou.get_labels(X_train_raw)))
        X_train_raw = []
if X_train_raw != []:
    X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
    Y_train = np.vstack((Y_train, ipinyou.get_labels(X_train_raw)))
    X_train_raw = []
print("all [{}] training records has been loaded".format(count))

print("start to load raw test data")
count = 0
first = True
f_test_log.seek(0)
for line in f_test_log:
    if first:
        first = False
        continue
    X_test_raw.append(line)
    count += 1
    if count % batch_size == 0:
        X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
        Y_test = np.vstack((Y_test, ipinyou.get_labels(X_test_raw)))
        X_test_raw = []
if X_test_raw != []:
    X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
    Y_test = np.vstack((Y_test, ipinyou.get_labels(X_test_raw)))
    X_test_raw = []
print("all [{}] test records has been loaded".format(count))

# impression evaluator
model = linear_model.SGDClassifier(loss='log', penalty='l2', alpha=1e-6, verbose=0, n_jobs=4, max_iter=100)
model.fit(X_train, Y_train.ravel())
predict_Y_train = model.predict_proba(X_train)[:, 1]
model.beta = predict_Y_train.sum()/Y_train.sum()
auc = sklearn.metrics.roc_auc_score(Y_train, predict_Y_train)
print('AUC in training data is: {}'.format(auc))

predict_Y_test = model.predict_proba(X_test)[:, 1]
auc = sklearn.metrics.roc_auc_score(Y_test, predict_Y_test)
print('AUC in test data is: {}'.format(auc))


pickle.dump(ipinyou, open(sys.argv[4]+'/encoder', 'wb'))  # the encoder
pickle.dump(X_train, open(sys.argv[4]+'/x_train', 'wb'))  # the encoded training data
pickle.dump(Y_train, open(sys.argv[4]+'/y_train', 'wb'))  # the encoded training data
pickle.dump(X_test, open(sys.argv[4]+'/x_test', 'wb'))    # the encoded test data
pickle.dump(model, open(sys.argv[4]+'/imp_eval_model', 'wb'))  # the impression evaluation model
pickle.dump(my_winner, open(sys.argv[4]+'/winning_model', 'wb')) # the winning function

f_train_log.close()
f_test_log.close()



