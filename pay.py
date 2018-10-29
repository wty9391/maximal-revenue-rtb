#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math

epsilon = sys.float_info.epsilon


class pay():
    def __init__(self,f_log):
        self.f_log = f_log
        self.name_col = {}  # feature_name:origin_index
        self.pay_pdf = {}
        self.pay_cdf = []
        self.pay = []
        self.amount = 0
        self.max_pay = 0
        
    def load(self):
        first = True
        count = 0
        for l in self.f_log:
            s = l.split('\t')
            if first:
                for i in range(0, len(s)):
                    self.name_col[s[i].strip()] = i
                pay_index = self.name_col['payprice']
                first = False
                continue
            win_price = float(s[pay_index])+epsilon
            win_price_int = math.floor(win_price)
            
            if win_price_int in self.pay_pdf:
                self.pay_pdf[win_price_int] += 1
            else:
                self.pay_pdf[win_price_int] = 1
                
            self.pay.append(win_price_int)
            count += 1

        self.amount = len(self.pay)
        self.max_pay = max(self.pay)
        for i in self.pay_pdf:
            self.pay_pdf[i] = self.pay_pdf[i]/self.amount

        self.pay_cdf.append(self.pay_pdf.get(0, 0))
        for x in range(1, self.max_pay):
                p = self.pay_pdf.get(x, 0) + self.pay_cdf[x-1]
                self.pay_cdf.append(p)
        self.pay_cdf.append(1.0)     
        self.pay_cdf[0] += epsilon
