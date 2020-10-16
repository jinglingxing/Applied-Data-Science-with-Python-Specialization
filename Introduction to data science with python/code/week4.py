#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 16:18:21 2019

@author: jinlingxing
"""
import numpy as np
#>>> n, p = 10, .5  # number of trials, probability of each trial
#>>> s = np.random.binomial(n, p, 1000)
# result of flipping a coin 10 times, tested 1000 times.

x = np.random.binomial(20, .5, 10000)
print((x>=15).mean())


#tornado 龙卷风
chance_of_tornado = 0.01

tornado_events = np.random.binomial(1, chance_of_tornado, 1000000)
    
two_days_in_a_row = 0
for j in range(1,len(tornado_events)-1):
    if tornado_events[j]==1 and tornado_events[j-1]==1:
        two_days_in_a_row+=1

print('{} tornadoes back to back in {} years'.format(two_days_in_a_row, 1000000/365))

