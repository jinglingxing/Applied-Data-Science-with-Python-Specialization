#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:34:43 2020

@author: jinlingxing
"""

#Assignment 4 - Understanding and Predicting Property Maintenance Fines
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
#import data
train_data = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/train.csv",sep = ',',encoding='cp1252',dtype={'zip_code': str, 'non_us_str_code': str, 'grafitti_status': str, 'violator_name':str, 
                        'mailing_address_str_number': str})
test_data = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/test.csv",sep = ',',encoding='cp1252')
address = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/addresses.csv",sep = ',',encoding='cp1252')
latlons = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/latlons.csv",sep = ',',encoding='cp1252')
add_lat = pd.merge(address,latlons, on = 'address')
train = pd.merge( add_lat, train_data,on ='ticket_id')
    # drop all unnecessary columns
train.drop(['agency_name', 'inspector_name', 'violator_name', 'non_us_str_code', 'violation_description', 
                    'grafitti_status', 'state_fee', 'admin_fee', 'ticket_issued_date', 'hearing_date',
                    # columns not available in test
                    'payment_amount', 'balance_due', 'payment_date', 'payment_status', 
                    'collection_status', 'compliance_detail', 
                    # address related columns
                    'violation_zip_code', 'country', 'address', 'violation_street_number',
                    'violation_street_name', 'mailing_address_str_number', 'mailing_address_str_name', 
                    'city', 'state', 'zip_code', 'address'], axis=1, inplace=True)
test = pd.merge(add_lat, test_data, on = 'ticket_id')
train['lat'] = train['lat'].fillna(train['lat'].mean())
train['lon'] = train['lon'].fillna(train['lon'].mean())
test['lat'] = test['lat'].fillna(test['lat'].mean())
test['lon'] = test['lon'].fillna(test['lon'].mean())


#drop null data
#get train_data and target
train = train[np.isfinite(train['compliance'])]
y = train.iloc[:,-1]
y_bincount = np.bincount(y)
paid_per = y_bincount[1]/y.count()
#0.0725 it's an imbalanced classification problem. 

#fit_transform
label_encoder = preprocessing.LabelEncoder()
train = train.apply(label_encoder.fit_transform)


#train_model
X_train, X_test, y_train, y_test = train_test_split(train.ix[:, train.columns != 'compliance'], train['compliance'])
rf = RandomForestRegressor()
grid_values = {'n_estimators': [10, 100], 'max_depth': [None, 30]}
grid_rf_auc = GridSearchCV(rf, param_grid=grid_values, scoring='roc_auc')
grid_rf_auc.fit(X_train, y_train)
print('Grid best parameter (max. AUC): ', grid_rf_auc.best_params_)
print('Grid best score (AUC): ', grid_rf_auc.best_score_)
result = pd.DataFrame(grid_rf_auc.predict(test), test.ticket_id)