#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:34:43 2020

@author: jinlingxing
"""

#Assignment 3 - Evaluation

#In this assignment you will train several models and evaluate how effectively they predict instances of fraud using data
#based on this dataset from Kaggle.
#Each row in fraud_data.csv corresponds to a credit card transaction. 
#Features include confidential variables V1 through V28 as well as Amount which is the amount of the transaction.
#The target is stored in the class column, where a value of 1 corresponds to an instance of fraud and 0 corresponds 
#to an instance of not fraud.

import numpy as np
import pandas as pd

#Question 1
#Import the data from fraud_data.csv. What percentage of the observations in the dataset are instances of fraud?
#This function should return a float between 0 and 1.
def answer_one():
    count = 0
    df = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/fraud_data.csv")
    for i in range(len(df)):
        if df['Class'][i]==1:
            count += 1
        per = count/len(df)
    return  per# Return your answer
#0.016410823768035772
    
# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/jinlingxing/dataScienceStudy/applied-machine-learning-in-python/course3_downloads/fraud_data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#Question 2
#Using X_train, X_test, y_train, and y_test (as defined above), train a dummy classifier that classifies 
#everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
#This function should a return a tuple with two floats, i.e. (accuracy score, recall score).
def answer_two():
    from sklearn.dummy import DummyClassifier
    dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train, y_train)
    y_dummy_prediction = dummy_majority.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_dummy_prediction)#0.9852507374631269
    from sklearn.metrics import recall_score 
    rec = recall_score(y_test, y_dummy_prediction) #0
    return [acc,rec]
# [0.9852507374631269, 0.0]
    
#Question 3    
#Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. 
#What is the accuracy, recall, and precision of this classifier?
#This function should a return a tuple with three floats, i.e. (accuracy score, recall score, precision score).

def answer_three():
    from sklearn.svm import SVC
    svm = SVC(gamma = 'scale').fit(X_train, y_train)
    y_pre = svm.predict(X_test)
    from sklearn.metrics import accuracy_score, recall_score, precision_score
    acc = accuracy_score(y_test, y_pre)
    rec = recall_score(y_test, y_pre)
    pre = precision_score(y_test, y_pre)
    return [acc,rec,pre]

#Question 4    
    #Using the SVC classifier with parameters {'C': 1e9, 'gamma': 1e-07},
    # what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
    #This function should return a confusion matrix, a 2x2 numpy array with 4 integers.

def answer_four():
    from sklearn.svm import SVC 
    from sklearn.metrics import confusion_matrix
    svm = SVC(gamma = 1e-07, C = 1e9).fit(X_train, y_train)
    y_score = svm.decision_function(X_test)
    y_score = np.where(y_score > -220,1,0)
    confusion = confusion_matrix(y_test, y_score)
    return confusion
#array([[5320,   24],
#       [  14,   66]])
    
# if you use the decision_function in LinearSVC classifier, the relation between those two will be more clear!
# Because then decision_function will give you scores for each class label (not same as SVC) and
# predict will give the class with the best score.

#Question 5
#Train a logisitic regression classifier with default parameters using X_train and y_train.
#For the logisitic regression classifier, create a precision recall curve and a roc curve
# using y_test and the probability estimates for X_test (probability it is fraud).
    
#Looking at the precision recall curve, what is the recall when the precision is 0.75? -----it's 0.8
#Looking at the roc curve, what is the true positive rate when the false positive rate is 0.16?
#This function should return a tuple with two floats, i.e. (recall, true positive rate)

def answer_five():
    from sklearn.linear_model import LogisticRegression
    logreg = LogisticRegression().fit(X_train, y_train)
    
    from sklearn.metrics import precision_recall_curve
    y_scores_lr = logreg.decision_function(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)

    from sklearn.metrics import roc_curve, auc
    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    precision_index = np.argwhere(precision==0.75)
    recall_specified = recall[precision_index]  #0.825
    
    fpr_index = np.argwhere(fpr_lr == 0.16)
    print(fpr_index)
    tpr_specified = tpr_lr[fpr_index]
    
    
    import matplotlib.pyplot as plt
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

    return [0.825, 0.935]
#(0.825, 0.935)
    
 
#Question 6
#Perform a grid search over the parameters listed below for a Logisitic Regression classifier,
# using recall for scoring and the default 3-fold cross validation.
#'penalty': ['l1', 'l2']
#'C':[0.01, 0.1, 1, 10, 100]
#From .cv_results_, create an array of the mean test scores of each parameter combination. i.e.
#        l1	l2
#0.01	?	?
#0.1	?	?
#1	    ?	?
#10	    ?	?
#100	?	?

#This function should return a 5 by 2 numpy array with 10 floats.
#Note: do not return a DataFrame, just the values denoted by '?' above in a numpy array.
def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    grid_values = {'penalty': ['l1','l2'], 'C': [0.01,0.1, 1, 10, 100], }
    lr = LogisticRegression()
    grid_lr_recall = GridSearchCV(lr, param_grid = grid_values, scoring = 'recall')
    grid_lr_recall.fit(X_train, y_train)
    results = grid_lr_recall.cv_results_
    test_scores = np.vstack((results['split0_test_score'], results['split1_test_score'], results['split2_test_score']))
    return  test_scores.mean(axis=0).reshape(5, 2)# Return your answer
# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
    plt.yticks(rotation=0);

GridSearch_Heatmap(answer_six())