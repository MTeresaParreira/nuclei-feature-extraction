# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:05:06 2019

@author: hemaxi
"""

" hyperparameter optimization for SVM "


from sklearn.model_selection import train_test_split
# =============================================================================
# from k_means import build_color_array
# 
# =============================================================================
from sklearn.utils import class_weight
from sklearn import svm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")



df = pd.read_pickle(r'')


#input dapi features
matrix = df.as_matrix(columns = ['norm_area', 'norm_intensity'])
X = matrix.astype(float).reshape(matrix.shape)

#output FUCCI labels
y_aux = df.as_matrix(columns = ['Automatic Label'])
y = y_aux.astype(float).reshape(y_aux.shape)
y = y.ravel()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


y_train = y_train.ravel()
y_test = y_test.ravel()

# =============================================================================
# # Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
#                      'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
#                     {'kernel': ['poly'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000],
#                      'degree': [1, 2, 3, 4, 5]}
#                    ]
# =============================================================================



scores = ['f1_macro']

#scores = {'precision': 'precision_macro', 'recall': 'recall_macro', 'f1score': 'f1_macro'}

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()
    
    clf = GridSearchCV(SVC(class_weight = 'balanced'), tuned_parameters, cv=5,
                       scoring= score)
    clf.fit(X_train, y_train)
    
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    
    
