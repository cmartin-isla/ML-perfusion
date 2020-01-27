import numpy as np
from sklearn import datasets, svm
import matplotlib.pyplot as plt
import os
import pickle
import random
import numpy as np
import pandas as pd
from time import time
from scipy.stats import randint as sp_randint
from functools import reduce

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import classification_report,roc_curve
from scipy.stats import uniform, randint
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import interp
from feature_processing.utils import *
from sklearn.externals import joblib
from utils.avg_report import *
import logging
import datetime

reports=[]

fold = 0
t = 30
seed = 42
n_folds = 10
np.random.seed(seed)



logging.basicConfig(level=logging.DEBUG,filename='rf.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s')

logging.debug('EXPERIMENT RANDOM FOREST' + str(datetime.datetime.now())+'_________________________________________________________')
logging.debug('Seed:'+ str(seed))
logging.debug('Temporal dimension:'+ str(t))
logging.debug('Number of stratified folds:'+ str(n_folds))
logging.debug('Input file:'+'../../feature_extraction/extracted_features/raw_timeseries_bw25.json')


json_filepath = '../../feature_extraction/extracted_features/raw_timeseries_bw25.json'
X, y, feature_names, resampling = difference_features(json_filepath,t)

k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True) 

for trainval,test in k_fold.split(X,y):
    
    
    x_trainval = X[trainval,...]
    y_trainval = y[trainval,...]
    x_test = X[test,...]
    y_test = y[test,...]
    
    
    k_fold_2 = StratifiedKFold(n_splits=n_folds, shuffle=True)

    clf = RandomForestClassifier(n_estimators=500, class_weight="balanced")

    param_grid = {"max_depth":          [15,13,11,9,5,3,2,None],
                  "max_features":       [20,25,30,35,40,50,60,70,80,90,100],
                  "min_samples_split":  [2,3,5,7,10,15,20,25,30,35,40],
                  "bootstrap":          [True, False],
                  "criterion":          ["gini", "entropy"]}
    
    
    
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=k_fold_2, iid=False, n_jobs=7)
    grid_search.fit(x_trainval, y_trainval)
    

    joblib.dump(grid_search, 'models/rf_fold'+str(fold)+'_t_'+str(t)+'.pkl')
    
    y_pred =grid_search.best_estimator_.predict(x_test)
    report = classification_report(y_test, y_pred, output_dict = True)
    reports.append(report)
    
    logging.debug('FOLD:'+ str(fold))
    logging.debug('Test report:'+ json.dumps(report,indent = 2))

    
    fold = fold + 1
    
 
logging.debug('AVERAGE')
logging.debug('Average report:'+ json.dumps(avg_report(reports,10),indent = 2))