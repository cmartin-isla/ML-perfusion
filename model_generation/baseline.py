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
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC


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


json_filepath = '../feature_extraction/extracted_features/raw_timeseries_bw25.json'
X, y, feature_names, resampling = difference_features(json_filepath,t)



k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True)

clf = SVC(class_weight="balanced")

param_grid =  {'C':                    [0.1, 1, 10, 100, 1000],  
                            'gamma':                [1, 0.1, 0.01, 0.001, 0.0001], 
                            'kernel':               ['linear','rbf']}



grid_search = GridSearchCV(clf, param_grid=param_grid, cv=k_fold, n_jobs=7)
grid_search.fit(X, y)

print(grid_search.cv_results_)
