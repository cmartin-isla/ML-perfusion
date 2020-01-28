import numpy as np
from sklearn import datasets, svm
import matplotlib.pyplot as plt
import os
import pickle
import random
import numpy as np

from time import time
from scipy.stats import randint as sp_randint

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
import sklearn
#print(sklearn.__version__)
import numpy as np
from sklearn.metrics import classification_report,roc_curve

from scipy.stats import uniform, randint
from sklearn.svm import SVC

from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from scipy import interp
from EstimatorSelection.EstimatorSelectionHelper import *
from feature_processing.utils import *


def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
    
    
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
            
if __name__ == '__main__':          
     
    t = 30
    json_filepath = '../../feature_extraction/extracted_features/raw_timeseries_bw25_norm.json'
    X, y, feature_names, resampling , subjects = difference_features(json_filepath,t)
    X_2,y_2,subjects_2 = import_pickle_features('../../feature_extraction/features_mid/')
    
    print(X.shape)
    print(X_2.shape)
    for idx_1,idx_2 in zip(np.argsort(subjects),np.argsort(subjects_2)):
        s_x_1 = np.reshape(X[idx_1,...],(18,30))
        s_x_2 = np.reshape(X_2[idx_2,...],(30,18)).T
        #plt.imshow(np.hstack((s_x_1,s_x_2)))
        #plt.show()
        
    all_f1scores = []
    
    np.random.seed(42)
    
    cnt = 0
    k_fold = StratifiedKFold(n_splits=10, shuffle=True)
    all_maps = []
    f1scores = []
    for trainval,test in k_fold.split(X,y):
        
        x_trainval = X[trainval,...]
        y_trainval = y[trainval,...]
        
        x_test = X[test,...]
        y_test = y[test,...]
        print(y_trainval)
        print(y_test)
        
        
        
        k_fold_2 = StratifiedKFold(n_splits=10, shuffle=True)
        
        
        
        clf = RandomForestClassifier(n_estimators=500, class_weight="balanced")
        param_grid = {"max_depth": [15,13,11,9,5,3,2,None],
          "max_features": [20,25,30,35,40,50,60,70,80,90,100],
          "min_samples_split": [2,3,5,7,10,15,20,25,30,35,40],
          "bootstrap": [True, False],
          "criterion": ["gini", "entropy"]}
        
        param_grid = {"max_depth": [None],
          "max_features": ['auto'],
          "min_samples_split": [2,3,5,7,10,15,20,25,30,35,40],
          "bootstrap": [True],
          "criterion": ["gini"]}
        
        
        
        grid_search = GridSearchCV(clf, param_grid=param_grid, cv=k_fold_2, iid=False, n_jobs=6,verbose=1)
        start = time()
        grid_search.fit(x_trainval, y_trainval)
        model = 'models/RF_cv_'+str(cnt)+'_nofs.mod'
        pickle.dump(grid_search, open(model, 'wb'))

        
        
        """
        model = 'models/RF_cv_'+str(cnt)+'_nofs.mod'
        grid_search = pickle.load(open(model, 'rb'))
        """
        cnt = cnt + 1 
        y_pred =grid_search.best_estimator_.predict(x_test)
        #all_maps.append(grid_search.best_estimator_.feature_importances_)
        report = classification_report(y_test, y_pred)
        print(report)
        f1scores.append(float(report[199:203]))
    
    print(f1scores)
    print('mean-f1:',np.mean(f1scores))
    all_f1scores.append(np.mean(f1scores))
    
    
    
