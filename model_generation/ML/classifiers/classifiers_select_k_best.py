from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

import sklearn
print(sklearn.__version__)
from sklearn.impute import SimpleImputer



svc_ppl = Pipeline([
  ('scaler', MinMaxScaler()),
  ('reduce_dim', SelectKBest(chi2)),
  ('imputation',SimpleImputer(strategy = 'median', fill_value = 0)),
  ('classification', SVC(class_weight="balanced"))
])

rf_ppl = Pipeline([
  ('scaler', MinMaxScaler()),
  ('reduce_dim', SelectKBest(chi2)),
  ('imputation',SimpleImputer(strategy = 'median', fill_value = 0)),
  ('classification', RandomForestClassifier(n_estimators=200, class_weight="balanced"))
])

ab_ppl = Pipeline([
  ('scaler', MinMaxScaler()),
  ('reduce_dim', SelectKBest(chi2)),
  ('imputation',SimpleImputer(strategy = 'median', fill_value = 0)),
  ('classification', AdaBoostClassifier())
])

dim_reductors = [SelectKBest(chi2),
                 SelectKBest(f_classif),
                 SelectKBest(mutual_info_classif)]


n_components = [10,20,30,36]
  
clfs = {'SVM':  {'clf': svc_ppl,
                 'parms':  {'reduce_dim': dim_reductors,
                            'reduce_dim__k': n_components,
                            'classification__C':                    [0.1,0.5, 1,5],  
                            'classification__gamma':                [1,0.5, 0.1], 
                            'classification__kernel':               ['linear','rbf']}  ,
                },
                
        'RF':   {'clf': rf_ppl,
                 'parms': {'reduce_dim': dim_reductors,
                           'reduce_dim__k': n_components,
                           "classification__max_depth":            [9,5,3,2,None],
                           "classification__min_samples_split":    [2,3,5,7,10],
                           "classification__bootstrap":            [True, False]},
                },
        
        'AB':  {'clf': ab_ppl,
                 'parms': {'reduce_dim': dim_reductors,
                           'reduce_dim__k': n_components,
                        'classification__n_estimators':[100,150,200,250,300],
                           'classification__algorithm' : ['SAMME', 'SAMME.R']}
                },
        
         
        }



