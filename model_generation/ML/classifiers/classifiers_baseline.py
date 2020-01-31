from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute



"""
clfs = {'SVM':  {'clf': SVC(class_weight="balanced"),
                 'parms':  {'C':                    [0.1,0.5, 1,5, 10,50, 100,500, 1000],  
                            'gamma':                [1,0.5, 0.1,0.05, 0.01, 0.005, 0.001,0.0005, 0.0001], 
                            'kernel':               ['linear','rbf']}  ,
                },
                
        'RF':   {'clf': RandomForestClassifier(n_estimators=200, class_weight="balanced"),
                 'parms': {"max_depth":            [9,5,3,2,None],
                           "max_features":         [20,25,30,35,40],
                           "min_samples_split":    [2,3,5,7,10],
                           "bootstrap":            [True, False]},
                },
        
        'AB':  {'clf': AdaBoostClassifier(),
                 'parms': {'n_estimators':[100,150,200,250,300],
                           'learning_rate':[1.0,1.5,2.0,2.5,3,3.5,4.0]},
                           'algorithm' : ['SAMME', 'SAMME.R']
                },
        
         
        }


"""
clfs = {'SVM':  {'clf': SVC(class_weight="balanced"),
                 'parms':  {'C':                    [0.1,0.5, 1,5],  
                            'gamma':                [1,0.5, 0.1], 
                            'kernel':               ['linear','rbf']}  ,
                },
                
        'RF':   {'clf': RandomForestClassifier(n_estimators=200, class_weight="balanced"),
                 'parms': {"max_depth":            [9,5,3,2,None],
                           "min_samples_split":    [2,3,5,7,10],
                           "bootstrap":            [True, False]},
                },
        
        'AB':  {'clf': AdaBoostClassifier(),
                 'parms': {'n_estimators':[100,150,200,250,300]},
                           'algorithm' : ['SAMME', 'SAMME.R']
                },
        
         
        }


"""


clfs = {
        
        'AB':  {'clf': AdaBoostClassifier(),
                 'parms': {'n_estimators':[100,150,200,250,300]},
                           'algorithm' : ['SAMME', 'SAMME.R']
                },
        
         
        }

"""

