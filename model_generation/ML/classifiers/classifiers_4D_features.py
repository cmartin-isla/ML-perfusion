from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.decomposition import FactorAnalysis
from tsfresh.feature_extraction import extract_features, MinimalFCParameters


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from tsfresh.transformers import RelevantFeatureAugmenter,FeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

ppl = Pipeline([('fresh', FeatureAugmenter(n_jobs=6,column_id='id', column_sort='time',default_fc_parameters=MinimalFCParameters())),
                ('reduce_dim',FactorAnalysis(n_components = 30)),
                ('clf', AdaBoostClassifier())])
clfs = {
        
        'AB':  {'clf': ppl,
                 'parms': {'clf__n_estimators':[100],
                           'clf__algorithm' : ['SAMME']}
                },
        
         
        }




