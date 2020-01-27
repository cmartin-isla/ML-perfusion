from sklearn.model_selection import StratifiedKFold
from feature_processing.utils import *

import logging
import datetime
from sklearn.svm import SVC 
from EstimatorSelection.EstimatorSelectionHelper import *
from classifiers import *


t = 30
seed = 42
n_folds = 10
np.random.seed(seed)


json_filepath = '../../feature_extraction/extracted_features/raw_timeseries_bw25.json'
X, y, feature_names, resampling = difference_features(json_filepath,t)



helper = EstimatorSelectionHelper(clfs)
k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True) 

helper.fit(X, y, scoring=None, n_jobs=6, cv = k_fold)

file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
helper.score_summary(sort_by='max_score', out = file)

