from sklearn.model_selection import StratifiedKFold
from feature_processing.utils import *

import logging,os,pickle
import datetime
from sklearn.svm import SVC 
from EstimatorSelection.EstimatorSelectionHelper import *
from classifiers import *
from sklearn.model_selection import cross_val_score


t = 30
seed = 42
n_folds = 10
np.random.seed(seed)


json_filepath = '../../feature_extraction/extracted_features/raw_timeseries_bw25_norm.json'
X, y, feature_names, resampling, ids = difference_features(json_filepath,t)




    


helper = EstimatorSelectionHelper(clfs)
k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True) 

models = 'models/RandomForest/'

for file in os.listdir(models):
    print(file)
    grid_search = pickle.load(open(os.path.join('models/RandomForest',file), 'rb'))
    scores = cross_val_score(grid_search.best_estimator_, X, y, cv=k_fold)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

helper.fit(X, y, scoring=None, n_jobs=6, cv = k_fold)

file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv'
helper.score_summary(sort_by='max_score', out = file)

