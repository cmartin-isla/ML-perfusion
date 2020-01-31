from sklearn.model_selection import StratifiedKFold
from feature_processing.utils import *

import logging,os,pickle
import datetime
from sklearn.svm import SVC 
from EstimatorSelection.EstimatorSelectionHelper import *
from classifiers.classifiers_reduce_dim import *
from sklearn.model_selection import cross_val_score

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)


t = 10
seed = 42
n_folds = 10
np.random.seed(seed)

execution_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
features_folder = '../../feature_extraction/extracted_features/current_experiment'

if __name__ == '__main__':
    
    
    for file_in in os.listdir(features_folder):
        print('EXPERIMENT',file_in)
        json_filepath = os.path.join(features_folder,file_in)
        X, y, feature_names, resampling, ids = difference_features(json_filepath,t)
        
        
        helper = EstimatorSelectionHelper(clfs)
        k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True) 
        
        
        helper.fit(X, y, scoring=None, n_jobs=6, cv = k_fold)
        file_out = os.path.join('results',file_in.split('.')[0]+'_reduce_dim_'+ execution_time + '.csv')
        helper.score_summary(sort_by='mean_score', out = file_out)
        
     
    
        
        
        

