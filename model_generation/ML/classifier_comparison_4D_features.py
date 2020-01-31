from sklearn.model_selection import StratifiedKFold
from feature_processing.utils import *

import logging,os,pickle
import datetime
from sklearn.svm import SVC 
from EstimatorSelection.EstimatorSelectionHelper import *
from classifiers.classifiers_4D_features import *
from sklearn.model_selection import cross_val_score

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
        json_filepath = os.path.join(features_folder,file_in)
        X_ds, y = all_features_dataframe(json_filepath,t)
        X = pd.DataFrame(index=y.index)
        

        
        clfs['AB']['clf'].set_params(fresh__timeseries_container=X_ds)
        helper = EstimatorSelectionHelper(clfs)
        k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True) 
        
        
        helper.fit(X, y, scoring=None, n_jobs=6, cv = k_fold)
        file_out = os.path.join('results',file_in.split('.')[0]+'_4D_'+ execution_time + '.csv')
        helper.score_summary(sort_by='max_score', out = file_out)

     
    
        
        
        

