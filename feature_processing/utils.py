import json
import numpy as np
import collections
import pprint as pp
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rescale, resize, downscale_local_mean,rotate
import matplotlib.pyplot as plt


def import_json_timeseries(json_filepath):
    
    with open(json_filepath) as json_file:
        data = json.load(json_file, object_pairs_hook=collections.OrderedDict)
    
    return data

def resample_and_rescale(raw2D, resampling = 30, scaler = MinMaxScaler()):
    
    raw2D = np.array(raw2D).T
    raw2D = resize(np.array(raw2D),(resampling,raw2D.shape[1]),order = 0, mode = 'edge',anti_aliasing = False)
    
    scaler.fit(raw2D)
    raw2D = scaler.transform(raw2D)
    
    return raw2D
    
    
def difference_features(json_file,resampling = 30, verbose = 0):
    
    data = import_json_timeseries(json_file)
    
    feature_names = [k.replace('original_','') for k in data[list(data.keys())[0]]['basal_features'].keys()]
    
    
    X = []
    y = []
    
    for k,v in data.items():
        
        id = k
        subject = v
        outcome = subject['outcome']
        
        basal =resample_and_rescale([x for x in subject['basal_features'].values()],resampling = resampling)
        adeno =resample_and_rescale([x for x in subject['adeno_features'].values()],resampling = resampling)

        X.append(np.ravel((basal-adeno).T))
        y.append(outcome)
        
    X = np.array(X)
    y = np.array(y)

    
    if verbose:
        summary = {'X' : X.shape, 'Y' : y.shape, 'feature_names': feature_names, 'temporal_resampling': resampling}
        print('PREPROCESSING SUMMARY')
        pp.pprint(summary)
        
    return X,y,feature_names,resampling
            
    
    

