import json,os,pickle
import numpy as np
import pandas as pd
import collections
import pprint as pp
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import rescale, resize, downscale_local_mean,rotate
import matplotlib.pyplot as plt
from astropy.wcs.docstrings import row


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

def reduce_pca(X,var):  
    from sklearn.decomposition import PCA

    pca = PCA(n_components=var, whiten=True)
    pca.fit(np.nan_to_num(np.array(X)))
    X_pca = pca.transform(np.array(X))
    return X_pca
    
    
def difference_features(json_file,resampling = 30, verbose = 0):
    
    data = import_json_timeseries(json_file)
    
    feature_names = [k.replace('original_','') for k in data[list(data.keys())[0]]['basal_features'].keys()]
    
    
    X = []
    y = []
    subjects = []
    
    for k,v in data.items():
        
        id = k
        subject = v
        outcome = subject['outcome']
        
        basal =resample_and_rescale([x for x in subject['basal_features'].values()],resampling = resampling)
        adeno =resample_and_rescale([x for x in subject['adeno_features'].values()],resampling = resampling)
        
        X.append(np.ravel((basal-adeno).T))
        y.append(outcome)
        subjects.append(id)
        
    X = np.array(X)
    y = np.array(y)

    
    if verbose:
        summary = {'X' : X.shape, 'Y' : y.shape, 'feature_names': feature_names, 'temporal_resampling': resampling}
        print('PREPROCESSING SUMMARY')
        pp.pprint(summary)
        
    return X,y,feature_names,resampling,subjects

def numpy_ts_to_dataframe(np_ts_matrix,subjects,features):
    
    f = ['id','time']
    f.extend(features)
    
    rows = []
    for subject in range(np_ts_matrix.shape[0]):
        for time in range(np_ts_matrix.shape[1]):
            row = [subjects[subject],time]
            values =  np_ts_matrix[subject,time,...]
            row.extend(values)
            rows.append(row)
            
    df = pd.DataFrame(rows, columns=f)
            
    """
    for time in range(np_ts_matrix.shape[0]):
        features_time = np_ts_matrix[time,:]
    """ 
    return df

def all_features_dataframe(json_file,resampling = 30, verbose = 0):
    
    data = import_json_timeseries(json_file)
    
    feature_names = [k.replace('original_','') for k in data[list(data.keys())[0]]['basal_features'].keys()]
    feature_names_B_A = [f + '_B' for f in feature_names]
    feature_names_A = [f + '_A' for f in feature_names]

    feature_names_B_A.extend(feature_names_A)
    
    
    X = []
    y = []
    subjects = []
    
    for k,v in data.items():
        
        id = k
        subject = v
        outcome = subject['outcome']
        
        basal =resample_and_rescale([x for x in subject['basal_features'].values()],resampling = resampling)
        adeno =resample_and_rescale([x for x in subject['adeno_features'].values()],resampling = resampling)

        X.append(np.hstack((basal,adeno)))
        y.append(outcome)
        subjects.append(id)
    
    
    id_to_target = {}
    for out, subject in zip(y,subjects):
        id_to_target[subject] = out
        
    y = pd.Series(id_to_target)
    df = numpy_ts_to_dataframe(np.array(X),subjects,feature_names_B_A)
    
    if verbose:
        summary = {'X' : X.shape, 'Y' : y.shape, 'feature_names': feature_names, 'temporal_resampling': resampling}
        print('PREPROCESSING SUMMARY')
        pp.pprint(summary)
        
    return df,y


def import_pickle_features(path):
    
    X = []
    y = []
    subjects = []
    for file in sorted(os.listdir(path)):
        data_mid = pickle.load(open(os.path.join(path,file), "rb"))
        
        subjects.append(file.split('_')[1].split('.')[0])
        #basal_mid = np.nan_to_num(data_mid[0])
        #adeno_mid = np.nan_to_num(data_mid[1])
        
        basal_mid = data_mid[0]
        adeno_mid = data_mid[1]
        data = (basal_mid-adeno_mid)
      
        X.append(np.ravel(data))
        y.append(data_mid[2])
        
    X = np.array(X)
    y = np.array(y)
    return X,y,subjects
            
    
def resort_dictionary(unsorted_dict):
    
    keys = np.sort(list(unsorted_dict.keys()))
    list_of_tuples = [(key, unsorted_dict[key]) for key in keys]
    sorted_dict = collections.OrderedDict(list_of_tuples)
        
    return sorted_dict

