import os,cv2,glob, datetime, pickle
import nibabel as nib

import numpy as np
import pandas as pd


import scipy,skimage
from skimage import feature
from skimage import exposure,measure
from skimage import data, color
from skimage.segmentation import active_contour
from skimage.transform import rescale, resize, downscale_local_mean,rotate

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import zoom

import matplotlib.pyplot as plt


import SimpleITK as sitk

from radiomics import featureextractor as fe, gldm
import pprint as pp

import scipy.stats as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.svm import SVC

from global_variables import *
from visualization.utils import *
from visualization.image_utils import *
from feature_processing.utils import *

import collections
import pprint as pp


folder = '../dataset/mid/'
n_files = len(sorted(os.listdir(folder)))
files = sorted(os.listdir(folder))


def unzigzag_features(features_2D):
    
    #from 2d array (each row a feature, each column a temporal instant, n_columns) to 1D vector

    return features_2D.reshape(-1, features_2D.shape[0]*features_2D.shape[1])

def zigzag_features(features_1D, shape):
    
    #from 2d array (each row a feature, each column a temporal instant, n_columns) to 1D vector

    return features.reshape(shape)
    
    

def extract_features(file_img,file_msk):
    
    #EXTRACTOR
    extractor = fe.RadiomicsFeaturesExtractor()
    extractor.settings['normalize'] = True
    extractor.settings['removeOutliers'] = True
    extractor.settings['force2D'] = True
    extractor.settings['binWidth'] = 25
    
    #ENABLED FEATURES
    
    #extractor.enableAllFeatures()
    #extractor.enableFeatureClassByName('shape')
    
    
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')
    #extractor.enableFeatureClassByName('glrlm')
    #extractor.enableFeatureClassByName('gldm')
    
    
    #LOAD NIFTIS
    img_data = nib.load(file_img)
    msk_data = nib.load(file_msk)
    
    print(img_data.get_fdata().shape,msk_data.get_fdata().shape)
    print(file_img,file_msk)
    
    
    #CONVERT TO LISTS
    img_lst = []
    msk_lst = []
    comb_lst = []
    
    for x in range(img_data.get_fdata().shape[0]):
        img_lst.append(img_data.get_fdata()[x,...])
        msk_lst.append(msk_data.get_fdata()[x,...])
    
    all_final_features = [] 
    all_features_dict = collections.OrderedDict()
    
    t = 0
    for img,msk in zip(img_lst,msk_lst):
        if len(np.unique(msk)) < 2:
            continue
        
        msk[msk == 2] = 0
        comb_lst.append(img*msk)
        slice_msk= np.expand_dims(msk.astype(int),2)  
        slice_img= np.expand_dims(img.astype(int),2)  
        
        
        
        sitk_img= sitk.GetImageFromArray(np.array(slice_img, dtype=np.int16))   
        sitk_msk= sitk.GetImageFromArray(np.array(slice_msk, dtype=np.int16)) 
      

        features = extractor.execute(sitk_img, sitk_msk)        
        features = resort_dictionary(features)
        
        
        final_features = []
        
        for k,v in features.items():

            if not '_original' in k and 'original_' in k:
                
                try:
                    all_features_dict[k].append(v)
                except:
                    all_features_dict[k] = []
                    all_features_dict[k].append(v)
                    
                final_features.append(v)
                print(k,v)
   
        all_final_features.append(final_features)
        
        t = t + 1
    
    array_features = np.array(all_final_features)
    array_features = resize(array_features,(30,array_features.shape[1]),order = 0,mode = 'edge',anti_aliasing = False)
    

    
    scaler = MinMaxScaler()
    scaler.fit(array_features)
    array_features = scaler.transform(array_features) 
    plt.imshow(array_features.T)
    #pickle.dump(array_features.T, open("features_basal_0.f", "wb")) 

    plt.show()
    #plt.imshow(array_features)
    
    
    #plt.imshow(array_features.T, cmap = plt.get_cmap('jet'))
    #show_imagegrid(comb_lst,3,int(np.ceil(len(comb_lst)/3)))

    #plot_3d_bar(array_features.T)
    #plot_sequences(array_features.T)
    #pickle.dump(array_features.T, open("features_basal_0.f", "wb")) 
    #plt.show()
    
    return all_features_dict
                          
          
all_df = []
numbers = []
all_patients_dict = {}

for index in range(0,n_files-1,4):
    
    
    
    adeno = files[index]
    adeno_gt = files[index + 1]

    
    basal = files[index + 2]
    basal_gt = files[index + 3]
  
    
    adeno_nii = nib.load(os.path.join(folder,adeno)).get_fdata()
    basal_nii = nib.load(os.path.join(folder,basal)).get_fdata()
    
    if basal_nii.shape[0] > adeno_nii.shape[0]:
            print('inspect................')
            adeno = files[index + 2]
            adeno_gt = files[index + 3]
            
            basal = files[index ]
            basal_gt = files[index + 1]
         
    adeno_fea = extract_features(os.path.join(folder,adeno),os.path.join(folder,adeno_gt))    
     
    basal_fea = extract_features(os.path.join(folder,basal),os.path.join(folder,basal_gt))
    

    
    number = adeno.split('_')[1]
    
    all_patients_dict[number] = {}
    all_patients_dict[number]['adeno_features'] = adeno_fea
    all_patients_dict[number]['basal_features'] = basal_fea
    all_patients_dict[number]['outcome'] = outcomes[number]
    
    #data = [basal_fea,adeno_fea,outcomes[number]]
    #pickle.dump(data, open(os.path.join('./features_mid',"features_"+number+".f"), "wb")) 

    #d = pickle.load(open("UKBB_ACDC_dataset.ds", "rb"))
    #plt.imshow((basal_fea-adeno_fea).T)  
    #plt.show()

#pickle.dump(all_patients_dict, open(os.path.join('extracted_features',"raw_timeseries_bw25.ts"), "wb")) 
#pp.pprint(all_patients_dict,indent = 4)


import json
from utils.to_json import *
with open('raw_timeseries_bw25_norm.json', 'w') as fp:
    
    ret = to_json(all_patients_dict)
    fp.write(ret)
    #json.dump(all_patients_dict, fp, sort_keys=True, indent=2)

           
        
    
