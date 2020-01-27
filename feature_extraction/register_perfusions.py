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
from skimage import transform


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


print(outcomes)



def register_perfusion(file_img_stress,file_msk_stress,file_img_rest,file_msk_rest):
    
    
    
    #LOAD NIFTIS
    img_data_stress = nib.load(file_img_stress)
    msk_data_stress = nib.load(file_msk_stress)
    
    img_data_rest = nib.load(file_img_rest)
    msk_data_rest = nib.load(file_msk_rest)
    
    #CONVERT TO LISTS
    img_lst_stress = [img_data_stress.get_fdata()[t,...] for t in range(img_data_stress.get_fdata().shape[0])]
    msk_lst_stress = [msk_data_stress.get_fdata()[t,...] for t in range(msk_data_stress.get_fdata().shape[0])]
    
    img_lst_rest = [img_data_rest.get_fdata()[t,...] for t in range(img_data_rest.get_fdata().shape[0])]
    msk_lst_rest = [msk_data_rest.get_fdata()[t,...] for t in range(msk_data_rest.get_fdata().shape[0])]
    
    #ID'S
    id = file_img_rest.split('_')[1]
    outcome = outcomes[id]
    
    
    #registration
    n_points = 20
    
    rest = []
    
    for img,msk in zip(img_lst_rest,msk_lst_rest):
        if len(np.unique(msk)) < 2:
            continue
        
        msk[msk == 2] = 0
        img = img*msk
        img_out= obtain_polars(img,msk,n_points=20)
        rest.append(img_out)
        
        
    stress = []
    
    for img,msk in zip(img_lst_stress,msk_lst_stress):
        if len(np.unique(msk)) < 2:
            continue
        
        msk[msk == 2] = 0
        img = img*msk
        img_out= obtain_polars(img,msk,n_points=20)
        stress.append(img_out)
    
    
    stress_arr = np.array(stress)
    rest_arr = np.array(rest) 
    
    stress_arr = zoom(stress_arr,(rest_arr.shape[0]/stress_arr.shape[0],1.0,1.0))
    
    
    
    for idx in range(stress_arr.shape[0]):
        stress_img = stress_arr[idx,...]
        rest_img = rest_arr[idx,...]
        
        rest_img = transform.match_histograms(rest_img,stress_img)
        plt.gcf().canvas.flush_events()

        plt.imshow(np.hstack((stress_img,rest_img)))
        plt.show(block = False)
        plt.title(id+' '+str(outcome))
        plt.pause(interval = 0.1)
    
   
    
    return None

                          
folder = '../dataset/mid/'
n_files = len(sorted(os.listdir(folder)))
files = sorted(os.listdir(folder))        

numbers = []
for index in range(0,n_files,4):
    
    adeno = files[index]
    adeno_gt = files[index + 1]

    basal = files[index + 2]
    basal_gt = files[index + 3]
    
    
    print('ADENO')
    print(adeno)
    print(adeno_gt)
    
    print('BASAL')
    print(basal)
    print(basal_gt)
    
    
    register_perfusion(os.path.join(folder,adeno),
                       os.path.join(folder,adeno_gt),
                       os.path.join(folder,basal),
                       os.path.join(folder,basal_gt))
  
    
    