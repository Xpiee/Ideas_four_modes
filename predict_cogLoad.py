import collections
import pandas as pd
import numpy as np
import os
import pickle
import neurokit2 as nk
import scipy
from scipy.stats import skew, kurtosis, iqr

# from main_utils import *
# from main_functions import *

import compute_ecg_eda_features

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import biosppy
import pyhrv.tools as tools
from pyhrv.hrv import hrv

from biosppy.signals.ecg import correct_rpeaks, extract_heartbeats
from biosppy.signals.ecg import *

import socket
import json 
import keyboard
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from config import SELECTCOLS


def get_predictions(ecgDF, edaDF, subBaseline, classifiers, subID=0, sessID=0, isBaseline=False):

    '''
    inputs: dataframe of ecg and eda signals
    output: prediction scores from the classifiers
    '''

    # assuming shimmers will output dataframe for ecg and eda
    # get features for ecg and eda -> Dataframe

    ecgFeat = compute_ecg_eda_features.extract_ecg_features(ecgDF) # default sample rate: 512.
    edaFeat = compute_ecg_eda_features.extract_eda_features(edaDF) # default sample rate: 128.

    featDF = pd.concat([ecgFeat, edaFeat], axis = 1)

 
    # clf1, clf2 = classifiers
    clf1 = classifiers

    selected_cols = SELECTCOLS[:-3]
   
    featDF = featDF[selected_cols].copy()

    if isBaseline:
        featDF.to_csv('py_baseline_{}_{}.csv'.format(subID, sessID), index=False)
        return ([0], [0])
    else:
        if not os.path.isfile(f'ectracted_features_unnormalized_{subID}_{sessID}.csv'):
            featDF.to_csv(f'ectracted_features_unnormalized_{subID}_{sessID}.csv')
        else:
            featDF.to_csv(f'ectracted_features_unnormalized_{subID}_{sessID}.csv', header=False, mode='a')
            
        featDF = featDF.sub(subBaseline[selected_cols].iloc[0])
        featDF = featDF.div(subBaseline[selected_cols].iloc[0])
        if not os.path.isfile(f'ectracted_features_{subID}_{sessID}.csv'):
            featDF.to_csv(f'ectracted_features_{subID}_{sessID}.csv')
        else:
            featDF.to_csv(f'ectracted_features_{subID}_{sessID}.csv', header=False, mode='a')

        # featDF.to_csv('test_data.csv', index=False)
        proba = clf1.predict_proba(featDF)
        return proba 

def load_classifiers(path1, path2=None):
    clf1 = pickle.load(open(path1, 'rb'))
    return clf1 # 


def predict_fucntion(ecgDF, edaDF, subBaseline, subID, sessID, isBaseline=False, classifiers=None):

    '''
    ecg_: array
    eda_: array

    '''

    ecgDF = ecgDF.drop(columns=['dummy'])
    proba = get_predictions(ecgDF, edaDF, subBaseline, classifiers, subID, sessID, isBaseline) # removed second classifier
    return proba 

