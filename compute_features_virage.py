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

import main_utils_1
from main_utils_1 import mk_dirs
import main_feature_functions
import compute_ecg_eda_features

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import biosppy
import pyhrv.tools as tools
from pyhrv.hrv import hrv

from biosppy.signals.ecg import correct_rpeaks, extract_heartbeats
from biosppy.signals.ecg import *

import socket
from datetime import datetime
import warnings

# from realtime_datacollection.main_utils import mk_dirs
warnings.filterwarnings("ignore")
import time

ecg_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']
eda_cols = ['Timestamp', 'GSR Conductance CAL']                

####################################### VIRAGE FEATURES #######################################

main_folder = "X:/Four Modes/Virage/Filtered/ECG_EDA"
savePath = 'X:/Four Modes/Virage/Extracted'
mk_dirs(savePath)
save_folder = os.path.join(savePath, 'ECG_EDA_Features')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for folder in os.listdir(main_folder):
    if os.path.isdir(os.path.join(main_folder, folder)):
        subjet_save_folder = os.path.join(save_folder, folder)
        if not os.path.exists(subjet_save_folder):
            os.mkdir(subjet_save_folder)
        for file in os.listdir(os.path.join(main_folder, folder)):
            subj_id = int(folder)
            if os.path.isfile(os.path.join(main_folder, folder, file)):
                if "ecg_level" in file:
                    sess_id = file[-5]
                    print('Working on file {}/ecg_level_{}.csv'.format(subj_id, sess_id))
                    ecgDF = pd.read_csv(os.path.join(main_folder, folder, file), skipinitialspace=True) # skiprows=1,, names=ecg_cols 
                    ecg_len = ecgDF.shape[0]
                    ecg_step_size = 10000
                    ecg_seg_features = []
                    firststamp = ecgDF['Timestamp'].iloc[0]
                    ecg_time = firststamp
                    curr_ind = 0
                    while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
                            ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
                            ecg_features = compute_ecg_eda_features.extract_ecg_features_only(ecg_seg.copy()) # default sample rate: 512.
                            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            ecg_features['subj_id'] = subj_id
                            ecg_features['sess_id'] = sess_id
                            ecg_seg_features.append(ecg_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind
                        ecg_time = ecgDF['Timestamp'].iloc[next_ind]
                    if len(ecg_seg_features) > 0:
                        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
                        ecg_all_features.to_csv(os.path.join(subjet_save_folder, 'ecg_featurs_{}.csv'.format(sess_id)), index=False)
                    else:
                        print("Skipping as no signal present")
                if "eda_level" in file:
                    sess_id = file[-5]
                    print('Working on file {}/eda_level_{}.csv'.format(subj_id, sess_id))
                    edaDF = pd.read_csv(os.path.join(main_folder, folder, file), skipinitialspace=True) # skiprows=1, , names=eda_cols 
                    eda_len = edaDF.shape[0]
                    eda_step_size = 10000
                    eda_seg_features = []
                    firststamp = edaDF['Timestamp'].iloc[0]
                    eda_time = firststamp
                    curr_ind = 0
                    while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
                            eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
                            eda_features = compute_ecg_eda_features.extract_eda_features_only(eda_seg.copy()) # Default sample rate: 128.
                            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            eda_features['subj_id'] = subj_id
                            eda_features['sess_id'] = sess_id
                            eda_seg_features.append(eda_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind 
                        eda_time = edaDF['Timestamp'].iloc[next_ind]
                    if len(eda_seg_features) > 0:
                        eda_all_features = pd.concat(eda_seg_features, axis=0)
                        eda_all_features.to_csv(os.path.join(subjet_save_folder, 'eda_featurs_{}.csv'.format(sess_id)), index=False)
                    else:
                        print("Skipping as no signal present")

# ####################################### BASELINE 3 FEATURES #######################################

baseline_folder = "X:/Four Modes/Virage/Filtered/Base3_ECG_EDA"
baseline_save_folder = os.path.join('X:/Four Modes/Virage/Extracted', 'ECG_EDA_Base3_Features')
if not os.path.exists(baseline_save_folder):
    os.mkdir(baseline_save_folder)

for folder in os.listdir(baseline_folder):
    if os.path.isdir(os.path.join(baseline_folder, folder)):
        subjet_save_folder = os.path.join(baseline_save_folder, folder)
        if not os.path.exists(subjet_save_folder):
            os.mkdir(subjet_save_folder)
        for file in os.listdir(os.path.join(baseline_folder, folder)):
            subj_id = int(folder)
            if os.path.isfile(os.path.join(baseline_folder, folder, file)):
                if "ecg_baseline" in file:
                    print('Working on file {}/ecg_baseline.csv'.format(subj_id))
                    ecgDF = pd.read_csv(os.path.join(baseline_folder, folder, file), skipinitialspace=True) # skiprows=1, , names=ecg_cols
                    ecg_len = ecgDF.shape[0]
                    ecg_step_size = 10000
                    ecg_seg_features = []
                    firststamp = ecgDF['Timestamp'].iloc[0]
                    ecg_time = firststamp
                    curr_ind = 0
                    while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
                            ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
                            ecg_features = compute_ecg_eda_features.extract_ecg_features_only(ecg_seg.copy()) # default sample rate: 512.
                            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            ecg_features['subj_id'] = subj_id
                            ecg_seg_features.append(ecg_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind 
                        ecg_time = ecgDF['Timestamp'].iloc[next_ind]
                    if len(ecg_seg_features) > 0:
                        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
                        ecg_all_features.to_csv(os.path.join(subjet_save_folder, 'ecg_baseline_featurs.csv'), index=False)
                    else:
                        print("Skipping as no signal present")
                if "eda_baseline" in file:
                    print('Working on file {}/eda_baseline.csv'.format(subj_id))
                    edaDF = pd.read_csv(os.path.join(baseline_folder, folder, file), skipinitialspace=True) # skiprows=1, , names=eda_cols
                    eda_len = edaDF.shape[0]
                    eda_step_size = 10000
                    eda_seg_features = []
                    firststamp = edaDF['Timestamp'].iloc[0]
                    eda_time = firststamp
                    curr_ind = 0
                    while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
                            eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
                            eda_features = compute_ecg_eda_features.extract_eda_features_only(eda_seg.copy()) # default sample rate: 128.
                            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            eda_features['subj_id'] = subj_id
                            eda_seg_features.append(eda_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind 
                        eda_time = edaDF['Timestamp'].iloc[next_ind]
                    if len(eda_seg_features) > 0:
                        eda_all_features = pd.concat(eda_seg_features, axis=0)
                        eda_all_features.to_csv(os.path.join(subjet_save_folder, 'eda_baseline_featurs.csv'), index=False)
                    else:
                        print("Skipping as no signal present")

main_folder = "X:/Four Modes/Virage/Filtered/Base2_ECG_EDA"
savePath = 'X:/Four Modes/Virage/Extracted'
mk_dirs(savePath)
save_folder = os.path.join(savePath, 'ECG_EDA_Base2_Features')
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

for folder in os.listdir(main_folder):
    if os.path.isdir(os.path.join(main_folder, folder)):
        subjet_save_folder = os.path.join(save_folder, folder)
        if not os.path.exists(subjet_save_folder):
            os.mkdir(subjet_save_folder)
        for file in os.listdir(os.path.join(main_folder, folder)):
            subj_id = int(folder)
            if os.path.isfile(os.path.join(main_folder, folder, file)):
                if "ecg_level" in file:
                    sess_id = file[-5]
                    print('Working on file {}/ecg_level_{}.csv'.format(subj_id, sess_id))
                    ecgDF = pd.read_csv(os.path.join(main_folder, folder, file), skipinitialspace=True) # skiprows=1,, names=ecg_cols 
                    ecg_len = ecgDF.shape[0]
                    ecg_step_size = 10000
                    ecg_seg_features = []
                    firststamp = ecgDF['Timestamp'].iloc[0]
                    ecg_time = firststamp
                    curr_ind = 0
                    while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
                            ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
                            ecg_features = compute_ecg_eda_features.extract_ecg_features_only(ecg_seg.copy()) # default sample rate: 512.
                            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            ecg_features['subj_id'] = subj_id
                            ecg_features['sess_id'] = sess_id
                            ecg_seg_features.append(ecg_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind
                        ecg_time = ecgDF['Timestamp'].iloc[next_ind]
                    if len(ecg_seg_features) > 0:
                        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
                        ecg_all_features.to_csv(os.path.join(subjet_save_folder, 'ecg_featurs_{}.csv'.format(sess_id)), index=False)
                    else:
                        print("Skipping as no signal present")
                if "eda_level" in file:
                    sess_id = file[-5]
                    print('Working on file {}/eda_level_{}.csv'.format(subj_id, sess_id))
                    edaDF = pd.read_csv(os.path.join(main_folder, folder, file), skipinitialspace=True) # skiprows=1, , names=eda_cols 
                    eda_len = edaDF.shape[0]
                    eda_step_size = 10000
                    eda_seg_features = []
                    firststamp = edaDF['Timestamp'].iloc[0]
                    eda_time = firststamp
                    curr_ind = 0
                    while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-2]:
                        try:
                            next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
                            eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
                            eda_features = compute_ecg_eda_features.extract_eda_features_only(eda_seg.copy()) # Default sample rate: 128.
                            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                            eda_features['subj_id'] = subj_id
                            eda_features['sess_id'] = sess_id
                            eda_seg_features.append(eda_features)
                        except Exception as e:
                            print(e)
                        curr_ind = next_ind 
                        eda_time = edaDF['Timestamp'].iloc[next_ind]
                    if len(eda_seg_features) > 0:
                        eda_all_features = pd.concat(eda_seg_features, axis=0)
                        eda_all_features.to_csv(os.path.join(subjet_save_folder, 'eda_featurs_{}.csv'.format(sess_id)), index=False)
                    else:
                        print("Skipping as no signal present")