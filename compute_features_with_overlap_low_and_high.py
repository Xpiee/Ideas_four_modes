import pandas as pd
import numpy as np
import os
import neurokit2 as nk

## from main_utils import *  # Do not use
## from main_functions import * # Do not use

import compute_ecg_eda_features

import pyhrv.time_domain as td
import pyhrv.frequency_domain as fd

import biosppy
import pyhrv.tools as tools
from pyhrv.hrv import hrv

from biosppy.signals.ecg import correct_rpeaks, extract_heartbeats
from biosppy.signals.ecg import *

# from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
import time

def extract_features_with_overlap(ecg_file, eda_file, subj_id, sess_id, file_timestamp, save_path, file_name_prefix):
    print('Reading {}'.format(ecg_file))
    ecgDF =  pd.read_csv(ecg_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['data_received_time', 'dummy'])
    ecgDF.reset_index(inplace=True, drop=True)
    print('Reading {}'.format(eda_file))
    edaDF = pd.read_csv(eda_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'GSR Conductance CAL'])
    edaDF = edaDF.drop(columns=['data_received_time'])
    edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]  
    edaDF.reset_index(inplace=True, drop=True) 

    ecg_window_size = eda_window_size = 10 * 1000
    ecg_step_size = eda_step_size = 0.5 * 1000
    
    #################################################
    ### ECG
    #################################################
    ecg_seg_features = []
    firststamp = ecgDF['Timestamp'].iloc[0]
    ecg_time = firststamp
    curr_ind = 0
    i = 1
    print('Extcting ecg features')
    while ecg_time + ecg_window_size <= ecgDF['Timestamp'].iloc[-1]:
        end_ind = (ecgDF['Timestamp'] > ecg_time + ecg_window_size).argmax() - 1
        ecg_seg = ecgDF.iloc[curr_ind:end_ind, :].copy()
        try:
            ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy())
            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            ecg_features['subj_id'] = subj_id
            ecg_features['sess_id'] = sess_id
            ecg_seg_features.append(ecg_features)
        except ValueError as e:
            print(e)
        curr_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1 
        ecg_time = ecgDF['Timestamp'].iloc[curr_ind]
        i += 1
    if len(ecg_seg_features) != 0:
        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
        ecg_feature_file_name = os.path.join(save_path, '{}_ecg_featurs_winsize_{:.2f}_stepsize_{:.2f}_{}_{}_{}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000, subj_id, sess_id, file_timestamp))
        ecg_all_features.to_csv(ecg_feature_file_name, index=False)
        print(f"Saving {ecg_feature_file_name}")
                    
        
    ##########################################################
    ## EDA
    ##########################################################
    eda_seg_features = []
    firststamp = edaDF['Timestamp'].iloc[0]
    eda_time = firststamp
    curr_ind = 0
    print('Extcting eda features')
    while eda_time + eda_window_size <= edaDF['Timestamp'].iloc[-1]:
        end_ind = (edaDF['Timestamp'] > eda_time + eda_window_size).argmax() - 1
        eda_seg = edaDF.iloc[curr_ind:end_ind, :].copy()
        try:
            eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy())
            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            eda_features['subj_id'] = subj_id
            eda_features['sess_id'] = sess_id
            eda_seg_features.append(eda_features)
        except ValueError as e:
            print(e)
        curr_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
        eda_time = edaDF['Timestamp'].iloc[curr_ind]
    if len(eda_seg_features) != 0:
        eda_all_features = pd.concat(eda_seg_features, axis=0)
        eda_feature_file_name = os.path.join(save_path, '{}_eda_featurs_winsize_{:.2f}_stepsize_{:.2f}_{}_{}_{}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000, subj_id, sess_id, file_timestamp))
        eda_all_features.to_csv(eda_feature_file_name, index=False)
        print(f'Saving {eda_feature_file_name}')
    print('Extraction done!')

def extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, file_name_prefix):
    ecg_file = os.path.join(highlowfile_path, f"{file_name_prefix}_ecg.csv")
    eda_file = os.path.join(highlowfile_path, f"{file_name_prefix}_eda.csv")
    print('Reading {}'.format(ecg_file))
    ecgDF =  pd.read_csv(ecg_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['data_received_time', 'dummy'])
    ecgDF.reset_index(inplace=True, drop=True)
    print('Reading {}'.format(eda_file))
    edaDF = pd.read_csv(eda_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'GSR Conductance CAL'])
    edaDF = edaDF.drop(columns=['data_received_time'])
    edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]  
    edaDF.reset_index(inplace=True, drop=True) 

    print("Lenght of {} ecg: {}s".format(file_name_prefix,  ecgDF.shape[0] / 512))
    print("Lenght of {} eda: {}s".format(file_name_prefix, edaDF.shape[0] / 128))

    ecg_window_size = eda_window_size = 10 * 1000
    ecg_step_size = eda_step_size = 0.5 * 1000
    
    #################################################
    ### ECG
    #################################################
    ecg_seg_features = []
    firststamp = ecgDF['Timestamp'].iloc[0]
    ecg_time = firststamp
    curr_ind = 0
    i = 1
    print('Extcting ecg features')
    while ecg_time + ecg_window_size <= ecgDF['Timestamp'].iloc[-1]:
        end_ind = (ecgDF['Timestamp'] > ecg_time + ecg_window_size).argmax() - 1
        ecg_seg = ecgDF.iloc[curr_ind:end_ind, :].copy()
        try:
            ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy()) # default sample rate: 512.
            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            ecg_features['subj_id'] = subj_id
            ecg_features['sess_id'] = experiment_id
            ecg_seg_features.append(ecg_features)
        except ValueError as e:
            print(e)
        curr_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1 
        ecg_time = ecgDF['Timestamp'].iloc[curr_ind]
        i += 1
    if len(ecg_seg_features) != 0:
        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
        ecg_feature_file_name = os.path.join(highlowfile_path, '{}_ecg_featurs_winsize_{:.2f}_stepsize_{:.2f}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000))
        ecg_all_features.to_csv(ecg_feature_file_name, index=False)
        print(f"Saving {ecg_feature_file_name}")
                    
        
    ##########################################################
    ## EDA
    ##########################################################
    eda_seg_features = []
    firststamp = edaDF['Timestamp'].iloc[0]
    eda_time = firststamp
    curr_ind = 0
    print('Extcting eda features')
    while eda_time + eda_window_size <= edaDF['Timestamp'].iloc[-1]:
        end_ind = (edaDF['Timestamp'] > eda_time + eda_window_size).argmax() - 1
        eda_seg = edaDF.iloc[curr_ind:end_ind, :].copy()
        try:
            eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy()) # default sample rate: 128.
            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            eda_features['subj_id'] = subj_id
            eda_features['sess_id'] = experiment_id
            eda_seg_features.append(eda_features)
        except ValueError as e:
            print(e)
        curr_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
        eda_time = edaDF['Timestamp'].iloc[curr_ind]
    if len(eda_seg_features) != 0:
        eda_all_features = pd.concat(eda_seg_features, axis=0)
        eda_feature_file_name = os.path.join(highlowfile_path, '{}_eda_featurs_winsize_{:.2f}_stepsize_{:.2f}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000))
        eda_all_features.to_csv(eda_feature_file_name, index=False)
        print(f'Saving {eda_feature_file_name}')
    print('Extraction done!')




def extract_features_with_overlap_from_sess1(sess_path, subj_id, experiment_id, file_name_prefix, window_size=10, step_size=1):
    ecg_file = os.path.join(sess_path, f"{file_name_prefix}_ecg.csv")
    eda_file = os.path.join(sess_path, f"{file_name_prefix}_eda.csv")
    print('Reading {}'.format(ecg_file))
    ecgDF =  pd.read_csv(ecg_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['data_received_time', 'dummy'])
    ecgDF = ecgDF.iloc[60*512:120*512, :] # assume that we have three miniutes low or high
    ecgDF.reset_index(inplace=True, drop=True)
    print('Reading {}'.format(eda_file))
    edaDF = pd.read_csv(eda_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'GSR Conductance CAL'])
    edaDF = edaDF.drop(columns=['data_received_time'])
    edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]  
    ecgDF = ecgDF.iloc[60*128:120*128, :]     # assume that we have three miniutes low or high
    edaDF.reset_index(inplace=True, drop=True) 

    

    ecg_window_size = eda_window_size = int(10 * 1000)
    ecg_step_size = eda_step_size = int(0.5 * 1000)
    
    #################################################
    ### ECG
    #################################################
    ecg_seg_features = []
    firststamp = ecgDF['Timestamp'].iloc[0]
    ecg_time = firststamp
    curr_ind = 0
    i = 1
    print('Extcting ecg features')
    while ecg_time + ecg_window_size <= ecgDF['Timestamp'].iloc[-1]:
        end_ind = (ecgDF['Timestamp'] > ecg_time + ecg_window_size).argmax() - 1
        ecg_seg = ecgDF.iloc[curr_ind:end_ind, :].copy()
        try:
            ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy()) # default sample rate: 512.
            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            ecg_features['subj_id'] = subj_id
            ecg_features['sess_id'] = experiment_id
            ecg_seg_features.append(ecg_features)
        except ValueError as e:
            print(e)
        curr_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1 
        ecg_time = ecgDF['Timestamp'].iloc[curr_ind]
        i += 1
    if len(ecg_seg_features) != 0:
        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
        ecg_feature_file_name = os.path.join(sess_path, '{}_ecg_featurs_sess1_winsize_{:.2f}_stepsize_{:.2f}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000))
        ecg_all_features.to_csv(ecg_feature_file_name, index=False)
        print(f"Saving {ecg_feature_file_name}")
                    
        
    ##########################################################
    ## EDA
    ##########################################################
    eda_seg_features = []
    firststamp = edaDF['Timestamp'].iloc[0]
    eda_time = firststamp
    curr_ind = 0
    print('Extcting eda features')
    while eda_time + eda_window_size <= edaDF['Timestamp'].iloc[-1]:
        end_ind = (edaDF['Timestamp'] > eda_time + eda_window_size).argmax() - 1
        eda_seg = edaDF.iloc[curr_ind:end_ind, :].copy()
        try:
            eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy()) # default sample rate: 128.
            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, 0))
            eda_features['subj_id'] = subj_id
            eda_features['sess_id'] = experiment_id
            eda_seg_features.append(eda_features)
        except ValueError as e:
            print(e)
        curr_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
        eda_time = edaDF['Timestamp'].iloc[curr_ind]
    if len(eda_seg_features) != 0:
        eda_all_features = pd.concat(eda_seg_features, axis=0)
        eda_feature_file_name = os.path.join(sess_path, '{}_eda_featurs_sess1_winsize_{:.2f}_stepsize_{:.2f}.csv'.format(file_name_prefix, ecg_window_size/1000, ecg_step_size/1000))
        eda_all_features.to_csv(eda_feature_file_name, index=False)
        print(f'Saving {eda_feature_file_name}')
    print('Extraction done!')