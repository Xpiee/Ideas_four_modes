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

def process_matbII_data(main_folder, save_folder, ecg_cols, eda_cols, isMatlab=False):
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
                    if "ecg_exp" in file:
                        sess_id = file[-5]
                        print('Working on file {}/ecg_level_{}.csv'.format(subj_id, sess_id))
                        ecgDF = pd.read_csv(os.path.join(main_folder, folder, file), skiprows=1, skipinitialspace=True, names=ecg_cols)
                        # ecgDF.drop(columns='dummy', inplace=True)
                        ecg_len = ecgDF.shape[0]
                        ecg_step_size = 10000
                        ecg_seg_features = []
                        firststamp = ecgDF['Timestamp'].iloc[0]
                        ecg_time = firststamp
                        curr_ind = 0
                        while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-1]:
                            next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
                            ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
                            try:
                                ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy()) # Default sample rate: 512.
                                ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                                ecg_features['subj_id'] = subj_id
                                ecg_features['sess_id'] = sess_id
                                ecg_seg_features.append(ecg_features)
                            except ValueError as e:
                                print(e)
                            curr_ind = next_ind
                            ecg_time = ecgDF['Timestamp'].iloc[next_ind]
                        if len(ecg_seg_features) != 0:
                            ecg_all_features = pd.concat(ecg_seg_features, axis=0)
                            ecg_all_features.to_csv(os.path.join(subjet_save_folder, 'ecg_featurs_{}.csv'.format(sess_id)), index=False)
                    if "eda_exp" in file:
                        sess_id = file[-5]
                        print('Working on file {}/eda_level_{}.csv'.format(subj_id, sess_id))
                        edaDF = pd.read_csv(os.path.join(main_folder, folder, file), skiprows=1, skipinitialspace=True, names=eda_cols)
                        if isMatlab:
                            edaDF['GSR Conductance CAL'] = 1000. / edaDF['GSR Conductance CAL'].values
                        eda_len = edaDF.shape[0]
                        eda_step_size = 10000
                        eda_seg_features = []
                        firststamp = edaDF['Timestamp'].iloc[0]
                        eda_time = firststamp
                        curr_ind = 0
                        while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-1]:
                            next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
                            eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
                            try:
                                eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy()) # default sample rate: 128.
                                eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
                                eda_features['subj_id'] = subj_id
                                eda_features['sess_id'] = sess_id
                                eda_seg_features.append(eda_features)
                            except ValueError as e:
                                print(e)
                            curr_ind = next_ind 
                            eda_time = edaDF['Timestamp'].iloc[next_ind]
                        if len(eda_seg_features) != 0:
                            eda_all_features = pd.concat(eda_seg_features, axis=0)
                            eda_all_features.to_csv(os.path.join(subjet_save_folder, 'eda_featurs_{}.csv'.format(sess_id)), index=False)


def ecg_extract_window_features(ecgDF, subj_id, sess_id=1, sRate=512., savePath=None):
    ecg_len = ecgDF.shape[0]
    ecg_step_size = 10000
    ecg_seg_features = []
    firststamp = ecgDF['Timestamp'].iloc[0]
    ecg_time = firststamp
    curr_ind = 0
    while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-1]:
        next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
        ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
        try:
            ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy()) # Default sample rate: 512.
            ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
            ecg_features['subj_id'] = subj_id
            ecg_features['sess_id'] = sess_id
            ecg_seg_features.append(ecg_features)
        except ValueError as e:
            print(e)
        curr_ind = next_ind
        ecg_time = ecgDF['Timestamp'].iloc[next_ind]
    if len(ecg_seg_features) != 0:
        ecg_all_features = pd.concat(ecg_seg_features, axis=0)
        ecg_all_features.to_csv(os.path.join(savePath, 'ecg_featurs_{}.csv'.format(sess_id)), index=False)


def eda_extract_window_features(edaDF, subj_id, sess_id=1, isDPZ=False, sRate=128., savePath=None):
    if isDPZ:
        edaDF['GSR Conductance CAL'] = 1000. / edaDF['GSR Conductance CAL'].values
    eda_len = edaDF.shape[0]
    eda_step_size = 10000
    eda_seg_features = []
    firststamp = edaDF['Timestamp'].iloc[0]
    eda_time = firststamp
    curr_ind = 0
    while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-1]:
        next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
        eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
        try:
            eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy()) # default sample rate: 128.
            eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
            eda_features['subj_id'] = subj_id
            eda_features['sess_id'] = sess_id
            eda_seg_features.append(eda_features)
        except ValueError as e:
            print(e)
        curr_ind = next_ind 
        eda_time = edaDF['Timestamp'].iloc[next_ind]
    if len(eda_seg_features) != 0:
        eda_all_features = pd.concat(eda_seg_features, axis=0)
        eda_all_features.to_csv(os.path.join(savePath, 'eda_featurs_{}.csv'.format(sess_id)), index=False)


if __name__ == '__main__':

    # isDPZ = False # True if extracting features for DPZ
    isDPZ = True # True if extracting features for DPZ

    if isDPZ == False:
        ecg_cols = ['Timestamp', 'ECG LL-RA CAL',
                    'ECG LA-RA CAL', 'ECG Vx-RL CAL']
        eda_cols = ['Timestamp', 'GSR Conductance CAL']                
            
        main_folder = "X:/RealTimeSegment/MatbII/Raw/ECG_EDA"
        savePath = 'X:/RealTimeSegment/MatbII/Extracted'
        mk_dirs(savePath)
        save_folder = os.path.join(savePath, 'ECG_EDA_Features')

        process_matbII_data(main_folder, save_folder, ecg_cols, eda_cols, isDPZ)

    else:
        subj_id = ['Dirk', 'Prithila', 'Zunayed']
        sess_id = 1
        
        ecg_cols = ['Timestamp', 'ECG LL-RA CAL',
                    'ECG LA-RA CAL', 'dummy' , 'ECG Vx-RL CAL']
        eda_cols = ['Timestamp', 'GSR Conductance CAL']

        dataPathDPZ = 'X:/RealTimeSegment/New Subjects Data/drik_prithial_zunayed'
        savePath = 'X:/RealTimeSegment/New Subjects Data/DPZ'
        mk_dirs(savePath)
        for sub in subj_id:
            subDataPathDPZ = os.path.join(dataPathDPZ, sub)
            ecgFile = os.path.join(subDataPathDPZ, f'ecg_{sess_id}.csv')
            edaFile = os.path.join(subDataPathDPZ, f'eda_{sess_id}.csv')
            ecgDF = pd.read_csv(ecgFile)
            ecgDF.columns = ecg_cols
            ecgDF.drop(columns='dummy', inplace=True)
            edaDF = pd.read_csv(edaFile)
            edaDF.columns = eda_cols

            savePathSub = os.path.join(savePath, sub)

            mk_dirs(savePathSub)
            ecg_extract_window_features(ecgDF, sub, sess_id, 512., savePathSub)
            eda_extract_window_features(edaDF, sub, sess_id, True, 128., savePathSub)

##########################################################################################

# if __name__ == '__main__':
#     subj_id = ['Dirk', 'Prithila', 'Zunayed']
#     sess_id = 1

#     ecg_cols = ['Timestamp', 'ECG LL-RA CAL',
#                 'ECG LA-RA CAL', 'ECG Vx-RL CAL']
#     eda_cols = ['Timestamp', 'GSR Conductance CAL']
#     dataPathDPZ = 'X:/RealTimeSegment/New Subjects Data/drik_prithial_zunayed'
#     savePath = 'X:/RealTimeSegment/New Subjects Data/DPZ'
#     mk_dirs(savePath)
#     for sub in subj_id:
#         subDataPathDPZ = os.path.join(dataPathDPZ, sub)
#         ecgFile = os.path.join(subDataPathDPZ, f'ecg_{sess_id}.csv')
#         edaFile = os.path.join(subDataPathDPZ, f'eda_{sess_id}.csv')

#     baseline_path_DPZ = r"X:/RealTimeSegment/New Subjects Data/drik_prithial_zunayed_baseline/"
#     savePath = r"X:/RealTimeSegment/New Subjects Data/drik_prithial_zunayed/BaseFeatures"
#     mk_dirs(savePath)
#     for subs in subj_id:
#         subBasePath = os.path.join(baseline_path_DPZ, subs)
#         ecg_baseline_file = os.path.join(subBasePath, f"baseline_ecg_{sess_id}.csv")
#         eda_baseline_file = os.path.join(subBasePath, f"baseline_eda_{sess_id}.csv")

#         baseline_extract_features_DPZ(ecg_baseline_file, eda_baseline_file, subs, sess_id, savePath)


########################################################################
# ######################################################################                        

# baseline_folder = "C:/Users/hbb1/Downloads/matbii/ECG_EDA_baseline"
# baseline_save_folder = os.path.join('C:/Users/hbb1/Downloads/matbii', 'ECG_EDA_baseline_Features')
# if not os.path.exists(baseline_save_folder):
#     os.mkdir(baseline_save_folder)

# for folder in os.listdir(baseline_folder):
#     if os.path.isdir(os.path.join(baseline_folder, folder)):
#         subjet_save_folder = os.path.join(baseline_save_folder, folder)
#         if not os.path.exists(subjet_save_folder):
#             os.mkdir(subjet_save_folder)
#         for file in os.listdir(os.path.join(baseline_folder, folder)):
#             subj_id = int(folder)
#             if os.path.isfile(os.path.join(baseline_folder, folder, file)):
#                 if "ecg_baseline" in file:
#                     print('Working on file {}/ecg_baseline.csv'.format(subj_id))
#                     ecgDF = pd.read_csv(os.path.join(baseline_folder, folder, file), skiprows=1, skipinitialspace=True, names=ecg_cols)
#                     ecg_len = ecgDF.shape[0]
#                     ecg_step_size = 10000
#                     ecg_seg_features = []
#                     firststamp = ecgDF['Timestamp'].iloc[0]
#                     ecg_time = firststamp
#                     curr_ind = 0
#                     while ecg_time + ecg_step_size <= ecgDF['Timestamp'].iloc[-1]:
#                         next_ind = (ecgDF['Timestamp'] > ecg_time + ecg_step_size).argmax() - 1
#                         ecg_seg = ecgDF.iloc[curr_ind:next_ind, :].copy()
#                         try:
#                             ecg_features = compute_ecg_eda_features.extract_ecg_features(ecg_seg.copy())
#                             ecg_features['Timestamp'] = int(np.round((ecg_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
#                             ecg_features['subj_id'] = subj_id
#                             ecg_seg_features.append(ecg_features)
#                         except ValueError as e:
#                             print(e)
#                         curr_ind = next_ind 
#                         ecg_time = ecgDF['Timestamp'].iloc[next_ind]
#                     if len(ecg_seg_features) != 0:
#                         ecg_all_features = pd.concat(ecg_seg_features, axis=0)
#                         ecg_all_features.to_csv(os.path.join(subjet_save_folder, 'ecg_baseline_featurs.csv'), index=False)
#                 if "eda_baseline" in file:
#                     print('Working on file {}/eda_baseline.csv'.format(subj_id))
#                     edaDF = pd.read_csv(os.path.join(baseline_folder, folder, file), skiprows=1, skipinitialspace=True, names=eda_cols)
#                     eda_len = edaDF.shape[0]
#                     eda_step_size = 10000
#                     eda_seg_features = []
#                     firststamp = edaDF['Timestamp'].iloc[0]
#                     eda_time = firststamp
#                     curr_ind = 0
#                     while eda_time + eda_step_size <= edaDF['Timestamp'].iloc[-1]:
#                         next_ind = (edaDF['Timestamp'] > eda_time + eda_step_size).argmax() - 1
#                         eda_seg = edaDF.iloc[curr_ind:next_ind, :].copy()
#                         try:
#                             eda_features = compute_ecg_eda_features.extract_eda_features(eda_seg.copy())
#                             eda_features['Timestamp'] = int(np.round((eda_seg['Timestamp'].iloc[0] - firststamp) / 1000, -1))
#                             eda_features['subj_id'] = subj_id
#                             eda_seg_features.append(eda_features)
#                         except ValueError as e:
#                             print(e)
#                         curr_ind = next_ind 
#                         eda_time = edaDF['Timestamp'].iloc[next_ind]
#                     if len(eda_seg_features) != 0:
#                         eda_all_features = pd.concat(eda_seg_features, axis=0)
#                         eda_all_features.to_csv(os.path.join(subjet_save_folder, 'eda_baseline_featurs.csv'), index=False)
