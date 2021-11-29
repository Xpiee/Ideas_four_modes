import collections
import pandas as pd
import numpy as np
import os
import pickle
import neurokit2 as nk
import scipy
from scipy.stats import skew, kurtosis, iqr

## from main_utils import *  # Do not use
## from main_functions import * # Do not use

# import main_utils_1
# import main_feature_functions
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

from main_utils_1 import mk_dirs
warnings.filterwarnings("ignore")

from Training_Code.config import SELECTCOLS

from stdbaseline import checkZeroRound, standardize_baseline_features 


def baseline_extract_features(ecg_baseline_file, eda_baseline_file, subID, sessID, save_path, file_datetime):
    print("Extracting Features.....")
    ecgDF =  pd.read_csv(ecg_baseline_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['data_received_time', 'dummy'])
    edaDF = pd.read_csv(eda_baseline_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'GSR Conductance CAL'])
    edaDF = edaDF.drop(columns=['data_received_time'])
    edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]
    
    ecgFeat = compute_ecg_eda_features.extract_ecg_features(ecgDF, ecg_sample_rt=512., dropCent=0.5)
    edaFeat = compute_ecg_eda_features.extract_eda_features(edaDF, eda_sample_rt=128., dropCent=0.5)

    ecgFeat.to_csv('ecg_baseline_{}_{}.csv'.format(subID, sessID), index=False)
    edaFeat.to_csv('eda_baseline_{}_{}.csv'.format(subID, sessID), index=False)

    featDF = pd.concat([ecgFeat, edaFeat], axis = 1)

    ############# NEEDS TO BE UPDATED BASED ON NEW COULMNS #############

    selected_cols = SELECTCOLS[:-3]
    featDF = featDF[selected_cols].copy()

    # selected_cols = ['ecg_HRV_IQRNN', 'ecg_HRV_MadNN','ecg_HRV_MeanNN',
    #         'ecg_HRV_MedianNN','ecg_HRV_RMSSD',
    #         'ecg_HRV_SD1', 'ecg_HRV_SD1SD2', 'ecg_HRV_SDNN', 'ecg_HRV_SDSD', 'ecg_HRV_pNN20',
    #         'ecg_HRV_pNN50', 'ecg_area_ts', 'ecg_entropy_features', 'ecg_iqr_features',
    #         'ecg_kurtosis_features', 'ecg_mad_ts', 'ecg_mean_features', 'ecg_median_features',
    #         'ecg_skew_features', 'ecg_sq_area_ts', 'ecg_std_features', 'eda_area_ts',
    #         'eda_entropy_features', 'eda_iqr_features', 'eda_kurtosis_features',
    #         'eda_mad_ts', 'eda_mean_features', 'eda_median_features',
    #         'eda_skew_features', 'eda_sq_area_ts', 'eda_std_features', 'ph_area_ts',
    #         'ph_entropy_features', 'ph_iqr_features', 'ph_kurtosis_features',
    #         'ph_mad_ts', 'ph_mean_features', 'ph_median_features','ph_skew_features',
    #         'ph_sq_area_ts', 'ph_std_features', 'ton_area_ts', 'ton_entropy_features',
    #         'ton_iqr_features', 'ton_kurtosis_features', 'ton_mad_ts', 'ton_mean_features',
    #         'ton_median_features', 'ton_skew_features', 'ton_sq_area_ts', 'ton_std_features',
    #         'ecg_hr_max', 'ecg_hr_mean', 'ecg_hr_min', 'ecg_hr_std', 'ecg_nni_counter', 'ecg_nni_diff_max',
    #         'ecg_nni_diff_mean', 'ecg_nni_max', 'ecg_nni_mean', 'ecg_nni_min', 'ecg_hf_peak',
    #         'ecg_ulf_abs', 'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs',  
    #         'ecg_lf_norm', 'ecg_hf_norm', 'ecg_lf_hf', 'ecg_tot_pwr']

    ############# ^^^^ NEEDS TO BE UPDATED BASED ON NEW COULMNS ^^^^ #############            


    featDF.to_csv(os.path.join(save_path, 'py_baseline_{}_{}_{}.csv'.format(subID, sessID, file_datetime)), index=False)
    print("Feature extraction done!")

def baseline_extract_features_phase2(subj_id, experiment_id, baseline_sess_id, baseline_path):
    print("Extracting Features.....")
    ecg_baseline_file = os.path.join(baseline_path, f"baseline_ecg_{baseline_sess_id}.csv")
    eda_baseline_file = os.path.join(baseline_path, f"baseline_eda_{baseline_sess_id}.csv")
    ecgDF =  pd.read_csv(ecg_baseline_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['data_received_time', 'dummy'])
    edaDF = pd.read_csv(eda_baseline_file, header=None, skiprows=1, names=['data_received_time', 'Timestamp', 'GSR Conductance CAL'])
    edaDF = edaDF.drop(columns=['data_received_time'])
    edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]
    # ecg_ = ecg_[512*60:-512*60]
    # eda_ = eda_[128*60:-128*60]
    
    print("Lenght of baseline ecg: {}s".format(ecgDF.shape[0] / 512))
    print("Lenght of baseline eda: {}s".format(edaDF.shape[0] / 128))


    ecgFeat = compute_ecg_eda_features.extract_ecg_features(ecgDF)
    edaFeat = compute_ecg_eda_features.extract_eda_features(edaDF)

    ecgFeat.to_csv(os.path.join(baseline_path, f"baseline_ecg_features_{baseline_sess_id}.csv"), index=False)
    edaFeat.to_csv(os.path.join(baseline_path, f"baseline_eda_features_{baseline_sess_id}.csv"), index=False)

    featDF = pd.concat([ecgFeat, edaFeat], axis = 1)

    ############# NEEDS TO BE UPDATED BASED ON NEW COULMNS #############

    selected_cols = SELECTCOLS[:-3]

    # selected_cols = ['ecg_HRV_IQRNN', 'ecg_HRV_MadNN','ecg_HRV_MeanNN',
    #         'ecg_HRV_MedianNN','ecg_HRV_RMSSD',
    #         'ecg_HRV_SD1', 'ecg_HRV_SD1SD2', 'ecg_HRV_SDNN', 'ecg_HRV_SDSD', 'ecg_HRV_pNN20',
    #         'ecg_HRV_pNN50', 'ecg_area_ts', 'ecg_entropy_features', 'ecg_iqr_features',
    #         'ecg_kurtosis_features', 'ecg_mad_ts', 'ecg_mean_features', 'ecg_median_features',
    #         'ecg_skew_features', 'ecg_sq_area_ts', 'ecg_std_features', 'eda_area_ts',
    #         'eda_entropy_features', 'eda_iqr_features', 'eda_kurtosis_features',
    #         'eda_mad_ts', 'eda_mean_features', 'eda_median_features',
    #         'eda_skew_features', 'eda_sq_area_ts', 'eda_std_features', 'ph_area_ts',
    #         'ph_entropy_features', 'ph_iqr_features', 'ph_kurtosis_features',
    #         'ph_mad_ts', 'ph_mean_features', 'ph_median_features','ph_skew_features',
    #         'ph_sq_area_ts', 'ph_std_features', 'ton_area_ts', 'ton_entropy_features',
    #         'ton_iqr_features', 'ton_kurtosis_features', 'ton_mad_ts', 'ton_mean_features',
    #         'ton_median_features', 'ton_skew_features', 'ton_sq_area_ts', 'ton_std_features',
    #         'ecg_hr_max', 'ecg_hr_mean', 'ecg_hr_min', 'ecg_hr_std', 'ecg_nni_counter', 'ecg_nni_diff_max',
    #         'ecg_nni_diff_mean', 'ecg_nni_max', 'ecg_nni_mean', 'ecg_nni_min', 'ecg_hf_peak',
    #         'ecg_ulf_abs', 'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs',  
    #         'ecg_lf_norm', 'ecg_hf_norm', 'ecg_lf_hf', 'ecg_tot_pwr']

    ############# ^^^^ NEEDS TO BE UPDATED BASED ON NEW COULMNS ^^^^ #############            

    featDF.to_csv(os.path.join(baseline_path, f'baseline_features_{baseline_sess_id}_nonstandardized.csv'), index=False)

    columns_to_standardize = ['ecg_sq_area_ts', 'ecg_nni_counter', 'ecg_ulf_abs',
            'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs', 'ecg_tot_pwr', 'eda_area_ts', 'eda_sq_area_ts', 'ton_sq_area_ts', 'scrNumPeaks']

    num_seg = edaDF.shape[0] // int(128 * 10)

    featDF = standardize_baseline_features(featDF.copy(), columns_to_standardize, num_seg=num_seg)
    featDF['ecg_nni_counter'] = checkZeroRound(featDF['ecg_nni_counter'].values)
    featDF['scrNumPeaks'] = checkZeroRound(featDF['scrNumPeaks'].values)

    featDF.to_csv(os.path.join(baseline_path, f'baseline_features_{baseline_sess_id}.csv'), index=False)
    print("Feature extraction done!")

def baseline_extract_features_virage_matbii(ecg_baseline_file, eda_baseline_file, subject, savePath = None):
    print("Extracting Features.....")
    # ecg_baseline_file = os.path.join(baseline_path, f"baseline_ecg_{baseline_sess_id}.csv")
    # eda_baseline_file = os.path.join(baseline_path, f"baseline_eda_{baseline_sess_id}.csv")
    ecgDF =  pd.read_csv(ecg_baseline_file, header=None, skiprows=1, names=['Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'ECG Vx-RL CAL'])
    edaDF = pd.read_csv(eda_baseline_file, header=None, skiprows=1, names=['Timestamp', 'GSR Conductance CAL'])

    # edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]
    # ecg_ = ecg_[512*60:-512*60]
    # eda_ = eda_[128*60:-128*60]
    
    ecgFeat = compute_ecg_eda_features.extract_ecg_features(ecgDF, ecg_sample_rt=512., dropCent=0.5)
    edaFeat = compute_ecg_eda_features.extract_eda_features(edaDF, eda_sample_rt=128., dropCent=0.5)

    savePathSubject = os.path.join(savePath, subject)

    mk_dirs(savePathSubject)

    ecgFeat.to_csv(os.path.join(savePathSubject, "ecg_baseline_features_oneline.csv"), index=False)
    edaFeat.to_csv(os.path.join(savePathSubject, "eda_baseline_features_oneline.csv"), index=False)

    featDF = pd.concat([ecgFeat, edaFeat], axis = 1)

    ############# NEEDS TO BE UPDATED BASED ON NEW COULMNS #############
    # selected_cols = SELECTCOLS[:-3]
    # featDF = featDF[selected_cols].copy()
    ############# ^^^^ NEEDS TO BE UPDATED BASED ON NEW COULMNS ^^^^ #############            

    featDF.to_csv(os.path.join(savePath, subject, "baseline_features.csv"), index=False)
    print("Feature extraction done!")

def baseline_extract_features_DPZ(ecg_baseline_file, eda_baseline_file, subject, sessID, savePath=None):
    print("Extracting Features.....")
    ecgDF =  pd.read_csv(ecg_baseline_file, header=None, skiprows=1, names=['Timestamp', 'ECG LL-RA CAL',  'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL'])
    ecgDF = ecgDF.drop(columns=['dummy'])
    edaDF = pd.read_csv(eda_baseline_file, header=None, skiprows=1, names=['Timestamp', 'GSR Conductance CAL'])
    # edaDF.iloc[:, 1] = 1000. / edaDF.iloc[:, 1]
    edaDF['GSR Conductance CAL'] = 1000. / edaDF['GSR Conductance CAL'].values
    
    ecgFeat = compute_ecg_eda_features.extract_ecg_features(ecgDF, ecg_sample_rt=512., dropCent=0.5)
    edaFeat = compute_ecg_eda_features.extract_eda_features(edaDF, eda_sample_rt=128., dropCent=0.5)

    savePathSubject = os.path.join(savePath, subject)

    mk_dirs(savePathSubject)

    ecgFeat.to_csv(os.path.join(savePathSubject, f"ecg_baseline_features_oneline_{sessID}.csv"), index=False)
    edaFeat.to_csv(os.path.join(savePathSubject, f"eda_baseline_features_oneline_{sessID}.csv"), index=False)

    featDF = pd.concat([ecgFeat, edaFeat], axis = 1)

    ############# NEEDS TO BE UPDATED BASED ON NEW COULMNS #############
    # selected_cols = SELECTCOLS[:-3]
    # featDF = featDF[selected_cols].copy()
    ############# ^^^^ NEEDS TO BE UPDATED BASED ON NEW COULMNS ^^^^ #############

    featDF.to_csv(os.path.join(savePath, subject, f"baseline_features_{sessID}.csv"), index=False) # changed sessID
    print("Feature extraction done!")

''' Baseline Features for Virage and MatbII'''

# if __name__ == "__main__":
#     # subj_id = 7
#     # sess_id = 1
#     baseline_path_virage = r"X:/RealTimeSegment/Driving Simulator/Raw/ECG_EDA_baseline/"
#     baseline_path_matbii = r"X:/RealTimeSegment/MatbII/Raw/ECG_EDA_baseline/"
#     savePath = "X:/RealTimeSegment/Driving Simulator/Extracted/ECG_EDA_baseline_oneline"

#     listOfSubjects = os.listdir(baseline_path_virage)

#     for subs in listOfSubjects:
#         subBasePath = os.path.join(baseline_path_virage, subs)
#         ecg_baseline_file = os.path.join(subBasePath, "ecg_baseline.csv")
#         eda_baseline_file = os.path.join(subBasePath, "eda_baseline.csv")
#         baseline_extract_features_virage_matbii(ecg_baseline_file, eda_baseline_file, subs, savePath)

#     listOfSubjectsMatb = os.listdir(baseline_path_matbii)
#     savePathMatbii = "X:/RealTimeSegment/MatbII/Extracted/ECG_EDA_baseline_oneline"

#     for subs in listOfSubjectsMatb:
#         subBasePath = os.path.join(baseline_path_matbii, subs)
#         ecg_baseline_file = os.path.join(subBasePath, "ecg_baseline.csv")
#         eda_baseline_file = os.path.join(subBasePath, "eda_baseline.csv")
#         baseline_extract_features_virage_matbii(ecg_baseline_file, eda_baseline_file, subs, savePathMatbii)

''' Baseline Features for Dirk Prithila and Zunayed '''

if __name__ == "__main__":
    subj_id = ['Dirk', 'Prithila', 'Zunayed']
    sess_id = 1
    baseline_path_DPZ = r"X:/RealTimeSegment/New Subjects Data/drik_prithial_zunayed_baseline/"
    savePath = r"X:/RealTimeSegment/New Subjects Data/DPZBaseFeatures"
    mk_dirs(savePath)
    for subs in subj_id:
        subBasePath = os.path.join(baseline_path_DPZ, subs)
        ecg_baseline_file = os.path.join(subBasePath, f"baseline_ecg_{sess_id}.csv")
        eda_baseline_file = os.path.join(subBasePath, f"baseline_eda_{sess_id}.csv")

        baseline_extract_features_DPZ(ecg_baseline_file, eda_baseline_file, subs, sess_id, savePath)