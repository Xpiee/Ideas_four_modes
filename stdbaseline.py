import numpy as np
import pandas as pd
import os

from pandas.core.indexes import base

def number_of_segments(lengthBase, sampleRate, winSize):
    winSegment = winSize * sampleRate
    return lengthBase // winSegment

def standardize_baseline_features(featBaseDF, columns, num_seg):


    # Read base features for standardizing the columns based on numOfSeg
    # featBaseDF = pd.read_csv(baseFeaturePath)

    # standardize the columns by dividing the column values from numOfSeg 
    featBaseDF[columns] = featBaseDF[columns] / num_seg

    epsilon_ = 0.0001
    featBaseDF = featBaseDF.replace(0, value=epsilon_)

    return featBaseDF.copy()

def checkZeroRound(arr):
    sign_arr = np.sign(arr)
    arr_rounded = np.ceil(np.abs(arr))
    return sign_arr * arr_rounded

if __name__ == '__main__':

    ### provide root path for baseline oneline features ###

    # baseRPath = "X:/RealTimeSegment/Driving Simulator/Extracted/ECG_EDA_baseline_oneline"
    # saveBaseRPath = "X:/RealTimeSegment/Driving Simulator/Extracted/ECG_EDA_baseline_oneline_std"
    # root path for raw baseline signal
    # rawRPath = "X:/RealTimeSegment/Driving Simulator/Raw/ECG_EDA_baseline"

    baseRPath = "X:/RealTimeSegment/MatbII/Extracted/ECG_EDA_baseline_oneline"
    saveBaseRPath = "X:/RealTimeSegment/MatbII/Extracted/ECG_EDA_baseline_oneline_std"
    # root path for raw baseline signal
    rawRPath = "X:/RealTimeSegment/MatbII/Raw/ECG_EDA_baseline"

    if not os.path.exists(saveBaseRPath):
        os.makedirs(saveBaseRPath)

    subIDs = os.listdir(baseRPath)

    eda_sample_rate = 128.
    ecg_sample_rate = 512.
    winsize = 10
    
    ecgColumns = ['ecg_sq_area_ts', 'ecg_nni_counter', 'ecg_ulf_abs',
    'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs', 'ecg_tot_pwr']
    edaColumns = ['eda_area_ts', 'eda_sq_area_ts', 'ton_sq_area_ts', 'scrNumPeaks']

    for sub in subIDs:
        subPath = os.path.join(baseRPath, sub)
        subDirs = os.listdir(subPath)

        # raw file path
        subPathRaw = os.path.join(rawRPath, sub)
        rawEcgDF = os.path.join(subPathRaw, 'ecg_baseline.csv')
        rawEdaDF = os.path.join(subPathRaw, 'eda_baseline.csv')

        # read baseline features
        subPathFeat = os.path.join(baseRPath, sub)
        oneEcgDF = os.path.join(subPathFeat, 'ecg_baseline_features_oneline.csv')
        oneEdaDF = os.path.join(subPathFeat, 'eda_baseline_features_oneline.csv')

        ecgDF = standardize_baseline_features(rawEcgDF, oneEcgDF, ecgColumns , ecg_sample_rate, 10)
        edaDF = standardize_baseline_features(rawEdaDF, oneEdaDF, edaColumns , eda_sample_rate, 10)

        ecgDF['ecg_nni_counter'] = checkZeroRound(ecgDF['ecg_nni_counter'].values)
        edaDF['scrNumPeaks'] = checkZeroRound(edaDF['scrNumPeaks'].values)

        saveBaseRPathSub = os.path.join(saveBaseRPath, sub)

        if not os.path.exists(saveBaseRPathSub):
            os.makedirs(saveBaseRPathSub)
        ecgDF.to_csv(os.path.join(saveBaseRPathSub, 'ecg_baseline_features_oneline.csv'), index=False)
        edaDF.to_csv(os.path.join(saveBaseRPathSub, 'eda_baseline_features_oneline.csv'), index=False)