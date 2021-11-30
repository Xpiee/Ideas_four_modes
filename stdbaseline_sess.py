'''
Standardization of Session Based Baselines for MatbII and Virage

'''

import numpy as np
import pandas as pd
import os

def number_of_segments(lengthBase, sampleRate, winSize):
    winSegment = winSize * sampleRate
    return lengthBase // winSegment

def standardize_baseline_features(rawBaseSignalPath, baseFeaturePath, columns, sampleRate, winSize):

    # Read raw Baseline Files for determining the length of the signal
    rawBaseDF = pd.read_csv(rawBaseSignalPath)
    lengthBase = len(rawBaseDF)

    # get number of segments for this baseline signal
    numOfSeg = number_of_segments(lengthBase, sampleRate, winSize)

    # Read base features for standardizing the columns based on numOfSeg
    featBaseDF = pd.read_csv(baseFeaturePath)

    # standardize the columns by dividing the column values from numOfSeg 
    featBaseDF[columns] = featBaseDF[columns] / numOfSeg

    epsilon_ = 0.0001
    featBaseDF = featBaseDF.replace(0, value=epsilon_)

    return featBaseDF.copy()

def checkZeroRound(colAr):
    signAr = np.sign(colAr)
    rdAr = np.ceil(np.abs(colAr))
    actualAr = signAr * rdAr
    return actualAr

if __name__ == '__main__':

    ### provide root path for baseline oneline features ###

    dataName = {'MatbII': 'MatB-II_Clipped_Baseline',
                'Driving Simulator': 'Virage_Clipped_Baseline'}

    for key, data in dataName.items():

        # baseRPath = "X:/RealTimeSegment/Driving Simulator/Extracted/ECG_EDA_baseline_oneline"
        # saveBaseRPath = "X:/RealTimeSegment/Driving Simulator/Extracted/ECG_EDA_baseline_oneline_std"
        # root path for raw baseline signal
        # rawRPath = "X:/RealTimeSegment/Driving Simulator/Raw/ECG_EDA_baseline"

        baseRPath = f"X:/Four modes baseline/{data}/Extracted/ECG_EDA_baseline_oneline"
        saveBaseRPath = f"X:/Four modes baseline/{data}/Extracted/ECG_EDA_baseline_oneline_std"

        # root path for raw baseline signal
        rawRPath = f"X:/Four modes baseline/{data}/Raw/ECG_EDA_baseline"

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

            subDirs = [x for x in subDirs if '_baseline_features_' in x]
            subDirs = set([x.split('_')[1] for x in subDirs])
            
            subPathRaw = os.path.join(rawRPath, sub)
            subPathFeat = os.path.join(baseRPath, sub)

            for sess in subDirs:

                # raw file path
                rawEcgDF = os.path.join(subPathRaw, f'ecg_{sess}.csv') ##################
                rawEdaDF = os.path.join(subPathRaw, f'eda_{sess}.csv')

                # read baseline features
                oneEcgDF = os.path.join(subPathFeat, f'ecg_{sess}_baseline_features_oneline.csv')
                oneEdaDF = os.path.join(subPathFeat, f'eda_{sess}_baseline_features_oneline.csv')

                ecgDF = standardize_baseline_features(rawEcgDF, oneEcgDF, ecgColumns , ecg_sample_rate, 10)
                edaDF = standardize_baseline_features(rawEdaDF, oneEdaDF, edaColumns , eda_sample_rate, 10)

                ecgDF['ecg_nni_counter'] = checkZeroRound(ecgDF['ecg_nni_counter'].values)
                edaDF['scrNumPeaks'] = checkZeroRound(edaDF['scrNumPeaks'].values)

                saveBaseRPathSub = os.path.join(saveBaseRPath, sub)

                if not os.path.exists(saveBaseRPathSub):
                    os.makedirs(saveBaseRPathSub)
                ecgDF.to_csv(os.path.join(saveBaseRPathSub, f'ecg_{sess}_baseline_features_oneline.csv'), index=False)
                edaDF.to_csv(os.path.join(saveBaseRPathSub, f'eda_{sess}_baseline_features_oneline.csv'), index=False)