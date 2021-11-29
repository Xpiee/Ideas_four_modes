import os
from random import shuffle
import sys
# sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import os

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm
from sklearn.svm import SVC
from .config import DRIVPATH, MATPATH, BASPATH, SELECTCOLS, DROPCOLS
import pickle

# from .stdbaseline import checkZeroRound, standardize_baseline_features 

# selected_cols = ['ecg_HRV_IQRNN', 'ecg_HRV_MadNN','ecg_HRV_MeanNN',
# 'ecg_HRV_MedianNN','ecg_HRV_RMSSD',
# 'ecg_HRV_SD1', 'ecg_HRV_SD1SD2', 'ecg_HRV_SDNN', 'ecg_HRV_SDSD', 'ecg_HRV_pNN20',
# 'ecg_HRV_pNN50', 'ecg_area_ts', 'ecg_entropy_features', 'ecg_iqr_features',
# 'ecg_kurtosis_features', 'ecg_mad_ts', 'ecg_mean_features', 'ecg_median_features',
# 'ecg_skew_features', 'ecg_sq_area_ts', 'ecg_std_features', 'eda_area_ts',
# 'eda_entropy_features', 'eda_iqr_features', 'eda_kurtosis_features',
# 'eda_mad_ts', 'eda_mean_features', 'eda_median_features',
# 'eda_skew_features', 'eda_sq_area_ts', 'eda_std_features', 'ph_area_ts',
# 'ph_entropy_features', 'ph_iqr_features', 'ph_kurtosis_features',
# 'ph_mad_ts', 'ph_mean_features', 'ph_median_features','ph_skew_features',
# 'ph_sq_area_ts', 'ph_std_features', 'ton_area_ts', 'ton_entropy_features',
# 'ton_iqr_features', 'ton_kurtosis_features', 'ton_mad_ts', 'ton_mean_features',
# 'ton_median_features', 'ton_skew_features', 'ton_sq_area_ts', 'ton_std_features',
# 'ecg_hr_max', 'ecg_hr_mean', 'ecg_hr_min', 'ecg_hr_std', 'ecg_nni_counter', 'ecg_nni_diff_max', 'ecg_nni_diff_mean', 'ecg_nni_max',
# 'ecg_nni_mean', 'ecg_nni_min', 'ecg_hf_peak',
# 'ecg_ulf_abs', 'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs',  
# 'ecg_lf_norm', 'ecg_hf_norm', 'ecg_lf_hf', 'ecg_tot_pwr'] + ['complexity', 'label', 'scaled label']

# dropCols = ['ecg_HRV_CMSE', 'ecg_HRV_CorrDim', 'ecg_HRV_MSE', 'ecg_HRV_DFA', 'ecg_HRV_RCMSE']


def load_matb(mat_feat_path, label_col = 'scaled label', select_columns=None):
    matSubs = os.listdir(mat_feat_path)
    xtrainMat = pd.DataFrame()

    for smat in matSubs:
        trainMat = pd.read_csv(os.path.join(mat_feat_path, '{}'.format(smat)))
        trainMat[['scrAmpDF_min','scrRecoveryTime_min', 'scrRiseTime_min']].fillna(0)
        trainMat.replace([np.inf, -np.inf], 0, inplace=True)
        xtrainMat = xtrainMat.append(trainMat)
        xtrainMat.reset_index(drop=True, inplace = True)
        
    xtrainMat = xtrainMat[select_columns].copy()
    xtrainMat.dropna(inplace=True)
    yMat = list(xtrainMat[label_col].copy())
    
    xtrainMat.drop(columns=['label', 'complexity', 'scaled label'], inplace=True)
    return xtrainMat, yMat

def load_viragebase(bas_feat_path, label_col = 'label', select_columns=None):
    baseSubs = os.listdir(bas_feat_path)
    xtrainBas = pd.DataFrame()

    for sbas in baseSubs:
        trainBas = pd.read_csv(os.path.join(bas_feat_path, '{}'.format(sbas)))
        # trainBas.drop(columns=dropCols, inplace=True)
        trainBas[['scrAmpDF_min','scrRecoveryTime_min', 'scrRiseTime_min']].fillna(0)
        trainBas.replace([np.inf, -np.inf], 0, inplace=True)        
        xtrainBas = xtrainBas.append(trainBas)
        xtrainBas.reset_index(drop=True, inplace = True)
        
    xtrainBas = xtrainBas[select_columns[:-1]].copy()
    xtrainBas.dropna(inplace=True)
    yBas = list(xtrainBas[label_col].copy())

    xtrainBas.drop(columns=['label', 'complexity'], inplace=True)
    return xtrainBas, yBas


def load_virage(vir_feat_path, label_col = 'scaled label', select_columns=None):
    virSubs = os.listdir(vir_feat_path)
    xtrainDriv = pd.DataFrame()
        
    for sdriv in virSubs:
        if sdriv in ['1241.csv', '1337.csv']:
            continue        
        trainDriv = pd.read_csv(os.path.join(vir_feat_path, '{}'.format(sdriv)))
        trainDriv[['scrAmpDF_min','scrRecoveryTime_min', 'scrRiseTime_min']].fillna(0)
        trainDriv.replace([np.inf, -np.inf], 0, inplace=True)        
        xtrainDriv = xtrainDriv.append(trainDriv)
        xtrainDriv.reset_index(drop=True, inplace = True)

    XtrainDriv = xtrainDriv[select_columns].copy()
    XtrainDriv.dropna(inplace=True)
    ytrainDriv = list(XtrainDriv[label_col].copy())
    XtrainDriv.drop(columns=['label', 'complexity', 'scaled label'], inplace=True)

    return XtrainDriv.copy(), ytrainDriv


def trainClassifier(lowDF, highDF, savePath=None, saveName=None):

    '''
    Get low and high dataframe from test subject
    lowDF: DataFrame, normalized with baseline
    highDF: DataFrame, normalized with baseline

    '''
    yLow = [1] * len(lowDF)
    yHigh = [9] * len(highDF)

    lowhighTrainSub = pd.concat([lowDF, highDF], axis=0)
    yLowHigh = yLow + yHigh

    lowhighTrainSub = lowhighTrainSub[SELECTCOLS[:-3]].copy()

    xtrainMat, yMat = load_matb(MATPATH, 'scaled label', select_columns=SELECTCOLS)
    xtrainBas, yBas = load_viragebase(BASPATH, 'label', select_columns=SELECTCOLS)
    xtrainDriv, yDriv = load_virage(DRIVPATH, 'scaled label', select_columns=SELECTCOLS)

    xtrainDriv = xtrainDriv.append(xtrainMat)
    xtrainDriv = xtrainDriv.append(xtrainBas)
    xtrainDriv = xtrainDriv.append(lowhighTrainSub)

    yTrain = yDriv + yMat + yBas + yLowHigh

    X =  xtrainDriv.values

    for idx, val in enumerate(yTrain):
        if val <= 4:
            yTrain[idx] = 0
        else: yTrain[idx] = 1
        
    yTrain = np.array(yTrain)
    shuffle_ind = np.random.permutation(yTrain.shape[0])
    X = X[shuffle_ind]
    yTrain = yTrain[shuffle_ind]

    # Training the classifier
    paramsrf = {
        'n_estimators': 3000,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'max_depth': 20, # 'max_depth': 50,
        'bootstrap': False, 'verbose':0, 'n_jobs': -1, 'class_weight': 'balanced'
        }

    paramlgbm = {
        'n_estimators': 3000,
        'num_leaves': 23, #'num_leaves': 100,
        'learning_rate': 0.05,
        'class_weight': 'balanced',
        'random_state': 24
        }

    clf_rf = RandomForestClassifier(**paramsrf)
    # # clf_bgm = lightgbm.LGBMClassifier(n_estimators = 3000, num_leaves=100, learning_rate=0.05, class_weight='balanced')
    clf_bgm = lightgbm.LGBMClassifier(**paramlgbm)
    # clf_svm = SVC(C=10, probability=True, class_weight='balanced')

    estimatorList = [('rf', clf_rf), ('gbm', clf_bgm)] #[('rf', clf_rf), ('svm', clf_svm), ('gbm', clf_bgm)]
    eclf = VotingClassifier(estimators=estimatorList, voting='soft')
    hist = eclf.fit(X, yTrain)
    # # yPred = hist.predict(xtestMed)
    # # print(classification_report(ytestMed, yPred, zero_division=1))


    # hist = clf_rf.fit(X, yTrain)

    if savePath and saveName:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        estimatorPath = os.path.join(savePath, '{}.sav'.format(saveName))
        pickle.dump(hist, open(estimatorPath, 'wb'))

    return hist

def trainClassifier_temp(savePath=None, saveName=None):

    '''
    Get low and high dataframe from test subject
    lowDF: DataFrame, normalized with baseline
    highDF: DataFrame, normalized with baseline

    '''
    xtrainMat, yMat = load_matb(MATPATH, 'scaled label', select_columns=SELECTCOLS)
    xtrainBas, yBas = load_viragebase(BASPATH, 'label', select_columns=SELECTCOLS)
    xtrainDriv, yDriv = load_virage(DRIVPATH, 'scaled label', select_columns=SELECTCOLS)

    xtrainDriv = xtrainDriv.append(xtrainMat)
    xtrainDriv = xtrainDriv.append(xtrainBas)

    yTrain = yDriv + yMat + yBas 

    X =  xtrainDriv.values

    for idx, val in enumerate(yTrain):
        if val <= 4:
            yTrain[idx] = 0
        else: yTrain[idx] = 1

    # Training the classifier
    paramsrf = {
        'n_estimators': 3000,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'auto',
        'max_depth': 20, # 'max_depth': 50,
        'bootstrap': False, 'verbose':0, 'n_jobs': -1, 'class_weight': 'balanced'
        }

    paramlgbm = {
        'n_estimators': 3000,
        'num_leaves': 23, #'num_leaves': 100,
        'learning_rate': 0.05,
        'class_weight': 'balanced',
        'random_state': 24
        }

    clf_rf = RandomForestClassifier(**paramsrf)
    # # clf_bgm = lightgbm.LGBMClassifier(n_estimators = 3000, num_leaves=100, learning_rate=0.05, class_weight='balanced')
    clf_bgm = lightgbm.LGBMClassifier(**paramlgbm)
    # clf_svm = SVC(C=10, probability=True, class_weight='balanced')

    estimatorList = [('rf', clf_rf), ('gbm', clf_bgm)] #[('rf', clf_rf), ('svm', clf_svm), ('gbm', clf_bgm)]
    eclf = VotingClassifier(estimators=estimatorList, voting='soft')
    hist = eclf.fit(X, yTrain)
    # # yPred = hist.predict(xtestMed)
    # # print(classification_report(ytestMed, yPred, zero_division=1))


    # hist = clf_rf.fit(X, yTrain)

    if savePath and saveName:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        estimatorPath = os.path.join(savePath, '{}.sav'.format(saveName))
        pickle.dump(hist, open(estimatorPath, 'wb'))

    return hist

def join_normalize_ecg_eda(ecg_df, eda_df, baseline_df):
    joined_df = join_ecg_eda(ecg_df, eda_df)
    joined_df = joined_df[baseline_df.columns].sub(baseline_df.iloc[0])
    joined_df = joined_df[baseline_df.columns].div(baseline_df.iloc[0])
    return joined_df


def join_ecg_eda(ecg_df, eda_df):
    ecg_df.drop(columns=['sess_id', 'subj_id'], inplace=True)
    eda_df.drop(columns=['sess_id', 'subj_id'], inplace=True)
    # ecg_df.index = ecg_df['Timestamp'].values
    # eda_df.index = eda_df['Timestamp'].values
    ecg_df= ecg_df.drop( columns='Timestamp')
    min_len = min(ecg_df.shape[0], eda_df.shape[0])
    if ecg_df.shape[0] > min_len:
        ecg_df = ecg_df.iloc[:min_len]
    elif eda_df.shape[0] > min_len:
        eda_df = eda_df.iloc[:min_len]
    ecg_df.reset_index(drop=True, inplace=True)
    eda_df.reset_index(drop=True, inplace=True)
    joined_df = pd.concat([ecg_df, eda_df], axis = 1)
    return joined_df
   
   
def train(highlowpath, subj_id, baseline_path, baseline_sess_id, model_save_path, model_name, eda_sampling_rate=128, window_size=10.):  
    ecg_high_file = os.path.join(highlowpath, 'high_ecg_featurs_winsize_10.00_stepsize_0.50.csv')
    ecg_low_file = os.path.join(highlowpath, 'low_ecg_featurs_winsize_10.00_stepsize_0.50.csv')
    eda_high_file = os.path.join(highlowpath, 'high_eda_featurs_winsize_10.00_stepsize_0.50.csv')
    eda_low_file = os.path.join(highlowpath, 'low_eda_featurs_winsize_10.00_stepsize_0.50.csv')
    # baseline_eda_signal_fiile = os.path.join(baseline_path, 'baseline_eda_{}.csv').format(baseline_sess_id)
    baseline_file = os.path.join(baseline_path, 'baseline_features_{}.csv'.format(baseline_sess_id))
    
    # columns_to_standardize = ['ecg_sq_area_ts', 'ecg_nni_counter', 'ecg_ulf_abs',
    #         'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs', 'ecg_tot_pwr', 'eda_area_ts', 'eda_sq_area_ts', 'ton_sq_area_ts', 'scrNumPeaks']


    # baseline_eda_signal_df = pd.read_csv(baseline_eda_signal_fiile)
    # num_seg = baseline_eda_signal_df.shape[0] // int(eda_sampling_rate * window_size)

    ecg_high_df = pd.read_csv(ecg_high_file)
    ecg_low_df = pd.read_csv(ecg_low_file)
    eda_high_df = pd.read_csv(eda_high_file)
    eda_low_df = pd.read_csv(eda_low_file)
    
    baseline_df = pd.read_csv(baseline_file)
    
    # baseline_df = standardize_baseline_features(baseline_file, columns_to_standardize, num_seg=num_seg)
    # baseline_df['ecg_nni_counter'] = checkZeroRound(baseline_df['ecg_nni_counter'].values)
    # baseline_df['scrNumPeaks'] = checkZeroRound(baseline_df['scrNumPeaks'].values)

    high_df = join_normalize_ecg_eda(ecg_high_df, eda_high_df, baseline_df)
    low_df = join_normalize_ecg_eda(ecg_low_df, eda_low_df, baseline_df)
    
    print('Start training the model')
    trainClassifier(low_df, high_df, savePath=model_save_path, saveName=model_name)
    print('Training done!')


def train_noralized_by_low(highlowpath, model_save_path, model_name):  
    ecg_high_file = os.path.join(highlowpath, 'high_ecg_featurs_winsize_10.00_stepsize_0.50.csv')
    ecg_low_file = os.path.join(highlowpath, 'low_ecg_featurs_winsize_10.00_stepsize_0.50.csv')
    eda_high_file = os.path.join(highlowpath, 'high_eda_featurs_winsize_10.00_stepsize_0.50.csv')
    eda_low_file = os.path.join(highlowpath, 'low_eda_featurs_winsize_10.00_stepsize_0.50.csv')
        
    ecg_high_df = pd.read_csv(ecg_high_file)
    ecg_low_df = pd.read_csv(ecg_low_file)
    eda_high_df = pd.read_csv(eda_high_file)
    eda_low_df = pd.read_csv(eda_low_file)
    
    ecg_eda_low_joined = pd.DataFrame(join_ecg_eda(ecg_low_df.copy(), eda_low_df.copy()).median(axis=0)).transpose()
    
    ecg_eda_low_joined['ecg_nni_counter'] = checkZeroRound(ecg_eda_low_joined['ecg_nni_counter'].values)
    ecg_eda_low_joined['scrNumPeaks'] = checkZeroRound(ecg_eda_low_joined['scrNumPeaks'].values)

    high_df = join_normalize_ecg_eda(ecg_high_df.copy(), eda_high_df.copy(), ecg_eda_low_joined.copy())
    low_df = join_normalize_ecg_eda(ecg_low_df.copy(), eda_low_df.copy(), ecg_eda_low_joined.copy())
    
    print('Start training the model')
    trainClassifier(low_df, high_df, savePath=model_save_path, saveName=model_name)
    print('Training done!')
    

def train_temp(highlowpath, subj_id, baseline_path, baseline_sess_id, model_save_path, model_name, eda_sampling_rate=128, window_size=10.):    
    print('Start training the model')
    trainClassifier_temp(savePath=model_save_path, saveName=model_name)
    print('Training done!')


