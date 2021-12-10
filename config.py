import numpy as np
import pandas as pd

DRIVPATH = r'C:\Users\behnam\Documents\realtime_datacollection\Training_Code\datasets\virage_features\Norm_ECG_EDA_Features_Combined_scld'
MATPATH = r'C:\Users\behnam\Documents\realtime_datacollection\Training_Code\datasets\matbii_features\Norm_ECG_EDA_Features_Combined_scld'
BASPATH = r'C:\Users\behnam\Documents\realtime_datacollection\Training_Code\datasets\virage_features\Norm_ECG_EDA_Features_Baseline_Combined'

SELECTCOLS = ['ecg_HRV_IQRNN', 'ecg_HRV_MadNN','ecg_HRV_MeanNN',
'ecg_HRV_MedianNN','ecg_HRV_RMSSD', 'ecg_HRV_SD1', 'ecg_HRV_SD1SD2',
'ecg_HRV_SDNN', 'ecg_HRV_SDSD', 'ecg_HRV_pNN20', 'ecg_HRV_pNN50', 'ecg_area_ts',
'ecg_entropy_features', 'ecg_iqr_features', 'ecg_kurtosis_features', 'ecg_mad_ts',
'ecg_mean_features', 'ecg_median_features', 'ecg_skew_features', 'ecg_sq_area_ts',
'ecg_std_features', 'eda_area_ts', 'eda_entropy_features', 'eda_iqr_features',
'eda_kurtosis_features', 'eda_mad_ts', 'eda_mean_features', 'eda_median_features',
'eda_skew_features', 'eda_sq_area_ts', 'eda_std_features', 'ph_area_ts',
'ph_entropy_features', 'ph_iqr_features', 'ph_kurtosis_features',
'ph_mad_ts', 'ph_mean_features', 'ph_median_features','ph_skew_features',
'ph_sq_area_ts', 'ph_std_features', 'ton_area_ts', 'ton_entropy_features',
'ton_iqr_features', 'ton_kurtosis_features', 'ton_mad_ts', 'ton_mean_features',
'ton_median_features', 'ton_skew_features', 'ton_sq_area_ts', 'ton_std_features',
'ecg_hr_max', 'ecg_hr_mean', 'ecg_hr_min', 'ecg_hr_std', 'ecg_nni_counter', 'ecg_nni_diff_max',
'ecg_nni_diff_mean', 'ecg_nni_max', 'ecg_nni_mean', 'ecg_nni_min', 'ecg_hf_peak', 'ecg_ulf_abs',
'ecg_vlf_abs', 'ecg_lf_abs', 'ecg_hf_abs', 'ecg_lf_norm', 'ecg_hf_norm', 'ecg_lf_hf',
'ecg_tot_pwr',  'ecg_max_features', 'ecg_min_features', 'eda_max_features', 'eda_min_features',
'scrAmpDF_max', 'scrAmpDF_mean', 'scrAmpDF_median', 'scrHgt_max', 'scrHgt_mean',
'scrHgt_median', 'scrHgt_min', 'scrHgt_std', 'scrNumPeaks',
'scrRecoveryTime_max', 'scrRecoveryTime_mean', 'scrRecoveryTime_median', 
'scrRiseTime_max', 'scrRiseTime_mean', 'scrRiseTime_median'] + ['complexity', 'label', 'scaled label']

SELECTIMPORTANT = ['ecg_HRV_MeanNN', 'ecg_HRV_MedianNN', 'ecg_area_ts',
       'ecg_entropy_features', 'ecg_iqr_features', 'ecg_kurtosis_features',
       'ecg_mad_ts', 'ecg_mean_features', 'ecg_median_features',
       'ecg_skew_features', 'ecg_sq_area_ts', 'ecg_std_features',
       'eda_area_ts', 'eda_mean_features', 'eda_median_features',
       'eda_sq_area_ts', 'ecg_hr_max', 'ecg_hr_mean', 'ecg_hr_min',
       'ecg_nni_max', 'ecg_nni_mean', 'ecg_nni_min', 'ecg_hf_norm',
       'ecg_max_features', 'ecg_min_features', 'eda_max_features',
       'eda_min_features', 'scrAmpDF_max', 'scrAmpDF_mean']

DROPCOLS = ['ecg_HRV_CMSE', 'ecg_HRV_CorrDim', 'ecg_HRV_MSE', 'ecg_HRV_DFA', 'ecg_HRV_RCMSE']

SELECTGAZECOLS = ['LPupil_avg', 'LPupil_max', 'LPupil_min', 'RPupil_avg', 'RPupil_max', 'RPupil_min', 'Blink_count', 'Blink_duration_avg', 'Blink_duration_max', 'Fixation_count', 'Fixation_duration_avg',
'Fixation_duration_max', 
'Fixation_duration_min', 'Fixation_dispersion_avg', 'Fixation_dispersion_max', 'Fixation_dispersion_min', 'Saccade_count', 'Saccade_duration_avg', 'Saccade_duration_max', 'Saccade_duration_min', 
'Saccade_amplitude_avg', 'Saccade_amplitude_max', 'Saccade_amplitude_min', 'Saccade_peak_vel_avg', 'Saccade_peak_vel_max', 'Saccade_peak_vel_min', 'Saccade_peak_acc_avg', 'Saccade_peak_acc_max', 
'Saccade_peak_acc_min', 'Saccade_peak_dec_avg', 'Saccade_peak_dec_max', 'Saccade_peak_dec_min', 'Saccade_direction_avg', 'Saccade_direction_max', 'Saccade_direction_min'] 

SELECTEEGCOLS = ['deltapower_ch1', 'thetapower_ch1', 'alphapower_ch1', 'betapower_ch1', 'gammapower_ch1', 'delta_psd_avg_ch1', 'theta_psd_avg_ch1', 'alpha_psd_avg_ch1', 'beta_psd_avg_ch1',
'gamma_psd_avg_ch1',
'delta_psd_max_ch1', 'theta_psd_max_ch1', 'alpha_psd_max_ch1', 'beta_psd_max_ch1', 'gamma_psd_max_ch1', 'delta_psd_min_ch1', 'theta_psd_min_ch1', 'alpha_psd_min_ch1', 'beta_psd_min_ch1', 
'gamma_psd_min_ch1', 'delta_psd_med_ch1', 'theta_psd_med_ch1', 'alpha_psd_med_ch1', 'beta_psd_med_ch1', 'gamma_psd_med_ch1', 'deltapower_ch2', 'thetapower_ch2', 'alphapower_ch2', 'betapower_ch2', 
'gammapower_ch2', 'delta_psd_avg_ch2', 'theta_psd_avg_ch2', 'alpha_psd_avg_ch2', 'beta_psd_avg_ch2', 'gamma_psd_avg_ch2', 'delta_psd_max_ch2', 'theta_psd_max_ch2', 'alpha_psd_max_ch2', 
'beta_psd_max_ch2', 'gamma_psd_max_ch2', 'delta_psd_min_ch2', 'theta_psd_min_ch2', 'alpha_psd_min_ch2', 'beta_psd_min_ch2', 'gamma_psd_min_ch2', 'delta_psd_med_ch2', 'theta_psd_med_ch2', 
'alpha_psd_med_ch2', 'beta_psd_med_ch2', 'gamma_psd_med_ch2', 'deltapower_ch3', 'thetapower_ch3', 'alphapower_ch3', 'betapower_ch3', 'gammapower_ch3', 'delta_psd_avg_ch3', 'theta_psd_avg_ch3',
'alpha_psd_avg_ch3', 'beta_psd_avg_ch3', 'gamma_psd_avg_ch3', 'delta_psd_max_ch3', 'theta_psd_max_ch3', 'alpha_psd_max_ch3', 'beta_psd_max_ch3', 'gamma_psd_max_ch3', 'delta_psd_min_ch3', 
'theta_psd_min_ch3', 'alpha_psd_min_ch3', 'beta_psd_min_ch3', 'gamma_psd_min_ch3', 'delta_psd_med_ch3', 'theta_psd_med_ch3', 'alpha_psd_med_ch3', 'beta_psd_med_ch3', 'gamma_psd_med_ch3', 'deltapower_ch4', 'thetapower_ch4',
'alphapower_ch4', 'betapower_ch4', 'gammapower_ch4', 'delta_psd_avg_ch4', 'theta_psd_avg_ch4', 'alpha_psd_avg_ch4', 'beta_psd_avg_ch4', 'gamma_psd_avg_ch4', 'delta_psd_max_ch4',
'theta_psd_max_ch4', 'alpha_psd_max_ch4', 'beta_psd_max_ch4', 'gamma_psd_max_ch4', 'delta_psd_min_ch4', 'theta_psd_min_ch4', 'alpha_psd_min_ch4', 'beta_psd_min_ch4', 'gamma_psd_min_ch4',
'delta_psd_med_ch4', 'theta_psd_med_ch4', 'alpha_psd_med_ch4', 'beta_psd_med_ch4', 'gamma_psd_med_ch4', 'delta_fft_avg_ch1', 'theta_fft_avg_ch1', 'alpha_fft_avg_ch1', 'beta_fft_avg_ch1',
'gamma_fft_avg_ch1', 'delta_fft_max_ch1', 'theta_fft_max_ch1', 'alpha_fft_max_ch1', 'beta_fft_max_ch1', 'gamma_fft_max_ch1', 'delta_fft_min_ch1', 'theta_fft_min_ch1', 'alpha_fft_min_ch1',
'beta_fft_min_ch1', 'gamma_fft_min_ch1', 'delta_fft_med_ch1', 'theta_fft_med_ch1', 'alpha_fft_med_ch1', 'beta_fft_med_ch1', 'gamma_fft_med_ch1', 'delta_fft_avg_ch2', 
'theta_fft_avg_ch2', 'alpha_fft_avg_ch2', 'beta_fft_avg_ch2', 'gamma_fft_avg_ch2', 'delta_fft_max_ch2', 'theta_fft_max_ch2', 'alpha_fft_max_ch2', 'beta_fft_max_ch2', 'gamma_fft_max_ch2', 
'delta_fft_min_ch2', 'theta_fft_min_ch2', 'alpha_fft_min_ch2', 'beta_fft_min_ch2', 'gamma_fft_min_ch2', 'delta_fft_med_ch2', 'theta_fft_med_ch2', 'alpha_fft_med_ch2', 'beta_fft_med_ch2', 
'gamma_fft_med_ch2', 'delta_fft_avg_ch1.1', 'theta_fft_avg_ch3', 'alpha_fft_avg_ch3', 'beta_fft_avg_ch3', 'gamma_fft_avg_ch3', 'delta_fft_max_ch3', 'theta_fft_max_ch3', 'alpha_fft_max_ch3', 
'beta_fft_max_ch3', 'gamma_fft_max_ch3', 'delta_fft_min_ch3', 'theta_fft_min_ch3', 'alpha_fft_min_ch3', 'beta_fft_min_ch3', 'gamma_fft_min_ch3', 'delta_fft_med_ch3', 'theta_fft_med_ch3', 
'alpha_fft_med_ch3', 'beta_fft_med_ch3', 'gamma_fft_med_ch3', 'delta_fft_avg_ch4', 'theta_fft_avg_ch4', 'alpha_fft_avg_ch4', 'beta_fft_avg_ch4', 'gamma_fft_avg_ch4', 'delta_fft_max_ch4', 
'theta_fft_max_ch4', 'alpha_fft_max_ch4', 'beta_fft_max_ch4', 'gamma_fft_max_ch4', 'delta_fft_min_ch4', 'theta_fft_min_ch4', 'alpha_fft_min_ch4', 'beta_fft_min_ch4', 'gamma_fft_min_ch4',
'delta_fft_med_ch4', 'theta_fft_med_ch4', 'alpha_fft_med_ch4', 'beta_fft_med_ch4', 'gamma_fft_med_ch4', 'delta_spec_ch1', 'theta_spec_ch1', 'alpha_spec_ch1', 'beta_spec_ch1', 'gamma_spec_ch1',
'delta_spec_ch2', 'theta_spec_ch2', 'alpha_spec_ch2', 'beta_spec_ch2', 'gamma_spec_ch2', 'delta_spec_ch3', 'theta_spec_ch3', 'alpha_spec_ch3', 'beta_spec_ch3', 'gamma_spec_ch3',
'delta_spec_ch4', 'theta_spec_ch4', 'alpha_spec_ch4', 'beta_spec_ch4', 'gamma_spec_ch4', 'min_ch1', 'max_ch1', 'avg_ch1', 'med_ch1', 'var_ch1', 'std_ch1', 'hj_mob_ch1', 'hj_comp_ch1', 
'lz_ch1', 'hig_ch1', 'min_ch2', 'max_ch2', 'avg_ch2', 'med_ch2', 'var_ch2', 'std_ch2', 'hj_mob_ch2', 'hj_comp_ch2', 'lz_ch2', 'hig_ch2', 'min_ch3', 'max_ch3', 'avg_ch3', 'med_ch3', 'var_ch3',
'std_ch3', 'hj_mob_ch3', 'hj_comp_ch3', 'lz_ch3', 'hig_ch3', 'min_ch4', 'max_ch4', 'avg_ch4', 'med_ch4', 'var_ch4', 'std_ch4', 'hj_mob_ch4', 'hj_comp_ch4', 'lz_ch4', 'hig_ch4'] 