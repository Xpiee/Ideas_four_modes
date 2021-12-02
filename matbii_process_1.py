import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk

from main_utils_1 import eda_cleaner, eda_decom, impute_eda, mk_dirs, impute_ecg, ecg_cleaner

from virage_process_1 import add_signal_timestamp

def process_matbii(main_path, ecg_sample_rt=512, eda_sample_rt=128, savePath=None, isBaseline=False, droPcent=0.05):

    subjects_id = os.listdir(main_path)
    if isBaseline:
        exp_id = ['baseline.csv']
    else:
        exp_id = ['exp_0.csv', 'exp_1.csv', 'exp_2.csv', 'exp_3.csv']
    
    ecg_rd_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']
    eda_rd_cols = ['Timestamp', 'GSR Conductance CAL']            
    
    for sub_id in subjects_id:
        
        subject_path = os.path.join(main_path, sub_id, 'Sensor Data')
        print(sub_id)

        for xid in exp_id:
            try:
                if sub_id == '1544' and (xid in ['exp_1.csv', 'exp_3.csv', 'exp_2.csv']):
                    # for exp_1 ECG recording was stopped after 2 mins :(
                    # Shimmer ECG sensor was not configured for ECG; hence no ECG was recorded. 
                    continue
                csv_path = os.path.join(savePath, sub_id)
                save_df_ecg = os.path.join(csv_path, 'ecg_{}'.format(xid))
                save_df_eda = os.path.join(csv_path, 'eda_{}'.format(xid))

                if os.path.exists(save_df_ecg) and os.path.exists(save_df_eda):
                    print("Data File already exists! skipping imputation!")
                    continue

                read_path = os.path.join(subject_path, '{}'.format(xid))
                df = pd.read_csv(read_path, dtype='object')
                if df.columns[0] == '#INFO':
                    df_ecg = pd.read_csv(read_path, skiprows = 32, skipinitialspace=True, usecols=ecg_rd_cols)
                    df_eda = pd.read_csv(read_path, skiprows = 32, skipinitialspace=True, usecols=eda_rd_cols)
                else: 
                    df_ecg = pd.read_csv(read_path, usecols=ecg_rd_cols)
                    df_eda = pd.read_csv(read_path, usecols=eda_rd_cols)

                df_ecg.dropna(inplace=True) # removing all the nan rows
                df_eda.dropna(inplace=True) # removing all the nan rows

                # Putting a check if the signal data is not present in the csv then skip that subject
                if len(df_ecg) == 0:
                    print('Subject {} does not have ECG signal data for session: {}'.format(sub_id, xid))
                    continue

                # Putting a check if the signal data is not present in the csv then skip that subject
                if len(df_eda) == 0:
                    print('Subject {} does not have EDA signal data for session: {}'.format(sub_id, xid))
                    continue

                df_new_ecg = add_signal_timestamp(df_ecg, ecg_sample_rt)
                df_new_eda = add_signal_timestamp(df_eda, eda_sample_rt)

                num_drops_ecg = df_new_ecg['ECG LL-RA CAL'].isna().sum()
                num_drops_eda = df_new_eda['GSR Conductance CAL'].isna().sum()

                if num_drops_ecg > len(df_new_ecg) * droPcent:
                    print(f'Missing more than 5 percent for ECG {xid}')
                    continue

                if num_drops_eda > len(df_new_eda) * droPcent:
                    print(f'Missing more than 5 percent for EDA {xid}')
                    continue

                df_impute_ecg = impute_ecg(df_new_ecg.copy())
                
                # cleaning the ECG signals
                df_impute_clean_ecg = ecg_cleaner(df_impute_ecg.copy(), ecg_sample_rt)

                # cleaning the EDA signals
                df_impute_eda = impute_eda(df_new_eda.copy())

                # cleaning the EDA signals
                df_impute_clean_eda = eda_cleaner(df_impute_eda.copy(), eda_sample_rt)
                df_impute_clean_eda = eda_decom(df_impute_clean_eda.copy(), eda_sample_rt)                

                # creating the directory
                mk_dirs(csv_path)

                df_impute_clean_ecg.to_csv(os.path.join(csv_path, 'ecg_{}'.format(xid)), index=False)
                df_impute_clean_eda.to_csv(os.path.join(csv_path, 'eda_{}'.format(xid)), index=False)

            except FileNotFoundError:
                # exp_3 for subject 1674 was not recorded :(
                continue

# if __name__ == '__main__':
#     main_path = r"X:\IDEaS\Full\June_11_2021"
#     ecgHz = 512
#     edaHz = 128

#     savePath = r'X:\Four Modes\Matbii\Filtered\ECG_EDA'
#     process_matbii(main_path, ecg_sample_rt=ecgHz,
#                     eda_sample_rt=edaHz, savePath=savePath, isBaseline=False)

if __name__ == '__main__':
    main_path = r"X:\IDEaS\Full\June_11_2021"
    ecgHz = 512
    edaHz = 128

    savePath = r'X:\Four Modes\Matbii\Filtered\Base3_ECG_EDA'
    process_matbii(main_path, ecg_sample_rt=ecgHz,
                    eda_sample_rt=edaHz, savePath=savePath, isBaseline=True, droPcent=0.5)