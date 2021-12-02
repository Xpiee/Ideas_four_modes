import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk

from main_utils_1 import eda_cleaner, eda_decom, impute_eda, mk_dirs, impute_ecg, ecg_cleaner

def add_signal_timestamp(df, sample_rt):
    df.reset_index(drop=True, inplace=True) # resetting the index after dropping nan rows
    # converting the timestamps to float to make the data timestamps consistent
    df['Timestamp'] = df['Timestamp'].astype('float')

    # creating a list of all timestamps that should have been there if there was no missing datapoints.
    time_list = ([df.loc[0, 'Timestamp'] + (x * (1000/sample_rt)) for x in range(0, int((df.loc[df.index[-1], 'Timestamp'] - df.loc[0, 'Timestamp'])/(1000/sample_rt)) + 1)])
    
    # creating a dataframe from the time_list that has all the timestamps (missing + not missing)
    df_time = pd.DataFrame(time_list, columns = ['timestamp'])

    # rounding the timestamps to 1 place decimal as then it would be more easier to compare timestamps!
    df_time['timestamp'] = df_time['timestamp'].round(decimals = 1)
    df_time.index = df_time['timestamp'] # shifting the timestamps to index

    df['Timestamp'] = df['Timestamp'].round(decimals = 1)
    df.index = df['Timestamp']

    df_new = pd.concat([df_time, df], axis = 1)
    df_new.drop(columns = ['Timestamp'], inplace=True)
    df_new.reset_index(inplace=True, drop=True)

    return df_new.copy()

def process_virage(main_path, ecg_sample_rt=512, eda_sample_rt=128, savePath=None, isBaseline=False, droPcent=0.05):
# main_path = r"X:\IDEaS\Driving Simulator\Signals_cp"
# ecg_sample_rt = 512
    subjects_id = os.listdir(main_path)
    dirlist = os.listdir(os.path.join(main_path, subjects_id[0]))
    if isBaseline:
        exp_id = ['baseline.csv']
    else:
        exp_id = [x for x in dirlist if 'level_' in x] # op: ['level_1.csv', 'level_2.csv', ...]

    ecg_rd_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']
    eda_rd_cols = ['Timestamp', 'GSR Conductance CAL']            

    for sub_id in subjects_id:
        subject_path = os.path.join(main_path, sub_id)
        print(sub_id)

        for xid in exp_id:
            try:

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
                # csv_path = os.path.join(savePath, sub_id)

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
#     main_path = r"X:\IDEaS\Driving Simulator\Signals_cp"
#     ecgHz = 512
#     edaHz = 128

#     savePath = r'X:\Four Modes\Virage\Filtered\ECG_EDA'
#     process_virage(main_path, ecg_sample_rt=ecgHz,
#                     eda_sample_rt=edaHz, savePath=savePath, isBaseline=False)

if __name__ == '__main__':
    main_path = r"X:\IDEaS\Driving Simulator\Signals_cp"
    ecgHz = 512
    edaHz = 128

    savePath = r'X:\Four Modes\Virage\Filtered\Base3_ECG_EDA'
    process_virage(main_path, ecg_sample_rt=ecgHz,
                    eda_sample_rt=edaHz, savePath=savePath, isBaseline=True, droPcent=0.5)

