from numpy import savez_compressed
import pandas as pd
import os
from main_utils_1 import eda_cleaner, eda_decom, impute_eda, mk_dirs, impute_ecg, ecg_cleaner

from virage_process_1 import add_signal_timestamp

''' Re Structuring Session Based Baseline Signals from Virage and MatbII '''

def process_base2_ecg_eda(main_path, data, ecg_sample_rt=512, eda_sample_rt=128, savePath=None, droPcent=0.05):
    subjects_id = os.listdir(main_path)
    ecgrd_cols = ['Timestamp', 'ECG LL-RA CAL',
            'ECG LA-RA CAL', 'ECG Vx-RL CAL']
    edard_cols = ['Timestamp', 'GSR Conductance CAL']

    for sub_id in subjects_id:
        subject_path = os.path.join(main_path, sub_id)
        print(sub_id)

        dirlist = os.listdir(subject_path)
        exp_id = [x.split('_')[0] for x in dirlist]

        for xid in exp_id:
            try:
                csv_path = os.path.join(savePath, '{}'.format(sub_id))
                save_df_ecg = os.path.join(csv_path, 'ecg_{}'.format(xid))
                save_df_eda = os.path.join(csv_path, 'eda_{}'.format(xid))

                if os.path.exists(save_df_ecg) and os.path.exists(save_df_eda):
                    print("Data File already exists! skipping imputation!")
                    continue

                read_path = os.path.join(subject_path, '{}_baseline.csv'.format(xid))
                skipTheseRows = 0
                if data == 'MatB-II_Clipped_Baseline':
                    df = pd.read_csv(read_path, low_memory=False)
                    if df.columns[0] == '#INFO':
                        skipTheseRows = 32

                df_ecg = pd.read_csv(read_path, skiprows=skipTheseRows, skipinitialspace=True, usecols=ecgrd_cols)
                df_eda = pd.read_csv(read_path, skiprows=skipTheseRows, skipinitialspace=True, usecols=edard_cols)

                df_ecg.dropna(inplace=True) # removing all the nan rows
                df_eda.dropna(inplace=True) # removing all the nan rows

                # Putting a check if the signal data is not present in the csv then skip that subject
                if len(df_ecg) == 0:
                    print('Subject {} does not have ECG signal data for session: {}'.format(sub_id, xid))
                    continue
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

                mk_dirs(csv_path)
                df_impute_clean_ecg.to_csv(os.path.join(csv_path, 'ecg_{}.csv'.format(xid)), index=False)
                df_impute_clean_eda.to_csv(os.path.join(csv_path, 'eda_{}.csv'.format(xid)), index=False)

            except FileNotFoundError:
                continue

if __name__ == '__main__':
    ## Provide the unstructured synchronized session baseline signals path
    dataName = {'Matbii': 'MatB-II_Clipped_Baseline',
                'Virage': 'Virage_Clipped_Baseline'}

    for datakey, data in dataName.items():
        main_path = f"X:/Four modes baseline/{data}/Signals"
        savePath = f'X:/Four Modes/{datakey}/Filtered/Base2_ECG_EDA'

        ecgHz = 512
        edaHz = 128
        process_base2_ecg_eda(main_path, data=data, ecg_sample_rt=ecgHz,
                    eda_sample_rt=edaHz, savePath=savePath, droPcent=0.5)
