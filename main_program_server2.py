import socket
import os
import keyboard
import json
from compute_baseline_features import baseline_extract_features_phase2
from compute_features_with_overlap_low_and_high import extract_features_with_overlap_phase2, extract_features_with_overlap_from_sess1
from Training_Code.TrainPipeline import train, train_noralized_by_low, train_temp
import warnings 
import sys
from datetime import datetime
import shutil

def mkdir_backup_if_exists(filepath, filename):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        warnings.warn('classifier folder already exists!')
        if os.path.exists(os.path.join(filepath, f'{filename}.sav')):
            str_datetime = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            os.mkdir(os.path.join(filepath, str_datetime))
            shutil.move(os.path.join(filepath, f'{filename}.sav'), os.path.join(filepath, str_datetime))


if __name__ == "__main__":
 
    PORT = 65432 
    # HOST = '127.0.0.1'
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sckt:
        HOST = socket.gethostname()
        sckt.bind((HOST, PORT))
        sckt.settimeout(0.5)
        print(f'{HOST}:{PORT} started!')
        try:
            while True:
                try:
                    sckt.listen()
                    conn, addr = sckt.accept()
                    with conn:
                        data = conn.recv(131072)
                        data = json.loads(data.decode())
                        if data['type'] == 'baseline':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            baseline_extract_features_phase2(subj_id, experiment_id, baseline_id, baseline_path)
                            print('Ready for command...')
                        elif data['type'] == 'low':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, 'low')
                            print('Ready for command...')
                        elif data['type'] == 'high':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, 'high')
                            print('Ready for command...')
                        elif data['type'] == 'low_sess1':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            sess_id = data['sess_id']
                            classifier_id = data['classifier_id']
                            sess1_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\{}".format(subj_id, experiment_id, sess_id)
                            extract_features_with_overlap_from_sess1(sess1_path, subj_id, experiment_id, 'low')
                            print('Ready for command...')
                        elif data['type'] == 'high_sess1':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            sess_id = data['sess_id']
                            classifier_id = data['classifier_id']
                            sess1_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\{}".format(subj_id, experiment_id, sess_id)
                            extract_features_with_overlap_from_sess1(sess1_path, subj_id, experiment_id, 'high')
                            print('Ready for command...')
                        elif data['type'] == 'train':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            print(f'Train only {subj_id} {experiment_id}')
                            model_name = f'classifier_{classifier_id}'
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            mkdir_backup_if_exists(save_model_path, model_name)
                            train(highlowfile_path, subj_id, baseline_path, baseline_id, save_model_path, model_name)
                            print('Ready for command...')
                        elif data['type'] == 'train_temp':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            print(f'Train only {subj_id} {experiment_id}')
                            model_name = f'classifier_{classifier_id}'
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            mkdir_backup_if_exists(save_model_path, model_name)
                            train_temp(highlowfile_path, subj_id, baseline_path, baseline_id, save_model_path, model_name)
                            print('Ready for command...')
                        elif data['type'] == 'train_sess1':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            sess_id = data['sess_id']
                            model_name = f'classifier_{classifier_id}_sess1'
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\{}".format(subj_id, experiment_id, sess_id)
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            if not os.path.exists(save_model_path):
                                os.makedirs(save_model_path)
                            else:
                                warnings.warn('classifier folder already exists!')
                            train(highlowfile_path, subj_id, baseline_path, baseline_id, save_model_path, model_name)
                            print('Ready for command...')
                        elif data['type'] == 'high_and_train':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, 'high')
                            ############# training ########################
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            model_name = f'classifier_{classifier_id}'
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            mkdir_backup_if_exists(save_model_path, model_name)
                            train(highlowfile_path, subj_id, baseline_path, baseline_id, save_model_path, model_name)
                            print('Ready for command...')
                        elif data['type'] == 'high_and_train_sess1':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            sess_id = data['sess_id']
                            baseline_id = data['baseline_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\{}".format(subj_id, experiment_id, sess_id)
                            extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, 'high')
                            ############# training ########################
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            model_name = f'classifier_{classifier_id}_sess1'
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            if not os.path.exists(save_model_path):
                                os.makedirs(save_model_path)
                            else:
                                warnings.warn('classifier folder already exists!')
                            train(highlowfile_path, subj_id, baseline_path, baseline_id, save_model_path, model_name)
                            print('Ready for command...')
                        elif data['type'] == 'high_and_train_normalized_on_low':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            extract_features_with_overlap_phase2(highlowfile_path, subj_id, experiment_id, 'high')
                            ############# training ########################
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            model_name = f'classifier_{classifier_id}'
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            mkdir_backup_if_exists(save_model_path, model_name)
                            train_noralized_by_low(highlowfile_path, save_model_path, model_name)
                            print('Ready for command...')               
                        elif data['type'] == 'train_normalized_on_low':
                            subj_id = data['subj_id']
                            experiment_id = data['experiment_id']
                            classifier_id = data['classifier_id']
                            highlowfile_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\low_high".format(subj_id, experiment_id)
                            experiment_id = data['experiment_id']
                            baseline_id = data['baseline_id']
                            baseline_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\baseline".format(subj_id, experiment_id)
                            model_name = f'classifier_{classifier_id}'
                            save_model_path = r"\\FEAS-963PHMM\Users\behnam\Documents\MATLAB\realtime_datacollection\App\data_store\{}\{}\classifier".format(subj_id, experiment_id)
                            mkdir_backup_if_exists(save_model_path, model_name)
                            train_noralized_by_low(highlowfile_path, save_model_path, model_name)
                            print('Ready for command...')                                        
                        else:
                            print('Unrecognizable command')

                except socket.timeout:
                    pass
                if keyboard.is_pressed('q'):
                    sys.exit()
        except Exception as e:
            print(e)