clear all
close all

session_data.type = 'baseline';
subj_id = 8;
experiment_id = 1;
baseline_id = 1;
sess1_id = 'A1';
sess2_id = 'A2';
classifier_id = 1;
%%
current_sess_id = sess1_id;
%% Remote servers' addresses
LOCAL_HOST_ADDRESS = '127.0.0.1';
REMOTE_HOST_ADDRESS = '192.168.50.15';
PORT = 65432;
%% send the info to the python program
session_data.type = 'start';
session_data.subj_id = subj_id;
session_data.experiment_id =  experiment_id;
session_data.baseline_id = baseline_id;
session_data.sess_id = current_sess_id;
session_data.classifier_id = classifier_id;
message = jsonencode(session_data);
t = tcpclient(LOCAL_HOST_ADDRESS,PORT);
write(t, message)

%%
ecg_array = zeros(0, 5);
eda_array = zeros(0, 2);
eda_index = 1;
ecg_index = 1;

%% folder and files to store data
% data_path = 'data_store\'; %% make sure you have the directory created.
% fileNameEcg = strcat(data_path, 'ecg_', num2str(subj_id), '_', num2str(sess_id), '.csv');
% fileNameEda = strcat(data_path, 'eda_', num2str(subj_id), '_', num2str(sess_id), '.csv');
fileNameEcg = sprintf('C:\\Users\\behnam\\Documents\\MATLAB\\realtime_datacollection\\App\\data_store\\%d\\%d\\%s\\ecg.csv', subj_id, experiment_id, current_sess_id);
fileNameEda = sprintf('C:\\Users\\behnam\\Documents\\MATLAB\\realtime_datacollection\\App\\data_store\\%d\\%d\\%s\\eda.csv', subj_id, experiment_id, current_sess_id);
%% num of samples assignment
numSamplesecg = 0;
numSamplesgsr = 0;
%% set sampling rate
ECGSamplingRate = 512;
GSRSamplingRate = 128;
%% to stop the script, while no time is mentioned as duration
ecg_data_table = readtable(fileNameEcg);
eda_data_table = readtable(fileNameEda);
% remove the first and the last minutes
% ecg_data_table = ecg_data_table(1:6 * 60 * 512,:);
% eda_data_table = eda_data_table(1:6 * 60 * 128,:);
% ecg_data_table = ecg_data_table(60 * 512:end-60*512,:);
% eda_data_table = eda_data_table(60 * 128:end-60*128,:);

new_data_len = 10; %randi([1, 3],1);
ecg_data_len = new_data_len * 512 / 2;
eda_data_len = new_data_len * 128 /2;
tic
while (ecg_index + ecg_data_len <= size(ecg_data_table, 1))
   
    ecgdata = ecg_data_table(ecg_index: ecg_index + ecg_data_len-1, :);
    edadata = eda_data_table(eda_index: eda_index + eda_data_len-1, :);
    ecg_index = ecg_index + ecg_data_len;
    eda_index = eda_index + eda_data_len;
    ecg_array = [ecg_array; ecgdata{:, 2:end}];
    eda_array = [eda_array; edadata{:, 2:end}];

    if (size(ecg_array, 1) >= ecg_data_len * 2 && size(eda_array, 1) >=eda_data_len * 2)
        eda_array_to_write = [eda_array(:, 1), 1000./eda_array(:, 2)];
        writematrix(ecg_array, sprintf('C:\\Users\\behnam\\Documents\\MATLAB\\realtime_datacollection\\App\\data_store\\%d\\1\\temp\\ecg_data_10s_%s.csv', subj_id, current_sess_id))
        writematrix(eda_array_to_write, sprintf('C:\\Users\\behnam\\Documents\\MATLAB\\realtime_datacollection\\App\\data_store\\%d\\1\\temp\\eda_data_10s_%s.csv',subj_id, current_sess_id))
        ecg_array = ecg_array(end-(5119-10*512 /2):end, :);
        eda_array = eda_array(end-(1279-10*128 /2):end, :);
        session_data.type = 'data_ready';
        t = tcpclient(LOCAL_HOST_ADDRESS, PORT);
        write(t, jsonencode(session_data))

    end
    pause(3.5)
end
session_data.type = 'done';
t = tcpclient(LOCAL_HOST_ADDRESS, PORT);
write(t, jsonencode(session_data))

toc;                               % Stop timer and add to elapsed time
