from PyQt5 import QtWidgets, QtCore, QtGui, QtNetwork
import pyqtgraph as pg
import sys 
import pandas as pd
import numpy as np
import time

from pyqtgraph.graphicsItems.DateAxisItem import makeSStepper
from cogload_gradient_display import CogloadGradientDisplay
from prob_display import ProbDisplay
import json
import os 
import datetime
from predict_cogLoad import load_classifiers
from predict_cogLoad import predict_fucntion

class MainWindow(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setFixedSize(800, 600)
        
        # netowrk settings
        self.tcpServer = None
        self.PORT = 65432
        self.HOST_ADDRESS = QtNetwork.QHostAddress('127.0.0.1')
        
        # data
        self.subj_id = None
        self.experiment_id = None
        self.baseline_id = None
        self.sess_id = None
        self.classifier_id = None
        self.results_folder = None
        self.logfile_name = None
        self.baseline_path = None
        self.save_model_path = None
        self.classfier_path = None
        self.baseline_file = None
        self.baseline = None
        self.classifiers = None

        # widgets and layouts
        self.layoutV = QtWidgets.QVBoxLayout()
        self.layoutH = QtWidgets.QHBoxLayout()
        self.gradient_cogload = CogloadGradientDisplay()
        self.prob_disp = ProbDisplay()
        self.plotWidgetEcg = pg.PlotWidget(plotItem=pg.PlotItem(title='<h1 style="color:blue;">ECG LL-RA</h1>'))
        self.plotWidgetGsr = pg.PlotWidget(plotItem=pg.PlotItem(title='<h1 style="color:blue;">EDA</h1>'))
        self.cogLoadLabel = QtWidgets.QLabel() 
        self.messageArea = QtWidgets.QTextEdit()
        self.messageArea.setReadOnly(True)
        self.messages= []
        self.messageArea.setStyleSheet("background-color: black; color: green")
        self.cogLoadLabel.setFont(QtGui.QFont('Arial', 40))
        self.layoutH.addWidget(self.gradient_cogload)
        self.layoutV.addWidget(self.plotWidgetEcg)
        self.layoutV.addWidget(self.plotWidgetGsr)
        self.layoutV.addWidget(self.cogLoadLabel)
        self.layoutV.addWidget(self.prob_disp)
        self.layoutV.addWidget(self.messageArea)
        self.layoutH.addLayout(self.layoutV)
        self.setLayout(self.layoutH)
        # plotting settings
        self.pen_ecg = pg.mkPen(color=(255, 0, 255), width=2)
        self.pen_gsr = pg.mkPen(color=(0, 255, 255), width=2)
        color = self.palette().color(QtGui.QPalette.Window)
        self.plotWidgetEcg.setBackground(color)
        self.plotWidgetGsr.setBackground(color)
        self.add_message('Waiting for a connection...')
        # # test the program
        # self.i = 4
        # self.plot_test()
        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.plot_test)
        # self.timer.start(5000)

    def add_message(self, message, maxsize=4):
        if len(self.messages) == maxsize:
            self.messages = self.messages[1:]
        self.messages.append(message)
        self.display_messages()

    def display_messages(self):
        self.messageArea.setText('\n'.join(self.messages))

    # start the sever by listining to the port
    def start_server(self):
        self.tcpServer = QtNetwork.QTcpServer(self)
        if not self.tcpServer.listen(self.HOST_ADDRESS, self.PORT):
            print("Cannt listen to the port")
            self.close()
            return
        self.tcpServer.newConnection.connect(self.receive_connection)

    # get the data
    def receive_connection(self):
        # Get a QTcpSocket from the QTcpServer
        clientConnection = self.tcpServer.nextPendingConnection()
        # wait until the connection is ready to read
        clientConnection.waitForReadyRead()
        # read incomming data
        instr = clientConnection.readAll()
        # in this case we print to the terminal could update text of a widget if we wanted.
        received_command = json.loads(str(instr, encoding='ascii'))
        # get the connection ready for clean up
        clientConnection.disconnected.connect(clientConnection.deleteLater)
        # now disconnect connection.
        clientConnection.disconnectFromHost()
        # parse the data and run the model if requested
        self.run_received_command(received_command)
        

    def run_received_command(self, received_command):
        if received_command['type'] =='start':
            self.subj_id = received_command['subj_id']
            self.experiment_id = received_command['experiment_id']
            self.baseline_id = received_command['baseline_id']
            self.sess_id = received_command['sess_id']
            self.classifier_id = received_command['classifier_id']
            results_folder = "C:/Users/behnam/Documents/MATLAB/realtime_datacollection/App/data_store/{}/{}/results".format(self.subj_id, self.experiment_id)
            self.logfile_name = os.path.join(results_folder, 'log_file_session_{}.txt'.format(self.sess_id))
            self.baseline_path = "C:/Users/behnam/Documents/MATLAB/realtime_datacollection/App/data_store/{}/{}/baseline".format(self.subj_id, self.experiment_id)
            self.save_model_path = "C:/Users/behnam/Documents/MATLAB/realtime_datacollection/App/data_store/{}/{}/classifier".format(self.subj_id, self.experiment_id)
            self.classfier_path = os.path.join(self.save_model_path, f'classifier_{self.classifier_id}.sav')
            with open(self.logfile_name, 'a') as logfile:
                logfile.write('======================================\n')
                logfile.write('{}\n'.format(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
                logfile.write('Model name: {}\n'.format(self.classfier_path))
                logfile.write('received,computed,cogload,cogload_prob,comments\n')
            self.baseline_file = os.path.join(self.baseline_path, f'baseline_features_{self.baseline_id}.csv')
            self.baseline = pd.read_csv(self.baseline_file)
            self.classifiers = load_classifiers(self.classfier_path)
            self.add_message(f'Subject ID: {self.subj_id}, Experiment ID: {self.experiment_id}, Session ID: {self.sess_id}, Basline ID: {self.baseline_id}, Classifier ID: {self.classifier_id}')
        elif received_command['type'] == 'data_ready':
            timepstamp_recieved = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            time.sleep(2)
            self.add_message('Data received at {}'.format(timepstamp_recieved))
            ecg_cols = ['Timestamp', 'ECG LL-RA CAL',
                'ECG LA-RA CAL', 'dummy', 'ECG Vx-RL CAL']
            eda_cols = ['Timestamp', 'GSR Conductance CAL']
            ecg_file = f'C:/Users/behnam/Documents/MATLAB/realtime_datacollection/App/data_store/{self.subj_id}/{self.experiment_id}/temp/ecg_data_10s_{self.sess_id}.csv'
            eda_file = f'C:/Users/behnam/Documents/MATLAB/realtime_datacollection/App/data_store/{self.subj_id}/{self.experiment_id}/temp/eda_data_10s_{self.sess_id}.csv'
            try:
                ecgDF = pd.read_csv(ecg_file, skipinitialspace=True, header=None, names=ecg_cols)
                edaDF = pd.read_csv(eda_file, skipinitialspace=True, header=None, names=eda_cols)
                predicted_cog_load_prob = predict_fucntion(ecgDF, edaDF, self.baseline, subID=self.subj_id, sessID=self.sess_id, isBaseline=False, classifiers=self.classifiers)
                timestamp_computed = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                self.add_message('Cog load computed at {}'.format(timestamp_computed))
                self.update_dashboard(ecgDF, edaDF, np.argmax(predicted_cog_load_prob[0]), predicted_cog_load_prob[0])
                with open(self.logfile_name, 'a') as logfile:
                    logfile.write('{},{},{},{},\n'.format(timepstamp_recieved, timestamp_computed, np.argmax(predicted_cog_load_prob[0]),  predicted_cog_load_prob[0]))
            except Exception as e:
                print('Error: {}'.format(e))
        elif received_command['type']  == 'done':
             self.add_message('-------------------------------------')
             self.add_message('Done')
             self.add_message('-------------------------------------')
             self.messages = []
             self.add_message('Server is ready....')
        elif received_command['type']  == 'record_time':
            timepstamp_event = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            with open(self.logfile_name, 'a') as logfile:
                logfile.write(',,,,{}:{}\n'.format(timepstamp_event, received_command['message']))
                print('time recoreded at {}'.format(timepstamp_event))

    def plot_test(self):
        self.add_message(f'working on file {self.i+1}')
        self.display_messages()
        cogload = {4:1, 5:0, 6:1, 7:0}
        subjects = [4, 5, 6, 7]
        ecgDataFile = f'C:/Users/behnam/Documents/MATLAB/realtime_datacollection/ecg_data_10s_{self.i}_1.csv'
        edaDataFile = f'C:/Users/behnam/Documents/MATLAB/realtime_datacollection/eda_data_10s_{self.i}_1.csv'
        ecgDF = pd.read_csv(ecgDataFile)
        edaDF = pd.read_csv(edaDataFile)
        self.update_dashboard(ecgDF, edaDF, cogload[self.i])
        self.i += 1
        if self.i >= 7:
            self.timer.stop()
    
    def update_dashboard(self, ecgDf, gsrDF, cog_load, cog_load_prob=None):
        self.plotWidgetGsr.clear()
        self.plotWidgetEcg.clear()
        t_ecg = (ecgDf.iloc[:, 0].values - ecgDf.iloc[0, 0])/1000
        self.plotWidgetEcg.plot(t_ecg, ecgDf.iloc[:, 1].values, pen=self.pen_ecg)
        t_gsr = (gsrDF.iloc[:, 0].values - gsrDF.iloc[0, 0])/1000
        self.plotWidgetGsr.plot(t_gsr, gsrDF.iloc[:, 1].values, pen=self.pen_gsr)
        if cog_load_prob is None:
            self.cogLoadLabel.setText(f'Cognitive load: {cog_load}')
            self.prob_disp.update_cogload(0.5, 0.5)
        else:
            self.cogLoadLabel.setText(f'Cognitive load: {cog_load} | <font color="blue">{cog_load_prob[0]:.2f}</font>, <font color="red">{cog_load_prob[1]:.2f}</font>')
            self.prob_disp.update_cogload(cog_load_prob[0],cog_load_prob[0])
        self.gradient_cogload.update_cogload(cog_load=cog_load)

    

if __name__ == '__main__':
    cog_load = [7, 6, 5, 4]
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.start_server()
    main.show()
    sys.exit(app.exec_())
    