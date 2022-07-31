import queue
import sys
import time
import wave
from copy import deepcopy
from io import BytesIO
from multiprocessing import Process, Queue, set_start_method

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pygame
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure
from mido import MidiFile, tick2second
from pylsl import StreamInfo, StreamInlet, StreamOutlet, resolve_byprop
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot

import live_matplot_funcs
from os import path
import btm

matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()

class BRAIN_MUSIC_PLAYER(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        ui_path = path.join('UI', 'temp.ui')
        self.ui = uic.loadUi(path,self)
        self.resize(888, 600)
        
        # LSL stream
        #get and print list of detected lsl streams
        streams = resolve_byprop('type', 'EEG', timeout=2)
        print(streams)
        #choose the stream we want, since there is only one it'll be the first
        stream = streams[0]
        #number of samples per second 
        sample_rate = 250 
        #the object that lets us pull data from the stream 
        self.inlet = StreamInlet(stream, max_chunklen = sample_rate)
        
        #Flags
        self.music_on = False
        self.plot_on = False

        self.threadpool = QtCore.QThreadPool()    
        # self.threadpool.setMaxThreadCount(2)
        self.CHUNK = 250
        # self.mq = queue.Queue(maxsize=self.CHUNK)
        self.pq = Queue(maxsize=self.CHUNK)
        self.mq = Queue(maxsize=self.CHUNK)
        
        self.features_list= ['feature1','feature2','feature3']
        self.tmpfile = 'temp.wav'
        
        self.comboBox.addItems(self.features_list)
        self.comboBox.setCurrentIndex(0)
        
        # data canvas
        self.canvas = MplCanvas(self, width=5, height=1.5, dpi=100)
        self.ui.gridLayout.addWidget(self.canvas, 0, 0, 1, 1)
        self.preference_plot = None
        
        # music player
        self.mp = MplCanvas(self, width=5, height=1.5, dpi=100)
        self.ui.gridLayout.addWidget(self.mp, 2, 0, 1, 1)
        self.mreference_plot = None
        
        self.plotdata =  np.zeros(500)
        self.musicdata = np.zeros((500,500))
        
        # music timer
        self.music_timer = QtCore.QTimer()
        self.music_timer.setInterval(30) #msec
        self.music_timer.timeout.connect(self.update_music)
        self.music_timer.start()
        self.mdata=[0]
        self.feature = self.features_list[0]
        
        # data plot timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30) #msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        self.pdata = [0]
        
        # Button Events
        self.pushButton.clicked.connect(self.start_music)
        self.pushButton_3.clicked.connect(self.stop_music)
        self.pushButton_2.clicked.connect(self.start_plot)
        
        # Combobox Events
        self.comboBox.currentIndexChanged['QString'].connect(self.update_feature)
        
        # Checkbox Events
        self.checkBox.stateChanged.connect(self.update_channel)
        self.checkBox_2.stateChanged.connect(self.update_channel)
        self.checkBox_3.stateChanged.connect(self.update_channel)
        self.checkBox_4.stateChanged.connect(self.update_channel)

    def getData(self):
        QtWidgets.QApplication.processEvents()    
        CHUNK = self.CHUNK

        while(self.plot_on):
            QtWidgets.QApplication.processEvents()    
            samples,time = self.inlet.pull_chunk(timeout=.5, max_samples=CHUNK)
           
            self.pq.put_nowait(samples)
            
            if self.plot_on is False:
                break
    
        self.pushButton_2.setEnabled(True)
        
    def getAudio(self):
        QtWidgets.QApplication.processEvents()    

        mid = MidiFile('../data/Never-Gonna-Give-You-Up-2.mid')
        BUFFER_LEN = 5  # secs
        past_dur = 0  # secs
        past_msg_itr = 0
        tempo = None
        meta_mid = MidiFile(type=mid.type, ticks_per_beat=mid.ticks_per_beat)
        meta_mid.add_track(mid.tracks[0].name)
        for msg in mid.tracks[0]:
            if msg.is_meta:
                if msg.type != 'track_name':
                    meta_mid.tracks[0].append(msg)
            else:
                break

        #the max number of samples pulled from lsl stream
        srate = 250
        CHUNK = srate*BUFFER_LEN

        buff = np.zeros(srate*BUFFER_LEN)*8 #init array that will hold 2 seconds (500 samples) of eeg data to display
        #init color map which converts a numeric value to rgb color, the range of the value is between 0 and 1

        pygame.mixer.init()

        tot_msgs = len(mid.tracks[0])
        tstart = time.time()
        # time.sleep(BUFFER_LEN)
        while(self.music_on):
            
            QtWidgets.QApplication.processEvents()
            tstart = time.time()
            #pull a chunk of eeg data from lsl stream
            samples = self.plotdata
            #check that samples contains values

            if len(samples) > 0:
                #samples is a length <= 250 list of lists which contain one value of the eeg stream
                #convert it to a shape (1,250) array (1 channel, 250 timesteps)
                chunk_data = np.vstack(samples).T
                channel_1 = chunk_data[0] #get a shape (250,) array of data from the first (and only) channel

                #update data array
                data = live_matplot_funcs.update_data_array(buff,channel_1)

                dur = past_dur
                subset_midi = deepcopy(meta_mid)
                for itr in range(past_msg_itr, tot_msgs):
                    msg = mid.tracks[0][itr]
                    past_msg_itr += 1
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    if (
                        not msg.is_meta
                    ):
                        if msg.type == 'note_on':
                            msg.velocity  # ranges from 0-127
                            msg.velocity = np.clip((np.random.randint(-32, 32) + msg.velocity,), 30, 115)[0]
                        # https://music.stackexchange.com/questions/86241/how-can-i-split-a-midi-file-programatically
                        curr_time = tick2second(msg.time, mid.ticks_per_beat, tempo)
                        if dur + curr_time - past_dur > BUFFER_LEN:
                            past_dur = dur
                            break
                        dur += curr_time
                        if dur >= past_dur:
                            subset_midi.tracks[0].append(msg)
                subset_midi.tracks[0].append(mid.tracks[0][-1])

                bytestream = BytesIO()
                subset_midi.save(file=bytestream)
                bytestream.seek(0)
                pygame.mixer.music.load(bytestream)
                pygame.mixer.music.play()
                if past_msg_itr >= tot_msgs:
                    self.music_on = False
                time.sleep(BUFFER_LEN - (time.time() - tstart))
            if self.music_on is False:
                break

        self.pushButton.setEnabled(True)
        
    def start_plot(self):
            
        self.pushButton_2.setEnabled(False)
        self.canvas.axes.clear()
        self.plot_on = True
        self.plot_worker = Worker(self.start_plot_stream, ) 
        self.threadpool.start(self.plot_worker)    
        self.preference_plot = None
        self.timer.setInterval(30) #msec

    def start_music(self):
        
        self.pushButton.setEnabled(False)
        self.mp.axes.clear()
        self.music_on = True
        self.music_worker = Worker(self.start_music_stream, )
        self.threadpool.start(self.music_worker)    
        self.mreference_plot = None
        self.music_timer.setInterval(30) #msec
            
    def stop_music(self):
        
        self.music_on = False
        # with self.mq.mutex:
        #     self.mq.queue.clear()
        

    def start_music_stream(self):
        self.getAudio()
        
    def start_plot_stream(self):
        self.getData()

    def update_feature(self,value):
        self.feature = self.features_list.index(value)
        print(self.feature)
        
    def update_channel(self,state):
        if state == QtCore.Qt.Checked:
            print('Checked')
        else:
            print('Unchecked')

    def update_plot(self):
        try:
            
            while self.plot_on:
                
                QtWidgets.QApplication.processEvents()    
                try: 
                    self.pdata = self.pq.get_nowait()
                    
                except queue.Empty:
                    break
                
                chunk_data = np.vstack(self.pdata).T
                new_data = chunk_data[0] #get a shape (250,)
                
                self.plotdata = live_matplot_funcs.update_data_array(self.plotdata, new_data)
                
                self.plotdata[ -len(new_data) : ] = new_data
      
                if self.preference_plot is None:
                    plot_refs = self.canvas.axes.plot( self.plotdata, color=(0,1,0.29))
                    self.preference_plot = plot_refs[0]    
                else:
                    self.preference_plot.set_ydata(self.plotdata)
                    
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            start, end = self.canvas.axes.get_ylim()
            self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
            self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.canvas.axes.set_ylim( ymin=-1, ymax=1)        

            self.canvas.draw()
        except Exception as e:
            pass
            # print(e)
    
    def update_music(self):
        pass
        # try:
            
        #     while self.music_on:
                
        #         QtWidgets.QApplication.processEvents()    
        #         try: 
        #             # self.mdata = self.mq.get_nowait()
        #             self.self.pdata = self.pq.get_nowait()
                    
        #         except queue.Empty:
        #             break

        #         chunk_data = np.vstack(self.pdata).T
        #         new_data = chunk_data[0] #get a shape (250,)
                
        #         self.plotdata = live_matplot_funcs.update_data_array(self.plotdata, new_data)
                
        #         # self.musicdata = np.roll(self.musicdata, -shift,axis = 0)
        #         # self.musicdata = self.mdata
        #         # self.ydata = self.musicdata[:]
        #         # self.mp.axes.set_facecolor((0,0,0))      
        #         # if self.mreference_plot is None:
        #         #     plot_refs = self.mp.axes.plot( self.ydata, color=(0,1,0.29))
        #         #     self.mreference_plot = plot_refs[0]    
        #         # else:
        #         #     self.mreference_plot.set_ydata(self.ydata)         
        #     # self.mp.axes.yaxis.grid(True,linestyle='--')
        #     # start, end = self.mp.axes.get_ylim()
        #     # self.mp.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
        #     # self.mp.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        #     # self.mp.axes.set_ylim( ymin=-1, ymax=1)        
        #     # self.mp.draw()
        # except Exception as e:
            
        #     pass


class Worker(QtCore.QRunnable):

    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):

        self.function(*self.args, **self.kwargs)     

if __name__ == '__main__':

    stream_process = Process(target=live_matplot_funcs.sendingData)
    stream_process.start()
    
    if stream_process.is_alive():
        print("streaming data...")
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = BRAIN_MUSIC_PLAYER()
    mainWindow.show()
    sys.exit(app.exec_())
