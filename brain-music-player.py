#%% Importing libraries
import queue
import sys
import time
from copy import deepcopy
from io import BytesIO
from multiprocessing import Queue

import matplotlib
import numpy as np
import pygame
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas
)
from matplotlib.figure import Figure
from mido import MidiFile, tick2second, Message
from PyQt5 import QtCore, QtWidgets, uic
from PyQt5.QtCore import pyqtSlot

import UI.live_matplot_funcs as live_matplot_funcs
import feature_extract
from os import path
from btm import BTM

matplotlib.use('Qt5Agg')
#%% 
# Plotting Class using Matplot
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        """
        Adding the matplotlib.pyplot plot to the widget in the UI
        Parameters
        ----------
        parent : instance, required
            Class Instance. The default is None.
        width : INT, optional
            The width of the plot in inches. The default is 5.
        height : INT, optional
            The height of the plot in inches. The default is 4.
        dpi : INT, optional
            The resolution of the figure in dots-per-inch. The default is 100.

        Returns
        -------
        Matplot Figure

        """
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()

# Music Player Class
class BRAIN_MUSIC_PLAYER(QtWidgets.QMainWindow):
    def __init__(self):
        
        QtWidgets.QMainWindow.__init__(self)
        
        # Loading the UI file
        ui_path = path.join('UI', 'temp.ui')
        self.ui = uic.loadUi(ui_path,self)
        self.resize(888, 600)
        
        # Initializing the data streaming class
        self.btm = BTM()
        self.btm.connect(5)

        #Flags
        self.music_on = False
        self.plot_on = False

        self.threadpool = QtCore.QThreadPool()    
        
        self.CHUNK = 250
        
        self.pq = Queue(maxsize=self.CHUNK)
        self.mq = Queue(maxsize=self.CHUNK)
        
        self.features_list= ['None','Variance','Alpha/Beta']
        self.feature = 0
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
        self.mbuffer = []
        
        self.plotdata =  np.zeros(500)
        self.musicdata = np.zeros((500,500))
        
        # music timer
        self.music_timer = QtCore.QTimer()
        self.music_timer.setInterval(0) #msec
        self.music_timer.timeout.connect(self.update_music)
        self.music_timer.start()
        self.mdata=[0]
        # self.feature = self.features_list[0]
        
        # data plot timer
        self.timer = QtCore.QTimer()
        self.timer.setInterval(30) #msec
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        # self.pdata = [0]
        
        # Button Events
        self.pushButton.clicked.connect(self.start_music)
        self.pushButton_3.clicked.connect(self.stop_music)
        self.pushButton_2.clicked.connect(self.start_plot)
        self.pushButton_4.clicked.connect(self.stop_plot)
        
        # Combobox Events
        self.comboBox.currentIndexChanged['QString'].connect(self.update_feature)
        
        # Radiobuttons for selecting the channels
        self.radioButton.toggled.connect(lambda:self.update_channel(self.radioButton))
        self.radioButton_2.toggled.connect(lambda:self.update_channel(self.radioButton_2))
        self.radioButton_3.toggled.connect(lambda:self.update_channel(self.radioButton_3))
        self.radioButton_4.toggled.connect(lambda:self.update_channel(self.radioButton_4))

        # feature value
        self.label_3.setText("0")

    def getData(self):
        """
        Takes out samples of data from the Muse device and feed it into the queue

        Returns
        -------
        None.

        """
        QtWidgets.QApplication.processEvents()    
        # Starts taking in data when "start" button is pressed
        while(self.plot_on):
            QtWidgets.QApplication.processEvents()    
            
            samples = self.btm.stream_update()
            # print("getData" , self.plot_on)
            self.pq.put_nowait(samples)
            
            if self.plot_on is False:
                break
    
        self.pushButton_2.setEnabled(True)

    def getAudio(self):
        """
        Function to get the modified audio

        Returns
        -------
        None.

        """
        QtWidgets.QApplication.processEvents()    
        midi_path = path.join('data', 'Never-Gonna-Give-You-Up-2.mid')
        mid = MidiFile(midi_path)
        BUFFER_LEN = self.btm.buffer_len  # secs
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

        pygame.mixer.init()

        tot_msgs = len(mid.tracks[0])
        tstart = time.time()
        # time.sleep(BUFFER_LEN)
        LOW, HIGH = 0, 127
        while(self.music_on):
            
            QtWidgets.QApplication.processEvents()
            tstart = time.time()

            modifier = 0
            if self.feature == 1:
                modifier = feature_extract.get_one_feature(
                    self.btm.eeg_buffer,
                    "variance",
                    self.btm.freqs
                )
            elif self.feature == 2:
                modifier = feature_extract.get_one_feature(
                    self.btm.eeg_buffer,
                    "a_to_b",
                    self.btm.freqs
                )
                # ab_ratio = feats['alpha'] / feats['beta']
                # print(ab_ratio)
            
            if len(self.mbuffer) > 20:
                self.mbuffer.pop(0)
            self.mbuffer.append(modifier)

            self.label_3.setText(str(modifier))
            dur = past_dur
            subset_midi = deepcopy(meta_mid)
            for itr in range(past_msg_itr, tot_msgs):
                msg = mid.tracks[0][itr]
                past_msg_itr += 1
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                if not msg.is_meta:
                    if msg.type in ('note_on', 'note_off') and self.feature > 0:
                        msg.velocity  # ranges from 0-127
                        # dev = np.random.normal(scale=var)
                        # subset_midi.tracks[0].append(Message(
                        #     'pitchwheel',
                        #     channel=0,
                        #     pitch=round(min(max(dev, LOW), HIGH)),
                        #     time=msg.time
                        # ))
                        mod = ('velocity', 'note')[1]
                        if mod == 'velocity':
                            msg.velocity = (100*round(modifier)) % HIGH
                        elif mod == 'note':
                            msg.note = msg.note + round(modifier) % HIGH
                            # round(min(max(dev + msg.velocity, LOW), HIGH))
                    # https://music.stackexchange.com/questions/86241/how-can-i-split-a-midi-file-programatically
                    curr_time = tick2second(msg.time, mid.ticks_per_beat, tempo)
                    if dur + curr_time - past_dur > BUFFER_LEN:
                        past_dur = dur
                        break
                    dur += curr_time
                    subset_midi.tracks[0].append(msg)
            subset_midi.tracks[0].append(mid.tracks[0][-1])

            bytestream = BytesIO()
            subset_midi.save(file=bytestream)
            bytestream.seek(0)
            pygame.mixer.music.load(bytestream)
            pygame.mixer.music.play()
            if past_msg_itr >= tot_msgs:
                self.music_on = False
            if self.music_on is False:
                break
            time.sleep(BUFFER_LEN - (time.time() - tstart))

        self.pushButton.setEnabled(True)
        
    def start_plot(self):
        """
        Initializing the variables for plotting the live EEG stream

        Returns
        -------
        None.

        """
        self.btm.init_buffer()

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
        self.music_timer.setInterval(0) #msec
            
    def stop_music(self):
        """
        Function to stop playing the background music

        Returns
        -------
        None.

        """
        self.music_on = False
        # with self.mq.mutex:
        #     self.mq.queue.clear()
        
    def stop_plot(self):
        """
        Function to stop streaming the EEG activity

        Returns
        -------
        None.

        """
        self.plot_on = False
        self.btm.init_buffer()
        self.mbuffer = []

    def start_music_stream(self):
        self.getAudio()
        
    def start_plot_stream(self):
        self.getData()

    def update_feature(self,value):
        self.feature = self.features_list.index(value)
        print(self.feature)
        

    def update_channel(self,button):
        """
        Function to update the channel based on selection of specific radiobutton

        Parameters
        ----------
        button : PyQT RadioButton
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if button.text() == "TP9":
            if button.isChecked():
                self.btm.set_channel([0])
                self.btm.init_buffer()
                self.mbuffer = []
        elif button.text() == "AF7":
            if button.isChecked():
                self.btm.set_channel([1])
                self.btm.init_buffer()
                self.mbuffer = []
        elif button.text() == "AF8":
            if button.isChecked():
                self.btm.set_channel([2])
                self.btm.init_buffer()
                self.mbuffer = []
        elif button.text() == "TP10":
            if button.isChecked():
                self.btm.set_channel([3])
                self.btm.init_buffer()
                self.mbuffer = []



    def update_plot(self):
        """
        Function to plot the live streaming EEG data

        Returns
        -------
        None.

        """
        try:
            while self.plot_on:
                
                QtWidgets.QApplication.processEvents()    
                # try: 
                #     self.pdata = self.pq.get_nowait()

                # except queue.Empty:
                #     break

                # print("update_plot", self.plot_on)
                self.plotdata = self.btm.eeg_buffer
                if self.preference_plot is None:
                    plot_refs = self.canvas.axes.plot(self.plotdata, color=(0,1,0.29))
                    self.preference_plot = plot_refs[0]    
                else:
                    self.preference_plot.set_ydata(self.plotdata)

                if self.plot_on is False:
                    break
                    
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            # start, end = self.canvas.axes.get_ylim()
            # self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
            # self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

            self.canvas.axes.set_ylim( ymin=-10, ymax=10)        

            self.canvas.draw()
        except Exception as e:
            pass
            # print(e)
    
    def update_music(self):
        # pass 
        try:
            while self.music_on:
                
                QtWidgets.QApplication.processEvents()    
                # try: 
                #     # self.mdata = self.mq.get_nowait()
                #     self.self.pdata = self.pq.get_nowait()
                    
                # except queue.Empty:
                #     break

                # chunk_data = np.vstack(self.pdata).T
                # new_data = chunk_data[0] #get a shape (250,)
                
                self.mplotdata = self.mbuffer
                
                # self.musicdata = np.roll(self.musicdata, -shift,axis = 0)
                # self.musicdata = self.mdata
                # self.ydata = self.musicdata[:]
                # self.mp.axes.set_facecolor((0,0,0))      
                if self.mreference_plot is None:
                    plot_refs = self.mp.axes.plot( self.mplotdata, color=(0,1,0.29))
                    self.mreference_plot = plot_refs[0]    
                else:
                    self.mreference_plot.set_ydata(self.mplotdata)  

            self.mp.axes.yaxis.grid(True,linestyle='--')
            # start, end = self.mp.axes.get_ylim()
            # self.mp.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
            # self.mp.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.mp.axes.set_ylim( ymin=-1, ymax=10)        
            self.mp.draw()
        except Exception as e:
            
            pass


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

    # Initializing the PyQT Application
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = BRAIN_MUSIC_PLAYER()
    mainWindow.show()
    sys.exit(app.exec_())
