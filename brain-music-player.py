import sys
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import queue
import numpy as np
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5 import uic
from PyQt5.QtCore import pyqtSlot
import wave, pyaudio
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
from multiprocessing import Process, Queue,set_start_method
import live_matplot_funcs
from os import path
import btm

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
        CHUNK = self.CHUNK
        
        wf = wave.open(self.tmpfile, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        frames_per_buffer=CHUNK)
        self.samplerate = wf.getframerate()
        while(self.music_on):
            
            QtWidgets.QApplication.processEvents()    
            data = wf.readframes(CHUNK)
            audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
            audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
            # Normalise float32 array                                                   
            max_int16 = 2**15
            audio_normalised = audio_as_np_float32 / max_int16
            
            self.mq.put_nowait(audio_normalised)
            stream.write(data)
            
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
        with self.mq.mutex:
            self.mq.queue.clear()
        

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
        try:
            
            while self.music_on:
                
                QtWidgets.QApplication.processEvents()    
                try: 
                    self.mdata = self.mq.get_nowait()
                    
                except queue.Empty:
                    break
                
                shift = len(self.mdata)
                
                self.musicdata = np.roll(self.musicdata, -shift,axis = 0)
                
                self.musicdata = self.mdata
                self.ydata = self.musicdata[:]
                
                self.mp.axes.set_facecolor((0,0,0))
                
      
                if self.mreference_plot is None:
                    plot_refs = self.mp.axes.plot( self.ydata, color=(0,1,0.29))
                    self.mreference_plot = plot_refs[0]    
                else:
                    self.mreference_plot.set_ydata(self.ydata)
                                
            self.mp.axes.yaxis.grid(True,linestyle='--')
            start, end = self.mp.axes.get_ylim()
            self.mp.axes.yaxis.set_ticks(np.arange(start, end, 0.5))
            self.mp.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.mp.axes.set_ylim( ymin=-1, ymax=1)        

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

    stream_process = Process(target=live_matplot_funcs.sendingData)
    stream_process.start()
    
    if stream_process.is_alive():
        print("streaming data...")
    
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = BRAIN_MUSIC_PLAYER()
    mainWindow.show()
    sys.exit(app.exec_())