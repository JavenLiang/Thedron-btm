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


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        fig.tight_layout()

class BRAIN_MUSIC_PLAYER(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.ui = uic.loadUi('temp.ui',self)
        self.resize(888, 600)
        
        #Flags
        self.music_on = False
        self.plot_on = False

        self.threadpool = QtCore.QThreadPool()    
        self.threadpool.setMaxThreadCount(2)
        self.CHUNK = 1024
        self.mq = queue.Queue(maxsize=self.CHUNK)
        self.pq = queue.Queue()
        
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
        
        self.plotdata =  np.zeros((500,500))
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
        
        wf = wave.open(self.tmpfile, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        frames_per_buffer=CHUNK)
        self.samplerate = wf.getframerate()
        while(self.plot_on):
            
            QtWidgets.QApplication.processEvents()    
            data = wf.readframes(CHUNK)
            audio_as_np_int16 = np.frombuffer(data, dtype=np.int16)
            audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
            # Normalise float32 array                                                   
            max_int16 = 2**15
            audio_normalised = audio_as_np_float32 / max_int16
            
            self.pq.put_nowait(audio_normalised)
            stream.write(data)
            
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
            
            while  self.plot_on:
                
                QtWidgets.QApplication.processEvents()    
                try: 
                    self.pdata = self.pq.get_nowait()
                    
                except queue.Empty:
                    break
                
                shift = len(self.pdata)
                
                self.plotdata = np.roll(self.plotdata, -shift,axis = 0)
                
                self.plotdata = self.pdata
                self.ydata = self.plotdata[:]
                
                # self.mp.axes.set_facecolor((0,0,0))
                
      
                if self.preference_plot is None:
                    plot_refs = self.canvas.axes.plot( self.ydata, color=(0,1,0.29))
                    self.preference_plot = plot_refs[0]    
                else:
                    self.preference_plot.set_ydata(self.ydata)
                    

            
            self.canvas.axes.yaxis.grid(True,linestyle='--')
            start, end = self.canvas.axes.get_ylim()
            self.canvas.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
            self.canvas.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.canvas.axes.set_ylim( ymin=-1, ymax=1)        

            self.canvas.draw()
        except Exception as e:
            
            pass
    
    def update_music(self):
        try:
            
            while  self.music_on:
                
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
            self.mp.axes.yaxis.set_ticks(np.arange(start, end, 0.1))
            self.mp.axes.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
            self.mp.axes.set_ylim( ymin=-1, ymax=1)        

            self.mp.draw()
        except Exception as e:
            
            pass

# www.pyshine.com
class Worker(QtCore.QRunnable):

    def __init__(self, function, *args, **kwargs):
        super(Worker, self).__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):

        self.function(*self.args, **self.kwargs)        



app = QtWidgets.QApplication(sys.argv)
mainWindow = BRAIN_MUSIC_PLAYER()
mainWindow.show()
sys.exit(app.exec_())