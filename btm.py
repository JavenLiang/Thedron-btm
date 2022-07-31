import numpy as np
from pylsl import StreamInlet, resolve_byprop 
from Process_Features.feature_extract import feature_extract
from scipy.signal import butter, lfilter
import csv

class BTM:
    def __init__(self):
        self.buffer_len = 20           ### length of the buffer (sec)
        self.freqs = 0                ### variable for input frequency
        self.data_chunk = 0.25        ### data chunk each time buffer is updated

    #### running functions
    def connect(self, chunk_length):
        """
        Connect to a running data stream (from muselsl or petal)

        input:
        chunk_length: max chunk length for StreamInlet, use small number for real-time app

        output:
        stream_in: data stream object
        eeg_time_corr: time correction to data stream
        """
        print("Connecting to device")
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('No EEG stream')

        print("Acquiring data")
        self.stream_in = StreamInlet(streams[0], max_chunklen=chunk_length)
        eeg_time_corr = self.stream_in.time_correction()
        return self.stream_in, eeg_time_corr

    def init_buffer(self, channels_num):
        """
        Initialize buffer

        input:
        channels_num: number of channels

        output:
        eeg_buffer: buffer np ndarray
        """
        stream_info = self.stream_in.info()
        self.freqs = int(stream_info.nominal_srate())
        self.eeg_buffer = np.zeros((int(self.freqs * self.buffer_len), channels_num))
        return self.eeg_buffer

    def run(self, channels):
        """
        Running the application

        input:
        channels: the channels to output (0, 1, 2, 3)
        """
        eeg_buffer = self.init_buffer(len(channels))
        try:
            while True:
                eeg_buffer = self.stream_update(channels)
                # print(feature_extract(eeg_buffer))
                print(eeg_buffer)

                # Enable for generating sample data
                # if eeg_buffer[0,0] != 0:
                #     with open('eegs.csv', 'w', newline='') as csvfile:
                #         for row in eeg_buffer:
                #             print
                #             spamwriter = csv.writer(csvfile, delimiter=' ')
                #             spamwriter.writerow(row)
                #     break


        except Exception as e:
            print(e)
            print("Ending")

    def stream_update(self, channels):
        eeg_data, timestamp = self.stream_in.pull_chunk(
            timeout=1, max_samples=int(self.data_chunk * self.freqs))

        ch_data = np.array(eeg_data)[:, channels]
        eeg_buffer = self.update_buffer(eeg_buffer, ch_data)
        return eeg_buffer

    #### Helper functions
    def get_buffer_len(self):
        """
        Return the lenght of buffer
        """
        return self.buffer_len

    def set_channel(self, ch):
        """
        Set the channels to output
        """
        self.channel = ch

    def update_buffer(self, data_buffer, new_data):
        """
        Updating the data buffer

        input: 
        data_buffer: current buffer to be updated
        new_data: new data to be entered into buffer
        """
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, data_buffer.shape[1])

        #%% init global vars
        cutoff_freq = 55 #remove freq. above 55Hz, temp value

        #low pass filter
        normalized_cutoff_freq = 2 * cutoff_freq / self.freqs
        numerator_coeffs, denominator_coeffs = butter(5, normalized_cutoff_freq) #hardcode 5 as order for now
        new_data = lfilter(numerator_coeffs, denominator_coeffs, new_data)

        new_buffer = np.concatenate((data_buffer, new_data), axis=0)
        new_buffer = new_buffer[new_data.shape[0]:, :]

        return new_buffer
        

if __name__ == "__main__":
    
    btm = BTM()
    stream_in, __ = btm.connect(5)
    btm.run([0, 1, 2, 3])

