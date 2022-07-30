import numpy as np
from pylsl import StreamInlet, resolve_byprop 

class BTM:
    def __init__(self):
        self.buffer_len = 5
        self.len_epoch = 1
        self.epoch_overlap = 0.75
        self.freqs = 0

    #### running functions
    def connect(self, sample_rate):
        print("Connecting to device")
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('No EEG stream')

        print("Acquiring data")
        self.stream_in = StreamInlet(streams[0], max_chunklen=sample_rate)
        eeg_time_corr = self.stream_in.time_correction()
        return self.stream_in, eeg_time_corr

    def init_buffer(self):
        stream_info = self.stream_in.info()
        self.freqs = int(stream_info.nominal_srate())
        self.eeg_buffer = np.zeros((int(self.freqs * self.buffer_len), 1))
        return self.eeg_buffer


    def run(self, channels):
        eeg_buffer = self.init_buffer()
        try:
            while True:
                eeg_data, timestamp = self.stream_in.pull_chunk(
                    timeout=1, max_samples=int((self.len_epoch - self.epoch_overlap) * self.freqs))

                ch_data = np.array(eeg_data)[:, channels]

                eeg_buffer = self.update_buffer(eeg_buffer, ch_data)
                print(eeg_buffer)
        except Exception as e:
            print(e)
            print("Ending")

    #### Helper functions
    def get_buffer_len(self):
        return self.buffer_len

    def set_channel(self, ch):
        self.channel = ch

    def update_buffer(self, data_buffer, new_data):
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, data_buffer.shape[1])

        new_buffer = np.concatenate((data_buffer, new_data), axis=0)
        new_buffer = new_buffer[new_data.shape[0]:, :]

        return new_buffer
        

if __name__ == "__main__":
    
    btm = BTM()
    stream_in, __ = btm.connect(250)
    btm.init_buffer()
    btm.run([0])

