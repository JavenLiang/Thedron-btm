import numpy as np
from pylsl import StreamInlet, resolve_byprop 

class BTM:
    def __init__(self):
        self.buffer_len = 5           ### length of the buffer (sec)
        self.freqs = 0                ### variable for input frequency
        self.data_chunk = 0.25        ### data chunk each time buffer is updated

    #### running functions
    def connect(self, sample_rate):
        """
        Connect to a running data stream (from muselsl or petal)

        input:
        sample_rate: target sampling frequency

        output:
        stream_in: data stream object
        eeg_time_corr: time correction to data stream
        """
        print("Connecting to device")
        streams = resolve_byprop('type', 'EEG', timeout=2)
        if len(streams) == 0:
            raise RuntimeError('No EEG stream')

        print("Acquiring data")
        self.stream_in = StreamInlet(streams[0], max_chunklen=sample_rate)
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
                eeg_data, timestamp = self.stream_in.pull_chunk(
                    timeout=1, max_samples=int(self.data_chunk * self.freqs))

                ch_data = np.array(eeg_data)[:, channels]
                # print(ch_data)
                eeg_buffer = self.update_buffer(eeg_buffer, ch_data)
                print(eeg_buffer)


        except Exception as e:
            print(e)
            print("Ending")

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

        new_buffer = np.concatenate((data_buffer, new_data), axis=0)
        new_buffer = new_buffer[new_data.shape[0]:, :]

        return new_buffer
        

if __name__ == "__main__":
    
    btm = BTM()
    stream_in, __ = btm.connect(250)
    btm.run([0, 1, 2, 3])

