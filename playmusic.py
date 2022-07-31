# apt-get install libasound2-dev
# sudo apt-get install libjack0 libjack-dev jackd2
import time

import numpy as np
import pygame

from mido import tick2second, MidiFile
from copy import deepcopy
from io import BytesIO

mid = MidiFile('./data/Never-Gonna-Give-You-Up-2.mid')
BUFFER_LEN = 5  # secs
cnt = 0
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
time.sleep(BUFFER_LEN)
while True:
    tstart = time.time()
    #pull a chunk of eeg data from lsl stream
    samples, timestamps = inlet.pull_chunk(timeout=.5, max_samples=CHUNK)
    #check that samples contains values

    if len(samples) > 0:
        #samples is a length <= 250 list of lists which contain one value of the eeg stream
        #convert it to a shape (1,250) array (1 channel, 250 timesteps)
        chunk_data = np.vstack(samples).T
        channel_1 = chunk_data[0] #get a shape (250,) array of data from the first (and only) channel

        #update data array
        data = update_data_array(buff,channel_1)

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
                    cnt += 1
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
        if past_msg_itr >= tot_msgs: break
        time.sleep(BUFFER_LEN - (time.time() - tstart))
