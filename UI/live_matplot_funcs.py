import numpy as np
import time
from pylsl import StreamInfo, StreamOutlet
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop
import matplotlib as mp
from multiprocessing import Process, Queue,set_start_method

def generateNoisyWave(times, freq, amp, noise):
    
    # This simplifies code later, this basically just creates our noise for us
    if(not isinstance(times, float)):
        noiseArray = noise * np.random.randn(len(times))
    else:
        noiseArray = noise * np.random.randn(1)
    
    
    sineWave = amp * np.sin(freq * 2 * np.pi * times)
    return sineWave + noiseArray

def sendingData():
    '''
    Sends simulated EEG data to lsl stream
    the alpha:beta ratio alternates high and low
    every 5 seconds
    '''
    # initialize info for lsl stream
    info = StreamInfo('BioSemi', 'EEG', 1, 250, 'float32','ff05:113d:6fdd:2c17:a643:ffe2:1bd1:3cd2')
    # next make an outlet to push the eeg samples to
    outlet = StreamOutlet(info)
    # array of 250 timesteps between 0 and 1 
    tsteps = np.linspace(0, 1, 250)

    focused = (0.1 , 0.3) #low alpha amp, high beta amp
    unfocused = (0.5,0.2) #high alpha amp, low beta amp
 
    alpha_amp,beta_amp = focused
    attention = True

    count = 0
    while True:
        #change focus to unfocused or vice versa, every 10 seconds
        if count%(250*10) == 0:
            if [alpha_amp,beta_amp] == focused:
                alpha_amp,beta_amp = unfocused
            else:
                alpha_amp,beta_amp = focused
            
        t = tsteps[count % 250] #value used to make the waves
       
        #get current value of 8hz alpha wave and 30hz beta wave
        alpha = generateNoisyWave(t,8,alpha_amp,noise=0.07) 
        beta = generateNoisyWave(t,30,beta_amp,noise=0.05) 
        

        #sum together the alpha and beta valies
        current_sample = alpha+beta
        #just incase we want to change the number of channels 
        # print([current_sample])
        outlet.push_sample([current_sample])

        count+=1 #increment count

        time.sleep(1/250) #wait 
        
def update_data_array(data,new_data):
    data = np.roll(data,-len(new_data)) #roll data to left
    data[-len(new_data):] = new_data #add new data to right
    return data

