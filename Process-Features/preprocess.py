# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:26:23 2022

@author: Seth Campbell

credit:
    -streaming data code inspired by "Python IV workshop" from NeurTechAlberta 
    -low pass filter from https://www.adamsmith.haus/python/answers/how-to-create-a-low-pass-filter-in-python
"""
def feature_extract(eeg_chunk, s_rate):
    """receives a chunk of eeg data and performs basic preprocessing and feature extraction
    
    input:
        eeg_chunk - numpy array of chunk of eeg data, columns as channels, rows as values in time
        s_rate - sampling rate of data
        
    output:
        feature_dict - dictionary of features. Currently only "variance" 
    """
    #%% import libraries
    import numpy as np
    #from pylsl import StreamInfo, StreamOutlet
    #from pylsl import StreamInlet, resolve_byprop
    from scipy.signal import butter, lfilter
    
    #%% init global vars
    #s_rate = 250 #temp value until confirmed
    win_size = 10 #temp value, window size for input stream
    cutoff_freq = 55 #remove freq. above 55Hz, temp value
    
    #%% setup input stream
    #streams = resolve_byprop('type', 'EEG', timeout=2) #change timeout later?
    #stream = streams[0] #should be only 1 stream (with 1 channel for now)
        
    #object to receive input form stream
    #inlet = StreamInlet(stream, max_chunklen = win_size) 

        
    #%%% preprocess signal
    #handle if stream is empty (if empty, use previous eeg chunk?)
    
    #receive a window of data from lsl stream
    #eeg_chunk, ts = inlet.pull_chunk(timeout=.5, max_samples= win_size) #timeout should be 0 to be as fast as possible?
    
    #low pass filter
    normalized_cutoff_freq = 2 * cutoff_freq / s_rate
    numerator_coeffs, denominator_coeffs = butter(5, normalized_cutoff_freq) #hardcode 5 as order for now
    feeg_chunk = lfilter(numerator_coeffs, denominator_coeffs, eeg_chunk)
    
    #remove eyeblink/jaw artifacts
    
    
    #%%% compute features
    #calc variance
    eeg_var = np.var(feeg_chunk)
    
    #normalize features(?)
    
    #gather features into dictionary
    feature_dict = {
        "variance": eeg_var        
    }
    
    return feature_dict
  