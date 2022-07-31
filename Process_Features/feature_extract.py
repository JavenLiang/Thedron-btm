# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:26:23 2022

@author: Seth Campbell

credit:
    -streaming data code inspired by "Python IV workshop" from NeurTechAlberta 
    -low pass filter from https://www.adamsmith.haus/python/answers/how-to-create-a-low-pass-filter-in-python
"""
def feature_extract(eeg_chunk):
    """receives a chunk of eeg data and performs basic preprocessing and feature extraction
    
    input:
        eeg_chunk - numpy array of chunk of eeg data, columns as channels, rows as values in time
        s_rate - sampling rate of data
        
    output:
        feature_dict - dictionary of features. Currently only "variance" 
    """
    #%% import libraries
    import numpy as np
    from scipy.signal import butter, lfilter
    
    #%% init global vars
    # cutoff_freq = 55 #remove freq. above 55Hz, temp value
        
    #%%% preprocess signal
    
    #if more than one channel, average them all
    if eeg_chunk.shape[1] > 1:
        eeg_chunk = eeg_chunk.mean(axis=1)
    
    #low pass filter
    # normalized_cutoff_freq = 2 * cutoff_freq / s_rate
    # numerator_coeffs, denominator_coeffs = butter(5, normalized_cutoff_freq) #hardcode 5 as order for now
    # feeg_chunk = lfilter(numerator_coeffs, denominator_coeffs, eeg_chunk)
    
    #remove eyeblink/jaw artifacts
    
    #%%% compute features
    #calc variance
    # eeg_var = np.var(feeg_chunk)
    eeg_var = np.var(eeg_chunk)
    
    #normalize features(?)
    
    #gather features into dictionary
    feature_dict = {
        "variance": eeg_var        
    }
    
    return feature_dict
  