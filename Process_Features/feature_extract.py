# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 13:26:23 2022

@author: Seth Campbell

credit:
    -streaming data code inspired by "Python IV workshop" from NeurTechAlberta 
    -low pass filter from https://www.adamsmith.haus/python/answers/how-to-create-a-low-pass-filter-in-python
"""
#%% import libraries
import numpy as np


def next_power_2(i):
    """Return first power of 2 that is larger than i"""
    n = 1
    while n < i:
        n *= 2
    return n


def compute_band_powers(eegdata, fs):
    """Extract band powers from EEG.
    input:
        eegdata - numpy array, rows = samples, columns = channels
        fs - sampling frequency of data
    
    output:
        band_powers - numpy array, rows = band (order: delta, theta, alpha, beta, gamma), columns = channels
    """
    
    winSampleLength, nbCh = eegdata.shape

    #smooth via hamming window
    w = np.hamming(winSampleLength) #hamming window with size of number of samples (should be close to normal buffersize if no transmission issues)
    dataWinCentered = eegdata - np.mean(eegdata, axis=0)  # center by subtracting mean 
    dataWinCenteredHam = (dataWinCentered.T * w).T #apply hamming window

    #Fourier transform 
    NFFT = next_power_2(winSampleLength) #helps speed up fft using a window with length of power of 2
    Y = np.fft.fft(dataWinCenteredHam, n=NFFT, axis=0) / winSampleLength
    PSD = 2 * np.abs(Y[0:int(NFFT / 2), :]) #power spectral density
    f = fs / 2 * np.linspace(0, 1, int(NFFT / 2)) #freq. scale

    #calc averge power in five specific bands
    ix_delta, = np.where(f < 4) 
    meanDelta = np.mean(PSD[ix_delta, :], axis=0)
    ix_theta, = np.where((f >= 4) & (f <= 8))
    meanTheta = np.mean(PSD[ix_theta, :], axis=0)
    ix_alpha, = np.where((f >= 8) & (f <= 12))
    meanAlpha = np.mean(PSD[ix_alpha, :], axis=0)
    ix_beta, = np.where((f >= 12) & (f < 30))
    meanBeta = np.mean(PSD[ix_beta, :], axis=0)
    ix_gamma, = np.where((f >= 30) & (f < 55))
    meanGamma = np.mean(PSD[ix_gamma, :], axis=0)

    #gather results
    band_powers = np.concatenate((meanDelta, meanTheta, meanAlpha, meanBeta, meanGamma), axis=0)
    #band_powers = np.log10(band_powers) #experiment with disabling log transformation

    return band_powers.reshape(5,nbCh) #rehape so that each row is a different band


def feature_extract(eeg_chunk, s_rate):
    """receives a chunk of eeg data and performs basic preprocessing and feature extraction
    
    input:
        eeg_chunk - numpy array of chunk of eeg data, columns as channels, rows as values in time
        s_rate - sampling rate of data
        
    output:
        feature_dict - dictionary of features. Currently only "variance" 
    """
        
    #%% preprocess signal
    
    # #if more than one channel, average them all (for early prototyping only)
    # if eeg_chunk.shape[1] > 1:
    #     eeg_chunk = eeg_chunk.mean(axis=1)
    
    #remove eyeblink/jaw artifacts
    
    #%% compute features
    #calc variance
    eeg_var = np.var(eeg_chunk,axis=0) #variance of each channel (column)
    eeg_var = eeg_var.mean() #for early prototpying, just average all channels variance
    
    #calc band powers
    band_powers = compute_band_powers(eeg_chunk, s_rate)
    
    #for early prototype, average across all channels
    delta = band_powers[0].mean()
    theta = band_powers[1].mean()
    alpha = band_powers[2].mean()
    beta  = band_powers[3].mean()
    gamma = band_powers[4].mean()
    
    #normalize features(?)
    
    #gather features into dictionary
    feature_dict = {
        "variance": eeg_var,
        "delta":  delta,
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma
    }
    
    return feature_dict
  