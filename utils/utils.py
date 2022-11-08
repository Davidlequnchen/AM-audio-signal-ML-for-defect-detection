## Required python libraries
import pandas as pd
import numpy as np
import scipy as sp
import os
import math
from skimage.transform import resize
from skimage import data
import skimage

import scaleogram as scg 
from glob import glob
import scipy
from scipy.signal import welch
import wave                    # library handles the parsing of WAV file headers
import pywt
import soundfile as sf
## Audio signal processing libraries
import librosa

from matplotlib import rc
# rc('font', **{'family':'DejaVu Sans Mono'})
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.collections import PatchCollection
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

colour = ["#e9150d", '#999999', 'c', 'Brown', "#fbab17",'#333333', "#0515bf", "#10a310"]


def simple_visualization(sound, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,

    librosa.display.waveshow(sound, sr=sr, alpha=alpha, label = 'original signal')
    axs.set_xlabel('Time', fontsize = 18)
    axs.set_ylabel('Amplitute', fontsize = 18)

    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)


def equalized_signal_visualization(noisy, cleaned, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,

    librosa.display.waveshow(noisy, sr=sr, alpha=alpha, label = 'original signal')
    librosa.display.waveshow(cleaned,sr=sr, alpha=0.5, label = 'eqaulized signal')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)

    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)

def bandpass_signal_visualization(equalized, cleaned, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,

    librosa.display.waveshow(equalized, sr=sr, alpha=alpha, label = 'equalized')
    librosa.display.waveshow(cleaned,sr=sr, alpha=0.5, label = 'bandpass filtered')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)

    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)


def final_signal_visualization(filtered, final, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,

    librosa.display.waveshow(filtered, sr=sr, alpha=alpha, label = 'bandpass filtered signal')
    librosa.display.waveshow(final,sr=sr, alpha=0.5, label = 'final extracted signal')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)

    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)

def two_step_cleaned_signal_visualization(original, equalized, cleaned, sr=44100, alpha = 1):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,
    
    librosa.display.waveshow(original, sr=sr, alpha=alpha, label = 'original signal')
    librosa.display.waveshow(equalized, sr=sr, alpha=1, label = 'after applying equalizer')
    librosa.display.waveshow(cleaned,sr=sr, alpha=0.6, label = 'denoised signal w equalizer and bandpass filter')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)

    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)

def three_step_cleaned_signal_visualization(original, equalized, bandpassed, final_cleaned, sr=44100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,
    
    librosa.display.waveshow(original, sr=sr, alpha=1, label = 'original signal')
    librosa.display.waveshow(equalized, sr=sr, alpha=0.85, label = 'after equalization')
    librosa.display.waveshow(bandpassed,sr=sr, alpha=0.7, label = 'after bandpass filter')
    librosa.display.waveshow(final_cleaned,sr=sr, alpha=0.5, label = 'final extracted audio')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)

def three_step_cleaned_signal_visualization_smooth(original, equalized, bandpassed, final_cleaned, sr=44100, N = 100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,
    
    original = pd.Series(original).rolling(window=N).mean().iloc[N-1:].values
    equalized = pd.Series(equalized).rolling(window=N).mean().iloc[N-1:].values
    bandpassed = pd.Series(bandpassed).rolling(window=N).mean().iloc[N-1:].values
    final_cleaned = pd.Series(final_cleaned).rolling(window=N).mean().iloc[N-1:].values
    
    
    librosa.display.waveshow(original, sr=sr, alpha=1, label = 'original signal')
    librosa.display.waveshow(equalized, sr=sr, alpha=0.85, label = 'after equalization')
    librosa.display.waveshow(bandpassed,sr=sr, alpha=0.7, label = 'after bandpass filter')
    librosa.display.waveshow(final_cleaned,sr=sr, alpha=0.5, label = 'final extracted audio')
    axs.set_xlabel('Time', fontsize = 16)
    axs.set_ylabel('Normalized Amplitute', fontsize = 16)
    axs.legend(loc = 3)
    fig.suptitle("Time-domain visualisation", fontsize = 20,  y=1)

    
def two_step_cleaned_signal_visualization_fft(original, equalized, bandpassed, 
                                                f_ratio=0.5, log_x=False, log_y=False, sr=44100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,
    
    X_original_mag = np.absolute(np.fft.fft(original))
    X_equalized_mag = np.absolute(np.fft.fft(equalized))
    X_bandpassed_mag = np.absolute(np.fft.fft(bandpassed))
    
    f_original = np.linspace(0, sr, len(X_original_mag))
    f_bins_original = int(len(X_original_mag)*f_ratio)  

    f_equalized = np.linspace(0, sr, len(X_equalized_mag))
    f_bins_equalized = int(len(X_equalized_mag)*f_ratio) 
    
    f_bandpassed = np.linspace(0, sr, len(X_bandpassed_mag))
    f_bins_bandpassed = int(len(X_bandpassed_mag)*f_ratio) 
    
    plt.plot(f_original[:f_bins_original], X_original_mag[:f_bins_original], alpha=1, label = 'original signal')
    plt.plot(f_equalized[:f_bins_equalized], X_equalized_mag[:f_bins_equalized],alpha=0.8, label = 'after equalization')
    plt.plot(f_bandpassed[:f_bins_bandpassed], X_bandpassed_mag[:f_bins_bandpassed],alpha=0.7, label = 'final extracted audio')
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Magnitute', fontsize=14)
    plt.title("FFT comparison", fontsize=18)
    
    # ax.set_ylim(0, 10000)
    if log_x:
        axs.set_xscale('log') 
    if log_y:
        axs.set_yscale('log') 

    axs.legend(loc = 3)



def three_step_cleaned_signal_visualization_fft(original, equalized, bandpassed, final_cleaned, 
                                                f_ratio=0.5, log_x=False, log_y=False, sr=44100):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(10, 5)) #constrained_layout=True,
    
    X_original_mag = np.absolute(np.fft.fft(original))
    X_equalized_mag = np.absolute(np.fft.fft(equalized))
    X_bandpassed_mag = np.absolute(np.fft.fft(bandpassed))
    X_final_mag = np.absolute(np.fft.fft(final_cleaned))
    
    f_original = np.linspace(0, sr, len(X_original_mag))
    f_bins_original = int(len(X_original_mag)*f_ratio)  

    f_equalized = np.linspace(0, sr, len(X_equalized_mag))
    f_bins_equalized = int(len(X_equalized_mag)*f_ratio) 
    
    f_bandpassed = np.linspace(0, sr, len(X_bandpassed_mag))
    f_bins_bandpassed = int(len(X_bandpassed_mag)*f_ratio) 

    f_final = np.linspace(0, sr, len(X_final_mag))
    f_bins_final = int(len(X_final_mag)*f_ratio) 
    
    plt.plot(f_original[:f_bins_original], X_original_mag[:f_bins_original], alpha=1, label = 'original signal')
    plt.plot(f_equalized[:f_bins_equalized], X_equalized_mag[:f_bins_equalized],alpha=0.8, label = 'after equalization')
    plt.plot(f_bandpassed[:f_bins_bandpassed], X_bandpassed_mag[:f_bins_bandpassed],alpha=0.7, label = 'bandpass filtered')
    plt.plot(f_final[:f_bins_final], X_final_mag[:f_bins_final], alpha=0.5, label = 'final extracted audio')
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Magnitute', fontsize=14)
    plt.title("FFT comparison", fontsize=18)
    
    # ax.set_ylim(0, 10000)
    if log_x:
        axs.set_xscale('log') 
    if log_y:
        axs.set_yscale('log')
    axs.legend(loc = 3) 

def three_step_cleaned_signal_visualization_fft_smooth(original, equalized, bandpassed, final_cleaned, 
                                                f_ratio=0.5, log_x=False, log_y=False, sr=44100, N_smooth = 1000):
    fig, axs = plt.subplots(1, 1, tight_layout = True, figsize=(8, 3)) #constrained_layout=True,
    
    X_original_mag = np.absolute(np.fft.fft(original))
    X_equalized_mag = np.absolute(np.fft.fft(equalized))
    X_bandpassed_mag = np.absolute(np.fft.fft(bandpassed))
    X_final_mag = np.absolute(np.fft.fft(final_cleaned))
    
    # freqeuncy bands
    f_original = np.linspace(0, sr, len(X_original_mag))
    f_bins_original = int(len(X_original_mag)*f_ratio)  

    f_equalized = np.linspace(0, sr, len(X_equalized_mag))
    f_bins_equalized = int(len(X_equalized_mag)*f_ratio) 
    
    f_bandpassed = np.linspace(0, sr, len(X_bandpassed_mag))
    f_bins_bandpassed = int(len(X_bandpassed_mag)*f_ratio) 

    f_final = np.linspace(0, sr, len(X_final_mag))
    f_bins_final = int(len(X_final_mag)*f_ratio) 
    
    ## get profile
    # high_idx, low_idx = hl_envelopes_idx(X_original_mag)
    # plt.plot(f_original[high_idx], X_original_mag[high_idx], alpha=1, label = 'original signal')
    # plt.plot(f_original[low_idx], X_original_mag[low_idx], alpha=1, label = 'original signal')
    # plt.plot(f_original, X_original_mag, alpha=1, label = 'original signal')
    
    N = N_smooth
    X_original_mag_mean = pd.Series(X_original_mag[:f_bins_original]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_original[:f_bins_original-N+1], X_original_mag_mean, alpha=1, label = 'original signal')
    
    X_equalized_mag_mean = pd.Series(X_equalized_mag[:f_bins_equalized]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_equalized[:f_bins_equalized-N+1], X_equalized_mag_mean, alpha=1, label = 'equalized signal')
    
    X_bandpassed_mag_mean = pd.Series(X_bandpassed_mag[:f_bins_bandpassed]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_bandpassed[:f_bins_bandpassed-N+1], X_bandpassed_mag_mean, alpha=1, label = 'bandpass filtered signal')
    
    X_final_mag_mean = pd.Series(X_final_mag[:f_bins_final]).rolling(window=N).mean().iloc[N-1:].values
    plt.plot( f_final[:f_bins_final-N+1], X_final_mag_mean, alpha=1, label = 'laser-material intercation sound')
     
    
    # ax.set_ylim(0, 10000)
    if log_x and not log_y:
        axs.set_xscale('log') 
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitute', fontsize=16)
        plt.title("Log scale freqeuncy comparison", fontsize=18)
    if log_y and not log_x:
        axs.set_yscale('log')
        plt.xlabel('Frequency (Hz)', fontsize=16)
        plt.ylabel('Magnitute (db)', fontsize=16)
        plt.title("Log scale magnitude comparison", fontsize=18)
    # axs.legend(loc = 3) 
    axs.legend(loc='best')




def three_step_fft_individual_plot(original, equalizer,bandpass, estimates1, estimates0, sampling_rate=44100, f_ratio=0.5 ):
    fig, axs = plt.subplots(
    nrows=5,
    ncols=1,
    sharey=False,
    # figsize=(14, 7),
    figsize=(14, 12),
    # dpi=800
    tight_layout=True
    )

    X_mag_original = calculate_and_plot_magnitude_spectrum(original, sampling_rate, 'Original signal (noisy)' , axs.flat[0], f_ratio=f_ratio)
    X_mag_equalized = calculate_and_plot_magnitude_spectrum(equalizer, sampling_rate, 'Step 1: equalized signal' , axs.flat[1], f_ratio=f_ratio)
    X_mag_bandpass = calculate_and_plot_magnitude_spectrum(bandpass, sampling_rate, 'Step 2: Bandpass filtered signal' , axs.flat[2], f_ratio=f_ratio)
    X_mag_original = calculate_and_plot_magnitude_spectrum(estimates1, sampling_rate, 'Step 3(a): Sound source separated: laser-material interaction sound' , axs.flat[3], f_ratio=f_ratio)
    X_mag_original = calculate_and_plot_magnitude_spectrum(estimates0, sampling_rate, 'step 3(b): Sound source separated: noise component' , axs.flat[4], f_ratio=f_ratio)
    # estimates[1].audio_data[0], hpss_estimates[0].audio_data[0]

    fig.tight_layout();

    # plt.show()
    # plt.clf()

def annot_max(x,y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text= "x={:.3f}, y={:.3f}".format(xmax, ymax)
    if not ax:
        ax=plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops=dict(arrowstyle="->") #,connectionstyle="angle,angleA=0,angleB=60"
    kw = dict(xycoords='data',textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top") #, ha="right", va="top"
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.5,0.56), **kw, fontsize=14)


def calculate_and_plot_magnitude_spectrum(signal, sr, title, ax, f_ratio=0.5):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
    
    ax.plot(f[:f_bins], X_mag[:f_bins])
    ax.set_xlabel('Frequency (Hz)', fontsize=16)
    ax.set_ylabel('Magnitute', fontsize=16)
    ax.set_title(title , fontsize=18)
    # ax.set_ylim(0, 7500)
    
    ymax = max(X_mag)
    xpos = np.where(X_mag == ymax)
    xmax = f[xpos]
    
    # annot_max(f[:f_bins], X_mag[:f_bins], ax=ax)
    
    return X_mag

def plot_magnitude_spectrum_single(signal, sr, title, f_ratio=0.5, log_x=False, log_y=False):
    X = np.fft.fft(signal)
    X_mag = np.absolute(X)
    
    # fig, ax = plt.figure(figsize=(13, 4))
    fig, ax = plt.subplots(figsize = (10,3))
    
    f = np.linspace(0, sr, len(X_mag))
    f_bins = int(len(X_mag)*f_ratio)  
    
    plt.plot(f[:f_bins], X_mag[:f_bins])
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.ylabel('Magnitute', fontsize=14)
    plt.title(title, fontsize=18)
    
    # ax.set_ylim(0, 10000)
    if log_x:
        ax.set_xscale('log') 
    if log_y:
        ax.set_yscale('log')

def plot_spectrogram_ax(Y, sr,  ax, hop_length, y_axis="log"):
    # plt.figure(figsize=(13, 5))
    img=librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis,
                             ax = ax,
                             cmap="magma", #'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                             vmin = -40, vmax = 20)#vmin = -70, vmax = 45

    fig.colorbar(img, ax=ax, format="%+2.f dB")
    # plt.set_clim(-80,50)
    # ax.set_ylim([-0.45, 0.45])
    ax.set_ylabel('Frequency', fontsize = 12)
    ax.set_xlabel('Time', fontsize = 12)  #(μs)


def plot_spectrogram(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))
    
    Stft = librosa.stft(Y, n_fft=frame_size, hop_length=hop_length)
    Y_sample  = np.abs(Stft) ** 2
    Y_log_sample = librosa.power_to_db(Y_sample)

    img=librosa.display.specshow(Y_log_sample, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                # cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                vmin = vmin, vmax = vmax
                                )
    plt.colorbar(img, format="%+2.f dB")
    plt.title(title, fontsize=18, pad=1.2)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)



def mfcc_feature_normalize(data):               # data is in numpy array (40,x)
    data_normalize = []
    data_max = np.max(data, axis =0)       # max value per row, 
    data_min = np.min(data, axis =0)       # min value per row
    for i in range(len(data)):
        data_nor = [(x - data_min[i])/(data_max[i] - data_min[i]) for x in data[i]] #normalize each value in 'x'
        data_normalize.append(data_nor)
    return np.asarray(data_normalize)      #convert back to numpy array

def plot_mfcc(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))

    # get the mfcc feature
    mfccs = librosa.feature.mfcc(y =Y, sr = sr, n_mfcc = 20)

    # Normalize mfccs to be within [0,1]
    mfccs_sc = mfcc_feature_normalize(mfccs)

    img=librosa.display.specshow(mfccs_sc, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                # cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                # vmin = vmin, vmax = vmax
                                )
    # plt.colorbar(img, format="%+5.f dB")
    plt.colorbar(img)
    plt.title(title, fontsize=18)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)

def plot_mfcc_fixed_colorbar(Y, sr, title, frame_size=1024, hop_length = 512, y_axis="log", vmin=-80, vmax=40, cmap="magma"):
    plt.figure(figsize=(8, 5))

    # get the mfcc feature
    mfccs = librosa.feature.mfcc(y =Y, sr = sr, n_mfcc = 20)

    # Normalize mfccs to be within [0,1]
    mfccs_sc = mfcc_feature_normalize(mfccs)

    img=librosa.display.specshow(mfccs_sc, 
                                sr=sr, 
                                hop_length=hop_length, 
                                x_axis="time", 
                                y_axis=y_axis,
                                # cmap=cmap, #'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet'
                                vmin = vmin, vmax = vmax
                                )
    # plt.colorbar(img, format="%+5.f dB")
    plt.colorbar(img)
    plt.title(title, fontsize=18)
    plt.ylabel('Frequency', fontsize = 16)
    plt.xlabel('Time', fontsize = 16)  #(μs)

    
def plot_wavelet_family(wavelet = "morl"):
    # choice: 'mexh', 'morl', 'shan1.5-1.0'
    scg.plot_wav(wavelet, figsize=(14,5))  


def plot_scaleogram(signal, sr, title, wavelet = 'shan1.5-1.0', period_length = 600, scale_resolution=50, coi = False, yscale = 'linear'):
    
    # plot heartbeat in time domain
    fig1, ax1 = plt.subplots(1,1, figsize = (9, 3));
    ax1.plot(signal, linewidth = 3, color = 'blue')
    # ax1.set_xlim(0, signal_length)
    ax1.set_title('Time-domain signal')

    # choose default wavelet function
    scg.set_default_wavelet(wavelet)

    # range of scales to perform the transform
    scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution))   # give len of 120

    # plot scalogram
    scg.cws(signal, scales=scales, figsize = (10, 5), coi = coi, ylabel = 'Period',
            xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
            cmap="jet", yscale=yscale
            )

    print("Wavelet function used to compute the transform:", scg.get_default_wavelet(), 
      "(", pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")


def denoised_original_signal_visualization(sample_original_list,sample_cleaned_list, sampling_rate=44100):
    fig, axs = plt.subplots(2, 2, tight_layout = True, figsize=(12, 10)) #constrained_layout=True,

    axs[0,0] = plt.subplot(2, 2, 1)
    librosa.display.waveshow(sample_original_list[0], sr=sampling_rate, alpha=1, label = 'original signal')
    librosa.display.waveshow(sample_cleaned_list[0], sr=sampling_rate, alpha=0.6,label = 'denoised signal')
    axs[0,0].set_title('Experiment 1', fontsize = 16, pad=10 )
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Amplitute')
    axs[0,0].legend(loc = 3)
    # axs[0,0].set_ylim((-0.45, 0.45))
    fig.suptitle("Time-domain visualisation of raw and denoised signal", fontsize = 20,  y=1.0005)

    axs[0,1] = plt.subplot(2, 2, 2)
    librosa.display.waveshow(sample_original_list[1], sr=sampling_rate, alpha=1, label = 'original signal')
    librosa.display.waveshow(sample_cleaned_list[1], sr=sampling_rate, alpha=0.6,label = 'denoised signal')
    axs[0,1].set_title('Experiment 2',  fontsize = 16, pad=10)
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Amplitute')
    axs[0,1].legend(loc = 3)
    # axs[0,1].set_ylim((-0.45, 0.45))

    axs[1,0] = plt.subplot(2, 2, 3)
    librosa.display.waveshow(sample_original_list[2], sr=sampling_rate, alpha=1, label = 'original signal')
    librosa.display.waveshow(sample_cleaned_list[2], sr=sampling_rate, alpha=0.6,label = 'denoised signal')
    axs[1,0].set_title('Experiment 3',  fontsize = 16, pad=10)
    axs[1,0].set_xlabel('Time')
    axs[1,0].set_ylabel('Amplitute')
    axs[1,0].legend(loc = 3)
    # axs[1,0].set_ylim((-0.45, 0.45))

    axs[1,1] = plt.subplot(2, 2, 4)
    librosa.display.waveshow(sample_original_list[3], sr=sampling_rate, alpha=1, label = 'original signal')
    librosa.display.waveshow(sample_cleaned_list[3], sr=sampling_rate, alpha=0.6,label = 'denoised signal')
    axs[1,1].set_title('Experiment 4',  fontsize = 16, pad=10)
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('Amplitute')
    axs[1,1].legend(loc = 3)
    # axs[1,1].set_ylim((-0.45, 0.45))



def extract_wavelet_transform(signal, sr, title, wavelet = 'shan1.5-1.0', period_length = 2000, scale_resolution=50, coi = False):
    # range of scales to perform the transform
    scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution))   # give len of 120
    coef, freq = pywt.cwt(signal, scales, wavelet, 1, method='conv')
    return coef, freq
    

def extract_wavelet_transform(signal, sr, wavelet = 'shan1.5-1.0', period_length = 200, scale_resolution=5):
    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet, dt = 1/44100)   # give len of 120
    scales=np.arange(1, period_length)
    coef, freq = pywt.cwt(signal, scales, wavelet)
    return coef, freq

def plot_wavelet_transform_resize(signal, sr, wavelet = 'shan1.5-1.0', period_length = 200, scale_resolution=5, rescale_size=120):
    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet, dt = 1/44100)   # give len of 120
    # scales=np.arange(1, period_length)
    scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution))   #len(scales) : 120
    # coef, freq = pywt.cwt(signal, scales, wavelet)
    coef, freq = scg.fastcwt(signal, scales, wavelet)
    
    # resize the 2D cwt coeffs 
    rescale = skimage.transform.resize(abs(coef), (len(scales), rescale_size), mode = 'constant')
    
    fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8));
    
    axs[0].imshow(abs(coef), cmap = 'jet', aspect = 'auto')
    axs[0].set_title("original")
    axs[1].imshow(rescale, cmap = 'jet', aspect = 'auto')
    axs[1].set_title("resize from 6k to 120 in time axis")


def extract_wavelet_transform_resize(signal, sr, wavelet = 'shan1.5-1.0', period_length = 200, scale_resolution=5, rescale_size=120):
    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet, dt = 1/44100)   # give len of 120
    # scales=np.arange(1, period_length)
    scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution))   #len(scales) : 120
    # coef, freq = pywt.cwt(signal, scales, wavelet)
    coef, freq = scg.fastcwt(signal, scales, wavelet)
    
    # resize the 2D cwt coeffs 
    rescale = skimage.transform.resize(abs(coef), (len(scales), rescale_size), mode = 'constant')
    
    return rescale


def extract_wavelet_transform_fast(signal, sr, wavelet = 'shan1.5-1.0', period_length = 200, scale_resolution=5):
    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet, dt = 1/44100)   # give len of 120
    scales=np.arange(1, period_length)
    coef, freq = scg.fastcwt(signal, scales, wavelet)
    return coef, freq

def plot_wavelet_transform(signal, sr, title, wavelet = 'shan1.5-1.0', period_length = 200, 
                           scale_resolution=5, set_colorbar = False, cmin=0, cmax=3):
    coef, freq = extract_wavelet_transform_fast(signal, sr, wavelet = wavelet, 
                                                period_length = period_length, scale_resolution=scale_resolution)
    
    plt.figure(figsize=(8, 4))
    # # fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 10));
    # fig, (ax1, ax2) = plt.subplots(figsize=(10, 10), nrows=2)
    # # plot audio in time domain
    # ax1.plot(signal, linewidth = 3, color = 'blue')
    # # axs[0].set_title(df_ab['label'][select_list[row]])
    # wavelet = ax2.imshow(abs(coef), cmap = 'jet', aspect = 'auto') #aspect = 'auto'
    # if (set_colorbar==True):
    #     fig.colorbar(wavelet, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.9, (cmin=0, cmax=3))
    # else:
    #     fig.colorbar(wavelet, ax=ax2, location='right', anchor=(0, 0.3), shrink=0.9)
    
    wavelet = plt.imshow(abs(coef), cmap = 'jet', aspect = 'auto') #aspect = 'auto'
    if (set_colorbar==True):
        plt.clim(cmin,cmax)
    else:
        # plt.clim(-4,4)
        pass
    
        
    
def plot_scaleogram(signal, sr, title, wavelet = 'shan1.5-1.0', period_length = 600, scale_resolution=50, coi = False, 
                    yscale = 'linear', set_colorbar = False, cmin=0, cmax=3):
    # plot heartbeat in time domain
    fig1, ax1 = plt.subplots(1,1, figsize = (9, 3));
    ax1.plot(signal, linewidth = 3, color = 'blue')
    # ax1.set_xlim(0, signal_length)
    ax1.set_title('Time-domain signal')

    # choose default wavelet function
    scg.set_default_wavelet(wavelet)

    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet)   # give len of 120
    scales=np.arange(1, period_length)
    # plot scalogram
    if (coi == False and set_colorbar==False):
        scg.cws(signal, scales=scales, figsize = (10, 5), coi = coi, ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale
                )
    elif (set_colorbar==False and coi == True):
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi='O',
                coikw={'alpha':1.0, 'facecolor':'pink', 'edgecolor':'green',
                'linewidth':5} ,
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                )
        
    elif (set_colorbar==True and coi == False):
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi=False,
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                clim=(cmin, cmax)
                )
    else:
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi='O',
                coikw={'alpha':1.0, 'facecolor':'pink', 'edgecolor':'green', 'linewidth':5},
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                clim=(cmin, cmax)
                )
    
  
    print("Wavelet function used to compute the transform:", scg.get_default_wavelet(), 
      "(", pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")
    
    
def plot_scaleogram_resize(signal, sr, title, wavelet = 'shan1.5-1.0', period_length = 600, scale_resolution=50, coi = False, 
                    yscale = 'linear', set_colorbar = False, cmin=0, cmax=3):
    # plot heartbeat in time domain
    fig1, ax1 = plt.subplots(1,1, figsize = (9, 3));
    ax1.plot(signal, linewidth = 3, color = 'blue')
    # ax1.set_xlim(0, signal_length)
    ax1.set_title('Time-domain signal')

    # choose default wavelet function
    scg.set_default_wavelet(wavelet)

    # range of scales to perform the transform
    # scales = scg.periods2scales(np.arange(1, period_length+1, scale_resolution), wavelet)   # give len of 120
    scales=np.arange(1, period_length)
    # plot scalogram
    if (coi == False and set_colorbar==False):
        scg.cws(signal, scales=scales, figsize = (10, 5), coi = coi, ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale
                )
    elif (set_colorbar==False and coi == True):
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi='O',
                coikw={'alpha':1.0, 'facecolor':'pink', 'edgecolor':'green',
                'linewidth':5} ,
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                )
        
    elif (set_colorbar==True and coi == False):
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi=False,
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                clim=(cmin, cmax)
                )
    else:
        scg.cws(signal, scales=scales, figsize = (10, 5), 
                coi='O',
                coikw={'alpha':1.0, 'facecolor':'pink', 'edgecolor':'green', 'linewidth':5},
                ylabel = 'Period',
                xlabel = 'Time', title = 'AM acoustic signal Scaleogram ' + title ,
                cmap="jet", yscale=yscale,
                clim=(cmin, cmax)
                )
    
  
    print("Wavelet function used to compute the transform:", scg.get_default_wavelet(), 
      "(", pywt.ContinuousWavelet(scg.get_default_wavelet()).family_name, ")")