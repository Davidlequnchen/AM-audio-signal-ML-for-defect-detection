### Define function to remove the high frequency using DWT
'''
Reference:
- https://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter/12233959#12233959
- https://python-acoustics.github.io/python-acoustics/signal.html#acoustics.signal.bandpass_filter

'''


from scipy.signal import butter, lfilter
import numpy as np
from acoustics.signal import bandpass_filter, bandpass, highpass
import scaleogram as scg 
from glob import glob
import scipy
from scipy.signal import welch
import wave                    # library handles the parsing of WAV file headers
import pywt
import soundfile as sf



def wavelet_bandpass_filter(x, thresh_lower = 0.05, thresh_upper = 0.9999, wavelet='coif6', level = 5):
    
    # Decompose to get the wavelet coefficients
    coeff = pywt.wavedec(x, wavelet=wavelet, mode="per", level= level)
     
    # Calculte the univeral threshold
    # the 0.3 value is tweak until I can reconstruct all the peak (without lefting out the small peak)
    thresh_lower = thresh_lower * np.nanmax(x)       
    coeff[1:] = ( pywt.threshold( i, value=thresh_lower, mode='greater' ) for i in coeff[1:] )
    thresh_upper = thresh_upper * np.nanmax(x)       
    # coeff[1:] = ( pywt.threshold( i, value=thresh_upper, mode='less' ) for i in coeff[1:] )
    
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec( coeff, wavelet = wavelet, mode='per' )


def lowpassfilter(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    coeff = pywt.wavedec(signal, wavelet, mode="per" )
    coeff[1:] = (pywt.threshold(i, value=thresh, mode="soft" ) for i in coeff[1:])
    reconstructed_signal = pywt.waverec(coeff, wavelet, mode="per" )
    return reconstructed_signal


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs=44100, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def acoustic_bandpass_filter (signal, lowcut, highcut, fs=44100, order=8):
    rect = bandpass(signal, lowcut, highcut, fs, order=8, zero_phase=False)
    return rect


def acoustic_highpass_filter (signal, cutoff, fs=44100, order=8):
    rect = highpass(signal, cutoff, fs, order=8, zero_phase=False)
    return rect