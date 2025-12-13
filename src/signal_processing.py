from scipy.signal import butter, filtfilt
import numpy as np

def bandpass(signal, fs, low=0.7, high=4.0):
    """
    Bandpass filter to keep only heartbeat frequencies.

    signal: list or np.array of green channel values
    fs: sampling rate (frames per second)
    low: low cutoff frequency (Hz)
    high: high cutoff frequency (Hz)
    """
    signal = np.array(signal)
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)
    return filtered
