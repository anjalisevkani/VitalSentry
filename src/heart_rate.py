import numpy as np

def calculate_bpm(signal, fs):
    """
    signal: 1-D filtered green signal (numpy array)
    fs: sampling rate (frames per second)
    """
    # FFT
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)

    # Find peak frequency in the FFT
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    # Convert to BPM
    bpm = peak_freq * 60
    return bpm
def calculate_respiration(signal, fs, low=0.1, high=0.5):
    """
    Calculate respiration rate from low-frequency signal.

    signal: 1-D filtered green signal (numpy array)
    fs: sampling rate (frames per second)
    low: low cutoff (Hz) ~ 0.1
    high: high cutoff (Hz) ~ 0.5
    """
    from scipy.signal import butter, filtfilt
    import numpy as np

    # Bandpass filter for breathing
    b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal)

    # FFT to find peak frequency
    fft = np.fft.rfft(filtered)
    freqs = np.fft.rfftfreq(len(filtered), 1/fs)
    peak_idx = np.argmax(np.abs(fft))
    peak_freq = freqs[peak_idx]

    # Convert Hz to breaths per minute
    rr = peak_freq * 60
    return rr
