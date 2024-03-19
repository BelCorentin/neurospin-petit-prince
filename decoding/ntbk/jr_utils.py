import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
from scipy.signal import correlate
from tqdm import tqdm
from pydantic import BaseModel
import typing as tp
mne.set_log_level(False)

def resample_signal(signal, original_freq, target_freq):
    
    # Calculate the number of samples in the resampled signal
    num_samples_target = int(len(signal) * target_freq / original_freq)
    
    # Resample the signal
    resampled_signal = resample(signal, num_samples_target)
    resampled_signal /= resampled_signal.max(0, keepdims=True)
    
    # Create the time vector for the resampled signal
    resampled_time = np.linspace(0, len(signal) / original_freq, num_samples_target, endpoint=False)
    
    return resampled_signal, target_freq, resampled_time

def gaussian_kernel(size, sigma):
    position = np.arange(size) - size // 2
    kernel_raw = np.exp(-0.5 * (position / sigma) ** 2)
    kernel_normalized = kernel_raw / np.sum(kernel_raw)
    return kernel_normalized
    
def lowpass(data, sfreq, wsize=20):
    window_size = int(.500 * sfreq)
    filter_kernel = gaussian_kernel(window_size, wsize)
    out = np.convolve(data, filter_kernel, mode='same')
    return out / out.max()

def align_series(x, y, stretches, decim=1):
    X, Y = x, y
    if decim:
        x = x[::decim]
        y = y[::decim]

    # pad
    assert len(x) == len(y) # FIXME
    if len(x) < len(y):
        z = np.zeros(len(y)-len(x))
        x = np.r_[z, x]
    elif len(x) > len(y):
        z = np.zeros(len(x)-len(y))
        y = np.r_[z, y]
    n = len(x)

    R = []
    corrs = []
    b_offsets = []
    R = []
    
    for stretch in tqdm(stretches):
        new_length = int(stretch * n)
        y_hat = resample(y, new_length)
        r = correlate(x, y_hat)
        R.append(r.max())

    best = np.argmax(R)
    strech = stretches[best]

    # offset
    new_length = int(stretch * len(X))
    Y_hat = resample(Y, new_length)
    Y_hat = Y
    r = correlate(X, Y_hat)
    best = np.argmax(r)
    offsets = np.arange(-len(X) + 1, len(Y_hat))
    offset = offsets[best]

    # offset
    #new_length = int(stretch * len(X))
    #Y_hat = resample(Y, new_length)
    #r = correlate(X, Y_hat)
    #best = np.argmax(r)
    #offsets = np.arange(-len(X) + 1, len(Y))
    #offset = offsets[best]
    
    return strech, offset, r
    

def resample_safe(x, target_length):
    idx = np.linspace(0., len(x)-1, target_length).astype(int)
    return x[idx]
    
class Align(BaseModel):
    stretches: tp.List[float] = np.linspace(.99, 1.02, 500)
    decim: int = 1
    freq: float

    _stretch: float
    _offset: float
    _corr: tp.Any

    def fit(self, X, Y):
        X = lowpass(X, self.freq)
        Y = lowpass(Y, self.freq)
        self._stretch, self._offset, self._corr = align_series(Y, X, self.stretches, self.decim)
        return self

    def predict(self, X):
        Y_hat = resample_safe(X, int(self._stretch*len(X)))

        pad = np.zeros(int(np.abs(self._offset)))
        if self._offset>0:
            Y_hat = np.r_[pad, Y_hat]
        elif self._offset<0:
            Y_hat = np.r_[Y_hat, pad]
        return Y_hat