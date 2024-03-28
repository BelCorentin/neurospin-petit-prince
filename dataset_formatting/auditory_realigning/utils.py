import wave

def get_wav_duration(filename):
    with wave.open(filename, 'rb') as wav_file:
        # Get the total number of frames in the WAV file
        frames = wav_file.getnframes()
        
        # Get the frame rate (number of frames per second)
        frame_rate = wav_file.getframerate()
        
        # Calculate the duration in seconds
        duration = frames / float(frame_rate)
        
        return duration

import numpy as np 
from scipy.io import wavfile
import matplotlib.pyplot as plt

CHAPTER_PATHS = [
    "ch1-3.wav",
    "ch4-6.wav",
    "ch7-9.wav",
    "ch10-12.wav",
    "ch13-14.wav",
    "ch15-19.wav",
    "ch20-22.wav",
    "ch23-25.wav",
    "ch26-27.wav",
]

def gen_audio_words_length():
    audio_word_trigger_length = []
    audio_trigger_length = []

    for i, chapter in enumerate(CHAPTER_PATHS):
        wav_file = f'/media/co/T7/workspace-LPP/data/MEG/LPP/PallierListen2023/download/sourcedata/stimuli/audio/{chapter}'
        duration = get_wav_duration(wav_file)
        fs, data = wavfile.read(wav_file)

        # Getting the channel for the triggers encoded in the 2nd part of the wav
        data = data[:,1]
        data = np.round(data / 30000)
        diff = np.diff(data)
        audio_triggers = np.where(diff == 1)[0]
        word_total_length =  (audio_triggers[-1] - audio_triggers[0] )/ fs

        audio_word_trigger_length.append(word_total_length)
        audio_trigger_length.append(duration)
    
    return audio_word_trigger_length
        
    
    
import mne
import logging
import mne_bids
import pandas as pd
from pathlib import Path

# Set the logger level to WARNING to reduce verbosity
logger = logging.getLogger('mne')
logger.setLevel(logging.ERROR)


#path = '/home/co/data/LPP_MEG_auditory'
path = Path("/media/co/T7/workspace-LPP/data/MEG/LPP/PallierListen2023/download")

def get_raw(subject, run):
    run_id = '0' + str(run) 
    task = 'listen'
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task,
        datatype="meg",
        root=path,
        run=run_id,
    )
    raw = mne_bids.read_raw_bids(bids_path)
    triggers = mne.find_events(raw, stim_channel="STI008", shortest_event=1)
    
    # Generate event_file path
    event_file = path / f"sub-{bids_path.subject}"
    event_file = event_file / f"ses-{bids_path.session}"
    event_file = event_file / "meg"
    event_file = str(event_file / f"sub-{bids_path.subject}")
    event_file += f"_ses-{bids_path.session}"
    event_file += f"_task-{bids_path.task}"
    event_file += f"_run-{bids_path.run}_events.tsv"
    assert Path(event_file).exists()

    meta = pd.read_csv(event_file, sep="\t")

    meta["word"] = meta["trial_type"].apply(
            lambda x: eval(x)["word"] if type(eval(x)) == dict else np.nan)

    # Remove the empty words:

    meta.loc[meta['word'] == ' ', 'word'] = None

    # Drop the rows containing NaN values in the text column
    meta = meta.dropna(subset=['word'])


    return raw, triggers[:,0], meta

def save_multiplot(subject, run):   
    raw, word_triggers, events = get_raw(str(subject), run)    
    # MEG events

    raw.pick(['stim', 'misc'])
    events_trigger = mne.find_events(raw, stim_channel="STI008",  shortest_event=1)
    times = np.copy(raw.times)
    values = np.zeros_like(times)
    values[events_trigger[:, 0]-raw.first_samp] = 1

    plt.plot(times, values)

    # Wav ground truth

    from utils import CHAPTER_PATHS

    wav_fs, wav = wavfile.read(f'/media/co/T7/workspace-LPP/data/MEG/LPP/PallierListen2023/download/sourcedata/stimuli/audio/{CHAPTER_PATHS[run-1]}')
    wav, wav_fs, wav_times = resample_signal(wav, wav_fs, raw.info['sfreq'])

    wav_triggers = np.clip(wav[:, 1], 0, 1)

    # df = pd.read_csv('infos.csv')
    # offset = (df[(df.subject == subject) & (df.run == run)].offset.values[0]) / raw.info['sfreq']
    # stretch = df[(df.subject == subject) & (df.run == run)].stretch.values[0]

    offset = word_triggers[0] - raw.first_samp
    offset = offset / raw.info['sfreq']

    # And add the initial time between the first word trigger and the first sample (eg. 3.05 for run 1)
    offset -= events.onset[0]

    # Get the length of the recorded word triggers
    meg_word_length = (word_triggers[-1] - word_triggers[0]) / raw.info['sfreq']
    # Deduct the stretch factor
    audio_word_length = np.load('audio_word_length.npy')
    stretch = meg_word_length / audio_word_length[run-1]

    # plt.plot(offset+(wav_times*stretch), wav_triggers)

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    xlim_pairs = [(0, 20), (20, 21), (120, 170), (200, 210), (204, 205), (400, 420), (510, 511), (520, 590), (580, 581)]

    for i, ax in enumerate(axs.flatten()):
        ax.plot(times, values)
        ax.plot(offset+(wav_times*stretch), wav_triggers)
        ax.set_xlim(xlim_pairs[i])

    plt.savefig(f'figures/Multiplot_{subject}_{run}.png')
    plt.close()
    # Flush the plot 
    plt.clf()
    plt.cla()
    
    # del raw
    # import gc
    # gc.collect()


    return stretch, offset


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
    for stretch in tqdm(stretches):
        new_length = int(stretch * n)
        y_hat = resample(y, new_length)
        r = correlate(x, y_hat)
        R.append(r.max())

    best = np.argmax(R)
    strech = stretches[best]

    # offset
    r = correlate(X, Y)
    best = np.argmax(r)
    offsets = np.arange(-len(X) + 1, len(Y))
    offset = offsets[best]
    
    return strech, offset
    

def resample_safe(x, target_length):
    idx = np.linspace(0., len(x)-1, target_length).astype(int)
    return x[idx]
    
class Align(BaseModel):
    stretches: tp.List[float] = np.linspace(.99, 1.02, 500)
    decim: int = 1
    freq: float

    _stretch: float
    _offset: float

    def fit(self, X, Y):
        X = lowpass(X, self.freq)
        Y = lowpass(Y, self.freq)
        self._stretch, self._offset = align_series(Y, X, self.stretches, self.decim)
        return self

    def predict(self, X):
        Y_hat = resample_safe(X, int(self._stretch*len(X)))

        pad = np.zeros(int(np.abs(self._offset)))
        if self._offset>0:
            Y_hat = np.r_[pad, Y_hat]
        elif self._offset<0:
            Y_hat = np.r_[Y_hat, pad]
        return Y_hat