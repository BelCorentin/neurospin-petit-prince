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