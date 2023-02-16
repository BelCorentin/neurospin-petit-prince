"""

DATASET related functions

"""


# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from wordfreq import zipf_frequency
from Levenshtein import editops

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from utils import match_list

def get_path(name='LPP_read'):
    if name == 'LPP_read':
        # TASK = "read"
        data = Path("/home/is153802/workspace_LPP/data/MEG/LPP/BIDS_lecture")

    elif name == 'LPP_listen':
        # TASK = "listen"
        data = Path("/home/is153802/workspace_LPP/data/MEG/LPP/BIDS")
    else:
        return (f'{name} is an invalid name. \n\
        Current options: LPP_read and LPP_listen')
    return data

# Epoching and decoding


def epoch_data(subject, run_id, task, path, filter=True):

    task = 'read' 
    print("Running the script on RAW data:")
    print(f"run {run_id}, subject: {subject}")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task, 
        datatype="meg",
        root=path,
        run=run_id,
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.pick_types(meg=True, stim=True)
    raw.load_data()
    raw = raw.filter(0.5, 20)

    event_file = path / f"sub-{bids_path.subject}"
    event_file = event_file / f"ses-{bids_path.session}"
    event_file = event_file / "meg"
    event_file = str(event_file / f"sub-{bids_path.subject}")
    event_file += f"_ses-{bids_path.session}"
    event_file += f"_task-{bids_path.task}"
    event_file += f"_run-{bids_path.run}_events.tsv"
    assert Path(event_file).exists()
    # read events
    meta = pd.read_csv(event_file, sep="\t")
    events = mne.find_events(
        raw, stim_channel="STI101",  shortest_event=1
    )
    word_length_meg = events[:, 2] - 2048  # Remove first event: chapter start and remove offset
    word_len_meta = meta.word.apply(len)
    i,j = match_list(word_len_meta, word_length_meg)
    events = events[j]
    # events = events[i]  # events = words_events[i]
    meta = meta.iloc[i].reset_index()

    meta['start'] = events[:, 0]/raw.info['sfreq']

    epochs = mne.Epochs(
        raw, events, metadata=meta, tmin=-0.1, tmax=0.6, decim=20, baseline=(-0.1, 0.0)
    )
    epochs.load_data()
    epochs = epochs.pick_types(meg=True, stim=False, misc=False)

    return epochs


def get_subjects(path):
    subjects = pd.read_csv(str(path) + "/participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    # subjects = np.delete(subjects, subjects.shape[0]-1)
    # Let's sort this array before outputting it!
    int_subjects = np.sort([int(subj) for subj in subjects])
    subjects = [str(subj) for subj in int_subjects]

    return subjects


def concac_runs(subject, task, filter, path, RUN=9):
    epochs = []

    for run_id in range(1, RUN + 1):
        print(".", end="")
        epo = epoch_data(subject, "%.2i" % run_id, task, path, filter)
        epo.metadata["label"] = f"run_{run_id}"
        epochs.append(epo)
    for epo in epochs:
        epo.info["dev_head_t"] = epochs[0].info["dev_head_t"]

    epochs = mne.concatenate_epochs(epochs)
    return epochs

