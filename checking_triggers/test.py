# Open the raw file for a random sub, random run
filepath = '/media/co/T7/workspace-LPP/data/MEG/LPP/PallierRead2023/download/sub-26/ses-01/meg/sub-26_ses-01_task-read_run-01_meg.fif' 

import mne
raw = mne.io.read_raw_fif(filepath, preload=True, allow_maxshield=True, verbose=True)
print(raw)

import mne
mne.viz.set_3d_backend("notebook")

raw.plot()