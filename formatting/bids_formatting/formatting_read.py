"""
The role of this script is to:

1)
Transform the raw data organization generated by the MEG,
that does not follow the same rules:
ex: some files are named sub_blocX_raw.fif where some others are sub_rX_raw.fif

The point will be to format everything under a stable and normalized format:
BIDS which will simplify postprocessing.

2)
Run the MNE_BIDS_PIPELINE in order to run a Maxwell filter on the raw data:
the raw data generated by Elekta needs to have this step done before going
any further.

3)
We'll need to add the text annotation / word events in
the different subject folders for other analysis.
The reference is the fMRI events file from:
https://openneuro.org/datasets/ds003643/versions/1.0.4

"""

# Imports ######

from __future__ import annotations
import pandas as pd
import os
import re
import mne
from pathlib import Path
from mne_bids import BIDSPath, write_raw_bids
# from CONST import dict_nip_to_sn

dict_nip_to_sn = {'cb_666666': '1',
                  'rb_666666': '2',
                  'dl_230038': '3',
                  'yb_220174': '4',
                  'mn_230056': '5',
                  'eg_220435': '6'}

# Get the vX used for the recording
subj_version = {'1': '1', '2': '2', '3': '2', '4': '2', '5': '2', '6': '2'}

# 1) Raw format to BIDS #####

#  CONST ###
BASE_PATH = Path('/home/is153802/workspace_LPP/data/MEG/LPP/')
# BIDS_PATH = BASE_PATH / 'LPP_bids'
BIDS_PATH = BASE_PATH / 'BIDS_lecture'
RAW_DATA_PATH = BASE_PATH / 'raw_lecture'
TASK = 'read'

# For each of these folders, go into the sub folder
# (that has the name of a subject)
for folder in RAW_DATA_PATH.iterdir():
    sub_dir = RAW_DATA_PATH / folder
    # for sub in sub_dir.iterdir():
    for sub_ in os.listdir(sub_dir):
        # Changing the subject to the actual NIP / dict:
        nip = str(folder).split('/')[-1]
        # print(nip)

        sub = dict_nip_to_sn[nip]
        if ((BIDS_PATH/f'sub-{sub}').exists()):
            print(f'Subject {sub} BIDS folder already exists')
            continue
        # Get the list of runs
        run_dir = sub_dir / sub_
        for file in os.listdir(run_dir):
            file = str(file)
            print((file))

            try:
                run = re.search(r"r([^']*)_raw.fif", file).group(1)
            # Two cases: filenames is sub_r{run_number}_raw
            # or sub_run{run_number}_raw
            # so it's we are in the second case, ignore the
            # first re and keep the 2nd result
                if len(run) > 2 and (not run.__contains__('_')):
                    run = re.search(r"run([^']*)_raw.fif", file).group(1)
                elif run.__contains__('_'):
                    run = re.search(r"_r([^']*)_raw.fif", file).group(1)
                    if len(run) > 2:
                        run = re.search(r"_run([^']*)_raw.fif", file).group(1)

            except Exception:
                print(f"No run found for file: {file}")
                continue

            # Check if the BIDS dataset already exists:
            sub = str(sub)
            fname = f"sub-{sub}/ses-01/meg/sub-{sub}_ses-01_task-\
                    {TASK}_run-0{run}_meg.fif"
            if (BIDS_PATH / fname).exists():
                print(f"The file {fname} already exists: not created again.")
                continue
            # Open the raw file
            raw = mne.io.read_raw_fif(run_dir / file, allow_maxshield=True)

            # Create a BIDS path with the correct parameters
            bids_path = BIDSPath(subject=sub, session='01', run='0'+str(run),
                                 datatype='meg', root=BIDS_PATH)
            bids_path.task = TASK

            # Write the BIDS path from the raw file
            write_raw_bids(raw, bids_path=bids_path, overwrite=True)


# Putting the generated annotation files (one for each run) in the correct
# directories: the processed one and the regular one
for sub in os.listdir(BIDS_PATH):
    if sub.__contains__('sub-'):
        version = subj_version[sub[4:]]  # Find version of timings used
        annotation_folder = f'/home/is153802/code/experiments/formatting/decoding_tsv_v{version}'

        SUBJ_PATH_BIDS = BIDS_PATH / f'{sub}/ses-01/meg'
        # files = os.listdir(SUBJ_PATH_FILT)
        files_bids = os.listdir(SUBJ_PATH_BIDS)
        for file in files_bids:
            try:
                run = re.search(r"_run-0([^']*)_meg.fif", file).group(1)
                # print("File for which an events one will be created: "+file)
            except Exception:
                continue

            annot = f'{annotation_folder}/run{run}_v{version}.tsv'

            df = pd.read_csv(annot, sep='\t')
            df.to_csv(f'{SUBJ_PATH_BIDS}/{sub}_ses-01_task-{TASK}_run-0{run}_events.tsv', sep='\t')
            # print(f"File created:  + {sub}_ses-01_task-{TASK}_run-0{run}_events.tsv")

print(f"\n \n ***************************************************\
\n Script finished!\n \
***************************************************\
\n Folder created: \n For bids: {BIDS_PATH} ")