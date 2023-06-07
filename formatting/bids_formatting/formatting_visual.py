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
                  'eg_220435': '6',
                  'gb_220537': '7',
                  'am_230061': '8',
                  'jd_220636': '9',
                  'jm_100042': '10',
                  'es_220094': '11',
                  'aa_230065': '12',
                  'll_180197': '13',
                  'am_220107': '14',
                  'ci_210203': '15',
                  'pl_230089': '16',
                  'jm_230095': '17',
                  'aj_220730': '18',
                  'fv_230122': '19',
                  'vr_230124': '20',
                  'gd_230114': '21',
                  'um_230121': '22',
                  'tv_230127': '23',
                  'ym_220242': '24',
                  'ac_230112': '25',
                  'po_230175': '26',
                  'jr_230176': '27',
                  'ml_110339': '28',
                  'li_230200': '29',
                  'nv_230178': '30',
                  'jf_230204': '31',
                  'cd_230186': '32',
                  're_230199': '33',
                  'nc_230202': '34',
                  'mm_230182': '35',
                  'sm_230170': '36',
                  'fd_110104': '37'
                  }

# 1) Raw format to BIDS #####

#  CONST ###
BASE_PATH = Path('/home/is153802/data')
# BIDS_PATH = BASE_PATH / 'LPP_bids'
BIDS_PATH = BASE_PATH / 'LPP_MEG_visual'
RAW_DATA_PATH = BASE_PATH / 'raw_visual'
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
        sub_number = int(sub[4:])
        if sub_number == 1:
            version = 1
        else:
            version = 2  # Find version of timings used
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
