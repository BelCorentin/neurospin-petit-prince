#!/usr/bin/python

import sys
import os

import mne
from mne_bids import BIDSPath, write_raw_bids  
import re
import numpy as np


# try for one file first

MEG_data_dir = r'C:\Users\Coco\Desktop\CEA\MEG\sf_180213'.replace('\\','/')
# Get list of patients
sub_list = os.listdir(MEG_data_dir)
 
# For each patient
for sub in sub_list:
    # Get the list of runs
    sub_dir = os.path.join(MEG_data_dir,sub)
    bloc_list = os.listdir(sub_dir)
    run_total = len(bloc_list)
    for file in bloc_list:
        # Use a regular expression to get the run number in the file name
        run = re.search(r"_r([^']*)_raw", file).group(1)
        # Open the raw file
        raw = mne.io.read_raw_fif(os.path.join(sub_dir,file),allow_maxshield=True)

        # Create a BIDS path with the correct parameters 
        bids_path = BIDSPath(subject=sub, session='01', run=run,datatype='meg', root='./bids_dataset')
        bids_path.task = "None"

        # Write the BIDS path from the raw file
        write_raw_bids(raw, bids_path=bids_path,overwrite = True)

