# Imports ######

from __future__ import annotations
import pandas as pd
import os
import re
import mne
from pathlib import Path
from mne_bids import BIDSPath, write_raw_bids

# Copying the new files first to the raw directory:

# cp -ru ~/neurospin/acquisition/neuromag/data/petit_prince
# /home/is153802/workspace_LPP/data/MEG/LPP/raw

# 1) Raw format to BIDS #####

#  CONST ###
BASE_PATH = Path('/home/is153802/data/')
# BIDS_PATH = BASE_PATH / 'LPP_bids'
BIDS_PATH = BASE_PATH / 'LPP_MEG_auditory'
RAW_DATA_PATH = BASE_PATH / 'raw_auditory'
TASK = 'listen'
annotation_folder = './annotations'



dict_nip_to_sn = {'ae_140329': '2', 'cc_150418': '3', 'cl_220500': '5',
                  'fr_190151': '7', 'hg_220389': '8', 'js_180232': '9',
                  'kc_160388': '10', 'kp_210315': '11', 'lg_170436': '12',
                  'lq_180242': '13', 'mb_180076': '14', 'mf_180020': '15',
                  'ml_100438': '16', 'ml_180010': '17', 'ml_220421': '18',
                  'pl_170230': '19', 'rt_220104': '20', 'sa_170217': '21',
                  'sf_180213': '22', 'eg_220435': '23', 'cl_190429': '24',
                  'ap_220150': '25', 'df_130078': '26', 'lp_090137': '4',
                  'ya_170284': '6', 'vc_200442': '27', 'cj_090334': '1',
                  'se_210401': '28', 'bk_220247': '29', 'sb_220619': '30',
                  'ya_220605': '31', 'nl_220497': '32', 'ay_220681': '33',
                  'td_220613': '34', 'jp_220691': '35', 'ap_120157': '36',
                  'md_220654': '37', 'jv_220664': '38', 'tc_200507': '39',
                  'ag_220624': '40', 'rh_220668': '41', 'cl_110710': '42',
                  'am_220666': '43', 'ml_220557': '44', 'dt_220722': '45',
                  'ia_220711': '46', 'lg_220612': '47', 'jm_220720': '48',
                  'cl_220706': '49', 'mf_210328': '50', 'ad_220723': '51',
                  'ar_220740': '52', 'al_220758': '53', 'rz_220739': '54',
                  'pj_150414': '55',
                  'gt_150298': '56',
                  'ad_140107': '57',
                  'gl_150316': '58',
                  }


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
            # Use a regular expression to get the run number in the file name
            # assert file.name.startswith('_r')
            # assert file.name.endswith('_raw')
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

# Formatting to be read by the MEG-MASC script

df = pd.read_csv('./annotations/annotation FR lppFR_word_information.csv')

df_ = pd.DataFrame(df['onset'])

df_['duration'] = df['offset'] - df['onset']

df_['trial_type'] = [{"kind": "word",
                     'word': df.loc[i, 'word'].replace("'", "")}
                     for i in range(df_.shape[0])]

df_.to_csv('./annotations/annotation_processed.tsv', sep='\t', index=False)

# Segmenting these annotations into different files for different runs

df = pd.read_csv('./annotations/annotation_processed.tsv', sep='\t')

df1 = df.iloc[0:1632, :]
df1.to_csv('./annotations/annotation_processed1.tsv', sep='\t', index=False)

# print(df1)
df2 = df.iloc[1632:3419, :]
df2.to_csv('./annotations/annotation_processed2.tsv', sep='\t', index=False)

# print(df2)
df3 = df.iloc[3419:5295, :]
df3.to_csv('./annotations/annotation_processed3.tsv', sep='\t', index=False)

# print(df3)
df4 = df.iloc[5295:6945, :]
df4.to_csv('./annotations/annotation_processed4.tsv', sep='\t', index=False)

# print(df4)
df5 = df.iloc[6945:8472, :]
df5.to_csv('./annotations/annotation_processed5.tsv', sep='\t', index=False)

# print(df5)
df6 = df.iloc[8472:10330, :]
df6.to_csv('./annotations/annotation_processed6.tsv', sep='\t', index=False)

# print(df6)
df7 = df.iloc[10330:12042, :]
df7.to_csv('./annotations/annotation_processed7.tsv', sep='\t', index=False)

# print(df7)
df8 = df.iloc[12042:13581, :]
df8.to_csv('./annotations/annotation_processed8.tsv', sep='\t', index=False)

# print(df8)
df9 = df.iloc[13581:15391, :]
df9.to_csv('./annotations/annotation_processed9.tsv', sep='\t', index=False)


# Putting the generated annotation files (one for each run) in the correct
# directories: the processed one and the regular one

for sub in os.listdir(BIDS_PATH):
    if sub.__contains__('sub-'):
        # SUBJ_PATH_FILT = PROC_DATA_PATH / f'{sub}/ses-01/meg'
        SUBJ_PATH_BIDS = BIDS_PATH / f'{sub}/ses-01/meg'
        # files = os.listdir(SUBJ_PATH_FILT)
        files_bids = os.listdir(SUBJ_PATH_BIDS)
        # for file in files:
        #     try:
        #         run = re.search(r"_run-0([^']*)_proc-filt_raw.fif",
        #                         file).group(1)
        #         print("File for which an events one will be created: "+file)
        #     except Exception:
        #         continue

        #     annot = f'{annotation_folder}/annotation_processed{run}.tsv'
        #     df = pd.read_csv(annot, sep='\t')
        #     df.to_csv(f'{SUBJ_PATH_FILT}/{sub}_ses-01_task-{TASK}_run-0{run}_events.tsv', sep='\t')
        #     print(f"File created:  + {sub}_ses-01_task-{TASK}_run-0{run}_events.tsv")
        for file in files_bids:
            try:
                run = re.search(r"_run-0([^']*)_meg.fif", file).group(1)
                # print("File for which an events one will be created: "+file)
            except Exception:
                continue

            annot = f'{annotation_folder}/annotation_processed{run}.tsv'

            df = pd.read_csv(annot, sep='\t')
            df.to_csv(f'{SUBJ_PATH_BIDS}/{sub}_ses-01_task-{TASK}_run-0{run}_events.tsv', sep='\t')
            # print(f"File created:  + {sub}_ses-01_task-{TASK}_run-0{run}_events.tsv")

print(f"\n \n ***************************************************\
\n Script finished!\n \
***************************************************\
\n Folder created: \n For bids: {BIDS_PATH} \n ")
