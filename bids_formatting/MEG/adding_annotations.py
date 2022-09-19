import os 
import pandas as pd
import re 

path = '/home/is153802/workspace_LPP/data/MEG/LPP/LPP_bids'


for sub in os.listdir(path):
    dir2 = os.path.join(path,f'{sub}/ses-01/meg')
    try:
        files = os.listdir(dir2)
        #print(files)
        for file in files:
            # run = re.search(r"_run-([^']*)_meg.fif", file).group(1)
            try:
                run = re.search(r"_run-0([^']*)_meg.fif", file).group(1)
                print("File for which an events one will be created: "+file)
            except:
                continue
            annot = f'/home/is153802/workspace_LPP/code/neurospin-petit-prince/bids_formatting/MEG/annotation_processed{run}.tsv'
            df = pd.read_csv(annot,sep='\t')
            df.to_csv(f'{dir2}/{sub}_ses-01_task-rest_run-0{run}_events.tsv',sep='\t')
            print(f"File created:  + {sub}_ses-01_task-rest_run-0{run}_events.tsv")
            
    except:
        print(f'{dir2} skipped: not a dir')
            
