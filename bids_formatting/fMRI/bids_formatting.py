import pandas as pd
import os 

# Const

filename_cn = "lppCN_word_information.csv"
filename_fr = "lppFR_word_information.csv"
filename_en = "lppEN_word_information.csv"


""" 
First step:
Create the three versions of the events.tsv 
and 
Populate the corresponding folders
"""

# Create the dfs of the annotations
try:
    df_cn = pd.read_csv(f'./annotation/CN/{filename_cn}')
    df_fr = pd.read_csv(f'./annotation/FR/{filename_fr}')
    df_en = pd.read_csv(f'./annotation/EN/{filename_en}')

except:
    print("Error: either this folder is already in the BIDS format, or the annotation folder is missing")

# Get the list of subjects (for which there will be an events.tsv file created)
filenames= os.listdir("./")
result = []

# Only keep folders
for filename in filenames: # loop through all the files and folders
    if os.path.isdir(os.path.join(os.path.abspath("."), filename)): # check whether the current object is a folder or not
        result.append(filename)

result.sort()

# Put them in their respective folders, with a corresponding name
for sub in result:
    # Save the events.tsv file in the folder, with the subj number in the filename
    if(sub.__contains__("sub-CN")):
        df_cn.to_csv(f'./{sub}/func/{sub}_task-IppCN_events.tsv',sep = '\t')
    elif(sub.__contains__("sub-FR")):
        df_fr.to_csv(f'./{sub}/func/{sub}_task-IppFR_events.tsv',sep = '\t')
    elif(sub.__contains__("sub-EN")):
        df_en.to_csv(f'./{sub}/func/{sub}_task-IppEN_events.tsv',sep = '\t')


print("Events annotations have successfully been updated to the BIDS format")
