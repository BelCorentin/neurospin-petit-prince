# Repository neurospin-petit-prince

This repo contains all the code for the Le Petit Prince experiment.

It is divided into 5 parts: 

- data: all the data files (parsers output, embeddings, path files)
## Note: 
You need to create both the data_path.txt and the origin.txt files
e.g:
``` 
echo "/my/path/to/lpp_dataset/" > data_path.txt
echo "my_computer" > origin.txt
```
Then add your my_computer to the function get_code_path in decoding/functions/utils.py

- dataset_formatting: all the ntbks and .py scripts to generate a BIDS dataset from the raw .fif files, and raw anat MRI

- decoding: ntbks, python functions where the decoding analysis is being run

- documentation: documentation about the LPP_experiment

- LPP_experiment: code and documentation about the visual MEG Experiment, STIM, etc.. 
## Note:
For the auditory experiment, it is available in this repository:
https://github.com/chrplr/lpp-paradigm/tree/merging-meg-fmri

- other_analysis: other analysis apart from the decoding (data mining, testing, etc..)


To start running the decoding notebooks:

```
pip install -r requirements.txt

cd decoding/ntbk

jupyter notebook .
```

## Note:

In order to start running the analysis, you need to have the LPP_MEG_{modality} folder available, and the data/data_path.txt & data/origin.txt both created, and integrated in the code:
decoding/functions/utils.py: get_code_path():91






