# Notebooks

In this folder are ordered two notebooks:

## Analysis
analysis.ipynb: where are run most analysis, using the higher level function: analysis / analysis_subject. It is also used to plot results.

## Smaller scale analysis
testing_functions.ipynb: to try to new analysis, on smaller samples for example

## Testing
functions.ipynb: where all functions are being tested, and you can get a clear understanding on the input / output of each function, as well as the general pipeline.

# Functions

This folder regroups all the functions organized under two subscripts:

## Dataset

This dataset.py file contains all the functions that are needed
to either read, preprocess, populate metadata, epoch, or in general deal
with the MEG data, metadata etc..

All the more general functions, not necessarily linked to the LPP
dataset will be put in utils.py instead

The general pipeline, when calling the higher level function: 

analysis_subject

Goes like:

analysis_subject, that returns the decoding score, for the parameters:
a subject, 
a modality (auditory or visual), 
a start (onset or offset), 
a level (word, constituent, sentence), 
a decoding_criterion (LASER embeddings, BOW embeddings, 1st word embeddings, ...)

calls itself 2 main functions:

    populate_metadata_epochs, that, in order to build epochs based on the previous parameters, will call:

        enrich_metadata(meta) that will populate the metadata, allowing us to match events and metadata, and use them for decoding

        select_meta_subset(meta, level, decoding_criterion) that will select the subset of events to use for epoching

        epoch_on_selection(raw, sel, start, level) that will build an epoch object on the previous selection

        apply_baseline that applies a baseline to the epochs object
        ## Note: for later, this apply_baseline would have to be applied before the epochs selection, now the question is how?

    and decoding_from_criterion, that will call:

        generate_embeddings: generate embeddings for the given epochss

        decod_Xy that will simply evaluate how much
            information can be infered from X to predict y.
            Training a RidgeCV and then calculating the
            Pearson R between predicted and test y


## Utils

This utils.py file will regroup all the functions that are not 
directly linked to the LPP dataset, such as training a 
Ridge Regression, generating embeddings from a text, etc..
