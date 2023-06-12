"""

DATASET related functions

"""


# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
# Tools
from pathlib import Path
import os
import subprocess
from utils import match_list, add_syntax, add_new_syntax

# CONST:

CHAPTERS = {
    1: "1-3",
    2: "4-6",
    3: "7-9",
    4: "10-12",
    5: "13-14",
    6: "15-19",
    7: "20-22",
    8: "23-25",
    9: "26-27",
}

# FUNC


def read_raw(subject, run_id, events_return=False, modality="visual"):
    print(f"Reading raw files for modality: {modality}")
    path = get_path(modality)
    task_map = {"auditory": "listen", "visual": "read", "fmri": "listen"}
    task = task_map[modality]
    print(f"\n Epoching for run {run_id}, subject: {subject}\n")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task,
        datatype="meg",
        root=path,
        run=run_id,
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.del_proj()  # To fix proj issues
    raw.pick_types(meg=True, stim=True)

    # Generate event_file path
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
    events = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

    # meta['word'] = meta['trial_type'].apply(lambda x: eval(x)['word'] if type(eval(x)) == dict else np.nan)
    # Initial wlength, as presented in the stimuli / triggers to match list
    meta["wlength"] = meta.word.apply(len)
    # Enriching the metadata with outside files:
    # path_syntax = get_code_path() / "data/syntax"
    path_syntax = get_code_path() / "data" / "syntax_new_no_punct"  # testing new syntax

    # Send raw metadata
    # meta = add_new_syntax(meta, path_syntax, int(run_id))
    meta = add_syntax(meta, path_syntax, int(run_id))

    # add sentence and word positions
    meta["sequence_id"] = np.cumsum(meta.is_last_word.shift(1, fill_value=False))
    for s, d in meta.groupby("sequence_id"):
        meta.loc[d.index, "word_id"] = range(len(d))

    # XXX FIXME
    # Making sure that there is no problem with words that contain ""
    meta.word = meta.word.str.replace('"', "")

    # Two cases for match list: is it auditory or visual ?
    if modality == 'auditory':
        word_events = events[events[:, 2] > 1]
        meg_delta = np.round(np.diff(word_events[:, 0]/raw.info['sfreq']))
        meta_delta = np.round(np.diff(meta.onset.values))
        i, j = match_list(meg_delta, meta_delta)
        assert len(i) > 1
        # events = events[i]  # events = words_events[i]

    # For auditory, we match on the time difference between triggers
    elif modality == "visual":
        # For visual, we match on the difference of word length 
        # encoded in the triggers
        # Here, events are the presented stimuli: with hyphens.
        # Have to make sure meta.word still contains the hyphens.
        # However, the meta.word might have lost the hyphens because
        # of the previous match hen adding syntax.

        i, j = match_list(events[:, 2], meta.wlength)
        assert len(i) > (0.9 * len(events))
        assert (events[i, 2] == meta.loc[j].wlength).mean() > 0.95

    meta["has_trigger"] = False
    meta.loc[j, "has_trigger"] = True

    # integrate events to meta for simplicity
    meta.loc[j, "start"] = events[i, 0] / raw.info["sfreq"]

    # preproc raw
    raw.load_data()
    raw = raw.filter(0.5, 20)

    if events_return:
        return raw, meta, events[i]

    else:
        return raw, meta


def get_path(name="visual"):
    path_file = Path("./../../data/data_path.txt")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
    if name == "visual":
        # TASK = "read"
        data = data / "LPP_MEG_visual"

    elif name == "auditory":
        # TASK = "listen"
        data = data / "LPP_MEG_auditory"
    elif name == "fmri":
        # TASK = "listen"
        data = data / "LPP_MEG_fMRI"
    else:
        print(f"{name} is an invalid name. \n\
        Current options: visual and auditory, fmri")

    return data


def get_code_path():
    path_file = Path("./../../data/origin.txt")
    with open(path_file, "r") as f:
        user = Path(f.readlines()[0].strip("\n"))
        user = str(user)
    if user == "XPS":
        # TASK = "read"
        data = get_path() / "../../code"
    elif user == "NS":
        # TASK = "listen"
        data = get_path() / "../../../../code/neurospin-petit-prince"
    elif user == "jeanzay":
        # TASK = "listen"
        data = Path("/gpfswork/rech/qtr/ulg98mt/code/neurospin-petit-prince")
    else:
        return f"{user} is an invalid name. \n\
        Current options: XPS and NS"
    return data


def get_subjects(path):
    subjects = pd.read_csv(str(path) + "/participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    # subjects = np.delete(subjects, subjects.shape[0]-1)
    # Let's sort this array before outputting it!
    int_subjects = np.sort([int(subj) for subj in subjects])
    subjects = [str(subj) for subj in int_subjects]

    return subjects


def add_embeddings(meta, run, level):

    """
    Function made to generate laser embeddings, store them,
    and add them to the metadata

    Does so for both constituent embeddings, and sentence ones


    """
    # Parse the metadata into constituents / sentences,
    # and generate txt files for each constituents / sentence
    # So that it can be parsed by LASER
    sentences = []
    current_sentence = []

    for index, row in meta.iterrows():

        # Append word to current sentence
        current_sentence.append(row['word'])

        # Check if end of sentence
        if level == 'sentence' and row['is_last_word']:
            # Join words into sentence string and append to list
            sentences.append(' '.join(current_sentence))
            # Reset current sentence
            current_sentence = []

        if level == 'constituent' and row['const_end']:
            # Join words into sentence string and append to list
            sentences.append(' '.join(current_sentence))
            # Reset current sentence
            current_sentence = []

    # Loop through sentences
    for i, sentence in enumerate(sentences):
        # Get sentence number
        sentence_num = i + 1

        # Create file name
        file_name = f'./embeds/txt/run{run}_{level}_{sentence_num}.txt'

        # Open text file 
        with open(file_name, 'w') as f:
            # Write sentence to file
            f.write(sentence)

    # Run LASER using the run number
    # path = Path('/home/is153802/github/LASER/tasks/embed')
    os.environ['LASER'] = '/home/is153802/github/LASER'

    for i, _ in enumerate(sentences):
        # Get sentence number
        sentence_num = i + 1

        txt_file = f"/home/is153802/code/decoding/local_testing/embeds/txt/run{run}_{level}_{sentence_num}.txt"
        emb_file = f"/home/is153802/code/decoding/local_testing/embeds/emb/run{run}_{level}_{sentence_num}.bin"
        if os.path.exists(emb_file):
            continue
        else:
            subprocess.run(['/bin/bash', '/home/is153802/github/LASER/tasks/embed/embed.sh', txt_file, emb_file])

    # Get the embeddings from the generated txt file, and add them to metadata
    dim = 1024
    embeddings = {}
    for index, sentence in enumerate(sentences):
        embeds = np.fromfile(
                f"{get_code_path()}/decoding/local_testing/embeds/emb/run{run}_{level}_{index+1}.bin",
                dtype=np.float32,
                count=-1,
                )
        embeds.resize(embeds.shape[0] // dim, dim)
        embeds = embeds.reshape(-1)
        embeddings[index] = embeds
    sent_index = 0
    embed_arrays = []
    for index, row in meta.iterrows():
        embed_arrays.append(embeddings[sent_index])
        # Check if end of sentence 
        if level == 'sentence' and row['is_last_word']:
            sent_index += 1
        elif level == 'constituent' and row['const_end']:
            sent_index += 1

    meta[f'embeds_{level}'] = embed_arrays
    
    return meta
