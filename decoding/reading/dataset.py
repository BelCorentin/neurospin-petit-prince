"""

DATASET related functions

"""

# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from wordfreq import zipf_frequency
from Levenshtein import editops

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from utils import match_list, add_syntax
import spacy

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


def get_path(name="LPP_read"):
    path_file = Path("./../../data/data_path.txt")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
    if name == "LPP_read":
        # TASK = "read"
        data = data / "BIDS_lecture"

    elif name == "LPP_listen":
        # TASK = "listen"
        data = data / "BIDS"
    else:
        return f"{name} is an invalid name. \n\
        Current options: LPP_read and LPP_listen"
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
    else:
        return f"{user} is an invalid name. \n\
        Current options: XPS and NS"
    return data


# Epoching and decoding


def epoch_data(
    subject,
    run_id,
    task,
    path,
    baseline_min=-0.2,
    baseline_max=0.8,
    filter=True,
    epoch_on="word",
    reference="start",
):

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
    raw.load_data()
    raw = raw.filter(0.5, 20)
    print(raw.info["sfreq"])
    print(type(raw.info["sfreq"]))

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
    if (
        bids_path.task == "read" and bids_path.subject == "2"
    ):  # A trigger value bug for this subject
        word_length_meg = (
            events[:, 2] - 2048
        )  # Remove first event: chapter start and remove offset
    else:
        word_length_meg = events[:, 2]
    # Here, the trigger value encoded the word length
    # which helps us realign triggers
    # From the event file / from the MEG events
    word_len_meta = meta.word.apply(len)
    i, j = match_list(word_len_meta, word_length_meg)
    events = events[j]
    meta = meta.iloc[i].reset_index()
    # The start parameter will help us
    # keep the link between raw events and metadata
    meta["start"] = events[:, 0] / raw.info["sfreq"]
    meta["condition"] = "sentence"
    meta = meta.sort_values("start").reset_index(drop=True)

    # Raw LPP textual data
    path_txt = get_code_path() / "data/txt_raw"
    # LPP Syntax data
    path_syntax = get_code_path() / "data/syntax"

    # Enriching the metadata with outside files:
    meta = add_syntax(meta, path_syntax, int(run_id))
    # Add the information on the sentence ending:
    # Only works for reading: TO FIX for listening... to see with Christophe
    # Also: only works for v2 (subject 1 (me) doesn't work )
    end_of_sentence = [
        True if meta.onset.iloc[i + 1] - meta.onset.iloc[i] > 0.7 else False
        for i, _ in enumerate(meta.values[:-1])
    ]
    end_of_sentence.append(True)
    meta["sentence_end"] = end_of_sentence

    # We are considering different cases:
    # Are we epoching on words, sentences, or constituents?
    # Different epoching for different analysis
    if epoch_on == "word" and reference == "start":
        # Default case, so nothing to change
        # Could be removed but kept for easy of reading
        happy = True
    # Word end
    if epoch_on == "word" and reference == "end":
        # Little hack: not really pretty but does the job
        # As epoching again uses the start column, we rename it like that
        # But it should be meta["end"] instead...
        meta["start"] = [row["start"] + row["duration"] for i, row in meta.iterrows()]

    # Sentence end
    elif epoch_on == "sentence" and reference == "end":
        # Add a LASER embeddings column for decoding
        dim = 1024
        embeds = np.fromfile(
            f"{get_code_path()}/data/laser_embeddings/emb_{CHAPTERS[int(run_id)]}.bin",
            dtype=np.float32,
            count=-1,
        )
        embeds.resize(embeds.shape[0] // dim, dim)
        print(meta)
        column = "sentence_end"
        value = True
        meta = meta[meta[column] == value]
        # TODO: rerun LASER
        print(embeds.shape[0], meta.shape[0])
        assert embeds.shape[0] == meta.shape[0]
        meta["laser"] = [emb for emb in embeds]
        print("Added embeddings")
    # Sentence start
    elif epoch_on == "sentence" and reference == "start":
        # Create a sentence-start column:
        # list_word_start = [
        #     True
        #     for i, is_last_word in enumerate(meta.is_last_word[:-1])
        #     if meta.is_last_word[i + 1]
        # ]
        list_word_start = [True]
        list_word_start_to_add = [
            True if meta.sentence_end[i - 1] else False
            for i, _ in enumerate(meta.sentence_end[1:])
        ]
        for boolean in list_word_start_to_add:
            list_word_start.append(boolean)
        meta["sentence_start"] = list_word_start
        column = "sentence_start"
        value = True
        meta = meta[meta[column] == value]
    # Constituent start
    elif epoch_on == "constituent" and reference == "start":
        # Create a constituent-start column:
        meta["constituent_start"] = [
            True for i, _ in enumerate(meta.is_last_word[1:]) if meta.n_closing > 1
        ]
        column = "constituent_start"
        value = True
        meta = meta[meta[column] == value]
    # Constituent end
    elif epoch_on == "constituent" and reference == "end":
        # Create a constituent-start column:
        meta["constituent_start"] = [
            True for i, _ in enumerate(meta.is_last_word[1:]) if meta.n_closing > 1
        ]
        column = "constituent_start"
        value = True
        meta = meta[meta[column] == value]
    epochs = mne.Epochs(
        raw, **mne_events(meta, raw), decim=20, tmin=baseline_min, tmax=baseline_max
    )
    # epochs = epochs['kind=="word"']
    # epochs.metadata["closing"] = epochs.metadata.closing_.fillna(0)
    epochs.load_data()
    epochs = epochs.pick_types(meg=True, stim=False, misc=False)
    data = epochs.get_data()

    # Scaling the data
    n_words, n_chans, n_times = data.shape
    vec = data.transpose(0, 2, 1).reshape(-1, n_chans)
    scaler = RobustScaler()
    idx = np.arange(len(vec))
    np.random.shuffle(idx)
    vec = scaler.fit(vec[idx[:20_000]]).transform(vec)
    # To try: sigmas = 7 or 15
    sigma = 7
    vec = np.clip(vec, -sigma, sigma)
    epochs._data[:, :, :] = (
        scaler.inverse_transform(vec)
        .reshape(n_words, n_times, n_chans)
        .transpose(0, 2, 1)
    )
    return epochs


def get_subjects(path):
    subjects = pd.read_csv(str(path) + "/participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    # subjects = np.delete(subjects, subjects.shape[0]-1)
    # Let's sort this array before outputting it!
    int_subjects = np.sort([int(subj) for subj in subjects])
    subjects = [str(subj) for subj in int_subjects]

    return subjects


def mne_events(meta, raw):
    events = np.ones((len(meta), 3), dtype=int)
    events[:, 0] = meta.start * raw.info["sfreq"]
    return dict(events=events, metadata=meta.reset_index())


def concac_runs(subject, task, path):
    RUN = 9
    epochs = []
    filter = True
    for run_id in range(1, RUN + 1):
        print(".", end="")
        epo = epoch_data(subject, "%.2i" % run_id, task, path, filter)
        epo.metadata["label"] = f"run_{run_id}"
        epochs.append(epo)
    for epo in epochs:
        epo.info["dev_head_t"] = epochs[0].info["dev_head_t"]

    epochs = mne.concatenate_epochs(epochs)
    return epochs


def epoch_runs(
    subject,
    run,
    task,
    path,
    baseline_min,
    baseline_max,
    epoch_on="word",
    reference="start",
):
    epochs = []
    for run_id in range(1, run + 1):
        print(".", end="")
        epo = epoch_data(
            subject,
            "%.2i" % run_id,
            task,
            path,
            baseline_min,
            baseline_max,
            epoch_on=epoch_on,
            reference=reference,
        )
        epo.metadata["label"] = f"run_{run_id}"
        epochs.append(epo)
    for epo in epochs:
        epo.info["dev_head_t"] = epochs[0].info["dev_head_t"]

    epochs = mne.concatenate_epochs(epochs)

    # Handling the data structure
    epochs.metadata["kind"] = epochs.metadata.trial_type.apply(
        lambda s: eval(s)["kind"]
    )
    epochs.metadata["word"] = epochs.metadata.trial_type.apply(
        lambda s: eval(s)["word"]
    )
    return epochs


def epoch_subjects(
    subjects,
    RUN,
    task,
    path,
    baseline_min,
    baseline_max,
    epoch_on="word",
    reference="start",
):
    epochs = []
    for subject in subjects:
        epo = epoch_runs(
            subject,
            RUN,
            task,
            path,
            baseline_min,
            baseline_max,
            epoch_on=epoch_on,
            reference=reference,
        )
        epochs.append(epo)
    for epo in epochs:
        epo.info["dev_head_t"] = epochs[0].info["dev_head_t"]

    epochs = mne.concatenate_epochs(epochs)

    # Handling the data structure
    epochs.metadata["kind"] = epochs.metadata.trial_type.apply(
        lambda s: eval(s)["kind"]
    )
    epochs.metadata["word"] = epochs.metadata.trial_type.apply(
        lambda s: eval(s)["word"]
    )
    return epochs


def epochs_slice(epochs, column, value, equal=True):
    meta = epochs.metadata
    if equal:
        subset = meta[meta[column] == value].level_0
    elif equal == "sup":
        subset = meta[meta[column] >= value].level_0
    elif equal == "inf":
        subset = meta[meta[column] <= value].level_0
    return epochs[subset]
