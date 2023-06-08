"""

DATASET related functions

"""


# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Tools
from pathlib import Path
from utils import match_list, add_syntax

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

    meta['word'] = meta['trial_type'].apply(lambda x: eval(x)['word'] if type(eval(x)) == dict else np.nan)
    # Initial wlength, as presented in the stimuli / triggers to match list
    meta["wlength"] = meta.word.apply(len)
    # Enriching the metadata with outside files:
    # path_syntax = get_code_path() / "data/syntax"
    path_syntax = get_code_path() / "data" / "syntax_new_no_punct"  # testing new syntax

    # Send raw metadata
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
        assert len(i) > 1000
        # events = events[i]  # events = words_events[i]

    # For auditory, we match on the time difference between triggers
    elif modality == "visual":
        # For visual, we match on the difference of word length encoded in the triggers
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


def sentence_epochs(subject):
    all_epochs = []
    for run_id in range(1, 10):
        print(".", end="")
        raw, meta = read_raw(subject=subject, run_id=run_id)

        # FIXME
        meta = meta.query("has_trigger").reset_index(drop=True)
        mne_events = np.ones((len(meta), 3), dtype=int)
        mne_events[:, 0] = meta.start * raw.info["sfreq"]
        sent_events = meta.query("word_id==0")
        assert len(sent_events)

        epochs = mne.Epochs(
            raw,
            mne_events[sent_events.index],
            metadata=sent_events,
            tmin=-0.500,
            tmax=2.0,
            decim=10,
            preload=True,
        )
        all_epochs.append(epochs)

    for epo in all_epochs:
        epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs


def word_epochs(subject):
    all_epochs = []
    for run_id in range(1, 9):
        print(".", end="")
        raw, meta = read_raw(subject=subject, run_id=run_id)

        # FIXME
        meta = meta.query("has_trigger").reset_index(drop=True)
        mne_events = np.ones((len(meta), 3), dtype=int)
        mne_events[:, 0] = meta.start * raw.info["sfreq"]
        word_events = meta
        assert len(word_events)

        epochs = mne.Epochs(
            raw,
            mne_events[word_events.index],
            metadata=word_events,
            tmin=-0.500,
            tmax=2.0,
            decim=10,
            preload=True,
        )
        all_epochs.append(epochs)

    for epo in all_epochs:
        epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs


def constituent_epochs(subject):
    all_epochs = []
    for run_id in range(1, 10):
        print(".", end="")
        raw, meta = read_raw(subject=subject, run_id=run_id)

        # FIXME
        meta = meta.query("has_trigger").reset_index(drop=True)
        mne_events = np.ones((len(meta), 3), dtype=int)
        mne_events[:, 0] = meta.start * raw.info["sfreq"]
        const_events = meta.query("constituent_start==True")
        assert len(const_events)

        epochs = mne.Epochs(
            raw,
            mne_events[const_events.index],
            metadata=const_events,
            tmin=-0.500,
            tmax=2.0,
            decim=10,
            preload=True,
        )
        all_epochs.append(epochs)

    for epo in all_epochs:
        epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs


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


# Epoching and decoding
def epoch_data(
    subject,
    run_id,
    task,
    path,
    baseline_min=-0.2,
    baseline_max=0.8,
    filter=True,
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

    # LPP Syntax data
    path_syntax = get_code_path() / "data/syntax"
    meta = add_syntax(meta, path_syntax, int(run_id))

    # add sentence and word positions
    meta["sequence_id"] = np.cumsum(meta.is_last_word.shift(1, fill_value=False))
    for s, d in meta.groupby("sequence_id"):
        meta.loc[d.index, "word_id"] = range(len(d))

    # Enriching the metadata with simple operations:
    # end of sentence information
    end_of_sentence = [
        True
        if str(meta.word.iloc[i]).__contains__(".")
        or str(meta.word.iloc[i]).__contains__("?")
        or str(meta.word.iloc[i]).__contains__("!")
        else False
        for i, _ in enumerate(meta.values[:-1])
    ]
    end_of_sentence.append(True)
    meta["sentence_end"] = end_of_sentence

    # sentence start information
    list_word_start = [True]
    list_word_start_to_add = [
        True if meta.sentence_end.iloc[i - 1] else False
        for i in np.arange(1, meta.shape[0])
    ]
    for boolean in list_word_start_to_add:
        list_word_start.append(boolean)
    meta["sentence_start"] = list_word_start

    # Will be done in a singular use case
    # # laser embeddings information
    # dim = 1024
    # embeds = np.fromfile(
    #     f"{get_code_path()}/data/laser_embeddings/emb_{CHAPTERS[int(run_id)]}.bin",
    #     dtype=np.float32,
    #     count=-1,
    # )
    # embeds.resize(embeds.shape[0] // dim, dim)
    # assert embeds.shape[0] == meta.shape[0]
    # meta["laser"] = [emb for emb in embeds]

    # constituent end information
    meta["constituent_end"] = [
        True if closing > 1 else False for i, closing in enumerate(meta.n_closing)
    ]

    # constituent start information
    list_constituent_start = [True]
    list_constituent_start_to_add = [
        True if meta.constituent_end.iloc[i - 1] else False
        for i in np.arange(1, meta.shape[0])
    ]
    for boolean in list_constituent_start_to_add:
        list_constituent_start.append(boolean)
    meta["constituent_start"] = list_constituent_start

    # Here, the trigger value encoded the word length
    # which helps us realign triggers
    # From the event file / from the MEG events
    word_len_meta = meta.word.apply(len)
    i, j = match_list(word_len_meta, word_length_meg)
    events = events[j]
    assert len(i) / meta.shape[0] > 0.8
    meta = meta.iloc[i].reset_index()

    # The start parameter will help us
    # keep the link between raw events and metadata
    meta["start"] = events[:, 0] / raw.info["sfreq"]
    meta["condition"] = "sentence"
    meta = meta.sort_values("start").reset_index(drop=True)
    meta["word_start"] = meta["start"]
    meta["word_end"] = meta["word_start"] + meta["duration"]

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


def epochs_slice(epochs, column, value=True, equal=True):
    meta = epochs.metadata
    if equal:
        subset = meta[meta[column] == value].level_0
    elif equal == "sup":
        subset = meta[meta[column] >= value].level_0
    elif equal == "inf":
        subset = meta[meta[column] <= value].level_0
    return epochs[subset]


#
# DEBUG FUNCTIONS
#


def word_epochs_debug(subject, run):
    all_epochs = []
    for run_id in range(1, run):
        print(".", end="")
        raw, meta = read_raw(subject=subject, run_id=run_id)

        # FIXME
        meta = meta.query("has_trigger").reset_index(drop=True)
        mne_events = np.ones((len(meta), 3), dtype=int)
        mne_events[:, 0] = meta.start * raw.info["sfreq"]
        word_events = meta
        assert len(word_events)

        epochs = mne.Epochs(
            raw,
            mne_events[word_events.index],
            metadata=word_events,
            tmin=-0.500,
            tmax=1.0,
            decim=10,
            preload=True,
        )
        all_epochs.append(epochs)

    for epo in all_epochs:
        epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs


def sentence_epochs_debug(subject, run):
    all_epochs = []
    for run_id in range(1, run):
        print(".", end="")
        raw, meta = read_raw(subject=subject, run_id=run_id)

        # FIXME
        meta = meta.query("has_trigger").reset_index(drop=True)
        mne_events = np.ones((len(meta), 3), dtype=int)
        mne_events[:, 0] = meta.start * raw.info["sfreq"]
        sent_events = meta.query("word_id==0")
        assert len(sent_events)

        # # laser embeddings information
        dim = 1024
        embeds = np.fromfile(
            f"{get_code_path()}/data/laser_embeddings/emb_{CHAPTERS[int(run_id)]}.bin",
            dtype=np.float32,
            count=-1,
        )
        embeds.resize(embeds.shape[0] // dim, dim)

        end_of_sentence = [
            True
            if str(meta.word.iloc[i]).__contains__(".")
            or str(meta.word.iloc[i]).__contains__("?")
            or str(meta.word.iloc[i]).__contains__("!")
            else False
            for i, _ in enumerate(meta.values[:-1])
        ]
        end_of_sentence.append(True)
        meta["sentence_end"] = end_of_sentence

        sent_events = meta.query("sentence_end==True")
        sent_events["laser"] = [emb for emb in embeds]
        assert embeds.shape[0] == sent_events.shape[0]

        epochs = mne.Epochs(
            raw,
            mne_events[sent_events.index],
            metadata=sent_events,
            tmin=-3.0,
            tmax=1.0,
            decim=10,
            preload=True,
        )
        all_epochs.append(epochs)

    for epo in all_epochs:
        epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs
