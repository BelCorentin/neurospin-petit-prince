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


# Epoching and decoding


def epoch_data(subject, run_id, task, path, filter=True):

    enrich = Enrich()

    task = "read"
    print("Running the script on RAW data:")
    print(f"run {run_id}, subject: {subject}")
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=task,
        datatype="meg",
        root=path,
        run=run_id,
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.pick_types(meg=True, stim=True)
    raw.load_data()
    raw = raw.filter(0.5, 20)

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
    if bids_path.task == "read" and bids_path.subject == "2":
        word_length_meg = (
            events[:, 2] - 2048
        )  # Remove first event: chapter start and remove offset
    else:
        word_length_meg = events[:, 2]
    word_len_meta = meta.word.apply(len)
    i, j = match_list(word_len_meta, word_length_meg)
    events = events[j]
    # events = events[i]  # events = words_events[i]
    meta = meta.iloc[i].reset_index()
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

    chapter = CHAPTERS[int(run_id)]

    meta["start"] = events[:, 0] / raw.info["sfreq"]
    meta["condition"] = "sentence"
    meta = meta.sort_values("start").reset_index(drop=True)

    # add parsing data

    # Raw textual data
    path_txt = path / "../../code/data/txt_raw"  # for XPS
    # path_txt = path / "../../../../code/neurospin-petit-prince/data/txt_raw"

    # Syntax data
    path_syntax = path / "../../code/data/syntax"  # for XPS
    # path_txt = path / "../../../../code/neurospin-petit-prince/data/syntax"  # for NS

    # Enriching the metadata with outside files:

    # meta = enrich(meta, path_txt / f"ch{chapter}.txt")
    # print(meta)
    meta = add_syntax(meta, path_syntax, int(run_id))

    epochs = mne.Epochs(raw, **mne_events(meta, raw), decim=20, tmin=-0.2, tmax=0.8)

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


class Enrich:
    """Enrich word dataframes (e.g. mne.Epochs.metadata)
    with syntactic information"""

    def __init__(
        self,
    ):
        model = "fr_core_news_sm"
        if not spacy.util.is_package(model):
            spacy.cli.download(model)

        self.nlp = spacy.load(model)

    def __call__(self, meta, txt_file):

        # read text file
        with open(txt_file, "r") as f:
            txt = f.read().replace("\n", "")

        # parse text file
        doc = self.nlp(txt)

        # add parse information to metadata
        parse_annots = []
        for sent_id, sent in enumerate(doc.sents):
            # HERE ADD ERIC DE LA CLERGERIE parser instead
            closings = parse(sent)
            assert len(closings) == len(sent)
            for word, closing in zip(sent, closings):
                parse_annots.append(
                    dict(
                        word_index=word.i - sent[0].i,
                        sequence_id=sent_id,
                        sequence_uid=str(sent),
                        closing=closing,
                        match_token=word.text,
                    )
                )

        # align text file and meg metadata
        def format_text(text):
            for char in "jlsmtncd":
                text = text.replace(f"{char}'", char)
            text = text.replace("œ", "oe")
            return text.lower()

        meg_words = meta.word.fillna("######").values
        text_words = [format_text(w.text) for w in doc]

        i, j = match_list(meg_words, text_words)

        # deal with missed tokens (e.g. wrong spelling, punctuation)
        assert len(parse_annots) == len(text_words)
        parse_annots = pd.DataFrame(parse_annots)
        parse_annots.closing = parse_annots.closing.fillna(0)
        parse_annots["closing_"] = 0
        parse_annots["missed_closing"] = 0
        missing = np.setdiff1d(range(len(parse_annots)), j)
        for missed in missing:
            current_closing = parse_annots.iloc[missed].closing
            prev_word = parse_annots.iloc[[missed - 1]].index
            if prev_word[0] >= 0:
                parse_annots.loc[prev_word, "missed_closing"] = current_closing
        parse_annots.closing_ = parse_annots.closing + parse_annots.missed_closing

        # Add new columns to original mne.Epochs.metadata
        # fill columns
        columns = (
            "word_index",
            "sequence_id",
            "sequence_uid",
            "closing_",
            "match_token",
        )
        for column in columns:
            meta[column] = None
            meta.loc[meta.iloc[i].index, column] = parse_annots[column].iloc[j].values
        return meta


def parse(sent):
    "identifies the number of closing nodes"

    def is_closed(node, position):
        """JR quick code to know whether is a word is closed given a word position"""
        if node.i > position:
            return False
        for child in node.children:
            if child.i > position:
                return False
            if not is_closed(child, position):
                return False
        return True

    closeds = []
    for current in range(1, len(sent) + 1):
        closed = 0
        for position, word in enumerate(sent):  # [:current]
            closed += is_closed(word, current)
        closeds.append(closed)

    closing = np.r_[np.diff(closeds), closeds[-1]]
    return closing


def epoch_runs(subject, run, task, path):
    epochs = []
    for run_id in range(1, run + 1):
        print(".", end="")
        epo = epoch_data(subject, "%.2i" % run_id, task, path)
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
