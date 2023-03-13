"""

General functions for decoding purposes

"""
# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from wordfreq import zipf_frequency
from Levenshtein import editops
from tqdm.notebook import trange
from scipy.stats import pearsonr
import spacy

nlp = spacy.load("fr_core_news_sm")

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

## CONST

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


def decod(X, y):
    assert len(X) == len(y)
    # define data
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 10)))
    cv = KFold(15, shuffle=True, random_state=0)

    # fit predict
    n, n_chans, n_times = X.shape
    if y.ndim == 1:
        y = np.asarray(y).reshape(y.shape[0], 1)
    R = np.zeros((n_times, y.shape[1]))

    for t in range(n_times):
        print(".", end="")
        rs = []
        # y_pred = cross_val_predict(model, X[:, :, t], y, cv=cv)
        for train, test in cv.split(X):
            model.fit(X[train, :, t], y[train])
            y_pred = model.predict(X[test, :, t])
            r = correlate(y[test], y_pred)
            rs.append(r)
        R[t] = np.mean(rs)
        # R[t] = correlate(y, y_pred)

    return R


# Function to correlate
def correlate(X, Y):
    if X.ndim == 1:
        X = np.array(X)[:, None]
    if Y.ndim == 1:
        Y = np.array(Y)[:, None]
    X = X - X.mean(0)
    Y = Y - Y.mean(0)

    SX2 = (X**2).sum(0) ** 0.5
    SY2 = (Y**2).sum(0) ** 0.5
    SXY = (X * Y).sum(0)
    return SXY / (SX2 * SY2)


def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def get_syntax(file):
    with open(file, "r") as f:
        txt = f.readlines()

    # parse syntactic trees
    out = []
    for sequence_id, sent in enumerate(txt):
        splits = sent.split("=")

        for prev, token in zip(splits, splits[1:]):
            out.append(
                dict(
                    pos=prev.split("(")[-1].split()[0],
                    word_id=int(prev.split()[-1]),
                    word=token.split(")")[0],
                    n_closing=token.count(")"),
                    sequence_id=sequence_id,
                    is_last_word=False,
                )
            )
        out[-1]["is_last_word"] = True

    synt = pd.DataFrame(out)

    # add deal with apostrophe
    out = []
    for sent, d in synt.groupby("sequence_id"):
        for token in d.itertuples():
            for tok in token.word.split("'"):
                out.append(dict(word=tok, n_closing=1, is_last_word=False, pos="XXX"))
            out[-1]["n_closing"] = token.n_closing
            out[-1]["is_last_word"] = token.is_last_word
            out[-1]["pos"] = token.pos
    return pd.DataFrame(out)


def format_text(text):
    for char in "jlsmtncd":
        text = text.replace(f"{char}'", char)
    text = text.replace("œ", "oe")
    return text.lower()


def add_syntax(meta, syntax_path, run):
    # get basic annotations
    meta = meta.copy().reset_index(drop=True)

    # get syntactic annotations
    syntax_file = syntax_path / f"ch{CHAPTERS[run]}.syntax.txt"
    synt = get_syntax(syntax_file)

    # align
    meta_tokens = meta.word.fillna("XXXX").apply(format_text).values
    synt_tokens = synt.word.apply(format_text).values

    i, j = match_list(meta_tokens, synt_tokens)
    print(meta_tokens, synt_tokens)
    assert (len(i) / len(meta_tokens)) > 0.8

    for key, default_value in dict(n_closing=1, is_last_word=False, pos="XXX").items():
        meta[key] = default_value
        meta.loc[i, key] = synt.iloc[j][key].values

    content_pos = ("NC", "ADJ", "ADV", "VINF", "VS", "VPP", "V")
    meta["content_word"] = meta.pos.apply(
        lambda pos: pos in content_pos if isinstance(pos, str) else False
    )
    return meta


# TO change
def analysis(raw, meta, data_path):
    # load MEG data
    raw.load_data()
    raw.filter(0.5, 20.0, n_jobs=-1)

    # get metadata
    meta = add_syntax(meta, data_path, run)

    # epoch
    def mne_events(meta):
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta.start * raw.info["sfreq"]
        return dict(events=events, metadata=meta.reset_index())

    epochs = mne.Epochs(
        raw, **mne_events(meta), decim=20, tmin=-0.2, tmax=1.5, preload=True
    )
    epochs = epochs['kind=="word"']

    scores = dict()
    scores["n_closing"] = decod(epochs, "n_closing")
    scores["n_closing_notlast"] = decod(
        epochs["content_word and not is_last_word"], "n_closing"
    )
    scores["n_closing_noun_notlast"] = decod(
        epochs['pos=="NC" and not is_last_word'], "n_closing"
    )
    return scores


def decod(epochs, target):
    model = make_pipeline(StandardScaler(), RidgeCV())
    cv = KFold(n_splits=5)

    y = epochs.metadata[target].values
    r = np.zeros(len(epochs.times))
    for t in trange(len(epochs.times)):
        X = epochs.get_data()[:, :, t]
        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            r[t] += pearsonr(y_pred, y[test])[0]
    r /= cv.n_splits
    return r


def create_target(decoding_criterion, epochs):
    if decoding_criterion == "embeddings":
        embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
        embeddings = np.array([emb for emb in embeddings])
        return embeddings
    elif decoding_criterion == "word_length":
        return epochs.metadata.word.apply(len)
