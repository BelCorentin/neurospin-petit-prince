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
import re
import string

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


# def decod(X, y):
#     assert len(X) == len(y)
#     # define data
#     model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 10)))
#     cv = KFold(15, shuffle=True, random_state=0)

#     # fit predict
#     n, n_chans, n_times = X.shape
#     if y.ndim == 1:
#         y = np.asarray(y).reshape(y.shape[0], 1)
#     R = np.zeros((n_times, y.shape[1]))

#     for t in range(n_times):
#         print(".", end="")
#         rs = []
#         # y_pred = cross_val_predict(model, X[:, :, t], y, cv=cv)
#         for train, test in cv.split(X):
#             model.fit(X[train, :, t], y[train])
#             y_pred = model.predict(X[test, :, t])
#             r = correlate(y[test], y_pred)
#             rs.append(r)
#         R[t] = np.mean(rs)
#         # R[t] = correlate(y, y_pred)

#     return R


# Function to return the Pearson correlation
# Between X and Y
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
    """
    Add the syntactic information to the existing metadata
    Feed the existing: word information with
    n_closing, number of closing nodes
    is_last_word, boolean value
    pos, the position in the sentence

    """
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


def format_text_meta(text):
    """
    Simple function to make sure there is
    no problematic characters
    """
    # I comment these lines as the new parsers handles it !
    for char in "jlsmtncd":
        text = text.replace(f"{char}'", char)

    text = text.replace("œ", "oe")
    return text.lower()


def format_text_syntax(text):
    """
    Simple function to make sure there is
    no problematic characters
    """
    # I comment these lines as the new parsers handles it !
    # for char in "jlsmtncd":
    #     text = text.replace(f"{char}'", char)
    # Instead:
    if text.strip(string.punctuation).eq(""):
        return ""

    text = df.replace("œ", "oe")
    return text.lower()


def format_text(text):
    """
    Simple function to make sure there is
    no problematic characters
    """
    # for char in "jlsmtncd":
    #     text = text.replace(f"{char}'", char)
    text = text.replace("œ", "oe")

    return text.lower()


def add_syntax(meta, syntax_path, run):
    """
    Use the get_syntax function to add it directly to
    the metadata from the epochs
    Basic problem with match list: new syntax has words like:
    "j" "avais"
    meta has:
    "j'avais"
    """
    # get basic annotations
    meta = meta.copy().reset_index(drop=True)

    # get syntactic annotations
    # syntax_file = syntax_path / f"ch{CHAPTERS[run]}.syntax.txt"
    syntax_file = (
        syntax_path / f"run{run}_v2_0.25_0.5-tokenized.syntax.txt"
    )  # testing new syntax
    synt = get_syntax(syntax_file)

    # Clean the meta tokens to match synt tokens
    meta_tokens = meta.word.fillna("XXXX").apply(format_text).values
    # Get the word after the hyphen to match the synt tokens
    meta_tokens = [stri.split("'")[1] if "'" in stri else stri for stri in meta.word]
    # Remove the punctuation
    translator = str.maketrans("", "", string.punctuation)
    meta_tokens = [stri.translate(translator) for stri in meta_tokens]

    # Handle synt tokens: they are split by hyphen
    synt_tokens = synt.word.apply(format_text).values
    # Remove the empty strings and ponct
    # punctuation_chars = set(string.punctuation)
    # synt_tokens = [
    #     stri
    #     for stri in synt_tokens
    #     if stri.strip() != "" and not any(char in punctuation_chars for char in stri)
    # ]

    i, j = match_list(meta_tokens, synt_tokens)
    assert (len(i) / len(meta_tokens)) > 0.8

    for key, default_value in dict(n_closing=1, is_last_word=False, pos="XXX").items():
        meta[key] = default_value
        meta.loc[i, key] = synt.iloc[j][key].values

    content_pos = ("NC", "ADJ", "ADV", "VINF", "VS", "VPP", "V")
    meta["content_word"] = meta.pos.apply(
        lambda pos: pos in content_pos if isinstance(pos, str) else False
    )
    return meta


def add_new_syntax(meta, syntax_path, run):
    """
    Use the get_syntax function to add it directly to
    the metadata from the epochs
    """
    # get basic annotations
    meta_ = meta.copy().reset_index(drop=True)

    # get syntactic annotations
    syntax_file = syntax_path / f"ch{CHAPTERS[run]}.syntax.txt"
    synt = get_syntax(syntax_file)

    # align

    # split hyphenated words
    meta_["clean_word"] = (
        meta_["word"].fillna("XXXX").str.replace(r"'", " '").str.split()
    )
    # explode list to create new rows for each token
    meta_ = meta_.explode("clean_word").reset_index(drop=True)
    meta_.word = meta_.word.str.lower()
    # create a set of all punctuation characters
    punct = set(string.punctuation)
    # remove punctuation from clean_word column
    meta_tokens = meta_["clean_word"].apply(
        lambda x: "".join([c for c in x if c not in punct])
    )
    meta_tokens = meta_tokens.str.lower()
    # meta_tokens = meta.word.fillna("XXXX").apply(format_text).values

    synt = synt[~synt["word"].str.strip(string.punctuation + " ").eq("")]
    synt_tokens = synt.word.str.lower()
    # synt_tokens = synt.word.apply(format_text_syntax).values
    print(synt_tokens[-50:], meta_tokens[-50:])

    i, j = match_list(meta_tokens, synt_tokens)
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
    """
    One function to rule them all:

    To be better defined before future refinement

    """
    # load MEG data
    raw.load_data()
    raw.filter(0.5, 20.0, n_jobs=-1)

    # get metadata
    meta = add_syntax(meta, data_path, run)

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
    """
    Run a RidgeCV to get the Pearson correlation between
    the predicted values and the actual values for a target

    The target can be anything as:
    word_length,
    spacy embeddings,
    syntactic informations, etc..

    """
    model = make_pipeline(StandardScaler(), RidgeCV())
    cv = KFold(n_splits=5)

    y_ini = epochs.metadata[target].values

    def reshape_y(y_ini, size):

        # Create an empty 2D array of size (134, 1024)
        y = np.empty((size, 1024))

        # Fill in the new array with data from the original arrays
        for i in range(len(y)):
            y[i] = y_ini[i]
        return y

    r = np.zeros(len(epochs.times))
    for t in trange(len(epochs.times)):
        X = epochs.get_data()[:, :, t]
        y = y_ini
        if target == "laser":
            y = reshape_y(y_ini, X.shape[0])
        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            r[t] += correlate(y_pred, y[test]).mean()
    r /= cv.n_splits
    return r


def create_target(decoding_criterion, epochs):
    """
    Trivial function to handle different use cases

    Not currently used, needs refinement
    """
    if decoding_criterion == "embeddings":
        embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
        embeddings = np.array([emb for emb in embeddings])
        return embeddings
    elif decoding_criterion == "word_length":
        return epochs.metadata.word.apply(len)
    elif decoding_criterion == "closing":
        target = "n_closing"
        return epochs.metadata[target].values


def save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R):
    """
    To save decoding results for later use
    eg: plotting, further analysis
    """
    np.save(
        f"./../results/{task}/decoding_{decoding_criterion}_{epoch_on}_{reference}_{sub}.npy",
        R,
    )
    return True


#
# DEBUG
#


def decod_debug(epochs, target):
    """
    Run a RidgeCV to get the Pearson correlation between
    the predicted values and the actual values for a target

    The target can be anything as:
    word_length,
    spacy embeddings,
    syntactic informations, etc..

    """
    model = make_pipeline(StandardScaler(), RidgeCV())
    cv = KFold(n_splits=5)

    y_ini = epochs.metadata[target].values

    def reshape_y(y_ini, size):

        # Create an empty 2D array of size (134, 1024)
        y = np.empty((size, 1024))

        # Fill in the new array with data from the original arrays
        for i in range(len(y)):
            y[i] = y_ini[i]
        return y

    r = np.zeros(len(epochs.times))
    for t in trange(len(epochs.times)):
        X = epochs.get_data()[:, :, t]
        y = y_ini
        if target == "laser":
            y = reshape_y(y_ini, X.shape[0])
        for train, test in cv.split(X, y):
            model.fit(X[train], y[train])
            y_pred = model.predict(X[test])
            r[t] += correlate(y_pred, y[test]).mean()
    r /= cv.n_splits
    return r


def decod_xy(X, y):
    assert len(X) == len(y)
    # define data
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 10)))
    cv = KFold(5, shuffle=True, random_state=0)

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


# TO TEST !!!!
def mne_events(meta, raw, start, level):
    if start == 'onset':
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta.start * raw.info["sfreq"]
        return dict(events=events, metadata=meta.reset_index())
    elif start == 'offset':
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta['{level}_stop'] * raw.info["sfreq"]
        return dict(events=events, metadata=meta.reset_index())

    else:
        print('start should be either onset or offset')
        return 0
