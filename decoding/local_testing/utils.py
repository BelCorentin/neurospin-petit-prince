"""

General functions for decoding purposes

"""

# ML/Data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from Levenshtein import editops
import string
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from joblib import Memory
from datetime import datetime, timedelta
import functools
import inspect
import shutil
import sys
import time

from functools import lru_cache

# CONST

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

# Create a joblib Memory object to handle caching
memory = Memory(location='./cache', verbose=0)


def disk_cache(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_size = sys.getsizeof(args) + sys.getsizeof(kwargs)
        if args_size > 10 * 1024 * 1024:
            raise ValueError("Arguments size exceeds 10MB limit.")
        if inspect.ismethod(func):
            instance = args[0]
            args = args[1:]
            cached_func = memory.cache(func.__func__)
            return cached_func(instance, *args, **kwargs)
        else:
            cached_func = memory.cache(func)
            return cached_func(*args, **kwargs)

    return wrapper
# Function to return the Pearson correlation
# Between X and Y


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
        print(
            f"{name} is an invalid name. \n\
        Current options: visual and auditory, fmri"
        )

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


def correlate(X, Y):
    """
    Calculates Pearson Correlation score between X and Y

    Input: X, Y two (n, m) arrays

    Returns: R, a (n) dimensional array
    """
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


@disk_cache
def add_syntax(meta, syntax_path, run):
    """
    Use the get_syntax function to add it directly to
    the metadata from the epochs
    Basic problem with match list: new syntax has words like:
    "j" "avais"
    meta has:
    "j'avais"

    That means there is a limitation in terms of matching we can do:
    Since what is presented is: "J'avais" but to parse the syntax, we need j + avais
    We'll never get a perfect match.
    Option chosen: keep only the second part (verb) and tag it as a VERB
    When aligning it with brain signals
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


def format_text(text):
    """
    Simple function to make sure there is
    no problematic characters
    """
    # for char in "jlsmtncd":
    #     text = text.replace(f"{char}'", char)
    text = text.replace("œ", "oe")

    return text.lower()


def add_new_syntax(meta, syntax_path, run):
    """
    Use the get_syntax function to add it directly to
    the metadata from the epochs
    """
    # get basic annotations
    meta_ = meta.copy().reset_index(drop=True)

    # get syntactic annotations
    syntax_file = (
        syntax_path / f"run{run}_v2_0.25_0.5-tokenized.syntax.txt"
    )  # testing new syntax
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

    i, j = match_list(meta_tokens, synt_tokens)
    assert (len(i) / len(meta_tokens)) > 0.95

    for key, default_value in dict(n_closing=1, is_last_word=False, pos="XXX").items():
        meta[key] = default_value
        meta.loc[i, key] = synt.iloc[j][key].values

    content_pos = ("NC", "ADJ", "ADV", "VINF", "VS", "VPP", "V")
    meta["content_word"] = meta.pos.apply(
        lambda pos: pos in content_pos if isinstance(pos, str) else False
    )
    return meta


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
    if start == "onset":
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta.start * raw.info["sfreq"]
        return dict(events=events, metadata=meta.reset_index())
    elif start == "offset":
        events = np.ones((len(meta), 3), dtype=int)
        events[:, 0] = meta[f"{level}_stop"] * raw.info["sfreq"]
        return dict(events=events, metadata=meta.reset_index())

    else:
        print("start should be either onset or offset")
        return 0


@lru_cache(maxsize=1500)
def get_embeddings(list_of_strings):
    """
    Retrieves the embeddings from the Sentence Transformer
    package.
    Input: A list of strings

    Returns the embeddings generated as a np array of size (384)
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(list_of_strings)

    return embeddings


@lru_cache(maxsize=1500)
def get_embeddings_disk(run_id, level_id, level):
    """
    Retrives the embeddings already generated by LASER
    that have been saved on disk

    Takes as input the run_id, level_id and level
    eg: run 1, sentence 4 for sentence

    Returns:
    A numpy array of shape 1024 of the embeddings values
    """
    dim = 1024

    embeds = np.fromfile(
        f"{get_code_path()}/decoding/local_testing/embeds/emb/run{run_id}_{level}_{level_id}.bin",
        dtype=np.float32,
        count=-1,
    )
    embeds.resize(embeds.shape[0] // dim, dim)
    embeds = embeds.reshape(-1)
    return embeds


def generate_embeddings(meta, level):
    """
    Generate embeddings from the metadata:
    Using the meta.{level}_words: use a sentence transformer
    to generate embeddings for it

    Returns: a list of embeddings a size meta.shape[0]
    """
    all_embeddings = []
    for level_id, df in meta.groupby(["run", f"{level}_id"]):
        complete_string = " ".join(df[f"{level}_words"].values[0])
        embeddings = get_embeddings(complete_string)
        all_embeddings.append(embeddings)
    return all_embeddings


@lru_cache(maxsize=1)
def load_spacy():
    nlp = spacy.load("fr_core_news_sm")
    return nlp


def generate_embeddings_sum(meta, level, nb_words):
    """
    Generate embeddings from the metadata:
    Using the meta.{level}_words: use the spacy embeddings
    of the sum of the words wanted

    Returns: a list of embeddings a size meta.shape[0]
    """
    all_embeddings = []
    nlp = load_spacy()
    for level_id, df in meta.groupby(["run", f"{level}_id"]):
        assert df.shape[0] == 1
        emb_list = []
        for word_i in range(nb_words):
            word = df[f"{level}_words"].values[0][word_i]
            emb = nlp(word).vector
            emb_list.append(emb)
        summed_emb = np.sum(emb_list, axis=0)
        all_embeddings.append(summed_emb)
    return all_embeddings


def generate_nth_embedding(meta, level, n_th_word):
    """
    Generate embeddings from the metadata:
    Using the meta.{level}_words: use the spacy embeddings
    of the n_th word wanted

    Returns: a list of embeddings a size meta.shape[0]
    """
    all_embeddings = []
    nlp = load_spacy()
    for level_id, df in meta.groupby(["run", f"{level}_id"]):
        assert df.shape[0] == 1
        emb_list = []
        word = df[f"{level}_words"].values[0][n_th_word-1]
        emb = nlp(word).vector
        emb_list.append(emb)
        summed_emb = np.sum(emb_list, axis=0)
        all_embeddings.append(summed_emb)
    return all_embeddings


def decoding_from_criterion(criterion, epochs, level, subject):
    """
    Input:
    - criterion: the criterion on which the decoding will
    be done (embeddings, wlength, w_freq, etc..)
    - dict_epochs: the dictionnary containing the epochs
    for each condition (starts x levels)
    - starts: (onset, offset)
    - levels: (word, sentence, constituent)

    Options for criterion:
    - embeddings: decodes on the embeddings based on the level: sentence,
        constituent or word.
    - embeddings_multiple_words{x}: where x is an integer between 1 and 3:
        decodes the sum of embeddings of the x following words
    - word_const_non_end / word_const_end: decodes the words embeddings
        of a subset: words that end a constituant vs words that don't
    - wlength: the length of the word

    Returns:
    Two dataframes:
    - all_scores: decoding scores for each subject / starts x levels
    - all_evos: ERP plots for each subject / starts x levels

    """

    all_scores = []
    
    # All epochs -> Decoding and generate evoked potentials
    if criterion == "embeddings" or criterion.__contains__('min'):
        criterion = f"emb_{level}"

    # decoding word emb
    epochs = epochs.pick_types(meg=True, stim=False, misc=False).load_data()
    X = epochs.get_data()

    if criterion == "emb_sentence" or criterion == "emb_constituent":
        print(f" {level} embeddings decoding")
        all_embeddings = generate_embeddings(epochs.metadata, level)
        embeddings = np.vstack(all_embeddings)

        # DANGER /!\ Source of hard to debug problems
        # Make sure that there aren't any constant / equal to zero dimension
        # Had a problem with that hard to debug where a dimension
        # of the sentence embeddings were constant...
        problematic_rows = np.where(embeddings.std(0) < 1e-3)

        # Add noise
        for prob in problematic_rows[0]:
            embeddings[:, prob] = 1e-3 * np.random.rand(embeddings.shape[0])

        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)

    # Summing multiple word embeddings case
    elif criterion.__contains__("multiple_words"):
        nb_words = criterion.split("multiple_words")[1][:1]
        print(f"Multiple word decoding: for {nb_words} words")
        print(f'For: {level}')
        embeddings = generate_embeddings_sum(epochs.metadata, level, int(nb_words))
        embeddings = np.array([emb for emb in embeddings])
        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)

    elif criterion.__contains__("only"):
        n_th_word = criterion.split("only")[1]
        print(f"{level} nth word embedding decoding: for {n_th_word} word")
        embeddings = generate_nth_embedding(epochs.metadata, level, int(n_th_word))
        embeddings = np.array([emb for emb in embeddings])
        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)

    #  Simple word embeddings
    elif criterion.__contains__("word"):
        # Same, with get_embeddings
        print("Word embeddings decoding")
        print(f'For: {level}')
        nlp = spacy.load("fr_core_news_sm")
        embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
        embeddings = np.array([emb for emb in embeddings])
        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)
    elif criterion == "wlength":
        print(f'Decoding word length for: {level}')
        y = epochs.metadata.wlength
        R_vec = decod_xy(X, y)
        scores = R_vec

    for t, score in enumerate(scores):
        all_scores.append(
            dict(
                subject=subject,
                score=score,
                t=epochs.times[t],
            )
        )

    return all_scores


def plot_scores(all_scores, levels, starts, decoding_criterion):
    # figure = plt.figure(figsize=(16, 10), dpi=80)
    fig, axes = plt.subplots(3, 2)
    for axes_, level in zip(axes, levels):
        for ax, start in zip(axes_, starts):
            cond1 = all_scores.level == f"{level}"
            cond2 = all_scores.start == f"{start}"
            data = all_scores[cond1 & cond2]
            y = []
            x = []
            for s, t in data.groupby("t"):
                score_avg = t.score.mean()
                y.append(score_avg)
                x.append(s)
            ax.plot(x, y)
            ax.set_title(f"{level} {start}")
            ax.axhline(y=0, color="r", linestyle="-")
    plt.suptitle(f"Decoding Performance for {decoding_criterion}")
