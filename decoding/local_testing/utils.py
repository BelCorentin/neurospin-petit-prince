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
from sentence_transformers import SentenceTransformer



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


# TODO: add lru_cache
def get_embeddings(list_of_strings):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(list_of_strings)

    return embeddings


# def generate_embeddings(meta, level):
#     # Group by sentence / constituent id:
#     for string in 
#     for level_id, df in meta.groupby(f'{level}_id'):
        
#         complete_string = ''
#         bsl = epochs.data[df.index[0], :, bsl_time].mean(-2)
#         # Remove basline to all words in the sentence
#         epochs._data[df.index] -= bsl[None, :, None]
#     return embeddings


def decoding_from_criterion(
    criterion, epochs, start, level, subject
):
    """
    Input:
    - criterion: the criterion on which the decoding will be done (embeddings, wlength, w_freq, etc..)
    - dict_epochs: the dictionnary containing the epochs for each condition (starts x levels)
    - starts: (onset, offset)
    - levels: (word, sentence, constituent)

    Returns:
    Two dataframes:
    - all_scores: decoding scores for each subject / starts x levels
    - all_evos: ERP plots for each subject / starts x levels

    """

    all_scores = []
    # All epochs -> Decoding and generate evoked potentials
    if criterion == "embeddings":
        criterion = f"emb_{level}"

    # decoding word emb
    epochs = epochs.load_data().pick_types(meg=True, stim=False, misc=False)
    X = epochs.get_data()

    if criterion == "emb_sentence" or criterion == "emb_constituent":
        # Calculate embeddings here
        # Like:
        embeddings = generate_embeddings(epochs.metadata[f'{level}_words'], level)
        embeddings = epochs.metadata[f"embeds_{level}"]
        embeddings = np.vstack(embeddings.values)
        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)
    # elif criterion == "emb_word" or criterion == 'embeddings_word_non_const_end' or criterion == 'embeddings_word_const_end':
    elif criterion.__contains__('word'):
        # Same, with get_embeddings
        nlp = spacy.load("fr_core_news_sm")
        embeddings = epochs.metadata.word.apply(
            lambda word: nlp(word).vector
        ).values
        embeddings = np.array([emb for emb in embeddings])
        R_vec = decod_xy(X, embeddings)
        scores = np.mean(R_vec, axis=1)
    elif criterion == "wlength":
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
    figure = plt.figure(figsize=(16, 10), dpi=80)
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
