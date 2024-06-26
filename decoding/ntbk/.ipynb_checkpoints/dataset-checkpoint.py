"""

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
import spacy
import matplotlib.pyplot as plt
from functools import lru_cache

# Homemade
from utils import (
    match_list,
    add_syntax,
    mne_events,
    decoding_from_criterion,
    get_code_path,
    get_path,
)

mne.set_log_level(verbose="error")
nlp = spacy.load("fr_core_news_sm")

# The dict setting the epoch windows times:
# Can be adapted depending on sentence / constituent length?
EPOCH_WINDOWS = {
    "word": {
        "onset_min": -0.3,
        "onset_max": 1.0,
        "offset_min": -1.0,
        "offset_max": 0.3,
    },
    "constituent": {
        "offset_min": -2.0,
        "offset_max": 0.5,
        "onset_min": -0.5,
        "onset_max": 2.0,
    },
    "sentence": {
        "offset_min": -4.0,
        "offset_max": 1.0,
        "onset_min": -1.0,
        "onset_max": 4.0,
    },
}


# Putting a cache here when iterating on notebooks,
# Or when doing multiple decodings for the same subject
@lru_cache(maxsize=9)
def read_raw(subject, run_id, events_return=False, modality="auditory"):
    """
    Epoching function that for a subject, modality and run returns the epochs
    object which triggers have been aligned from the
    MEG metadata and STIM events file.

    Args:
        - subject: str
        - run_id: str
        - events_return: bool
        - modality: str (visual or auditory)
    Returns:
        - mne.Epochs
    """
    #path = get_path(modality)
    # For testing
    path = Path("/media/co/T7/workspace-LPP/data/MEG/LPP/PallierListen2023/download")
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

    if modality == "auditory":
        meta["word"] = meta["trial_type"].apply(
            lambda x: eval(x)["word"] if type(eval(x)) == dict else np.nan
        )
    # Initial wlength, as presented in the stimuli / triggers to match list
    meta["wlength"] = meta.word.apply(len)
    meta["run"] = run_id
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

    # Making sure that there is no problem with words that contain ""
    meta.word = meta.word.str.replace('"', "")

    # Two cases for match list: is it auditory or visual ?
    if modality == "auditory":

        ## Updated code
        # word events
        # match events and metadata
        word_events = events[events[:, 2] > 1]
        meg_delta = np.diff(word_events[:, 0].astype(float) / raw.info["sfreq"])
        meta_delta = np.diff(meta.onset.values)
        pres = 1e2
        i, j = match_list(np.round(meg_delta*pres), np.round(meta_delta*pres))
        assert len(i) / len(meg_delta) > .70
        assert len(i) > 500

        # Not needed anymore, as we need the whole metadata for syntax and embeddings
        # meta = meta.iloc[j].reset_index(drop=True)
        # meta["start"] = word_events[i, 0] / raw.info["sfreq"]

        # Old code
        # word_events = events[events[:, 2] > 1]
        # meg_delta = np.round(np.diff(word_events[:, 0] / raw.info["sfreq"]))
        # meta_delta = np.round(np.diff(meta.onset.values))
        # i, j = match_list(meg_delta, meta_delta)
        # assert len(i) > 1
        # events = events[i]  # events = words_events[i]

    # For auditory, we match on the time difference between triggers
    elif modality == "visual":
        """
        For visual, we match on the difference of word length
        encoded in the triggers
        Here, events are the presented stimuli: with hyphens.
        Have to make sure meta.word still contains the hyphens.
        However, the meta.word might have lost the hyphens because
        of the previous match when adding syntax.
        Handle the first two subjects:
        """
        if subject == "1" or subject == "2":
            events[:, 2] = events[:, 2] - 2048
        i, j = match_list(events[:, 2], meta.wlength)
        assert len(i) > (0.8 * len(events))
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


def get_subjects(path):
    """
    Returns the subject list from the path chosen

    Args:
        - path: pathlib.Path
    Returns:
        - list(str)
    """
    subjects = pd.read_csv(str(path) + "/participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    # subjects = np.delete(subjects, subjects.shape[0]-1)
    # Let's sort this array before outputting it!
    int_subjects = np.sort([int(subj) for subj in subjects])
    subjects = [str(subj) for subj in int_subjects]

    return subjects


def enrich_metadata(meta):
    """
    Populate the metadata with information about onset, offset of
    different syntactic markers and categories

    Args:
        - meta: pd.DataFrame
    Returns:
        - pd.DataFrame
    """
    meta["word_onset"] = True
    meta["word_stop"] = meta.start + meta.duration
    meta["sentence_onset"] = meta.word_id == 0
    meta["prev_closing"] = meta["n_closing"].shift(1)
    meta["constituent_onset"] = meta.apply(
        lambda x: x["prev_closing"] > x["n_closing"] and x["n_closing"] == 1, axis=1
    )
    meta["constituent_onset"].fillna(False, inplace=True)
    meta["const_end"] = meta.constituent_onset.shift(-1)
    meta.drop("prev_closing", axis=1, inplace=True)

    # Adding the sentence stop info
    meta["sentence_id"] = np.cumsum(meta.sentence_onset)
    for s, d in meta.groupby("sentence_id"):
        meta.loc[d.index, "sent_word_id"] = range(len(d))
        meta.loc[d.index, "sentence_length"] = len(d)
        meta.loc[d.index, "sentence_start"] = d.start.min()
        last_word_duration = meta.loc[d.index.max(), "duration"]
        meta.loc[d.index, "sentence_stop"] = d.start.max() + last_word_duration

        # Adding the info for each word the information about the sentence / constituent
        # it is part of, in order to keep it when filtering on sentence / const later
        meta.at[d.index[0], "sentence_words"] = d.word.values

    # Adding the constituents stop info
    meta["constituent_id"] = np.cumsum(meta.constituent_onset)
    for s, d in meta.groupby("constituent_id"):
        meta.loc[d.index, "const_word_id"] = range(len(d))
        meta.loc[d.index, "constituent_length"] = len(d)
        meta.loc[d.index, "constituent_start"] = d.start.min()
        last_word_duration = meta.loc[d.index.max(), "duration"]
        meta.loc[d.index, "constituent_stop"] = d.start.max() + last_word_duration

        # Adding the info for each word the information about the sentence / constituent
        # it is part of, in order to keep it when filtering on sentence / const later
        meta.at[d.index[0], "constituent_words"] = d.word.values

    return meta


def select_meta_subset(meta, level, decoding_criterion):
    """
    Select only the rows containing the True for the conditions
    Simplified to only get for the onset: sentence onset epochs, constituent onset epochs,etc
    Args:
        - meta: pd.DataFrame
        - level: str
        - decoding_criterion: str
    Returns:
        - pd.DataFrame
    """

    # When running the analysis on all decoding criterions, to try to get more stable
    # results; especially when comparing only1,2,3,4, it might be better to set the decoding on
    # the same subset: select only sentences longer than 5 words; so the decoders are trained
    # on the same dataset

    if level == "sentence":
        sel = meta.query(f"{level}_onset==True and sentence_length >= 5")
        assert sel.shape[0] > 10
        return sel
    # In the case of sentences, it might be worth it to try to
    # Only select longer sentences ?
    # Like:
    # if level == sentence:
    # sel = meta.query(f'{level}_onset==True and len({levels}_words)>=5')
    if decoding_criterion == "embeddings_word_non_const_end":
        sel = meta.query(f"{level}_onset==True and is_last_word==False")
    elif decoding_criterion == "embeddings_word_const_end":
        sel = meta.query(f"{level}_onset==True and is_last_word==True")
    # For debugging sentence embeddings
    elif decoding_criterion.__contains__("embeddings_multiple_words"):
        min = decoding_criterion.split("multiple_words")[1]
        sel = meta.query(f"{level}_onset==True and {level}_length >= {min}")
    # For choosing subsets of sentence / constituent of minimal length:
    elif decoding_criterion.__contains__("min"):
        min = decoding_criterion.split("_min")[1]
        sel = meta.query(f"{level}_onset==True and {level}_length >= {min}")
    # Handling only n_th word embeddings
    elif decoding_criterion.__contains__("only"):
        min = decoding_criterion.split("only")[1]
        if level == "word":
            sel = meta.query(f"{level}_onset==True")
        else:
            sel = meta.query(f"{level}_onset==True and {level}_length >= {min}")
    else:
        sel = meta.query(f"{level}_onset==True")
    assert sel.shape[0] > 10
    return sel


def epoch_on_selection(raw, sel, start, level):
    """
    Epoching from the metadata having all onset events: if the start=Offset, the mne events
    Function will epoch on the offset of each level instead of the onset

    Args:
        - raw: mne.Raw
        - sel: str
        - start: str
        - level: str
    Returns:
        - mne.Epochs
    """
    epochs = mne.Epochs(
        raw,
        **mne_events(sel, raw, start=start, level=level),
        decim=10,
        tmin=EPOCH_WINDOWS[f"{level}"][f"{start}_min"],
        tmax=EPOCH_WINDOWS[f"{level}"][f"{start}_max"],
        event_repeated="drop",
        preload=True,
        baseline=None,
    )
    return epochs


def apply_baseline(epochs, level, tmin=-0.300, tmax=0):
    """
    To be applied at the beginning of the preproc
    TODO: check more thoroughly the exact part where it
    should be applied
    Applies a baseline to the epochs data

    Returns the epochs object baselined
    Args:
        - epochs: mne.Epochs
        - level: str
        - tmin: float
        - tmax: float
    Returns:
        - mne.Epochs
    """
    if level == "word":
        return epochs
    meta = epochs.metadata.copy()
    meta.reset_index(inplace=True)
    LEVELS = dict(sentence="sent_word_id", constituent="const_word_id")

    meta[f"{level}_uid"] = np.cumsum(epochs.metadata[LEVELS[level]] == 0) * -1
    bsl_time = (epochs.times >= tmin) * (epochs.times <= tmax)
    # For each sentence, baseline by the first word
    for sid, df in meta.groupby(f"{level}_uid"):
        # Basline activity of the first word
        bsl = epochs._data[df.index[0], :, bsl_time].mean(-2)
        # Remove basline to all words in the sentence
        epochs._data[df.index] -= bsl[None, :, None]
    return epochs


def populate_metadata_epochs(
    modality,
    subject,
    level,
    start,
    runs=9,
    decoding_criterion="embeddings",
):
    """
    Takes as input subject number, modality, level of epoching wanted (word, sentence or constituent)
    and start (onset or offset) as well as the number of total runs (for debugging).

    Returns an epochs object, for these particular parameters

    Args:
        - runs: int
        - subject: str
        - run_id: str
        - events_return: bool
        - modality: str (visual or auditory)
    Returns:
        - mne.Epochs
    """

    all_epochs = []
    # Iterating on runs, building the metadata and re-epoching
    for run in range(1, runs + 1):
        raw, meta_, events = read_raw(
            subject, run, events_return=True, modality=modality
        )
        meta = meta_.copy()

        # Add information about constituents onsets, offsets, etc..
        meta = enrich_metadata(meta)

        # Select the subset needed for the level (filter on sentence/constituent)
        sel = select_meta_subset(meta, level, decoding_criterion)

        # Add the embeddings to the metadata limited to the level

        epochs = epoch_on_selection(raw, sel, start, level)

        epochs = apply_baseline(
            epochs, level, tmin=EPOCH_WINDOWS[f"{level}"][f"{start}_min"], tmax=0
        )

        all_epochs.append(epochs)

    # Once we have the dict of epochs per condition full (epoching for each run for a subject)
    # we can concatenate them, and fix the dev_head

    # Concatenate epochs
    if len(all_epochs) != 1:
        # Handle the case where there is only one run
        for epo in all_epochs:
            epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]
    else:
        return all_epochs[0]

    epochs = mne.concatenate_epochs(all_epochs)

    return epochs


def analysis(modality, start, level, decoding_criterion):
    """
    Function similar to the analysis_subject one, except that subjects
    is a list of subject_id (list(string)) and the analysis_subject function
    will be ran for all subjects, then all_scores will be concatenated and outputted

    Returns all scores

    Args:
        - decoding_criterion: str
        - level: str
        - start: str
        - modality: str (visual or auditory)
    Returns:
        - pd.DataFrame

    """
    path = get_path(modality)
    subjects = get_subjects(path)
    all_scores = []
    if modality == "auditory":  # TODO REDO BIDS AND FIX THIS
        subjects = subjects[2:]
    for subject in subjects:
        scores = analysis_subject(subject, modality, start, level, decoding_criterion)
        all_scores.append(scores)

    file_path = f"./../data/results/all_scores_{modality}_{decoding_criterion}_{level}_{start}.csv"
    pd.DataFrame(all_scores).to_csv(file_path, index=False)

    return all_scores


def analysis_subject(
    subject, modality, start, level, decoding_criterion, runs=9, write=True
):
    """
    Decode for the criterion the correlation score between predicted
    and real criterion
    E.g.: for sentence level, gives the Pearson Correlation between fitted/predicted
    sentence embeddings & the real ones

    Returns a dataframe containing the scores, as well as saving it under ./results

    Args:
        - subject: str
        - decoding_criterion: str
        - level: str
        - start: str
        - modality: str (visual or auditory)
        - write: bool
    Returns:
        - pd.DataFrame
    """

    file_path = f"./../data/results/{modality}/{decoding_criterion}_{level}_{start}_sub{subject}.csv"
    if os.path.exists(file_path):
        print("Analysis already done")
        return None
    else:
        epochs = populate_metadata_epochs(
            modality,
            subject,
            level,
            start,
            runs=runs,
            decoding_criterion=decoding_criterion,
        )

        all_scores = decoding_from_criterion(decoding_criterion, epochs, level, subject)

        if write:
            pd.DataFrame(all_scores).to_csv(file_path, index=False)

        return all_scores


def load_scores(subject, level, start, decoding_criterion, modality):
    """
    Loading scores of a:
    - subject, for a level and given start, criterion and modality
    that has run the function analysis_subject on them

    Returns a pandas DF containing the scores, with columns:
    subject, t, score

    Args:
        - decoding_criterion: str
        - level: str
        - start: str
        - modality: str (visual or auditory)
    Returns:
        - pd.DataFrame
    """
    file = f"./../data/results/{modality}/{decoding_criterion}_{level}_{start}_sub{subject}.csv"
    scores = pd.read_csv(file)
    return scores


"""
The functions below are only for plotting purposes

Might be better to just fine-tune them to your own use
"""


def unique_plot(
    subject, level, start, decoding_criterion, modality, from_scores=False, scores=None
):
    if from_scores:
        data = pd.DataFrame(scores)
    else:
        data = load_scores(subject, level, start, decoding_criterion, modality)
    y = []
    x = []
    for s, t in data.groupby("t"):
        score_avg = t.score.mean()
        y.append(score_avg)
        x.append(s)
    plt.plot(x, y)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.suptitle(
        f"Decoding Performance for {decoding_criterion} and {modality} for sub-{subject}, epoched on {level} {start}"
    )


def sub_avg_plot(
    level, start, decoding_criterion, modality, from_scores=False, scores=None
):
    if from_scores:
        all_scores = pd.DataFrame(scores)
    else:
        path = get_path(modality)
        subjects = get_subjects(path)[2:]
        all_scores = load_scores(
            subjects[0], level, start, decoding_criterion, modality
        )
        # Create the sub dataframe
        for sub in subjects[1:]:
            try:
                all_scores = pd.concat(
                    [
                        all_scores,
                        load_scores(sub, level, start, decoding_criterion, modality),
                    ]
                )
            except Exception as e:
                print(f"No decoding data for sub-{sub}")
                print(e)
    y = []
    x = []
    for s, t in all_scores.groupby("t"):
        score_avg = t.score.mean()
        y.append(score_avg)
        x.append(s)
    plt.plot(x, y)
    plt.axhline(y=0, color="r", linestyle="-")
    plt.suptitle(
        f"Decoding Performance for {decoding_criterion} and {modality} for all_subjects, epoched on {level} {start}"
    )


## Group analysis plots, more complex


def labelize_criterion(criterion):
    if criterion.__contains__("only"):
        nb_only = criterion.split("only")[1]
        return f"sentence {nb_only}st word embedding"
    elif criterion == "bow":
        return "Bag of Words embedding"
    elif criterion == "embeddings":
        return "LASER embed"
    elif criterion.__contains__("multiple_words"):
        nb_only = criterion.split("multiple_words")[1]
        return f"sentence {nb_only} words embedding"
    elif criterion.__contains__("word"):
        return "first word embedding"


def plot_all_conditions_one_subject(subject, modalities, starts, criterions, level):
    figure = plt.figure(figsize=(32, 20), dpi=80)
    fig, axes = plt.subplots(2, 2)
    for axes_, modality in zip(axes, modalities):
        for ax, start in zip(axes_, starts):
            # For each criterion:
            # Plot the score associated

            # For embeddings (laser), bow, only1;2 etc .. it's direct
            # Then outside of the for crit in criterion, plot the sum of decoding scores for
            # all the only...
            for decoding_criterion in np.atleast_1d(criterions):

                data = load_scores(subject, level, start, decoding_criterion, modality)
                y = []
                x = []
                for s, t in data.groupby("t"):
                    score_avg = t.score.mean()
                    y.append(score_avg)
                    x.append(s)
                label = labelize_criterion(decoding_criterion)
                ax.fill_between(x, y, label=label, alpha=0.7)

                ax.set_title(f"{modality} {start}")
                ax.axhline(y=0, color="r", linestyle="-")

            # Summing decoding scores of n_th_words
            all_data = pd.DataFrame()
            for i in range(1, 6):
                data = load_scores(subject, level, start, f"only{i}", modality)
                all_data = pd.concat([all_data, data])
            y = []
            x = []
            for s, t in all_data.groupby("t"):
                score_summed = t.score.sum()
                y.append(score_summed)
                x.append(s)
            # ax.fill_between(x, y, label='sum of decoding performances', alpha=0.6)
            # ax.set_title(f"{modality} {start}")
            # ax.axhline(y=0, color="r", linestyle="-")

    plt.suptitle(f"Decoding Performance for {subject}")
    plt.legend()


def plot_sentence_simple(criterions, subject, start, modality):
    level = "sentence"
    figure = plt.figure(figsize=(32, 20), dpi=80)
    fig, axes = plt.subplots(1, 1)

    # For each criterion:
    # Plot the score associated

    # For embeddings (laser), bow, only1;2 etc .. it's direct
    # Then outside of the for crit in criterion, plot the sum of decoding scores for
    # all the only...

    for decoding_criterion in np.atleast_1d(criterions):
        data = load_scores(subject, level, start, decoding_criterion, modality)
        y = []
        x = []
        for s, t in data.groupby("t"):
            score_avg = t.score.mean()
            y.append(score_avg)
            x.append(s)
        label = labelize_criterion(decoding_criterion)
        axes.fill_between(x, y, label=label, alpha=0.7)

        axes.set_title(f"{modality} {start}")
        axes.axhline(y=0, color="r", linestyle="-")

    # Summing decoding scores of n_th_words
    all_data = pd.DataFrame()
    for i in range(1, 6):
        data = load_scores(subject, level, start, f"only{i}", modality)
        all_data = pd.concat([all_data, data])
    y = []
    x = []
    for s, t in all_data.groupby("t"):
        score_summed = t.score.sum()
        y.append(score_summed)
        x.append(s)
    # axes.fill_between(x, y, label='sum of decoding performances', alpha=0.6)
    # axes.set_title(f"{modality} {start}")
    # axes.axhline(y=0, color="r", linestyle="-")

    plt.suptitle(f"Decoding Performance for {subject}")
    plt.legend()


def plot_all_conditions(modalities, starts, criterions, level):
    """
    Plot all the conditions, for all subjects

    For onset, offset,
    For auditory, visual

    Plot each word embedding, the sentence embedding decoding
    """
    figure = plt.figure(figsize=(32, 20), dpi=80)
    fig, axes = plt.subplots(2, 2)
    for axes_, modality in zip(axes, modalities):
        path = get_path(modality)
        subjects = get_subjects(path)[2:]

        for ax, start in zip(axes_, starts):
            # For each criterion:
            # Plot the score associated

            # For embeddings (laser), bow, only1;2 etc .. it's direct
            # Then outside of the for crit in criterion, plot the sum of decoding scores for
            # all the only...
            for decoding_criterion in np.atleast_1d(criterions):

                # Iteration on subject to mean their data
                data = pd.DataFrame()
                for subject in subjects:
                    try:
                        data_one = load_scores(
                            subject, level, start, decoding_criterion, modality
                        )
                        data = pd.concat([data, data_one])
                    except Exception as e:
                        print(e)

                y = []
                x = []
                for s, t in data.groupby("t"):
                    score_avg = t.score.mean()
                    y.append(score_avg)
                    x.append(s)
                label = labelize_criterion(decoding_criterion)
                ax.fill_between(x, y, label=label, alpha=0.7)

                ax.set_title(f"{modality} {start}")
                ax.axhline(y=0, color="r", linestyle="-")

            # Summing decoding scores of n_th_words
            all_data = pd.DataFrame()
            for i in range(1, 6):
                data = load_scores(subject, level, start, f"only{i}", modality)
                all_data = pd.concat([all_data, data])
            y = []
            x = []
            for s, t in all_data.groupby("t"):
                score_summed = t.score.sum()
                y.append(score_summed)
                x.append(s)
            # ax.fill_between(x, y, label='sum of decoding performances', alpha=0.6)
            # ax.set_title(f"{modality} {start}")
            # ax.axhline(y=0, color="r", linestyle="-")

    plt.suptitle(f"Decoding Performance for {subject}")
    plt.legend()


# # OLD AND NOT WORKING ANYMORE

# def load_scores_debug(modality, decoding_criterion):
#     # subjects = get_subjects(path)
#     subjects = range(3, 17)
#     first_subject = subjects[0]
#     all_scores = pd.read_csv(
#         f"./results/scores_{modality}_{decoding_criterion}_sub{first_subject}.csv"
#     )
#     for subject in subjects[1:]:
#         file = f"./results/scores_{modality}_{decoding_criterion}_sub{subject}.csv"  # TO CHANGE
#         scores = pd.read_csv(file)
#         all_scores = pd.concat([all_scores, scores])

#     all = pd.DataFrame(all_scores)
#     return all


# def plot_scores_debug(modality, decoding_criterion):
#     levels = ("word", "constituent", "sentence")
#     starts = ("onset", "offset")
#     # For all subjects, there is a max:
#     all_scores = load_scores_debug(modality, decoding_criterion)

#     figure = plt.figure(figsize=(16, 10), dpi=80)
#     fig, axes = plt.subplots(3, 2)
#     for axes_, level in zip(axes, levels):
#         for ax, start in zip(axes_, starts):
#             cond1 = all_scores.level == f"{level}"
#             cond2 = all_scores.start == f"{start}"
#             data = all_scores[cond1 & cond2]
#             y = []
#             x = []
#             for s, t in data.groupby("t"):
#                 score_avg = t.score.mean()
#                 y.append(score_avg)
#                 x.append(s)
#             ax.fill_between(x, y)
#             ax.set_title(f"{level} {start}")
#             ax.axhline(y=0, color="r", linestyle="-")
#     plt.suptitle(f"Decoding Performance for {decoding_criterion} and {modality}")


# def plot_scores(modality, decoding_criterion):
#     """
#     Simple function to build a matplotlib fillbetween plot
#     of the decoding score on a window

#     """
#     levels = ("word", "constituent", "sentence")
#     starts = ("onset", "offset")
#     # For all subjects, there is a max:
#     all_scores, total_subjects = load_scores(modality, decoding_criterion)

#     figure = plt.figure(figsize=(16, 10), dpi=80)
#     fig, axes = plt.subplots(3, 2)
#     for axes_, level in zip(axes, levels):
#         for ax, start in zip(axes_, starts):
#             cond1 = all_scores.level == f"{level}"
#             cond2 = all_scores.start == f"{start}"
#             data = all_scores[cond1 & cond2]
#             y = []
#             x = []
#             for s, t in data.groupby("t"):
#                 score_avg = t.score.mean()
#                 y.append(score_avg)
#                 x.append(s)
#             ax.fill_between(x, y)
#             ax.set_title(f"{level} {start}")
#             ax.axhline(y=0, color="r", linestyle="-")
#     plt.suptitle(f"Decoding Performance for {decoding_criterion} and {modality} for {total_subjects} subs")


# def check_plotting_possible():
#     """
#     This function gives an overview of the type of results file
#     available, such that can be used to decide easily what to plot
#     using the plot_scores function
#     """
#     path = '.'
#     file_types = []
#     for filename in os.listdir(os.path.join(path, 'results')):
#         if filename.startswith("scores_"):
#             parts = filename.split("_")
#             file_type = parts[1]
#             # Check if second part is 'embeddings'
#             if parts[2] == 'embeddings':
#                 file_types.append(file_type)
#             else:
#                 # Get everything after file type until next '_'
#                 rest = "_".join(parts[2:])
#                 file_types.append(file_type + "_" + rest)
#     return list(set(file_types))


# def subs_to_plot(modality, decoding_criterion):
#     """
#     This function calculates the amount of subjects for which we can do
#     the plotting of their results
#     """
#     path = '.'
#     subs = []
#     for filename in os.listdir(os.path.join(path, 'results')):
#         if filename.startswith("scores_"):
#             subcategory = filename.split("_")[1].split("_sub")[0]
#             sub_number = filename.split("_")[1].split("_sub")[1].split(".")[0]
#             if subcategory == f'{modality}_{decoding_criterion}':
#                 subs.append(sub_number)
#     return subs
