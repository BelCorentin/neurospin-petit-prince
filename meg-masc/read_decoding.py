####################################################
####################################################
# Imports
####################################################
####################################################

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
from TO_EXCLUDE import to_exclude

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
mne.set_log_level(False)


class PATHS:
    data = Path("/home/is153802/workspace_LPP/data/MEG/LPP/BIDS_lecture")

TASK = "listen"

####################################################
####################################################
# Functions
####################################################
####################################################


# Epoching and decoding
def epoch_data(subject, run_id):

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=TASK,
        datatype="meg",
        root=PATHS.data,
        run=run_id,
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.pick_types(meg=True, stim=True)
    raw.load_data()
    raw = raw.filter(0.5, 20)

    event_file = PATHS.data / f"sub-{bids_path.subject}"
    event_file = event_file / f"ses-{bids_path.session}"
    event_file = event_file / "meg"
    event_file = str(event_file / f"sub-{bids_path.subject}")
    event_file += f"_ses-{bids_path.session}"
    event_file += f"_task-{bids_path.task}"
    event_file += f"_run-{bids_path.run}_events.tsv"
    assert Path(event_file).exists()
    # read events
    meta = pd.read_csv(event_file, sep="\t")
    events = mne.find_events(
        raw, stim_channel="STI101", shortest_event=1, min_duration=0.0010001
    )

    # match events and metadata
    word_events = events[events[:, 2] > 1]
    meg_delta = np.round(np.diff(word_events[:, 0] / raw.info["sfreq"]))
    meta_delta = np.round(np.diff(meta.onset.values))

    print(events)
    print(meta.onset.values)
    i, j = match_list(meg_delta, meta_delta)
    print(f"Len i : {len(i)} for run {run_id}")
    assert len(i) > 500
    events = word_events[i]
    # events = events[i]  # events = words_events[i]
    meta = meta.iloc[j].reset_index()

    epochs = mne.Epochs(
        raw, events, metadata=meta, tmin=-0.3, tmax=0.8, decim=10, baseline=(-0.2, 0.0)
    )

    data = epochs.get_data()
    epochs.load_data()

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


def decod(X, y):
    assert len(X) == len(y)
    # define data
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 10)))
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, n_chans, n_times = X.shape
    R = np.zeros(n_times)
    for t in range(n_times):
        print(".", end="")
        y_pred = cross_val_predict(model, X[:, :, t], y, cv=cv)
        R[t] = correlate(y, y_pred)
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


# Utils
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


def get_subjects():
    subjects = pd.read_csv(str(PATHS.data) + "/participants.tsv", sep="\t")
    subjects = subjects.participant_id.apply(lambda x: x.split("-")[1]).values
    # subjects = np.delete(subjects, subjects.shape[0]-1)
    # Let's sort this array before outputting it!
    int_subjects = np.sort([int(subj) for subj in subjects])
    subjects = [str(subj) for subj in int_subjects]

    return subjects


####################################################
####################################################
# Main
####################################################
# ##################################################

if __name__ == "__main__":

    report = mne.Report()

    subjects = get_subjects()

    RUN = 9

    print("\nSubjects for which the decoding will be tested: \n")
    print(subjects)

    for subject in subjects:

        if subject in to_exclude:
            continue

        print(f"Subject {subject}'s decoding started")
        epochs = []
        for run_id in range(1, RUN + 1):
            print(".", end="")
            epo = epoch_data(subject, "%.2i" % run_id)
            epo.metadata["label"] = f"run_{run_id}"
            epochs.append(epo)

        # Quick fix for the dev_head: has to be
        # fixed before doing source reconstruction
        for epo in epochs:
            epo.info["dev_head_t"] = epochs[0].info["dev_head_t"]
            # epo.info['nchan'] = epochs[0].info['nchan']

        epochs = mne.concatenate_epochs(epochs)

        # Get the evoked potential averaged on all epochs for each channel
        evo = epochs.average(method="median")
        evo.plot(spatial_colors=True)

        # Handling the data structure
        epochs.metadata["kind"] = epochs.metadata.trial_type.apply(
            lambda s: eval(s)["kind"]
        )
        epochs.metadata["word"] = epochs.metadata.trial_type.apply(
            lambda s: eval(s)["word"]
        )

        # Run a linear regression between MEG signals
        # and word frequency classification
        # X = epochs.get_data() # Regular data: mag & grad
        # X = epochs.copy().pick_types(meg='mag').get_data()  # Only mag data
        # X = epochs.copy().pick_types(meg='grad').get_data() # Only grad data
        X = epochs.get_data()  # Both mag and grad
        y = epochs.metadata.word.apply(lambda w: zipf_frequency(w, "fr"))
        R = decod(X, y)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        dec = plt.fill_between(epochs.times, R)
        # plt.show()
        report.add_evokeds(evo, titles=f"Evoked for sub {subject} ")
        report.add_figure(fig, title=f"decoding for subject {subject}")
        # report.add_figure(dec, subject, tags="word")
        report.save("./figures/decoding_raw.html", open_browser=False, overwrite=True)

        print("Finished!")