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


# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
mne.set_log_level(False)


mne.set_log_level(False)


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
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
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


class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        # assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
        # print(f"File opened: {data}")
        if str(data).__contains__("final"):
            print("Processed data (Maxwell filtered) used")
        if str(data).__contains__("BIDS"):
            print("Raw data (no filtering) used")
    # assert data.exists()


# For Dell ####
# TASK = "rest"
# subject = "220628"
# To simplify for the time being
# To run on the Neurospin workstation
# PATHS.data = Path("/home/co/workspace_LPP/data/MEG/LPP/LPP_bids")

# For NS ####
TASK = "listen"
subject = "1"
# To simplify for the time being
# To run on the Neurospin workstation
PATHS.data = Path("/home/is153802/data/BIDS_final")

epochs_final = []
ph_final = []

for run_id in np.arange(1, 10):

    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session="01",
        task=TASK,
        datatype="meg",
        root=PATHS.data,
        run="0" + str(run_id),
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
    print(event_file)
    assert Path(event_file).exists()
    # read events
    meta = pd.read_csv(event_file, sep="\t")
    events = mne.find_events(raw, stim_channel="STI101", shortest_event=1)

    phonemes = meta[meta.trial_type.str.contains("phoneme")]
    words = meta[meta.trial_type.str.contains("word")]

    # match events and metadata
    word_events = events[events[:, 2] > 1]
    meg_delta = np.round(np.diff(word_events[:, 0] / raw.info["sfreq"]))
    meta_delta = np.round(np.diff(meta.onset.values))

    i, j = match_list(meg_delta, meta_delta)

    assert len(i) > 500
    events = word_events[i]
    # events = events[i]  # events = words_events[i]
    meta = meta.iloc[j].reset_index()

    x = (events[0][0] / raw.info["sfreq"]) - list(meta.onset)[0]

    events_ph = ((phonemes.onset + x) * raw.info["sfreq"]).to_numpy(dtype="int")
    zeros = np.zeros(events_ph.shape)
    last_c = np.ones(events_ph.shape) * 128
    events_ph = np.stack((events_ph, zeros, last_c), axis=1)
    events_ph = events_ph.astype("int")
    # print(events_ph)

    epochs = mne.Epochs(
        raw,
        events_ph,
        metadata=phonemes,
        tmin=-0.1,
        tmax=0.5,
        decim=10,
        baseline=(-0.1, 0.0),
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

    epochs.metadata["label"] = f"run_{run_id}"

    epochs_final.append(epochs)

    for epo_ in epochs_final:
        epo_.info["dev_head_t"] = epochs_final[0].info["dev_head_t"]

    # epo.info['nchan'] = epochs[0].info['nchan']

    voiced_ph = np.array(
        [str(tupl).__contains__("non-voiced") for tupl in phonemes.iterrows()]
    )

    ph_final.append(voiced_ph)


epochs = mne.concatenate_epochs(epochs_final)

# Get the evoked potential averaged on all epochs for each channel
evo = epochs[0].average(method="median")
evo.plot(spatial_colors=True)

plt.savefig("./fig_evoked.png")

X = epochs.get_data()  # Both mag and grad

phonemes = np.array(ph_final)
y = phonemes.reshape((6229, 1))
R = decod(X, y)

fig, ax = plt.subplots(1, figsize=[6, 6])
dec = plt.fill_between(epochs[0].times, R)


plt.savefig("./fig_decode.png")
