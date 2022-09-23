"""
Check decoding scripts from JR King adapted to fit the LPP MEG Dataset and annotations 

-> No phonemes annotations / timestamps

However the rest is the same, as we have all the words onset, durations & co

Using a Ridge to decode the word frequency class (based off median)

"""

##########################################################################################################
# Imports
##########################################################################################################

# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, scale, RobustScaler
from sklearn.linear_model import Ridge*
from wordfreq import zipf_frequency
from Levenshtein import editops

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
from joblib import Parallel, delayed
matplotlib.use("Agg")
mne.set_log_level(False)




##########################################################################################################
# Const
##########################################################################################################

"""
Here, we use two datasets, both BIDS formatted

RAW: the raw files organised under the BIDS format
PROC: the processed files that have undergone a Maxwell Filter (Elektra MEG data)

# RAW = "~/workspace_LPP/data/MEG/LPP/LPP_bids"
# PROC = "~/workspace_LPP/data/MEG/LPP/final_all"

"""
class PATHS:
    path_file = Path("./data_path.txt")
    if not path_file.exists():
        data = Path(input("data_path?"))
        assert data.exists()
        with open(path_file, "w") as f:
            f.write(str(data) + "\n")
    with open(path_file, "r") as f:
        data = Path(f.readlines()[0].strip("\n"))
        print(f'File opened: {data}')
        if str(data).__contains__('final'):
            print("Processed data (Maxwell filtered) used")
        if str(data).__contains__('BIDS'):
            print("Raw data (no filtering) used")

    assert data.exists()

# To simplify for the time being

# To run on the Neurospin workstation
PATHS.data = Path("/home/is153802/workspace_LPP/data/MEG/LPP/LPP_bids")
# On the DELL
PATHS.data = Path("/home/co/workspace_LPP/data/MEG/LPP/LPP_bids")



##########################################################################################################
# Functions
##########################################################################################################

# Epoching and decoding

def epoch_data(subject, run_id):

    # define path
    bids_path = mne_bids.BIDSPath(
        subject=subject,
        session='01',
        task='rest',
        datatype="meg",
        root=PATHS.data,
        run=run_id,
        #processing = 'sss' # Uncomment this line when working with preprocessed files (undergone Max Filter)
    )

    raw = mne_bids.read_raw_bids(bids_path)
    raw.pick_types(meg=True, stim=True)
    raw.load_data()
    raw = raw.filter(.5, 20)

    event_file = PATHS.data/f'sub-{bids_path.subject}'
    event_file = event_file / f'ses-{bids_path.session}'
    event_file = event_file / 'meg'
    event_file = str(event_file / f'sub-{bids_path.subject}')
    event_file += f'_ses-{bids_path.session}'
    event_file += f'_task-{bids_path.task}'
    event_file += f'_run-{bids_path.run}_events.tsv'
    Path(event_file).exists()
    
    # read events
    meta = pd.read_csv(event_file, sep='\t')
    events = mne.find_events(raw, stim_channel='STI101')
    
    # match events and metadata
    word_events = events[events[:, 2]==128]
    meg_delta = np.round(np.diff(word_events[:, 0]/raw.info['sfreq']))
    meta_delta = np.round(np.diff(meta.onset.values))
    i, j = match_list(meg_delta, meta_delta)
    assert len(i) > 1000
    events = events[i]
    meta = meta.iloc[j].reset_index()

    epochs = mne.Epochs(raw, events, metadata=meta, tmin=-.3, tmax=.8, decim=10)
    return epochs


def decod(X, y):
    assert len(X) == len(y)
    
    # define data
    model = make_pipeline(StandardScaler(), Ridge())
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, n_chans, n_times = X.shape
    R = np.zeros(n_times)
    for t in range(n_times):
        print('.', end='')
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


# Scaling and clipping the noise that has an amplitude higher than 15 sigmas
def scale_epochs(data,epochs):
    n_words, n_chans, n_times = data.shape 
    vec = data.transpose(0, 2, 1).reshape(-1, n_chans)
    scaler = RobustScaler()
    idx = np.arange(len(vec))
    np.random.shuffle(idx)
    vec = scaler.fit(vec[idx[:20_000]]).transform(vec)
    vec = np.clip(vec, -15, 15)
    epochs._data[:,:,:] = scaler.inverse_transform(vec).reshape(n_words, n_times, n_chans).transpose(0, 2, 1)
    return epochs


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
    subjects = np.delete(subjects,subjects.shape[0]-1)
    print("\nSubjects for which the decoding will be tested: \n")
    print(subjects)


##########################################################################################################
# Main
##########################################################################################################

if __name__ == "__main__":


    report = mne.Report()
    subjects = get_subjects()

    for subject in subjects[[-1]]:
        print(f'Subject {subject}\'s decoding started')
    epochs = []
        for run_id in range(1, 10):
            print('.', end='')
            epo = epoch_data(subject, '%.2i' % run_id)
            epochs.append(epo)

        # Quick fix for the dev_head: has to be fixed before doing source reconstruction
        for epo in epochs:
            epo.info['dev_head_t'] = epochs[0].info['dev_head_t']


        epochs = mne.concatenate_epochs(epochs) 

        data = epochs.get_data()

        epochs_proc = scale_epochs(data,epochs)

        # Get the evoked potential averaged on all epochs for each channel
        evo = epochs.average()
        evo.plot(spatial_colors=True)

        # Handling the data structure
        epochs.metadata['kind'] = epochs.metadata.trial_type.apply(lambda s: eval(s)['kind'])
        epochs.metadata['word'] = epochs.metadata.trial_type.apply(lambda s: eval(s)['word'])

        # Run a linear regression between MEG signals and word frequency classification
        X = epochs.get_data()
        y = epochs.metadata.word.apply(lambda w: zipf_frequency(w, 'fr'))
        R = decod(X, y)

        dec  = plt.fill_between(epochs.times, R)

        report.add_figure(evo, subject, tags="evo_word")
        report.add_figure(dec, subject, tags="decoding word")

        report.save("decoding.html", open_browser=False, overwrite=True)



    
    # report = mne.Report()

    # decoding
    # all_results = list()
    # results = list()
    
    # # Parallelizing the for loop:
    # # Parallel(n_jobs=16,prefer="threads")(delayed(fig_gen)(subject) for subject in subjects)
    # Parallel(n_jobs=1,prefer="threads")(delayed(fig_gen)(subject) for subject in subjects)


    # # for subject in subjects:

    #     # print(subject)

    #     # out = _decod_one_subject(subject)
    #     # if out is None:
    #     #     continue

    #     # (
    #     #     fig_evo,
    #     #     fig_decod,
    #     #     results,
    #     #     fig_evo_ph,
    #     #     fig_decod_ph,
    #     #     results_ph,
    #     # ) = out

    #     # report.add_figure(fig_evo, subject, tags="evo_word")
    #     # report.add_figure(fig_decod, subject, tags="word")
    #     # report.add_figure(fig_evo_ph, subject, tags="evo_phoneme")
    #     # report.add_figure(fig_decod_ph, subject, tags="phoneme")

    #     # report.save("decoding.html", open_browser=False, overwrite=True)

    #     # all_results.append(results)
    #     # all_results.append(results_ph)
    #     # print("done")

    # pd.concat(all_results, ignore_index=True).to_csv("decoding_results.csv")