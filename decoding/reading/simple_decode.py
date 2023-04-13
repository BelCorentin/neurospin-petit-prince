# Simply test decoding for all subjects, gradually

from dataset import word_epochs_debug, get_path, get_subjects
from utils import save_decoding_results
from plot import plot_R
import mne
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
import numpy as np
from utils import correlate
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


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


path = get_path("LPP_read")
subjects = get_subjects(path)
report = mne.Report()

# WORDS

all_epochs = []
for sub in subjects:

    epochs = word_epochs_debug(sub, 9)
    all_epochs.append(epochs)

    # Decoding for one subject

    # First word:
    epochs = epochs.pick_types(meg=True, stim=False, misc=False)
    X = epochs.get_data()
    y = epochs.metadata.word.apply(len)
    R_vec = decod(X, y)

    decoding_criterion = "wordlengthtest"
    task = "read"
    reference = "start"
    epoch_on = "word"
    save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    dec = plt.fill_between(epochs.times, R_vec)
    # plt.show()
    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)

    report.add_evokeds(evo, titles=f"Evoked for sub {sub} ")
    report.add_figure(fig, title=f"decoding {decoding_criterion} for subject {sub}")

    # Second: embeddings
    import spacy

    nlp = spacy.load("fr_core_news_sm")
    epochs = epochs.pick_types(meg=True, stim=False, misc=False)
    X = epochs.get_data()
    embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
    embeddings = np.array([emb for emb in embeddings])
    R_vec = decod(X, embeddings)
    R_vec = np.mean(R_vec, axis=1)

    decoding_criterion = "wordembed"
    task = "read"
    reference = "start"
    epoch_on = "word"
    save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    dec = plt.fill_between(epochs.times, R_vec)
    # plt.show()

    report.add_figure(fig, title=f"decoding {decoding_criterion} for subject {sub}")

    report.save("./figures/decoding_test.html", open_browser=False, overwrite=True)


for epo in all_epochs:
    epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

epochs_all = mne.concatenate_epochs(all_epochs)


# Actually decode

# Word length

# Remove the STIM information before decoding it (or else we'll get a 100% accuracy since the word length info is in the STIM channels)
epochs_all = epochs_all.pick_types(meg=True, stim=False, misc=False)
X = epochs_all.get_data()
y = epochs_all.metadata.word.apply(len)
R_vec = decod(X, y)


decoding_criterion = "wordlengthtest"
task = "read"
reference = "start"
epoch_on = "word"
save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

fig, ax = plt.subplots(1, figsize=[6, 6])
dec = plt.fill_between(epochs.times, R_vec)
# plt.show()
evo = epochs.average(method="median")
evo.plot(spatial_colors=True)

report.add_evokeds(evo, titles=f"Evoked for sub {sub} ")
report.add_figure(fig, title=f"decoding {decoding_criterion} for all")

# Emb
nlp = spacy.load("fr_core_news_sm")
epochs = epochs.pick_types(meg=True, stim=False, misc=False)
X = epochs.get_data()
embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
embeddings = np.array([emb for emb in embeddings])
R_vec = decod(X, embeddings)
R_vec = np.mean(R_vec, axis=1)


fig, ax = plt.subplots(1, figsize=[6, 6])
dec = plt.fill_between(epochs.times, R_vec)

report.add_figure(fig, title=f"decoding {decoding_criterion} for all")

report.save("./figures/decoding_test.html", open_browser=False, overwrite=True)
