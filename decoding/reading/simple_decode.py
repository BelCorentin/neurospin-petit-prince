from dataset import read_raw, get_subjects, get_path, mne_events
from utils import decod_xy
import mne
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import match_list
import spacy

nlp = spacy.load("fr_core_news_sm")

all_evos = []
all_scores = []

path = get_path("LPP_read")
subjects = get_subjects(path)
task = "read"
# Debug
run = 1

epoch_windows = {
    "word": {"min": -1.0, "max": 0.3},
    "constituent": {"min": -2.0, "max": 0.5},
    "sentence": {"min": -4.0, "max": 1.0},
}


report = mne.Report()

for subject in subjects[-2:]:
    print()
    raw, meta_, events = read_raw(subject, run, events_return=True)
    meta = meta_.copy()
    # Metadata update
    # Word start
    meta["word_onset"] = True

    # Word end
    meta["word_offset"] = True

    # Sent start
    meta["sentence_onset"] = meta.word_id == 0

    # Sent stop
    meta["next_word_id"] = meta["word_id"].shift(-1)
    meta["sentence_offset"] = meta.apply(
        lambda x: True if x["word_id"] > x["next_word_id"] else False, axis=1
    )
    meta["sentence_offset"].fillna(False, inplace=True)
    meta.drop("next_word_id", axis=1, inplace=True)

    # Const start
    meta["prev_closing"] = meta["n_closing"].shift(1)
    meta["constituent_onset"] = meta.apply(
        lambda x: True
        if x["prev_closing"] > x["n_closing"] and x["n_closing"] == 1
        else False,
        axis=1,
    )
    meta["constituent_onset"].fillna(False, inplace=True)
    meta.drop("prev_closing", axis=1, inplace=True)

    # Const stop
    meta["next_closing"] = meta["n_closing"].shift(-1)
    meta["constituent_offset"] = meta.apply(
        lambda x: True if x["n_closing"] > x["next_closing"] else False, axis=1
    )
    meta["constituent_offset"].fillna(False, inplace=True)
    meta.drop("next_closing", axis=1, inplace=True)

    for start in ("onset", "offset"):
        # for level in ('word', 'constituent', 'sentence'):
        for level in ("sentence", "constituent", "word"):
            # Select only the rows containing the True for the conditions (sentence_end, etc..)
            sel = meta.query(f"{level}_{start}==True")
            print(sel)
            assert sel.shape[0] > 10  #
            # TODO check variance as well for sentences
            # Matchlist events and meta
            # So that we can epoch now that's we've sliced our metadata
            i, j = match_list(events[:, 2], sel.word.apply(len))
            sel = sel.reset_index().loc[j]
            epochs = mne.Epochs(
                raw,
                **mne_events(sel, raw),
                decim=10,
                tmin=epoch_windows[f"{level}"]["min"],
                tmax=epoch_windows[f"{level}"]["max"],
                event_repeated="drop",
            )  # n_words OR n_constitutent OR n_sentences

            # mean
            evo = (
                epochs.copy()
                .load_data()
                .pick_types(meg=True)
                .average(method="median")
                .get_data()
            )

            # decoding word emb
            epochs = epochs.load_data().pick_types(meg=True, stim=False, misc=False)
            X = epochs.get_data()
            embeddings = epochs.metadata.word.apply(
                lambda word: nlp(word).vector
            ).values
            embeddings = np.array([emb for emb in embeddings])
            print(embeddings)
            R_vec = decod_xy(X, embeddings)
            scores = np.mean(R_vec, axis=1)

            for t, score in enumerate(scores):
                all_evos.append(
                    dict(
                        subject=subject,
                        evo=evo,
                        start=start,
                        level=level,
                        t=epochs.times[t],
                    )
                )
                all_scores.append(
                    dict(
                        subject=subject,
                        score=score,
                        start=start,
                        level=level,
                        t=epochs.times[t],
                    )
                )

all_scores = pd.DataFrame(all_scores)

fig, axes = plt.subplots(3, 2)

for axes_, start in zip(axes, ("onset", "offset")):
    for ax, level in zip(axes_, ("word", "constituent", "sentence")):
        cond1 = all_scores.level == f"{level}"
        cond2 = all_scores["start"] == f"{start}"
        data = all_scores[cond1 & cond2]
        x = data["t"]
        y = data["score"]

        ax.plot(x, y)
        # sns.lineplot(ax=ax, x='t', y='score', data=all_scores.query('start==@start, level==@level'))

plt.savefig("./fig_plot.png")


# # Simply test decoding for all subjects, gradually

# from dataset import word_epochs_debug, get_path, get_subjects
# from utils import save_decoding_results
# from plot import plot_R
# import mne
# from sklearn.model_selection import KFold, cross_val_predict
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler, RobustScaler
# from sklearn.linear_model import RidgeCV
# import numpy as np
# from utils import correlate
# import matplotlib.pyplot as plt
# import matplotlib

# matplotlib.use("Agg")


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


# path = get_path("LPP_read")
# subjects = get_subjects(path)
# report = mne.Report()

# # WORDS

# all_epochs = []
# for sub in subjects:

#     epochs = word_epochs_debug(sub, 9)
#     all_epochs.append(epochs)

#     # Decoding for one subject

#     # First word:
#     epochs = epochs.pick_types(meg=True, stim=False, misc=False)
#     X = epochs.get_data()
#     y = epochs.metadata.word.apply(len)
#     R_vec = decod(X, y)

#     decoding_criterion = "wordlengthtest"
#     task = "read"
#     reference = "start"
#     epoch_on = "word"
#     save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

#     fig, ax = plt.subplots(1, figsize=[6, 6])
#     dec = plt.fill_between(epochs.times, R_vec)
#     # plt.show()
#     evo = epochs.average(method="median")
#     evo.plot(spatial_colors=True)

#     report.add_evokeds(evo, titles=f"Evoked for sub {sub} ")
#     report.add_figure(fig, title=f"decoding {decoding_criterion} for subject {sub}")

#     # Second: embeddings
#     import spacy

#     nlp = spacy.load("fr_core_news_sm")
#     epochs = epochs.pick_types(meg=True, stim=False, misc=False)
#     X = epochs.get_data()
#     embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
#     embeddings = np.array([emb for emb in embeddings])
#     R_vec = decod(X, embeddings)
#     R_vec = np.mean(R_vec, axis=1)

#     decoding_criterion = "wordembed"
#     task = "read"
#     reference = "start"
#     epoch_on = "word"
#     save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

#     fig, ax = plt.subplots(1, figsize=[6, 6])
#     dec = plt.fill_between(epochs.times, R_vec)
#     # plt.show()

#     report.add_figure(fig, title=f"decoding {decoding_criterion} for subject {sub}")

#     report.save("./figures/decoding_test.html", open_browser=False, overwrite=True)


# for epo in all_epochs:
#     epo.info["dev_head_t"] = all_epochs[1].info["dev_head_t"]

# epochs_all = mne.concatenate_epochs(all_epochs)


# # Actually decode

# # Word length

# # Remove the STIM information before decoding it (or else we'll get a 100% accuracy since the word length info is in the STIM channels)
# epochs_all = epochs_all.pick_types(meg=True, stim=False, misc=False)
# X = epochs_all.get_data()
# y = epochs_all.metadata.word.apply(len)
# R_vec = decod(X, y)


# decoding_criterion = "wordlengthtest"
# task = "read"
# reference = "start"
# epoch_on = "word"
# save_decoding_results(sub, decoding_criterion, task, reference, epoch_on, R_vec)

# fig, ax = plt.subplots(1, figsize=[6, 6])
# dec = plt.fill_between(epochs.times, R_vec)
# # plt.show()
# evo = epochs.average(method="median")
# evo.plot(spatial_colors=True)

# report.add_evokeds(evo, titles=f"Evoked for sub {sub} ")
# report.add_figure(fig, title=f"decoding {decoding_criterion} for all")

# # Emb
# nlp = spacy.load("fr_core_news_sm")
# epochs = epochs.pick_types(meg=True, stim=False, misc=False)
# X = epochs.get_data()
# embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
# embeddings = np.array([emb for emb in embeddings])
# R_vec = decod(X, embeddings)
# R_vec = np.mean(R_vec, axis=1)


# fig, ax = plt.subplots(1, figsize=[6, 6])
# dec = plt.fill_between(epochs.times, R_vec)

# report.add_figure(fig, title=f"decoding {decoding_criterion} for all")

# report.save("./figures/decoding_test.html", open_browser=False, overwrite=True)
