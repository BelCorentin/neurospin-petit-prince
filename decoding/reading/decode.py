from dataset import get_path, get_subjects, epoch_data, epoch_runs
from utils import decod, correlate, match_list, create_target, analysis
import mne_bids
from pathlib import Path
import pandas as pd
import numpy as np
import mne
import spacy
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from wordfreq import zipf_frequency
from Levenshtein import editops
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
mne.set_log_level(False)

"""
Set-up cases: what kind of decoding is being done:
- word length
- embeddings
- closing nodes
- etc...
"""
decoding_criterion = "word_length"

if decoding_criterion == "embeddings":
    nlp = spacy.load("fr_core_news_sm")


report = mne.Report()
path = get_path("LPP_read")
subjects = get_subjects(path)
RUN = 2
task = "read"

print("\nSubjects for which the decoding will be tested: \n")
print(subjects)

for subject in subjects:  # Ignore the first one
    print(f"Subject {subject}'s decoding started")
    epochs = epoch_runs(subject, RUN, task, path)

    # Get the evoked potential averaged on all epochs for each channel
    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)

    # Run a linear regression between MEG signals
    # and word frequency classification
    X = epochs.get_data()

    y = create_target(decoding_criterion, epochs)

    R_vec = decod(X, y)
    if decoding_criterion == "emdeddings":
        R_vec = np.mean(R_vec, axis=1)

    fig, ax = plt.subplots(1, figsize=[6, 6])
    dec = plt.fill_between(epochs.times, R_vec)
    # plt.show()
    report.add_evokeds(evo, titles=f"Evoked for sub {subject} ")
    report.add_figure(fig, title=f"decoding for subject {subject}")
    # report.add_figure(dec, subject, tags="word")
    report.save(
        f"./figures/reading_decoding_{decoding_criterion}.html",
        open_browser=False,
        overwrite=True,
    )

    print("Finished!")
