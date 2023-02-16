from dataset import get_path, get_subjects, epoch_data
from utils import decod, correlate, match_list
import mne_bids
from pathlib import Path
import pandas as pd
import numpy as np
import mne

from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from wordfreq import zipf_frequency
from Levenshtein import editops
import matplotlib.pyplot as plt

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
        X = epochs.get_data()
        y = epochs.metadata.word.apply(len)
        R = decod(X, y)

        fig, ax = plt.subplots(1, figsize=[6, 6])
        dec = plt.fill_between(epochs.times, np.squeeze(R))
        # plt.show()
        report.add_evokeds(evo, titles=f"Evoked for sub {subject} ")
        report.add_figure(fig, title=f"decoding for subject {subject}")
        # report.add_figure(dec, subject, tags="word")
        report.save("./figures/reading_decoding_word_length.html", open_browser=False, overwrite=True)

        print("Finished!")