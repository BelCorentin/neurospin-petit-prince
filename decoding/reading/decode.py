# Homemade imports
from dataset import get_path, get_subjects, epoch_runs
from utils import (
    decod,
    save_decoding_results,
)
from plot import plot_subject

# General imports
import numpy as np
import spacy
import mne
import hydra
from omegaconf import DictConfig

mne.set_log_level(False)


"""
Set-up cases: what kind of decoding is being done:
- word length
- embeddings
- closing nodes
- etc...

We are using Hydra configurations to simplify this process
"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    decoding_criterion = cfg.decod.decoding_criterion
    task = cfg.decod.task
    baseline_min = cfg.decod.baseline_min
    baseline_max = cfg.decod.baseline_max

    print("Decoding criterion chosen:", decoding_criterion)
    print("Modality of data (read or audio) chosen:", task)

    if decoding_criterion == "embeddings":
        nlp = spacy.load("fr_core_news_sm")
    elif decoding_criterion == "syntax":
        target = "n_closing"
    elif decoding_criterion == "word_length":
        target = "word_length"
    # To keep implementing

    report = mne.Report()
    path = get_path("LPP_read")
    subjects = get_subjects(path)
    RUN = 1
    task = "read"

    print("\nSubjects for which the decoding will be tested: \n")
    print(subjects)

    for subject in subjects[4]:  # Ignore the first one
        print(f"Subject {subject}'s decoding started")
        epochs = epoch_runs(subject, RUN, task, path)  # To do: add baseline as a param

        # Get the evoked potential averaged on all epochs for each channel
        # evo = epochs.average(method="median")
        # evo.plot(spatial_colors=True)

        # Run a linear regression between MEG signals
        # and word frequency classification
        # X = epochs.get_data()

        # y = create_target(decoding_criterion, epochs)

        R_vec = decod(epochs, target)
        if decoding_criterion == "embeddings":
            R_vec = np.mean(R_vec, axis=1)

        save_decoding_results(subject, decoding_criterion, task, R_vec)

        fig = plot_subject(subject, decoding_criterion, task)
        # plt.show()
        # report.add_evokeds(evo, titles=f"Evoked for sub {subject} ")
        report.add_figure(fig, title=f"decoding for subject {subject}")
        # report.add_figure(dec, subject, tags="word")
        report.save(
            f"./figures/{task}_decoding_{decoding_criterion}_{subject}.html",
            open_browser=False,
            overwrite=True,
        )
        report.save(
            f"./figures/{task}_decoding_{decoding_criterion}.html",
            open_browser=False,
            overwrite=True,
        )

        print("Finished!")


run()
