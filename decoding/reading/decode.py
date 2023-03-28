# Homemade imports
from dataset import get_path, get_subjects, epoch_runs, epochs_slice
from utils import (
    decod,
    save_decoding_results,
)

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

Hydra organization:

"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg: DictConfig) -> None:
    decoding_criterion = cfg.decod.decoding_criterion
    task = cfg.decod.task
    baseline_min = cfg.decod.baseline_min
    baseline_max = cfg.decod.baseline_max
    epoch_on = cfg.decod.epoch_on
    reference = cfg.decod.reference

    print("\nDecoding criterion chosen:", decoding_criterion)
    print("\nModality of data (read or audio) chosen:", task)
    print("\nStart of decoding window chosen:", baseline_min)
    print("\nEnd of decoding window chosen:", baseline_max)
    print("\nEpoching on:", epoch_on)
    print("\nStart or end of previous epoching reference:\n", reference)

    path = get_path("LPP_read")
    subjects = get_subjects(path)
    RUN = 9

    print("\nSubjects for which the decoding will be tested: \n")
    print(subjects)

    for subject in subjects[1:]:  # Removing subject 1 as it doesn't work
        print(f"Subject {subject}'s decoding started")
        epochs = epoch_runs(
            subject,
            RUN,
            task,
            path,
            baseline_min,
            baseline_max,
        )

        # Slice the epochs based on the epoch_criterion:
        column_to_slice_on = f"{epoch_on}_{reference}"
        if epoch_on == "sentence" or epoch_on == "word":  # eg: {sentence}_{end} or {word}_{start}
            epochs = epochs_slice(epochs, column_to_slice_on)
        elif epoch_on == "constituent":
            epochs = epochs_slice(epochs, column_to_slice_on, value=2, equal='sup')

        R_vec = decod(epochs, decoding_criterion)
        if decoding_criterion == "embeddings":
            R_vec = np.mean(R_vec, axis=1)

        save_decoding_results(
            subject, decoding_criterion, task, reference, epoch_on, R_vec
        )
        print("Decoding job finished!")


run()
