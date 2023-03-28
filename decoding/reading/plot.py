"""

Functions for plotting

"""
from dataset import get_path, get_code_path

import mne
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_evoked(epochs):

    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)


def plot_subject(sub, decoding_criterion, task, reference, epoch_on, min, max):
    """
    Function to plot the decoding score of a particular subject
    and for a particular decoding criterion
    eg: word length, embeddings, closing nodes

    Returns: matplotlib plot
    """

    path = get_code_path()
    # Format the file path

    # Open the pandas DataFrame containing the decoding values
    R = np.load(
        (path) / f"decoding/results/{task}/decoding_{decoding_criterion}_{epoch_on}_{reference}_{sub}.npy"
    )
    # Plot it
    times = np.linspace(min, max, R.shape[0])  # To do better at generalizing
    fig, ax = plt.subplots(1, figsize=[6, 6])
    dec = plt.fill_between(times, R)
    return fig
