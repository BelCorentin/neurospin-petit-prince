"""

Functions for plotting

"""
from dataset import get_path

import mne
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def plot_evoked(epochs):

    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)


def plot_subject(subject, decoding_criterion, task):
    """
    Function to plot the decoding score of a particular subject
    and for a particular decoding criterion
    eg: word length, embeddings, closing nodes

    Returns: matplotlib plot
    """
    if task == "read":
        task_path = "LPP_read"
    elif task == "listen":
        task_path = "LPP_listen"
    path = get_path(task_path)
    # Format the file path
    file_path = Path("to/define.tsv")

    # Open the pandas DataFrame containing the decoding values
    df = pd.open_csv(file_path, sep="\t")

    # Plot it
    plt.fill_between(df["R_score"], df["epochs.times"])
