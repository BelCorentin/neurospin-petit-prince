"""

Functions for plotting

"""

import mne
import matplotlib.pyplot as plt
import pandas as pd


def plot_evoked(epochs):

    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)


def plot_subject(subject, decoding_criterion):
    """
    Function to plot the decoding score of a particular subject
    and for a particular decoding criterion
    eg: word length, embeddings, closing nodes

    Returns: matplotlib plot
    """

    # Format the file path
    file_path = Path("to/define.tsv")

    # Open the pandas DataFrame containing the decoding values
    df = pd.open_csv(file_path, sep="\t")

    # Plot it
    plt.fill_between(df["R_score"], df["epochs.times"])
