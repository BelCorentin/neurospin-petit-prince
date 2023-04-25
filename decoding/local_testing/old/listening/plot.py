'''

Functions for plotting

'''

import mne
import matplotlib.pyplot as plt

def plot_evoked(epochs):

    evo = epochs.average(method="median")
    evo.plot(spatial_colors=True)