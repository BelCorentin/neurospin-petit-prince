# Homemade imports
from dataset import get_path, get_subjects, epoch_subjects, epochs_slice
from plot import plot_subject

# General imports
import numpy as np
import mne
import hydra
from omegaconf import DictConfig
import matplotlib

matplotlib.use("Agg")

mne.set_log_level(False)

# Later: integrate Hydra here as well. For now, just simple plotting of ERPS

report = mne.Report()
path = get_path("LPP_read")
subjects = get_subjects(path)
RUN = 1
baseline_min = -1.0
baseline_max = 1.0
task = "read"
print("\nSubjects for which the plotting will be done: \n")
print(subjects)

# DEBUG
subjects = subjects[4]
epochs = epoch_subjects(
    subjects, RUN, task, path, baseline_max=baseline_max, baseline_min=baseline_min
)






# Build a 3x2 plot, with for each condition (sentence, word, constituent), and for (start, end),
# the ERP associated
cond = ["sentence", "word", "constituent"]
cases = {"start", "end"}

# Plotting and adding to the report, the averaged ERPs of:
# words, sentences and constituents, centered at the beginning and end of each


# Need to map for:
# - end of word,
# - beginning of sentence (epochs[i+1] !danger limits)
# - beginning of constituent (epochs[i+1] !same danger)

evos = []
for condi in cond:
    for case in cases:
        # Slice the epochs based on the epoch_criterion:
        column_to_slice_on = f"{condi}_{case}"
        if condi == "sentence" or condi == "word":  # eg: {sentence}_{end} or {word}_{start}
            epochs = epochs_slice(epochs, column_to_slice_on)
        elif condi == "constituent":
            epochs = epochs_slice(epochs, column_to_slice_on, value=2, equal='sup')
        evo = epochs.average(method="median")
        evos.append(evo)
        evo.plot(spatial_colors=True)
        report.add_evokeds(evo, titles=f"Evoked for condition {column_to_slice_on}  ")

evokeds = dict(sentence=evos[0], word=evos[2], constituent=evos[4])

fig = mne.viz.plot_compare_evokeds(evokeds, combine="mean")

report.add_figure(fig, title="Evoked response comparaison")


report.save(
    f"./figures/{task}_ERP_all_cond.html",
    open_browser=False,
    overwrite=True,
)











# # Build a 3x2 plot, with for each condition (sentence, word, constituent), and for (start, end),
# # the ERP associated
# cond = {
#     "sentence": {"column": "is_last_word", "target": True},
#     "word": {"column": "kind", "target": "word"},
#     "constituent": {"column": "n_closing", "target": 3},
# }

# cases = {"start", "end"}

# # Plotting and adding to the report, the averaged ERPs of:
# # words, sentences and constituents, centered at the beginning and end of each


# # Need to map for:
# # - end of word,
# # - beginning of sentence (epochs[i+1] !danger limits)
# # - beginning of constituent (epochs[i+1] !same danger)

# evos = []
# for condi in cond:
#         for case in cases:
#         target = cond[condi]["target"]
#         col = cond[condi]["column"]
#         if col == "n_closing":  # To handle all n_closings
#             ep = epochs_slice(epochs, col, target, equal="sup")
#         else:
#             ep = epochs_slice(epochs, col, target)
#         # ep.average().plot(gfp='only')
#         evo = ep.average(method="median")
#         evos.append(evo)
#         evo.plot(spatial_colors=True)
#         report.add_evokeds(evo, titles=f"Evoked for condition {col}  ")

# evokeds = dict(sentence=evos[0], word=evos[2], constituent=evos[4])

# fig = mne.viz.plot_compare_evokeds(evokeds, combine="mean")

# report.add_figure(fig, title="Evoked response comparaison")


# report.save(
#     f"./figures/{task}_ERP_all_cond.html",
#     open_browser=False,
#     overwrite=True,
# )
# for condi in cond:
#     for case in cases:
#         target = cond[condi]["target"]
#         col = cond[condi]["column"]
#         if col == "n_closing":  # To handle all n_closings
#             ep = epochs_slice(epochs, col, target, equal="sup")
#         else:
#             ep = epochs_slice(epochs, col, target)
#         # ep.average().plot(gfp='only')
#         evo = ep.average(method="median")
#         evo.plot(spatial_colors=True)
#         report.add_evokeds(evo, titles=f"Evoked for condition {col} and case {case} ")


"""
Setup for plotting ERPs, and other decoding results plots

Hydra options:
- event_type: sentence, word, constituant
- case: before, after
- var: the variable on which the plot depends of
"""


# @hydra.main(version_base=None, config_path="conf_plt", config_name="config")
# def run(cfg: DictConfig) -> None:
#     event_type = cfg.plot.event_type
#     task = cfg.plot.task
#     case = cfg.plot.case
#     var = cfg.plot.var

#     report = mne.Report()
#     path = get_path("LPP_read")
#     subjects = get_subjects(path)
#     RUN = 9

#     print("\nSubjects for which the plotting will be done: \n")
#     print(subjects)

#     for subject in subjects:  # Ignore the first one
#         print(f"Subject {subject}'s epoching and plotting started")
#         epochs = epoch_runs(subject, RUN, task, path)  # To do: add baseline as a param

#         # Get the evoked potential averaged on all epochs for each channel
#         evo = epochs.average(method="median")
#         evo.plot(spatial_colors=True)
#         # if decoding_plot:
#         #     fig = plot_subject(subject, decoding_criterion, task)
#         # plt.show()
#         report.add_evokeds(evo, titles=f"Evoked for sub {subject} ")
#         # report.add_figure(fig, title=f"decoding for subject {subject}")
#         # report.add_figure(dec, subject, tags="word")
#         report.save(
#             f"./figures/{task}_ERP_{event_type}_{case}_{var}_{subject}.html",
#             open_browser=False,
#             overwrite=True,
#         )
#         report.save(
#             f"./figures/{task}_ERP_{event_type}_{case}_{var}.html",
#             open_browser=False,
#             overwrite=True,
#         )

#         print("Finished!")


# run()