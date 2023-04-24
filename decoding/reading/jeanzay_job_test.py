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
runs = 4

epoch_windows = {"word": {"onset_min": -0.3, "onset_max": 1.0, "offset_min": -1.0, "offset_max": 0.3},
                  "constituent": {"offset_min": -2.0, "offset_max": 0.5, "onset_min": -0.5, "onset_max": 2.0},
                  "sentence": {"offset_min": -4.0, "offset_max": 1.0, "onset_min": -1.0, "onset_max": 4.0}}


            
for subject in subjects[2:]:
    dict_epochs = dict() # DICT containing epochs grouped by conditions (start x level)
    # Dict init
    for start in ('onset', 'offset'): 
            for level in ('word', 'constituent', 'sentence'):
                epoch_key = f'{level}_{start}'
                dict_epochs[epoch_key] = [] 
    for run in range(1,runs+1):
        raw, meta_, events = read_raw(subject, run, events_return = True)
        meta = meta_.copy()
        # Metadata update
        # Word start
        meta['word_onset'] = True

        # Word end
        meta['word_offset'] = True

        # Sent start
        meta['sentence_onset'] = meta.word_id == 0

        # Sent stop
        meta['next_word_id'] = meta['word_id'].shift(-1)
        meta['sentence_offset'] = meta.apply(lambda x: True if x['word_id'] > x['next_word_id'] else False, axis=1)
        meta['sentence_offset'].fillna(False, inplace=True)
        meta.drop('next_word_id', axis=1, inplace=True)

        # Const start
        meta['prev_closing'] = meta['n_closing'].shift(1)
        meta['constituent_onset'] = meta.apply(lambda x: True if x['prev_closing'] > x['n_closing'] and x['n_closing'] == 1 else False, axis=1)
        meta['constituent_onset'].fillna(False, inplace=True)
        meta.drop('prev_closing', axis=1, inplace=True)

        # Const stop
        meta['next_closing'] = meta['n_closing'].shift(-1)
        meta['constituent_offset'] = meta.apply(lambda x: True if x['n_closing'] > x['next_closing'] else False, axis=1)
        meta['constituent_offset'].fillna(False, inplace=True)
        meta.drop('next_closing', axis=1, inplace=True)

        for start in ('onset', 'offset'): 
            # for level in ('word', 'constituent', 'sentence'):
            for level in ('sentence', 'constituent', 'word'):
                # Select only the rows containing the True for the conditions (sentence_end, etc..)
                sel = meta.query(f'{level}_{start}==True')
                assert sel.shape[0] > 10  #
                # TODO check variance as well for sentences
                # Matchlist events and meta
                # So that we can epoch now that's we've sliced our metadata
                i, j = match_list(events[:, 2], sel.word.apply(len))
                sel = sel.reset_index().loc[j]
                epochs = mne.Epochs(raw, **mne_events(sel, raw), decim = 10,
                                     tmin = epoch_windows[f'{level}'][f'{start}_min'],
                                       tmax = epoch_windows[f'{level}'][f'{start}_max'],
                                         event_repeated = 'drop', # check event repeated
                                            preload=True)  # n_words OR n_constitutent OR n_sentences
                epoch_key = f'{level}_{start}'
            
                dict_epochs[epoch_key].append(epochs)
        
            
    # Once we have the dict of epochs per condition full (epoching for each run for a subject)
    # we can concatenate them, and fix the dev_head             
    for start_ in ('onset', 'offset'): 
        for level_ in ('word', 'constituent', 'sentence'):
            epoch_key = f'{level_}_{start_}'
            all_epochs_chosen = dict_epochs[epoch_key]
            # Concatenate epochs
            for epo in all_epochs_chosen:
                epo.info["dev_head_t"] = all_epochs_chosen[1].info["dev_head_t"]

            dict_epochs[epoch_key] = mne.concatenate_epochs(all_epochs_chosen)

    # Now that we have all the epochs, rerun the plotting / decoding on averaged
    for start in ('onset', 'offset'): 
            for level in ('word', 'constituent', 'sentence'):  
                epoch_key = f'{level}_{start}'
                epochs = dict_epochs[epoch_key]
                # mean
                evo = epochs.copy().pick_types(meg=True).average(method='median')
                all_evos.append(dict(subject=subject, evo=evo, start=start, level=level))


                # decoding word emb
                epochs = epochs.load_data().pick_types(meg=True, stim=False, misc=False)
                X = epochs.get_data()
                embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
                embeddings = np.array([emb for emb in embeddings])
                R_vec = decod_xy(X, embeddings)
                scores = np.mean(R_vec, axis=1)

                for t, score in enumerate(scores):
                    all_scores.append(dict(subject=subject, score=score, start=start, level=level, t=epochs.times[t]))

all_scores = pd.DataFrame(all_scores)
all_evos = pd.DataFrame(all_evos)

all_scores.to_csv('./score.csv')
all_evos.to_csv('./evos.csv')
