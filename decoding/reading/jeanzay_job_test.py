from dataset import read_raw, get_subjects, get_path, mne_events
from utils import decod_xy
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import match_list

all_evos = []
all_scores = []

path = get_path("LPP_read")
subjects = get_subjects(path)
task = "read"
# Debug
run = 1


for subject in subjects[4]:
    print()
    raw, meta, events = read_raw(subject, run, events_return = True)
    for start in ('onset', 'offset'): 
        for level in ('word', 'constituent', 'sentence'):
            # Word start
            meta['word_onset'] = True

            # I don't really understand what to do here.. nothing for the moment
            # But I guess it will make sense once we introduce the baseline
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

            # Select only the rows containing the True for the conditions (sentence_end, etc..)
            sel = meta.query(f'{level}_{start}==True')

            # Matchlist events and meta
            # So that we can epoch now that's we've sliced our metadata
            i, j = match_list(events[:, 2], sel.word.apply(len))
            sel = sel.reset_index().loc[j]
            epochs = mne.Epochs(raw, **mne_events(sel, raw), event_repeated = 'drop')  # n_words OR n_constitutent OR n_sentences

            # mean
            evo = epochs.copy().load_data().pick_types(meg=True).average(method='median').get_data()

            # decoding word emb
            import spacy
            nlp = spacy.load("fr_core_news_sm")
            epochs = epochs.load_data().pick_types(meg=True, stim=False, misc=False)
            X = epochs.get_data()
            embeddings = epochs.metadata.word.apply(lambda word: nlp(word).vector).values
            embeddings = np.array([emb for emb in embeddings])
            R_vec = decod_xy(X, embeddings)
            scores = np.mean(R_vec, axis=1)

            for t, score in enumerate(scores):
                all_evos.append(dict(subject=subject, evo=evo, start=start, level=level, t=epochs.times[t]))
                all_scores.append(dict(subject=subject, score=score, start=start, level=level, t=epochs.times[t]))

all_scores = pd.DataFrame(all_scores)


fig, axes = plt.subplots(3, 2)

for axes_, start in zip(axes, ('onset', 'offset')):
    for ax, level in zip( axes_, ('word', 'constituent', 'sentence')):  
        cond1 = all_scores.level==f'{level}'
        cond2 = all_scores['start']==f'{start}'
        data = all_scores[ cond1 & cond2]
        x = data['t']
        y = data['score']
        
        ax.plot(x,y)
        # sns.lineplot(ax=ax, x='t', y='score', data=all_scores.query('start==@start, level==@level'))

plt.savefig('./test.png')