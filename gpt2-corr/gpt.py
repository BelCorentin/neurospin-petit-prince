#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mne
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import RidgeCV
from wordfreq import zipf_frequency
from Levenshtein import editops
import torch

# Tools
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
mne.set_log_level(False)


# Utils
def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def decod(X, y):
    assert len(X) == len(y)
    # define data
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 10)))
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, n_chans, n_times = y.shape
    R = np.zeros(n_times)
    for t in range(n_times):
        print(".", end="")
        y_pred = cross_val_predict(model, X, y[:, :, t], cv=cv)
        R[t] = correlate(y[:, :, t], y_pred)
    return R


# Function to correlate
def correlate(X, Y):
    correlation = np.corrcoef(X, Y)
    return np.mean(correlation)
    # if X.ndim == 1:
    #     X = np.array(X)[:, None]
    # if Y.ndim == 1:
    #     Y = np.array(Y)[:, None]
    # X = X - X.mean(0)
    # Y = Y - Y.mean(0)

    # SX2 = (X**2).sum(0) ** 0.5
    # SY2 = (Y**2).sum(0) ** 0.5
    # SXY = (X * Y).sum(0)
    # return SXY / (SX2 * SY2)


#

# ## Initial textual data
#
# The textual data that we'll use to compare GPT2 and MEG activations is from Le Petit Prince.
# Let's build an array with each words.

# In[2]:


file = "~/Downloads/lpp.csv"
df = pd.read_csv(file)
list_words = df.iloc[:, 1]
list_words_run1 = list_words[:1610]


# ## MEG Activations

# In[5]:


# Read raw file
raw_file = "/home/co/workspace_LPP/data/MEG/LPP/LPP_bids/sub-220628/ses-01/meg/sub-220628_ses-01_task-rest_run-01_meg.fif"
raw = mne.io.read_raw_fif(raw_file, allow_maxshield=True)

# Load data, filter
raw.pick_types(meg=True, stim=True)
raw.load_data()
raw = raw.filter(0.5, 20)

# Load events and realign them
event_file = "/home/co/workspace_LPP/data/MEG/LPP/LPP_bids/sub-220628/ses-01/meg/sub-220628_ses-01_task-rest_run-01_events.tsv"
meta = pd.read_csv(event_file, sep="\t")
events = mne.find_events(
    raw, stim_channel="STI101", shortest_event=1, min_duration=0.0010001
)

# match events and metadata
word_events = events[events[:, 2] > 1]
meg_delta = np.round(np.diff(word_events[:, 0] / raw.info["sfreq"]))
meta_delta = np.round(np.diff(meta.onset.values))

i, j = match_list(meg_delta, meta_delta)
events = word_events[i]
meta = meta.iloc[j].reset_index()

# Epoch on the aligned events and load the epoch data
epochs = mne.Epochs(
    raw, events, metadata=meta, tmin=-0.3, tmax=0.8, decim=10, baseline=(-0.2, 0.0)
)

data = epochs.get_data()
epochs.load_data()

# Scale the data
n_words, n_chans, n_times = data.shape
vec = data.transpose(0, 2, 1).reshape(-1, n_chans)
scaler = RobustScaler()
idx = np.arange(len(vec))
np.random.shuffle(idx)
vec = scaler.fit(vec[idx[:20_000]]).transform(vec)
sigma = 7
vec = np.clip(vec, -sigma, sigma)
epochs._data[:, :, :] = (
    scaler.inverse_transform(vec).reshape(n_words, n_times, n_chans).transpose(0, 2, 1)
)


# In[6]:


epochs.get_data().shape


# ## GPT 2 Activations

# In[7]:


from transformers import GPT2Config, GPT2Model, GPT2Tokenizer

# Initiate the model and its trained weights
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")


# In[8]:


embeddings_array = []  # The list containing the different embeddings
# Accessible under the parameters: embeddings_array[word_n, layer_n, 0, 0]

# For each word, tokenize it and get the embeddings from it
for word in list_words_run1:
    inputs = tokenizer(word, return_tensors="pt")

    outputs = model(**inputs, output_hidden_states=True)

    embeddings_array.append(outputs.hidden_states)


# In[9]:


len(embeddings_array)
(embeddings_array[0][12][0])


# In[11]:


embeddings_array[0][0].shape


# ## Correlation

# In[21]:


final_array = []
for tuple_ in embeddings_array:

    tensor = torch.cat(tuple_)
    print(tensor)
    tensor = tensor[:, 0, :]
    tensor.reshape(13, 1, 768)
    tensor = tensor.detach().numpy()
    final_array.append(tensor)

final_array = np.reshape(final_array, (1610, -1))


# In[27]:


X = final_array
X.shape


# In[25]:


y = epochs.get_data()
y.shape


# In[35]:


R = decod(X, y)


# In[ ]:


fig, ax = plt.subplots(1, figsize=[6, 6])
dec = plt.fill_between(epochs.times, R)


plt.save_fig("./dec_gpt.png")
# # Utils

# In[34]:


# Utils
def match_list(A, B, on_replace="delete"):
    """Match two lists of different sizes and return corresponding indice
    Parameters
    ----------
    A: list | array, shape (n,)
        The values of the first list
    B: list | array: shape (m, )
        The values of the second list
    Returns
    -------
    A_idx : array
        The indices of the A list that match those of the B
    B_idx : array
        The indices of the B list that match those of the A
    """

    if not isinstance(A, str):
        unique = np.unique(np.r_[A, B])
        label_encoder = dict((k, v) for v, k in enumerate(unique))

        def int_to_unicode(array: np.ndarray) -> str:
            return "".join([str(chr(label_encoder[ii])) for ii in array])

        A = int_to_unicode(A)
        B = int_to_unicode(B)

    changes = editops(A, B)
    B_sel = np.arange(len(B)).astype(float)
    A_sel = np.arange(len(A)).astype(float)
    for type_, val_a, val_b in changes:
        if type_ == "insert":
            B_sel[val_b] = np.nan
        elif type_ == "delete":
            A_sel[val_a] = np.nan
        elif on_replace == "delete":
            # print('delete replace')
            A_sel[val_a] = np.nan
            B_sel[val_b] = np.nan
        elif on_replace == "keep":
            # print('keep replace')
            pass
        else:
            raise NotImplementedError
    B_sel = B_sel[np.where(~np.isnan(B_sel))]
    A_sel = A_sel[np.where(~np.isnan(A_sel))]
    assert len(B_sel) == len(A_sel)
    return A_sel.astype(int), B_sel.astype(int)


def decod(X, y):
    assert len(X) == len(y)
    # define data
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 8, 10)))
    cv = KFold(5, shuffle=True, random_state=0)

    # fit predict
    n, n_chans, n_times = y.shape
    R = np.zeros(n_times)
    for t in range(n_times):
        print(".", end="")
        y_pred = cross_val_predict(model, X, y[:, :, t], cv=cv)
        R[t] = correlate(y[:, :, t], y_pred)
    return R


# Function to correlate
def correlate(X, Y):
    correlation = np.corrcoef(X, Y)
    return np.mean(correlation)
    # if X.ndim == 1:
    #     X = np.array(X)[:, None]
    # if Y.ndim == 1:
    #     Y = np.array(Y)[:, None]
    # X = X - X.mean(0)
    # Y = Y - Y.mean(0)

    # SX2 = (X**2).sum(0) ** 0.5
    # SY2 = (Y**2).sum(0) ** 0.5
    # SXY = (X * Y).sum(0)
    # return SXY / (SX2 * SY2)


#
