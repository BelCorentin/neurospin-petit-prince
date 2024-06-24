"""

This dataset.py file contains all the functions that are needed
to either read, preprocess, populate metadata, epoch, or in general deal
with the MEG data, metadata etc..

All the more general functions, not necessarily linked to the LPP
dataset will be put in utils.py instead

The general pipeline, when calling the higher level function: 

analysis_subject

Goes like:

analysis_subject, that returns the decoding score, for the parameters:
a subject, 
a modality (auditory or visual), 
a start (onset or offset), 
a level (word, constituent, sentence), 
a decoding_criterion (LASER embeddings, BOW embeddings, 1st word embeddings, ...)

calls itself 2 main functions:

    populate_metadata_epochs, that, in order to build epochs based on the previous parameters, will call:

        enrich_metadata(meta) that will populate the metadata, allowing us to match events and metadata, and use them for decoding

        select_meta_subset(meta, level, decoding_criterion) that will select the subset of events to use for epoching

        epoch_on_selection(raw, sel, start, level) that will build an epoch object on the previous selection

        apply_baseline that applies a baseline to the epochs object
        ## Note: for later, this apply_baseline would have to be applied before the epochs selection, now the question is how?

    and decoding_from_criterion, that will call:

        generate_embeddings: generate embeddings for the given epochss

        decod_Xy that will simply evaluate how much
            information can be infered from X to predict y.
            Training a RidgeCV and then calculating the
            Pearson R between predicted and test y



"""


# Neuro
import mne
import mne_bids

# ML/Data
import numpy as np
import pandas as pd

# Tools
from pathlib import Path
import os
import spacy
import matplotlib.pyplot as plt
from functools import lru_cache
import typing as tp
import itertools
import dataclasses


# Homemade
from utils import (
    match_list,
    add_syntax,
    mne_events,
    decoding_from_criterion,
    get_code_path,
    get_path,
)

mne.set_log_level(verbose="error")
nlp = spacy.load("fr_core_news_sm")

class NoApproximateMatch(ValueError):
    """Error raised when the function could not fully match the two list
    The error has a 'match' attribute holding the matches so far, for debugging
    """

    def __init__(self, msg: str, matches: tp.Any) -> None:
        super().__init__(msg)
        self.matches = matches


@dataclasses.dataclass
class Tolerance:
    """Convenience tool for check if a value is  within tolerance"""

    abs_tol: float
    rel_tol: float

    def __call__(self, value1: float, value2: float) -> bool:
        diff = abs(value1 - value2)
        tol = max(self.abs_tol, self.rel_tol * min(abs(value1), abs(value2)))
        return diff <= tol


@dataclasses.dataclass
class Sequence:
    """Handle for current information on the sequence matching"""

    sequence: tp.Sequence[float]  # the sequence to match
    current: int  # the current index for next match look-up
    matches: tp.List[int]  # the matches so far in the sequence

    def valid_index(self, shift: int = 0) -> bool:
        return self.current + shift < len(self.sequence)

    def diff(self, shift: int = 0) -> float:
        return self.sequence[self.current + shift] - self.last_value

    @property
    def last_value(self) -> float:
        return self.sequence[self.matches[-1]]

    def diff_to(self, ind: int) -> np.ndarray:
        r = self.matches[-1]
        sub = self.sequence[r : r + ind] if ind > 0 else self.sequence[r + ind : r]
        return np.array(sub) - self.last_value


def approx_match_samples(
    s1: tp.Sequence[float],
    s2: tp.Sequence[float],
    abs_tol: float,
    rel_tol: float = 0.003,
    max_missing: int = 3,
    first_match: tp.Tuple[int, int] | None = None,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """Approximate sample sequence matching
    Eg:
    seq0 = [1100, 2300, 3600]
    seq1 = [0, 1110, 3620, 6500]
    will match on 1100-1110 with tolerance 10,
    and then on 3600-3620 (as the diffs match with tolerance 10)

    Returns
    -------
    tuple of indices which match on the first list and the second list
    """
    if first_match is None:
        # we need to figure out the first match:
        # let's try on for several initial matches,
        # and pick the one that matches the most
        success: tp.Tuple[np.ndarray, np.ndarray] | None = None
        error: tp.Any = None
        for offsets in itertools.product(range(max_missing + 1), repeat=2):
            try:
                out = approx_match_samples(
                    s1, s2, abs_tol=abs_tol, rel_tol=rel_tol, max_missing=max_missing, first_match=offsets  # type: ignore
                )
                if success is None or len(out[0]) > len(success[0]):  # type: ignore
                    success = out
            except NoApproximateMatch as e:
                if error is None or error.matches[0][-1] < e.matches[0][-1]:
                    error = e
        if success is not None:
            return success
        raise error
    tolerance = Tolerance(abs_tol=abs_tol, rel_tol=rel_tol)
    seqs = (
        Sequence(s1, first_match[0] + 1, [first_match[0]]),
        Sequence(s2, first_match[1] + 1, [first_match[1]]),
    )
    while all(s.valid_index() for s in seqs):
        # if we match within the tolerance limit, then move on
        # otherwise move the pointer for the less advanced sequence
        if tolerance(seqs[0].diff(), seqs[1].diff()):
            for k, s in enumerate(seqs):
                s.matches.append(s.current)
                s.current += 1
        else:
            # move one step
            seqs[1 if seqs[1].diff() < seqs[0].diff() else 0].current += 1
        # allow for 1 extra (absolute) step if getting closer
        for k, seq in enumerate(seqs):
            other = seqs[(k + 1) % 2]
            if seq.valid_index(shift=1) and other.valid_index():
                # need to check 2 tolerance so that we can match farther
                # if it is closer
                if abs(seq.diff(1) - seq.diff()) <= 2 * abs_tol:
                    if abs(seq.diff(1) - other.diff()) < abs(seq.diff() - other.diff()):
                        seq.current += 1
        # if we are over the limit for matching, then abort
        if any(m.current - m.matches[-1] > max_missing + 1 for m in seqs):
            msg = f"Failed to match after indices {[s.matches[-1] for s in seqs]} "
            msg += f"(values {[s.last_value for s in seqs]}, {first_match=})\n"
            msg += f"(follows:\n {seqs[0].diff_to(10)}\n {seqs[1].diff_to(10)}"
            msg += f"(before:\n {seqs[0].diff_to(-10)}\n {seqs[1].diff_to(-10)}"
            out = tuple(np.array(s.matches) for s in seqs)  # type: ignore
            raise NoApproximateMatch(msg, matches=out)
    return tuple(np.array(s.matches) for s in seqs)  # type: ignore
