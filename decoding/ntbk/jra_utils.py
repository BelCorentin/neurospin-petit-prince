

import itertools

import typing as tp


import numpy as np


class NoApproximateMatch(ValueError):
    def __init__(self, msg: str, ind: int, matches: tp.Any) -> None:
        super().__init__(msg)
        self.ind = ind
        self.matches = matches


def approx_match_samples(
    s1: tp.List[int],
    s2: tp.List[int],
    tol: int,
    rel_tol: float = 0.003,
    max_missing: int = 3,
    first_match: tp.Tuple = None,
) -> tp.Tuple:
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
                    s1, s2, tol=tol, max_missing=max_missing, first_match=offsets  # type: ignore
                )
                if success is None or len(out[0]) > len(success[0]):  # type: ignore
                    success = out
            except NoApproximateMatch as e:
                if error is None or error.ind < e.ind:
                    error = e
        if success is not None:
            return success
        raise error
    seqs = (s1, s2)
    matched = ([first_match[0]], [first_match[1]])
    inds = [s + 1 for s in first_match]
    while all(ind < len(s) for ind, s in zip(inds, seqs)):
        # if we are over the missing limit, then stop
        diffs = [s[inds[k]] - s[matched[k][-1]] for k, s in enumerate(seqs)]
        # if we match within the tolerance limit, then move on
        # otherwise move the pointer for the less advanced sequence
        # print("tol",max(tol, rel_tol * min(diffs)))
        if abs(diffs[0] - diffs[1]) <= max(tol, rel_tol * min(diffs)):
            for k, m in enumerate(matched):
                m.append(inds[k])
                inds[k] += 1
        elif diffs[0] > diffs[1]:
            inds[1] += 1
        else:
            inds[0] += 1
        # if we are over the limit for matching, then abort
        if any(i - m[-1] > max_missing + 1 for i, m in zip(inds, matched)):
            last = [m[-1] for m in matched]
            vals = [s[i] for s, i in zip(seqs, last)]
            following = [np.array(s[i : i + 10]) - s[i] for s, i in zip(seqs, last)]
            before = [np.array(s[i - 10 : i]) - s[i] for s, i in zip(seqs, last)]

            msg = (
                f"Failed to match after indices {last} (values {vals}, {first_match=})\n"
            )
            msg += f"(follows:\n {following[0]}\n {following[1]}"
            msg += f"(before:\n {before[0]}\n {before[1]}"
            raise NoApproximateMatch(
                msg, ind=last[0], matches=tuple(np.array(m) for m in matched)
            )
    return tuple(np.array(m) for m in matched)  # type: ignore