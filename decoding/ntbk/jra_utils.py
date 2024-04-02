

import itertools

import typing as tp


import numpy as np

import dataclasses

class NoApproximateMatch(ValueError):
    def __init__(self, msg: str, ind: int, matches: tp.Any) -> None:
        super().__init__(msg)
        self.ind = ind
        self.matches = matches


@dataclasses.dataclass
class Tolerance:
    """Convenience tool for check if a value is  within tolerance"""

    abs_tol: float
    rel_tol: float

    def __call__(self, value1: float, value2) -> bool:
        diff = abs(value1 - value2)
        tol = max(self.abs_tol, self.rel_tol * min(abs(value1), abs(value2)))
        return diff <= tol

    def absolute(self, value1: float, value2: float) -> bool:
        diff = abs(value1 - value2)
        return diff <= self.abs_tol


@dataclasses.dataclass
class IndexDiff:
    index: int
    value: float


@dataclasses.dataclass
class Sequence:
    sequence: tp.List[float]
    current: int
    matches: tp.List[int]

    def diff(self, shift: int = 0) -> float:
        return self.sequence[self.current + shift] - self.last_value

    @property
    def last_value(self) -> float:
        return self.sequence[self.matches[-1]]

    def diff_to(self, ind: int) -> np.array:
        r = self.matches[-1]
        sub = self.sequence[r : r + ind] if ind > 0 else self.sequence[r + ind : r]
        return np.array(sub) - self.last_value


def approx_match_samples(
    s1: tp.List[int],
    s2: tp.List[int],
    abs_tol: float,
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
                    s1, s2, abs_tol=abs_tol, rel_tol=rel_tol, max_missing=max_missing, first_match=offsets  # type: ignore
                )
                if success is None or len(out[0]) > len(success[0]):  # type: ignore
                    success = out
            except NoApproximateMatch as e:
                if error is None or error.ind < e.ind:
                    error = e
        if success is not None:
            return success
        raise error
    tolerance = Tolerance(abs_tol=abs_tol, rel_tol=rel_tol)
    seqs = (
        Sequence(s1, first_match[0] + 1, [first_match[0]]),
        Sequence(s2, first_match[1] + 1, [first_match[1]]),
    )
    while all(s.current < len(s.sequence) for s in seqs):
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
            if seq.current + 1 < len(seq.sequence) and other.current < len(
                other.sequence
            ):
                # if tolerance.absolute(seq.diff(1), seq.diff()):
                if abs(seq.diff(1) - seq.diff()) < 2 * abs_tol:
                    if abs(seq.diff(1) - other.diff()) < abs(seq.diff() - other.diff()):
                        seq.current += 1
        # if we are over the limit for matching, then abort
        if any(m.current - m.matches[-1] > max_missing + 1 for m in seqs):
            msg = f"Failed to match after indices {[s.matches[-1] for s in seqs]} "
            msg += "(values {[s.last_value for s in seqs]}, {first_match=})\n"
            msg += f"(follows:\n {seqs[0].diff_to(10)}\n {seqs[1].diff_to(10)}"
            msg += f"(before:\n {seqs[0].diff_to(-10)}\n {seqs[1].diff_to(-10)}"
            out = tuple(np.array(s.matches) for s in seqs)
            raise NoApproximateMatch(msg, ind=seqs[0].last_value, matches=out)
    return tuple(np.array(s.matches) for s in seqs)