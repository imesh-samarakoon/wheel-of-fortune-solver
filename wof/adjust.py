"""What the beams missed, and how to score letter probabilities against the truth.

Beam search keeps a few dozen answers that together hold only a small share of the model's
probability. Normalizing over just those treats every other answer as impossible, so whatever
the top few answers agree on goes to ~100% and whatever they lack goes to ~0%. On an empty board
that is badly overconfident: 64 beams held about **5%** of the model's board-fitting probability
in a real measurement.

``hybrid_posterior`` is the fix that needs no extra data. Split the model's board-fitting answers
into the ones the beams found (set B) and everything else. For B we know each answer's probability
exactly. For everything else we use the sampled draws that landed outside B, each weighted by
P(model writes it) / P(the sampler drew it), averaged over all draws. That average is an unbiased
estimate of the probability outside B (and of how it splits across answers):

    weight(answer in B)        = P(answer | prompt)                       exact
    weight(sampled, not in B)  = (sum of its draws' importance weights) / number of draws
    P(letter in puzzle)        = sum of weights of answers containing it / sum of all weights

It is only as good as the samples: with few effective draws the outside-B part is noisy, which is
why a game's default is beams only (sampling costs about one beam per draw) and the answer judge
(``wof.judge``) carries the "the answer may be none of these" job instead.

Neither function uses the true answer to build anything. ``letter_scores`` compares a finished set
of letter probabilities against it afterwards.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .board import Board
from .posterior import AnswerPosterior, Candidate
from .search import SearchResult


@dataclass
class HybridInfo:
    found_mass: float       # total P(answer | prompt) of the answers the beams found (exact)
    missed_mass: float      # estimated P of every other board-fitting answer (from the samples)
    n_draws: int            # sampling draws attempted
    n_outside: int          # draws that landed on an answer the beams didn't find
    n_outside_answers: int  # distinct such answers
    outside_ess: float      # effective sample size of those draws (small = noisy estimate)

    @property
    def coverage(self) -> float:
        """Estimated share of the model's board-fitting probability the beams found."""
        total = self.found_mass + self.missed_mass
        return self.found_mass / total if total > 0 else 1.0


def hybrid_posterior(beam: SearchResult, sample: SearchResult) -> tuple[AnswerPosterior, HybridInfo]:
    """Beam answers at their exact probability, plus sampled answers the beams missed (estimated).

    Candidates carry ``source`` "beam" or "sample" so a printout can show which part is which.
    One small leak: an answer counts as "found" by whichever spellings the beams found, so a
    sampled draw of the same answer via a different tokenization is not added again.
    """
    if beam.method != "beam" or sample.method != "sample":
        raise ValueError("hybrid_posterior(beam_result, sample_result)")
    found = beam.answer_logweights()
    n = sample.n_draws or len(sample.paths)
    outside: dict[str, list[float]] = defaultdict(list)
    for p in sample.paths:
        if p.answer not in found:
            outside[p.answer].append(p.log_weight)
    log_n = math.log(n) if n else 0.0
    missed = {a: _logsumexp(ws) - log_n for a, ws in outside.items()}
    cands = [Candidate(a, w, "beam") for a, w in found.items()]
    cands += [Candidate(a, w, "sample") for a, w in missed.items()]
    all_lw = [w for ws in outside.values() for w in ws]
    ess = math.exp(2 * _logsumexp(all_lw) - _logsumexp([2 * w for w in all_lw])) if all_lw else 0.0
    info = HybridInfo(
        found_mass=sum(math.exp(w) for w in found.values()),
        missed_mass=sum(math.exp(w) for w in missed.values()),
        n_draws=n,
        n_outside=len(all_lw),
        n_outside_answers=len(outside),
        outside_ess=ess,
    )
    return AnswerPosterior(beam.board, cands), info


def letter_scores(presence: dict[str, float], board: Board, truth: str) -> dict[str, float]:
    """Brier score and log loss of P(letter in puzzle) against the true answer (lower = better).

    Only the letters in ``presence`` (i.e. the uncalled ones) count, and "in the puzzle" means in
    a slot that is still hidden. Brier 0 = perfect, 0.25 = saying 50% about everything.
    """
    hidden = {truth[i] for i in board.hidden_positions}
    eps = 1e-6
    n = len(presence) or 1
    brier = sum((p - (c in hidden)) ** 2 for c, p in presence.items()) / n
    logloss = -sum(math.log(max(eps, p if c in hidden else 1 - p)) for c, p in presence.items()) / n
    return {"brier": brier, "logloss": logloss}


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))
