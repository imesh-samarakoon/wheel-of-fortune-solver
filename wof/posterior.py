"""Letter probabilities from a posterior over whole answers.

This is the core idea of the project: we never ask a model "what letter goes
in slot 7?" directly. Language models think in tokens (``" THE"``, ``"ING"``),
not letters, so a per-slot letter distribution isn't something they expose.
Instead every candidate generator (the historical-puzzle lookup, an LLM, a
word list...) produces *whole candidate answers* with a log-weight, and
everything a player cares about falls out by marginalizing over them:

- ``slot_marginals()``          P(letter at each hidden slot)
- ``presence()``                P(letter appears anywhere in the puzzle)
- ``expected_count()``          E[# of times a letter appears] (money on the wheel)
- ``after_call(letter, ...)``   the posterior once a letter's been called
- ``entropy()``                 how uncertain we still are, in bits

The posterior is only over the candidates it was given (a truncated support),
so ``coverage`` matters: if the true answer was never generated, every number
here is confidently wrong. Evaluation tracks how often the truth is in the set.

An optional ``Unknown`` share says "the answer is none of these" explicitly (the
answer judge, ``wof.judge``, sets it when it rates every candidate as
nonsense). The candidates then share ``1 - unknown.mass`` between them, and
``presence()`` / ``expected_count()`` mix in the unknown answer's letters
(a stand-in, e.g. English letter frequencies). ``slot_marginals()`` and the
"what if I call this letter" methods (``outcomes``, ``after_call``,
``information_gain``, ``guess_right_after``) stay over the candidates only.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .board import HIDDEN, LETTERS, Board, normalize


@dataclass(frozen=True)
class Unknown:
    """The answer is none of the candidates: how likely that is, and what its letters look like."""

    mass: float                       # P(the answer is none of the candidates)
    presence: dict[str, float]        # P(letter in the hidden part | none of the candidates)
    expected_count: dict[str, float]  # E[# times in hidden slots | none of the candidates]


@dataclass(frozen=True)
class Candidate:
    answer: str
    logweight: float
    source: str = ""


class AnswerPosterior:
    def __init__(self, board: Board, candidates: list[Candidate], unknown: Unknown | None = None):
        self.board = board
        merged: dict[str, tuple[float, str]] = {}
        for c in candidates:
            ans = normalize(c.answer)
            if not board.consistent(ans):
                continue
            if ans in merged:
                # Same answer from two sources: keep the larger weight rather than
                # summing, so a source that lists duplicates can't inflate itself.
                prev_w, prev_s = merged[ans]
                if c.logweight > prev_w:
                    merged[ans] = (c.logweight, c.source or prev_s)
            else:
                merged[ans] = (c.logweight, c.source)
        self.answers = list(merged)
        self.sources = [merged[a][1] for a in self.answers]
        if self.answers:
            logw = [merged[a][0] for a in self.answers]
            m = max(logw)
            w = [math.exp(x - m) for x in logw]
            z = sum(w)
            self.probs = [x / z for x in w]
        else:
            self.probs = []
        # "None of these": with no candidates at all it is the whole posterior
        if unknown is not None and (unknown.mass > 0 or not self.answers):
            mass = min(max(unknown.mass, 0.0), 1.0) if self.answers else 1.0
            self.unknown: Unknown | None = Unknown(mass, unknown.presence, unknown.expected_count)
            self.probs = [p * (1 - mass) for p in self.probs]
        else:
            self.unknown = None

    @property
    def unknown_mass(self) -> float:
        """P(the answer is none of the candidates); 0 unless an ``Unknown`` share was given."""
        return self.unknown.mass if self.unknown else 0.0

    # --- basic views ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.answers)

    @property
    def empty(self) -> bool:
        return not self.answers

    def top(self, k: int = 10) -> list[tuple[str, float]]:
        ranked = sorted(zip(self.answers, self.probs), key=lambda t: -t[1])
        return ranked[:k]

    def prob_of(self, answer: str) -> float:
        answer = normalize(answer)
        for a, p in zip(self.answers, self.probs):
            if a == answer:
                return p
        return 0.0

    def entropy(self) -> float:
        """In bits; "none of these" counts as one more outcome."""
        ps = self.probs + ([self.unknown.mass] if self.unknown else [])
        return -sum(p * math.log2(p) for p in ps if p > 0)

    # --- letter marginals -----------------------------------------------------

    def slot_marginals(self) -> dict[int, dict[str, float]]:
        """{position: {letter: P}} for every hidden slot (candidates only: sums to 1 - unknown_mass)."""
        out: dict[int, dict[str, float]] = {i: defaultdict(float) for i in self.board.hidden_positions}
        for a, p in zip(self.answers, self.probs):
            for i in out:
                out[i][a[i]] += p
        return {i: dict(sorted(d.items(), key=lambda t: -t[1])) for i, d in out.items()}

    def presence(self, letters: str = LETTERS) -> dict[str, float]:
        """P(letter is somewhere in the hidden part of the puzzle), uncalled letters only."""
        out = {c: 0.0 for c in letters if c not in self.board.called}
        for a, p in zip(self.answers, self.probs):
            hidden = {a[i] for i in self.board.hidden_positions}
            for c in out:
                if c in hidden:
                    out[c] += p
        if self.unknown:
            for c in out:
                out[c] += self.unknown.mass * self.unknown.presence.get(c, 0.0)
        return dict(sorted(out.items(), key=lambda t: -t[1]))

    def expected_count(self, letters: str = LETTERS) -> dict[str, float]:
        out = {c: 0.0 for c in letters if c not in self.board.called}
        hidden = self.board.hidden_positions
        for a, p in zip(self.answers, self.probs):
            for i in hidden:
                if a[i] in out:
                    out[a[i]] += p
        if self.unknown:
            for c in out:
                out[c] += self.unknown.mass * self.unknown.expected_count.get(c, 0.0)
        return dict(sorted(out.items(), key=lambda t: -t[1]))

    # --- what calling a letter would do ---------------------------------------

    def _candidate_probs(self) -> list[float]:
        """The candidates' probabilities renormalized to sum to 1 (the unknown share left out)."""
        z = sum(self.probs)
        return [p / z for p in self.probs] if z > 0 else list(self.probs)

    def outcomes(self, letter: str) -> dict[tuple[int, ...], float]:
        """P(each possible reveal) if ``letter`` is called: positions it lands on -> probability.

        ``()`` is "not in the puzzle". This is exact over the candidate set (the unknown share, if
        any, is left out and the rest renormalized).
        """
        letter = letter.upper()
        out: dict[tuple[int, ...], float] = defaultdict(float)
        hidden = self.board.hidden_positions
        for a, p in zip(self.answers, self._candidate_probs()):
            out[tuple(i for i in hidden if a[i] == letter)] += p
        return dict(out)

    def after_call(self, letter: str, positions: tuple[int, ...] | list[int]) -> AnswerPosterior:
        """The candidates still consistent after the reveal (the unknown share is dropped)."""
        board = self.board.reveal_positions(letter, list(positions))
        cands = [Candidate(a, math.log(p), s) for a, p, s in zip(self.answers, self.probs, self.sources) if p > 0]
        return AnswerPosterior(board, cands)

    def expected_entropy_after(self, letter: str) -> float:
        """E[entropy of the posterior after calling ``letter``], in bits (over the candidates only)."""
        letter = letter.upper()
        groups: dict[tuple[int, ...], list[float]] = defaultdict(list)
        hidden = self.board.hidden_positions
        for a, p in zip(self.answers, self._candidate_probs()):
            groups[tuple(i for i in hidden if a[i] == letter)].append(p)
        total = 0.0
        for ps in groups.values():
            z = sum(ps)
            if z <= 0:
                continue
            total += z * -sum((q / z) * math.log2(q / z) for q in ps if q > 0)
        return total

    def information_gain(self, letters: str = LETTERS) -> dict[str, float]:
        """Expected bits learned per letter, over the candidates only."""
        h = -sum(p * math.log2(p) for p in self._candidate_probs() if p > 0)
        out = {c: h - self.expected_entropy_after(c) for c in letters if c not in self.board.called}
        return dict(sorted(out.items(), key=lambda t: -t[1]))

    def guess_right_after(self, letters: str = LETTERS) -> dict[str, float]:
        """P(the top answer is right once ``letter`` is called), per uncalled letter.

        Calling a letter splits the candidates by where (if anywhere) it shows up; whatever comes
        back, the solver would then guess the most likely answer left. So this is the sum, over the
        outcomes, of the best candidate's probability in each: a letter every candidate has in the
        same places (or none has) leaves it at P(top answer now), and a letter that separates the
        leading answers raises it. The "none of these" share can't be guessed, so it never counts.
        """
        letters_left = [c for c in letters if c not in self.board.called]
        hidden = self.board.hidden_positions
        out = {}
        for c in letters_left:
            best: dict[tuple[int, ...], float] = {}
            for a, p in zip(self.answers, self.probs):
                key = tuple(i for i in hidden if a[i] == c)
                best[key] = max(best.get(key, 0.0), p)
            out[c] = sum(best.values())
        return dict(sorted(out.items(), key=lambda t: -t[1]))

    # --- display ----------------------------------------------------------------

    def render_slots(self, k: int = 3) -> str:
        """The board with the top-``k`` letters (and probabilities) under each hidden slot."""
        marg = self.slot_marginals()
        lines = [self.board.template]
        for rank in range(k):
            row = []
            for i, c in enumerate(self.board.template):
                if c != HIDDEN:
                    row.append(" ")
                    continue
                ranked = list(marg.get(i, {}).items())
                row.append(ranked[rank][0].lower() if rank < len(ranked) else " ")
            lines.append("".join(row))
        return "\n".join(lines)
