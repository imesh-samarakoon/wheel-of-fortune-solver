"""Filling one board word at a time: per-word confidence and the words that could go in a word.

The beam search judges a partial answer by its log P so far, so an answer whose payoff comes late
gets pruned before the payoff. "A WISH COME TRUE" is the textbook case: once "COME TRUE" shows up
in every leading answer, "A Wish" is still an unlikely *start* (" A W" ranked ~17th), so the search
writes "A Fish Come True" / "A City Come True" instead, even though the model rates the whole of
"A Wish Come True" ~80x higher than either. ``LLMSolver.recombine`` can only swap in words some
partial answer wrote, and here none ever wrote "Wish".

So: when the rest of the phrase is settled, put every word from a word list that fits one open
board word into it and score each whole answer (``LLMSolver.fill_slots``). The late payoff counts,
and the candidates come from the word list, not from the search's left-to-right ranking.

This module is the model-free half:

- ``word_marginals``  P(board word i is w), summed over the candidate answers
- ``open_slots``      which board words to fill: every other word that still has a hidden letter
                      is confident (its most likely word >= ``min_confidence``)
- ``build_vocab``     a word list (with counts) from past puzzles' answers
- ``slot_words``      the words that fit board word i, most common first
- ``match_case``      write a word the way the word it replaces is written ("Wish" for "Fish")
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping

from .board import HIDDEN, LETTERS, Board, normalize


def board_words(board: Board) -> list[str]:
    """The board's words, as shown (``"A ____ _O_E TR_E"`` -> ``["A", "____", "_O_E", "TR_E"]``)."""
    return board.template.split(" ")


def hidden_words(board: Board) -> list[int]:
    """Indices of the board words that still have a hidden letter."""
    return [i for i, w in enumerate(board_words(board)) if HIDDEN in w]


def word_marginals(probs: Mapping[str, float], board: Board) -> list[dict[str, float]]:
    """For each board word, P(it is w) over the candidate answers, most likely first.

    ``probs``: answer -> probability or any non-negative weight (normalized here). Answers that
    don't fit the board are ignored. Only the candidates count, so a word in every candidate is at
    100% even if the candidates hold a sliver of the model's probability: it says where the found
    answers agree, not how likely the word truly is. That's what ``open_slots`` needs.
    """
    n = len(board_words(board))
    out: list[dict[str, float]] = [{} for _ in range(n)]
    fitting = [(normalize(a), p) for a, p in probs.items() if p > 0]
    fitting = [(a, p) for a, p in fitting if board.consistent(a)]
    z = sum(p for _, p in fitting)
    if not z:
        return out
    for a, p in fitting:
        for i, w in enumerate(a.split(" ")):
            out[i][w] = out[i].get(w, 0.0) + p / z
    return [dict(sorted(m.items(), key=lambda kv: -kv[1])) for m in out]


def top_word(marginal: Mapping[str, float]) -> tuple[str, float]:
    """The most likely word in one board word's marginal (``("", 0.0)`` if empty)."""
    return max(marginal.items(), key=lambda kv: kv[1], default=("", 0.0))


def open_slots(marginals: list[dict[str, float]], board: Board,
               min_confidence: float = 0.8) -> list[tuple[int, float]]:
    """The board words worth filling: (index, how sure we are of the rest of the phrase).

    Board word i qualifies when it has a hidden letter and every OTHER word with a hidden letter
    has a most-likely word at ``min_confidence`` or more (fully revealed words don't count: they're
    certain). The confidence returned is the weakest of those other words (1.0 if there are none).
    A word can be confident itself and still be filled: in "A FISH COME TRUE" at 99% the model is
    sure of FISH, and the point is to find out whether something else fits better.
    """
    hidden = hidden_words(board)
    out = []
    for i in hidden:
        others = [top_word(marginals[j])[1] for j in hidden if j != i]
        conf = min(others, default=1.0)
        if conf >= min_confidence:
            out.append((i, conf))
    return out


def build_vocab(answers: Iterable) -> Counter:
    """Every word in ``answers`` with how many answers it appears in (the ``slot_words`` ranking).

    ``answers``: strings or anything with an ``.answer`` (e.g. ``Puzzle``s). Build it from the
    example pool only (never the test puzzles), so the word list can't give a test answer away.
    """
    vocab: Counter = Counter()
    for a in answers:
        text = normalize(getattr(a, "answer", a))
        vocab.update(set(w for w in text.split(" ") if any(c in LETTERS for c in w)))
    return vocab


def slot_regex(board: Board, i: int) -> re.Pattern[str]:
    """A regex for the words that fit board word ``i`` (hidden = any uncalled letter)."""
    allowed = "".join(c for c in LETTERS if c not in board.called)
    slot = f"[{allowed}]" if allowed else "(?!)"
    return re.compile("".join(slot if c == HIDDEN else re.escape(c) for c in board_words(board)[i]))


def slot_words(board: Board, i: int, vocab: Mapping[str, int], limit: int | None = None) -> list[str]:
    """The words of ``vocab`` that fit board word ``i``, most common first (ties alphabetical)."""
    pattern = slot_regex(board, i)
    fits = [w for w in vocab if pattern.fullmatch(normalize(w))]
    fits.sort(key=lambda w: (-vocab[w], w))
    return fits[:limit] if limit is not None else fits


def match_case(word: str, like: str) -> str:
    """``word`` written in ``like``'s case: "WISH" like "Fish" -> "Wish", like "FISH" -> "WISH"."""
    letters = [c for c in like if c.isalpha()]
    if not letters:
        return word
    if all(c.isupper() for c in letters) and len(letters) > 1:
        return word.upper()
    if all(c.islower() for c in letters):
        return word.lower()
    if len(letters) == 1:  # "A" in title case: can't tell, title case it
        return word[:1].upper() + word[1:].lower()
    return "-".join(part[:1].upper() + part[1:].lower() for part in word.split("-"))
