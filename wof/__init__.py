"""A Wheel of Fortune solver: a probability for every letter, and which one to call.

The short version of how it works. A language model predicts *tokens* (" Burt", "aster"), never
single letters, so we never ask it "what letter is in slot 7?". Instead we make it write whole
answers that fit the board, keep the model's own probability for each, and marginalize:

    P(letter is in the puzzle) = total probability of the candidate answers that contain it

Modules, roughly in the order the solver uses them:

    board.py       the board: what's showing, what's been called, what still fits
    search.py      the model, searched token by token under the board's constraint (the core)
    posterior.py   candidate answers -> per-slot and per-letter probabilities
    judge.py       a second model's opinion: is this candidate real English or board-fitting junk?
    slots.py       filling one open board word from a word list (what the left-to-right search misses)
    policy.py      which letter to call: most likely / most money / most information / most decisive
    calling.py     play a whole puzzle, one letter at a time, re-running the model on each new board
    adjust.py      how much of the model's probability the search actually found, and scoring
    puzzles.py     the bundled puzzles and the separate pool used as prompt examples
    show.py        the printouts every step of the way

Start with ``explore.py`` (a walkthrough of one puzzle, step by step) or ``play.py`` (play a
puzzle and watch the solver work). Settings for both live in ``config.py``.
"""

from __future__ import annotations

from .board import Board, normalize
from .posterior import AnswerPosterior, Candidate
from .puzzles import Clue, Puzzle, load_puzzles

__all__ = ["AnswerPosterior", "Board", "Candidate", "Clue", "Puzzle", "load_puzzles", "normalize"]
