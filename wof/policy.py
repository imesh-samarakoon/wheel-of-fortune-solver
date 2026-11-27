"""Which letter to call next, given a posterior over answers.

Each policy returns a ranked {letter: score}. They answer different questions:

- ``most_likely``    maximize P(letter is in the puzzle) — avoid losing your turn
- ``most_money``     maximize E[# occurrences] — the wheel pays per occurrence
- ``most_info``      maximize expected information gain — get to the solve fastest
- ``most_decisive``  maximize P(the top answer is right after this call) — split the leading answers
- ``likely_then_decisive``  ``most_likely`` until a few answers hold most of the probability
                     and a quarter of the board is showing, then ``most_decisive``

``most_likely`` is the right play on the show: a letter that isn't there hands the wheel to an
opponent. If the only goal is to reach the answer in as few turns as possible, a miss costs no
more than a hit, and a letter every leading answer has in the same places teaches nothing about
which one it is. Late in a game that is exactly the trap: with "STRAWBERRY & PAPER FAN" and
"... PAPER JAM" in front, ``most_likely`` called B, Y, W and S (all certain, all in the same
places in every candidate) before a letter that could tell FAN from JAM. ``most_decisive`` calls
the letter that separates them. Early on, when the candidates may not include the answer at all,
revealing sure letters still helps the model find it, hence the switching policy.

Vowels cost $250 on the show and consonants are what the wheel pays for, so
``kind`` restricts a policy to one or the other.
"""

from __future__ import annotations

from collections.abc import Callable

from .board import CONSONANTS, LETTERS, VOWELS
from .posterior import AnswerPosterior

Policy = Callable[[AnswerPosterior, str], dict[str, float]]


def _letters(kind: str) -> str:
    return {"any": LETTERS, "consonant": "".join(sorted(CONSONANTS)), "vowel": "".join(sorted(VOWELS))}[kind]


def most_likely(post: AnswerPosterior, kind: str = "any") -> dict[str, float]:
    return post.presence(_letters(kind))


def most_money(post: AnswerPosterior, kind: str = "any") -> dict[str, float]:
    return post.expected_count(_letters(kind))


def most_info(post: AnswerPosterior, kind: str = "any") -> dict[str, float]:
    return post.information_gain(_letters(kind))


def most_decisive(post: AnswerPosterior, kind: str = "any") -> dict[str, float]:
    """P(the most likely answer is right after calling the letter): ``AnswerPosterior.guess_right_after``.

    A one-step lookahead on the thing a fast solver cares about, rather than entropy: it rewards
    splitting the leading answers and ignores how the long tail gets shuffled. Letters that can't
    change the top answer all tie at P(top answer now); ``calling.choose_letter`` then breaks the
    tie by expected occurrences, so with nothing to separate it calls a letter that's there.
    """
    return post.guess_right_after(_letters(kind))


def top_share(post: AnswerPosterior, k: int = 4) -> float:
    """How much probability the ``k`` most likely answers hold together ("none of these" counts against it)."""
    return sum(p for _, p in post.top(k))


def revealed_share(board) -> float:
    """Share of the board's letter slots already showing (0 = empty board, 1 = solved)."""
    shown = sum(1 for c in board.template if c in LETTERS)
    total = shown + board.n_hidden
    return shown / total if total else 1.0


def switch_when_close(early: Policy = most_likely, late: Policy = most_decisive, *, top: int = 4,
                      share: float = 0.5, min_revealed: float = 0.25) -> Policy:
    """``early`` until the ``top`` most likely answers hold at least ``share`` of the probability AND at
    least ``min_revealed`` of the board's letters are showing, then ``late``.

    The share counts "none of these" against the candidates, so after a wrong guess that leaves the
    answer looking like it may be missing, it goes back to revealing sure letters (which gives the
    model more of the board to search with) rather than splitting answers that are probably all wrong.

    ``min_revealed``: on a near-empty board the candidates' concentration can't be trusted (64 beams
    hold ~5% of the model's board-fitting probability there, and the judge's "none of these" share is
    small), so a confident-looking top 4 doesn't mean the answer is among them. In "THE SPEED OF
    LIGHT" the top 4 held ~70% on the empty board, so it switched at once and called A (it split
    BOARD from THING answers) and missed; E, which ``likely`` would have called, hit 3x and the next
    search found the answer. 0 = switch on concentration alone.
    """

    def policy(post: AnswerPosterior, kind: str = "any") -> dict[str, float]:
        close = top_share(post, top) >= share and revealed_share(post.board) >= min_revealed
        return (late if close else early)(post, kind)

    policy.__name__ = f"{getattr(early, '__name__', 'early')}_then_{getattr(late, '__name__', 'late')}"
    return policy


likely_then_decisive = switch_when_close(most_likely, most_decisive)

POLICIES: dict[str, Policy] = {"likely": most_likely, "money": most_money, "info": most_info,
                               "decisive": most_decisive, "likely_then_decisive": likely_then_decisive}


def best(post: AnswerPosterior, policy: Policy = most_info, kind: str = "any") -> str | None:
    scores = policy(post, kind)
    return next(iter(scores), None)
