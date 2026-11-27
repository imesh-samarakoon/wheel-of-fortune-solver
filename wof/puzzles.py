"""The puzzles: a bundled set to play, plus a separate pool shown to the model as examples.

Two groups, loaded from ``data/puzzles.json`` (see ``DATA_FILE``):

- ``Puzzles.play``      the puzzles you solve. ``play.py`` picks from these.
- ``Puzzles.examples``  worked examples put in the prompt (few-shot), and the word list
                        ``wof.slots`` fills single board words from.

**The two groups never share an answer.** That is the whole point of splitting them: an example
that gave away the answer would make the solver look far better than it is. ``load_puzzles``
checks it and raises if they overlap.

A ``Clue`` is everything the model is allowed to read: the category, any hint, the year and a
plain-English note on what an unusual category means. The answer itself only ever reaches the
model as an example from a *different* puzzle.

You can bring your own puzzles: write a JSON file in the same shape and pass it to
``load_puzzles(path)``, or set ``PUZZLES_FILE`` in ``config.py``.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .board import Board, normalize, unwritable

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "puzzles.json"


@dataclass(frozen=True)
class Clue:
    """Everything shown on the board besides the letters: what the model is allowed to read."""

    category: str
    hint: str = ""
    year: str = ""
    context: str = ""  # a note on what an unusual category means ("category_notes" in the JSON)


@dataclass(frozen=True)
class Puzzle:
    answer: str  # normalized: uppercase A-Z, punctuation kept as it appears on the board
    clue: Clue
    id: str = ""

    def board(self, called: str | set[str] = "") -> Board:
        """The board with ``called`` letters called (every occurrence revealed)."""
        return Board.from_answer(self.answer, set(called.upper()) if isinstance(called, str) else called)

    def __str__(self) -> str:
        return f"[{self.clue.category}] {self.answer}"


@dataclass
class Puzzles:
    play: list[Puzzle] = field(default_factory=list)
    examples: list[Puzzle] = field(default_factory=list)
    note: str = ""

    def __len__(self) -> int:
        return len(self.play)

    def get(self, which: str | int, seed: int = 0) -> Puzzle:
        """One puzzle to solve: its number in ``play`` (1-based), part of its answer, or "random"."""
        if which == "random":
            return random.Random(seed).choice(self.play)
        if isinstance(which, int) or str(which).strip().isdigit():
            n = int(which)
            if not 1 <= n <= len(self.play):
                raise IndexError(f"puzzle {n} doesn't exist: there are {len(self.play)} (1-{len(self.play)})")
            return self.play[n - 1]
        return self.find(str(which))

    def find(self, text: str) -> Puzzle:
        """The first puzzle whose answer contains ``text`` (case-insensitive)."""
        t = normalize(text)
        for p in self.play:
            if t in p.answer:
                return p
        where = " (it's in the example pool, so it can't be played: an example would give it away)" \
            if any(t in p.answer for p in self.examples) else ""
        raise KeyError(f"no puzzle matching {text!r}{where}")

    def categories(self) -> dict[str, int]:
        """{category: how many puzzles to play have it}, most common first."""
        counts: dict[str, int] = {}
        for p in self.play:
            counts[p.clue.category] = counts.get(p.clue.category, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _puzzle(row: dict, notes: dict[str, str], i: int, group: str) -> Puzzle:
    answer = normalize(row["answer"])
    if not answer:
        raise ValueError(f"{group}[{i}] has an empty answer")
    category = normalize(row["category"])
    return Puzzle(
        answer=answer,
        clue=Clue(category=category, hint=str(row.get("hint", "")).strip(),
                  year=str(row.get("year", "")).strip(),
                  context=str(row.get("context") or notes.get(category, "")).strip()),
        id=f"{group}-{i + 1}",
    )


def load_puzzles(path: str | Path | None = None) -> Puzzles:
    """Read a puzzle file (default ``data/puzzles.json``) into ``Puzzles``.

    The file is ``{"note": ..., "category_notes": {CATEGORY: note}, "examples": [...], "puzzles": [...]}``
    where each puzzle is ``{"answer": ..., "category": ..., "hint": ...}`` (hint optional). A
    category listed in ``category_notes`` gets that note as its clue's ``context``, so a note is
    written once rather than on every puzzle.
    """
    path = Path(path or DATA_FILE)
    if not path.exists():
        where = ("the bundled puzzle file is missing from this checkout" if path == DATA_FILE
                 else f"the repo's own is at {DATA_FILE}")
        raise FileNotFoundError(f"no puzzle file at {path}: {where}.")
    data = json.loads(path.read_text())
    notes = {normalize(k): v for k, v in (data.get("category_notes") or {}).items()}
    play = [_puzzle(r, notes, i, "puzzle") for i, r in enumerate(data.get("puzzles", []))]
    examples = [_puzzle(r, notes, i, "example") for i, r in enumerate(data.get("examples", []))]
    if not play:
        raise ValueError(f"{path} has no puzzles to play")
    overlap = {p.answer for p in play} & {p.answer for p in examples}
    if overlap:
        raise ValueError(f"{path}: {len(overlap)} answer(s) are in both groups, so an example would give a "
                         f"puzzle away: {', '.join(sorted(overlap)[:5])}")
    return Puzzles(play=play, examples=examples, note=str(data.get("note", "")))


def pick_examples(pool: list[Puzzle], target: Puzzle, n: int = 5, same_category: bool = True,
                  seed: int = 0) -> list[Puzzle]:
    """``n`` solved puzzles from ``pool`` to show the model before ``target`` (few-shot examples).

    ``same_category``: fill with the target's own category first, then any category, so the model
    sees how this kind of category is answered. Deterministic for a given ``seed`` and target, so
    re-running a cell shows the same prompt. ``pool`` is the example pool, which never contains a
    playable answer (``load_puzzles`` checks), so an example can't leak the answer.
    """
    rng = random.Random(f"{seed}|{target.answer}")
    pool = [p for p in pool if p.answer != target.answer]
    same = [p for p in pool if p.clue.category == target.clue.category] if same_category else []
    picked = rng.sample(same, min(n, len(same)))
    rest = [p for p in pool if p not in picked]
    picked += rng.sample(rest, min(n - len(picked), len(rest)))
    rng.shuffle(picked)
    return picked


def from_text(answer: str, category: str, hint: str = "", context: str = "", year: str = "") -> Puzzle:
    """A puzzle typed in by hand (``play.py --answer ... --category ...``)."""
    answer = normalize(answer)
    if not answer:
        raise ValueError("an answer is needed")
    if not any(c.isalpha() for c in answer):
        raise ValueError(f"{answer!r} has no letters to guess")
    bad = unwritable(answer)
    if bad:
        raise ValueError(f"{answer!r} contains {', '.join(repr(c) for c in bad)}, which the search can't "
                         f"write: a board position holding one of those can never be filled. Letters, "
                         f"digits, spaces and ' - & . , ! ? : \" / are fine.")
    return Puzzle(answer=answer, clue=Clue(normalize(category) or "PHRASE", hint.strip(), year.strip(),
                                           context.strip()), id="yours")
