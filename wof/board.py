"""The puzzle board: what's shown, what's been called, what's still hidden.

Everything in this package works on *normalized* answers: uppercase A-Z, with
every other character (spaces, apostrophes, hyphens, ampersands, ...) treated
as punctuation that is always shown on the board, exactly like the show.

A ``Board`` never knows the answer. It holds:

- ``template``: the answer with every letter replaced by ``_`` unless revealed,
  punctuation kept as-is (e.g. ``"_HE _A__ ___'_"``)
- ``called``: every letter called so far, present or not

``Board.from_answer`` builds one from a known answer (for simulation/eval);
``Board.consistent(candidate)`` is the one test every candidate generator and
the posterior rely on.
"""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass, field

LETTERS = string.ascii_uppercase
VOWELS = frozenset("AEIOU")
CONSONANTS = frozenset(LETTERS) - VOWELS  # Y counts as a consonant on the show
FREQUENCY_ORDER = "ETAOINSRHLDCUMFPGWYBVKXJQZ"  # English letter frequency, most common first
HIDDEN = "_"
# Everything a board may hold besides letters: digits, space, and every printable ASCII
# punctuation mark except "_", which is reserved for a hidden slot. These are always shown and
# never guessed. They also bound what the search may write (``wof.search.ANSWER_CHARS``): a
# character outside this set would leave a board position that no token at all can fill, which is
# why the set is deliberately generous rather than a list of the marks puzzles usually use.
BOARD_PUNCTUATION = " " + string.digits + "".join(c for c in string.punctuation if c != HIDDEN)


def normalize(text: str) -> str:
    """Uppercase, strip accents, collapse whitespace. Punctuation is kept."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.upper()).strip()


def is_letter(c: str) -> bool:
    return c in LETTERS


def unwritable(answer: str) -> list[str]:
    """Characters of ``answer`` that can't be on a board, so no search could ever write them.

    Anything outside ``LETTERS`` and ``BOARD_PUNCTUATION``, which in practice means ``_`` (reserved
    for a hidden slot) and non-ASCII that survives ``normalize`` (CJK, emoji, box drawing).
    """
    ok = set(LETTERS) | set(BOARD_PUNCTUATION)
    return sorted({c for c in normalize(answer) if c not in ok})


@dataclass(frozen=True)
class Board:
    template: str
    called: frozenset[str] = field(default_factory=frozenset)

    # --- construction ---------------------------------------------------------

    @classmethod
    def from_answer(cls, answer: str, called: set[str] | frozenset[str] = frozenset()) -> Board:
        answer = normalize(answer)
        called = frozenset(c.upper() for c in called)
        template = "".join(c if (not is_letter(c) or c in called) else HIDDEN for c in answer)
        return cls(template=template, called=called)

    # --- derived views --------------------------------------------------------

    @property
    def hidden_positions(self) -> list[int]:
        return [i for i, c in enumerate(self.template) if c == HIDDEN]

    @property
    def n_hidden(self) -> int:
        return self.template.count(HIDDEN)

    @property
    def solved(self) -> bool:
        return self.n_hidden == 0

    @property
    def absent(self) -> frozenset[str]:
        """Letters called that turned out not to be in the puzzle."""
        return frozenset(c for c in self.called if c not in self.template)

    @property
    def words(self) -> list[str]:
        return self.template.split(" ")

    @property
    def uncalled(self) -> list[str]:
        return [c for c in LETTERS if c not in self.called]

    def regex(self) -> re.Pattern[str]:
        """A regex matching exactly the normalized answers consistent with this board.

        A hidden slot can be any letter that has NOT been called: called letters
        that are in the puzzle were all revealed (the show reveals every
        occurrence), and called letters that aren't present can't be there.
        """
        allowed = "".join(c for c in LETTERS if c not in self.called)
        slot = f"[{allowed}]" if allowed else "(?!)"  # every letter called: a hidden slot can't exist
        parts = [slot if c == HIDDEN else re.escape(c) for c in self.template]
        return re.compile("".join(parts))

    def consistent(self, candidate: str) -> bool:
        candidate = normalize(candidate)
        return len(candidate) == len(self.template) and self.regex().fullmatch(candidate) is not None

    # --- updates ----------------------------------------------------------------

    def reveal(self, letter: str, answer: str) -> Board:
        """Call ``letter`` against a known answer (simulation)."""
        return Board.from_answer(answer, self.called | {letter.upper()})

    def reveal_positions(self, letter: str, positions: list[int]) -> Board:
        """Call ``letter`` live: the host showed it at ``positions`` (empty = not in puzzle)."""
        letter = letter.upper()
        chars = list(self.template)
        for p in positions:
            if chars[p] != HIDDEN:
                raise ValueError(f"position {p} is already showing {chars[p]!r}")
            chars[p] = letter
        return Board(template="".join(chars), called=self.called | {letter})

    def __str__(self) -> str:
        return self.template
