"""Call letters one at a time and watch the solver change its mind.

A game is a loop over turns. Each turn:

1. **Look.** Get a posterior over whole answers for the current board. With the LLM
   (``LLMTurnSolver``) that means running the model again: the prompt now shows the revealed
   letters, and the search only lets hidden slots take letters that haven't been called. So the
   model's token predictions really are recomputed from the new board, not just filtered.
2. **Record.** The most likely answer and the probability the solver gives it, where the true
   answer sits (probability and rank), P(letter in puzzle) for every uncalled letter, and how
   good those letter probabilities were against the truth (Brier, log loss).
3. **Call.** The policy's best uncalled letter (``wof.policy``: most likely to be in the puzzle by
   default). The host reveals every occurrence; we record how many there were.

It repeats until the board is full. The last turn sees a full board, so the solver is 100% sure
there and the final turn needs no model call.

**Solving.** With ``solve_at``, a turn can solve instead of calling a letter: once the solver's
confidence signal (``solve_signal``) is at or below ``solve_at``, it guesses its most likely answer
and calls no letter that turn. Right = it wins and the game stops. Wrong = that answer is ruled
out for the rest of the game (dropped from every later posterior, so it can never be the top
answer again) and the turn ends. The board hasn't changed, so the next turn reuses this turn's
posterior minus the wrong answer instead of running the model again on the same board; it may
guess again (its new top answer) or call a letter. Turn numbers therefore count letters called
plus wrong guesses; ``Turn.letters_before`` is the number of letters called before a turn.

``Game`` then answers the questions this was built for: when did the top answer become (and stay)
right, when did the solver first pass 50% / 90% / 99% on one answer and was it right then, and how
often did the letters it called actually turn up.

The loop itself knows nothing about models: anything that maps a ``Board`` to an
``AnswerPosterior`` works (tests use a fixed list of candidates). It may also return
``(posterior, info)``, where ``info`` is a dict stored on the turn (timings, coverage, ...).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .adjust import hybrid_posterior, letter_scores
from .board import CONSONANTS, FREQUENCY_ORDER, HIDDEN, LETTERS, VOWELS, Board, normalize
from .policy import Policy, most_likely
from .posterior import AnswerPosterior, Candidate, Unknown

CONFIDENCE_LEVELS = (0.5, 0.9, 0.99)
SOLVE_SIGNALS = ("expected_brier", "brier")

PosteriorFn = Callable[[Board], "AnswerPosterior | tuple[AnswerPosterior, dict]"]


@dataclass
class Turn:
    """What the solver believed at one point in the game, and the letter it called next."""

    number: int                       # 0 = the starting board, before this game called anything
    board: Board                      # the board the solver looked at
    n_candidates: int
    top: list[tuple[str, float]]      # the most likely answers, best first
    p_truth: float                    # probability given to the true answer (0 if not a candidate)
    truth_rank: int | None            # 1 = the top answer; None = not among the candidates
    entropy_bits: float
    presence: dict[str, float]        # P(letter in the hidden part of the puzzle), uncalled letters
    expected_count: dict[str, float]  # E[# times it appears in hidden slots]
    scores: dict[str, float]          # Brier / log loss of ``presence`` against the truth
    info: dict = field(default_factory=dict)
    # The call made after this turn (None on the last turn)
    called: str | None = None
    p_called: float = math.nan        # the P(in puzzle) the solver gave the letter it called
    expected_called: float = math.nan  # and how many times it expected it
    count: int = 0                    # how many times it actually appeared
    signal: float = math.nan          # the solve signal this turn (see ``solve_signal``); lower = surer
    guess: str | None = None          # the answer it tried to solve with this turn (then no letter)
    guess_correct: bool | None = None
    letters_before: int = 0           # letters this game had called before this turn
    p_unknown: float = 0.0            # P(the answer is none of the candidates); 0 without an answer judge
    held_back: bool = False           # the solve signal was low enough, but P(top answer) < min_top

    @property
    def top_answer(self) -> str | None:
        return self.top[0][0] if self.top else None

    @property
    def p_top(self) -> float:
        return self.top[0][1] if self.top else 0.0

    @property
    def top_correct(self) -> bool:
        return self.truth_rank == 1

    @property
    def hit(self) -> bool:
        return self.count > 0

    @property
    def n_hidden(self) -> int:
        return self.board.n_hidden


@dataclass
class Milestone:
    level: float          # e.g. 0.9
    turn: int | None      # first turn the top answer reached this probability (None = never)
    letters_called: int   # letters this game had called before that turn
    hidden_left: int      # hidden letters still on the board then
    answer: str | None    # the answer it was that sure of
    correct: bool


@dataclass
class Game:
    answer: str
    turns: list[Turn]
    stopped: str  # "guessed" (solved by guessing), "solved" (board full), "confident" (stop_at), "max letters"

    @property
    def guesses(self) -> list[Turn]:
        """The turns where it tried to solve, in order."""
        return [t for t in self.turns if t.guess]

    @property
    def wrong_guesses(self) -> list[str]:
        return [t.guess for t in self.guesses if not t.guess_correct]

    @property
    def won_by_guess(self) -> Turn | None:
        """The turn it solved the puzzle by guessing, if it did."""
        return next((t for t in self.turns if t.guess_correct), None)

    @property
    def calls(self) -> list[Turn]:
        """The turns that ended in a call, in order."""
        return [t for t in self.turns if t.called]

    @property
    def letters_called(self) -> str:
        return "".join(t.called for t in self.calls)

    @property
    def hits(self) -> int:
        return sum(t.hit for t in self.calls)

    @property
    def misses(self) -> int:
        return sum(not t.hit for t in self.calls)

    def right_from(self) -> int | None:
        """The first turn from which the top answer was the truth on every later turn too."""
        first = None
        for t in self.turns:
            if t.top_correct:
                first = t.number if first is None else first
            else:
                first = None
        return first

    def first_found(self) -> int | None:
        """The first turn the truth was among the candidates at all."""
        return next((t.number for t in self.turns if t.truth_rank is not None), None)

    def milestones(self, levels: tuple[float, ...] = CONFIDENCE_LEVELS) -> list[Milestone]:
        """For each level, the first turn the solver put at least that much on one answer."""
        out = []
        for level in levels:
            t = next((t for t in self.turns if t.p_top >= level), None)
            if t is None:
                out.append(Milestone(level, None, len(self.calls), 0, None, False))
            else:
                out.append(Milestone(level, t.number, t.letters_before, t.n_hidden, t.top_answer, t.top_correct))
        return out

    def wrong_and_sure(self, level: float = 0.9) -> list[Turn]:
        """Turns where the solver put at least ``level`` on an answer that was wrong."""
        return [t for t in self.turns if t.p_top >= level and not t.top_correct]

    def call_calibration(self) -> tuple[float, float]:
        """(mean P(in puzzle) of the letters it called, share of them that were in the puzzle)."""
        calls = [t for t in self.calls if not math.isnan(t.p_called)]
        if not calls:
            return math.nan, math.nan
        return sum(t.p_called for t in calls) / len(calls), sum(t.hit for t in calls) / len(calls)


# --- choosing a letter ---------------------------------------------------------------------------

def _letters(kind: str) -> str:
    return {"any": LETTERS, "consonant": "".join(sorted(CONSONANTS)), "vowel": "".join(sorted(VOWELS))}[kind]


def choose_letter(post: AnswerPosterior, policy: Policy = most_likely, kind: str = "any") -> str | None:
    """The policy's best uncalled letter.

    Ties (common: several letters every candidate shares, all at 100%) go to the letter expected
    to appear more often, then to the more common letter in English. If ``kind`` has no uncalled
    letters left, any letter will do. With no candidates at all, English frequency order decides.
    """
    board = post.board
    if not any(c not in board.called for c in _letters(kind)):
        kind = "any"
    pool = [c for c in _letters(kind) if c not in board.called]
    if not pool:
        return None
    freq = {c: k for k, c in enumerate(FREQUENCY_ORDER)}
    if post.empty:
        return min(pool, key=freq.__getitem__)
    scores = policy(post, kind)
    counts = post.expected_count(_letters(kind))
    return min(pool, key=lambda c: (-round(scores.get(c, 0.0), 9), -round(counts.get(c, 0.0), 9), freq[c]))


# --- the game ------------------------------------------------------------------------------------

def _look(posterior_fn: PosteriorFn, board: Board) -> tuple[AnswerPosterior, dict]:
    if board.solved:  # nothing hidden: the board is the answer, no model needed
        return AnswerPosterior(board, [Candidate(board.template, 0.0, "board")]), {"skipped": "board full"}
    out = posterior_fn(board)
    return out if isinstance(out, tuple) else (out, {})


def without(post: AnswerPosterior, ruled_out: set[str]) -> AnswerPosterior:
    """The posterior with ``ruled_out`` answers removed and the rest renormalized.

    A "none of these" share (``post.unknown``) is renormalized along with the candidates left.
    """
    if not ruled_out or not any(a in ruled_out for a in post.answers):
        return post
    keep = [(a, p, s) for a, p, s in zip(post.answers, post.probs, post.sources) if p > 0 and a not in ruled_out]
    unknown = post.unknown
    if unknown is not None:
        total = unknown.mass + sum(p for _, p, _ in keep)
        unknown = Unknown(unknown.mass / total if total > 0 else 1.0, unknown.presence, unknown.expected_count)
    return AnswerPosterior(post.board, [Candidate(a, math.log(p), s) for a, p, s in keep], unknown)


def expected_brier(presence: dict[str, float]) -> float:
    """The Brier score the solver expects for its own letter probabilities: mean p x (1 - p).

    It is the Brier it would get if its probabilities were exactly right, so it needs no answer:
    0 = it is certain of every letter, 0.25 = 50% on every letter. Uncalled letters only, the same
    letters ``Turn.scores["brier"]`` averages over.
    """
    if not presence:
        return 0.0
    return sum(p * (1 - p) for p in presence.values()) / len(presence)


def solve_signal_value(turn: Turn, signal: str = "expected_brier") -> float:
    """How sure the solver is this turn, lower = surer, compared against ``solve_at``.

    - ``"expected_brier"``: ``expected_brier(turn.presence)``, the solver's own estimate. Uses
      nothing the solver couldn't know.
    - ``"brier"``: the Brier score of its letter probabilities against the TRUE answer (the
      "Brier" column of ``print_game``). The solver can't know this in a real game, so a solver
      that waits for it is peeking at the answer to decide when to guess; useful only as a
      cheating upper bound.
    """
    if signal == "expected_brier":
        return expected_brier(turn.presence)
    if signal == "brier":
        return turn.scores.get("brier", 0.0)
    raise ValueError(f"signal must be one of {SOLVE_SIGNALS}, got {signal!r}")


def make_turn(number: int, post: AnswerPosterior, truth: str, info: dict | None = None, top_k: int = 10) -> Turn:
    """Everything worth recording about one posterior, measured against the true answer."""
    ranked = post.top(len(post))
    answers = [a for a, _ in ranked]
    presence = post.presence() if not post.empty else {c: 0.0 for c in post.board.uncalled}
    return Turn(
        number=number,
        board=post.board,
        n_candidates=len(post),
        top=ranked[:top_k],
        p_truth=post.prob_of(truth),
        truth_rank=answers.index(truth) + 1 if truth in answers else None,
        entropy_bits=post.entropy(),
        presence=presence,
        expected_count=post.expected_count() if not post.empty else {c: 0.0 for c in post.board.uncalled},
        scores=letter_scores(presence, post.board, truth) if presence else {},
        info=info or {},
        p_unknown=post.unknown_mass,
    )


def play_letters(
    answer: str,
    posterior_fn: PosteriorFn,
    policy: Policy = most_likely,
    kind: str = "any",
    start: Board | None = None,
    stop_at: float | None = None,
    max_letters: int = 26,
    top_k: int = 10,
    on_turn: Callable[[Turn], None] | None = None,
    solve_at: float | None = None,
    solve_signal: str = "expected_brier",
    min_top: float | None = None,
) -> Game:
    """Call letters one at a time against ``answer`` until the board is full or it solves.

    ``start``: the board to begin from (default: everything hidden). Letters already called on it
    count as called but not as this game's calls.
    ``solve_at``: solve instead of calling a letter whenever the turn's ``solve_signal``
    (``solve_signal_value``; lower = surer) is at or below this, e.g. 0.005. Right = the game stops
    ("guessed"); wrong = that answer is ruled out for good and the turn ends with no letter called.
    ``None`` = never guess, call letters until the board is full.
    ``min_top``: don't guess unless the top answer has at least this probability, even if the
    solve signal says to; the turn calls a letter instead and is marked ``held_back``. It matters
    with an answer judge (``LLMTurnSolver(judge=)``): the solve signal only looks at letters, and
    with a "none of these" share whose letters copy the candidates', letters can look certain while
    the judge doubts the answer. Without a judge it rarely binds (a signal of 0.005 already means
    ~95%). ``None`` = no such check.
    ``stop_at``: stop, without guessing, as soon as the top answer has at least this probability.
    ``None`` = don't.
    ``on_turn``: called with each turn as soon as it's recorded (to print progress live).
    """
    answer = normalize(answer)
    board = start or Board.from_answer(answer)
    if not board.consistent(answer):
        raise ValueError(f"the starting board {board.template!r} doesn't fit the answer {answer!r}")
    if solve_signal not in SOLVE_SIGNALS:
        raise ValueError(f"solve_signal must be one of {SOLVE_SIGNALS}, got {solve_signal!r}")
    turns: list[Turn] = []
    ruled_out: set[str] = set()  # answers it guessed wrong: never candidates again
    reuse: AnswerPosterior | None = None  # after a wrong guess: same board, so no new model run
    n_called = 0
    while True:
        if reuse is not None:
            post, info, reuse = reuse, {"reused": "same board as the last turn, minus the wrong guess"}, None
        else:
            post, info = _look(posterior_fn, board)
            post = without(post, ruled_out)
        turn = make_turn(len(turns), post, answer, info, top_k)
        turn.signal = solve_signal_value(turn, solve_signal)
        turn.letters_before = n_called
        turn.held_back = (solve_at is not None and bool(turn.top) and turn.signal <= solve_at
                          and min_top is not None and turn.p_top < min_top and not board.solved)
        turns.append(turn)
        if board.solved:
            stopped = "solved"
        elif (solve_at is not None and turn.top and turn.signal <= solve_at
              and (min_top is None or turn.p_top >= min_top)):
            # Solve instead of calling a letter
            turn.guess = turn.top_answer
            turn.guess_correct = turn.guess == answer
            if on_turn:
                on_turn(turn)
            if turn.guess_correct:
                return Game(answer, turns, "guessed")
            ruled_out.add(turn.guess)
            reuse = without(post, {turn.guess})
            continue
        elif stop_at is not None and turn.p_top >= stop_at:
            stopped = "confident"
        elif n_called >= max_letters:
            stopped = "max letters"
        else:
            stopped = ""
        letter = None if stopped else choose_letter(post, policy, kind)
        if letter is None:
            if on_turn:
                on_turn(turn)
            return Game(answer, turns, stopped or "no letters left")
        turn.called = letter
        turn.p_called = turn.presence.get(letter, 0.0)
        turn.expected_called = turn.expected_count.get(letter, 0.0)
        turn.count = sum(answer[i] == letter for i in board.hidden_positions)
        n_called += 1
        if on_turn:
            on_turn(turn)
        board = board.reveal(letter, answer)


# --- the LLM as a posterior function ---------------------------------------------------------------

class LLMTurnSolver:
    """Runs the LLM search on whatever board it's handed: ``play_letters``' posterior function.

    Every call builds a fresh prompt from the board (so the revealed letters are in it), runs beam
    search under the board's constraint and, with ``n_samples``, adds sampled answers the beams
    missed (``adjust.hybrid_posterior``). The info dict it returns per turn holds the search
    results, timing, and how much of the model's probability the found answers carry.

    ``carry``: re-score the previous turn's top ``carry`` answers that still fit the new board and
    the new search didn't find, and add them (each is one forward pass of the model, under this
    turn's prompt, so its probability is on the same footing as the fresh answers). Without it the
    search starts from scratch every turn, and a long board's answer can be found on one turn and
    lost on the next just because the beams missed it. 0 = off.

    ``case``: passed to ``LLMSolver.beam_search`` (and to sampling and scoring). The default,
    ``case="title"`` (the examples' style; ``"same"`` = whatever ``style`` is), stops the model
    from copying the revealed board line in capitals, which is what makes letter-by-letter play
    work at all: copying is so cheap that junk like "ZEEPING THE ZAIN THING" otherwise crowds the
    beams out once letters are showing. ``None`` = either case.

    ``example_reveal`` / ``show_called``: how the examples' boards look and whether every board
    gets a "Called:" line (``llm.example_reveals`` / ``llm.build_prompt``). ``example_reveal="same"``
    shows each example with the letters called so far on this board revealed, rebuilt every turn;
    the default, ``"random"``, shows each example at its own point in a game (fixed across turns),
    which did best in the first real comparison. ``"none"`` + ``show_called=False`` = the old prompt.

    ``expand``: next tokens each partial answer tries (``LLMSolver.beam_search``); ``None`` = ``beams``.

    ``fill_slots``: settings for ``LLMSolver.fill_slots``, which must include ``vocab`` (e.g.
    ``dict(vocab=slots.build_vocab(examples), min_confidence=0.8)``), or ``None`` = off. After
    the search and ``carry``: once every other word of the phrase is settled (its
    most likely word at ``min_confidence`` or more among the answers found), each open board word
    gets every word from ``vocab`` that fits it, and each new answer is scored whole. Catches the
    answer whose start looks unlikely but whose end pays off ("A WISH COME TRUE": the search
    wrote "A Fish Come True"). A few hundred model passes on the turns it fires, none otherwise.
    One extra key, ``refill``, is this class's own: ``"changed"`` (default) re-fills a board word
    only when it changed since the turn it was last filled (a new letter in it, or a different
    rest of the phrase), since what that fill found is still carried forward; ``"always"`` = every
    turn, as before. ``max_slots=1`` (an ``LLMSolver.fill_slots`` setting) fills only the least
    settled word per turn.

    ``judge``: an answer judge (``wof.judge``, e.g. ``JevJudge()``) asked once per turn
    about every candidate: is it real, familiar English that fits the category? Its scores reweight
    the posterior (``apply_judge``) and add an explicit "none of these" share, so a turn whose
    candidates are all nonsense stops being confident in one of them. ``judge_weight`` tempers the
    scores (0 = ignore them), ``judge_prior_unknown`` is P(none of these) before the judge on an
    empty board, scaled down by the share of letters still hidden (a fixed share would keep "none
    of these" at ~10% even with one judged-real answer left, so the solver would never guess; with
    ``n_samples``, ``1 - coverage`` is used instead), ``judge_real_rate`` how highly the judge
    typically rates a real answer: a number, or {category: rate} with "*" as the fallback (see
    ``judge.calibrate_real_rate``); ``None`` (default) = the judge's own measured rates
    (``JevJudge.real_rates``, i.e. ``judge.REAL_RATES``), else 0.9. ``unknown_letters``: what an answer outside the
    candidates looks like letter by letter (default ``judge.candidate_letters``: like the candidates
    before the judge, so the judge changes calls only by reweighting; ``judge.english_letters`` =
    English frequencies, board-blind). Pair it with ``play_letters(min_top=)`` so the solver
    doesn't guess an answer the judge doubts. ``None`` = no judge (as before).
    With ``truth`` given, the judge is also asked about the true answer, for display only.

    ``truth``: if given, the true answer is also scored directly each turn (one extra forward
    pass, no search), so you can see how likely the model rates it even on the turns the search
    missed it. It never affects which answers are found or which letter is called -- it is a
    measurement, not a hint.

    Carrying needs the previous turn, so the solver remembers it. Use one ``LLMTurnSolver`` per
    game; it also forgets on its own when handed a board that isn't a later turn of the last one.
    """

    def __init__(self, solver, clue, examples=(), *, beams: int = 16, keep: int | None = None,
                 max_per_first_word: int | None = 4, n_samples: int = 0, temperature: float = 1.0,
                 sample_seed: int = 0, style: str = "title", show_board: bool = True,
                 carry: int = 10, case: str | None = "same",
                 example_reveal: str = "random", show_called: bool = True, reveal_seed: int = 0,
                 fill_slots: dict | None = None, expand: int | None = None,
                 truth: str | None = None,
                 judge=None, judge_weight: float = 1.0, judge_prior_unknown: float = 0.1,
                 judge_real_rate: float | dict[str, float] | None = None, unknown_letters=None):
        from .judge import candidate_letters
        from .search import build_prompt, example_reveals, styled

        self._build_prompt, self._styled, self._reveals = build_prompt, styled, example_reveals
        self.example_reveal, self.show_called, self.reveal_seed = example_reveal, show_called, reveal_seed
        self.solver, self.clue, self.examples = solver, clue, list(examples)
        self.beams, self.keep, self.max_per_first_word = beams, keep, max_per_first_word
        self.n_samples, self.temperature, self.sample_seed = n_samples, temperature, sample_seed
        self.style, self.show_board, self.truth = style, show_board, truth
        self.carry = carry
        self.case = style if case == "same" else case
        self.fill_slots = fill_slots
        self.expand = expand
        self.judge, self.judge_weight = judge, judge_weight
        self.judge_prior_unknown, self.judge_real_rate = judge_prior_unknown, judge_real_rate
        self.unknown_letters = unknown_letters or candidate_letters
        self._last: AnswerPosterior | None = None
        self._filled: set = set()  # what fill_slots already filled this game (its ``skip``)

    def reset(self) -> None:
        """Forget the previous turn (``carry`` and ``fill_slots``' memory start over)."""
        self._last = None
        self._filled = set()

    def _later_turn(self, board: Board) -> bool:
        """Is ``board`` a later turn of the game the last call looked at?"""
        last = self._last
        return (last is not None and len(last.board.template) == len(board.template)
                and last.board.called <= board.called)

    def _carried(self, board: Board, found: set[str]) -> list[str]:
        """The previous turn's top answers that still fit ``board`` and weren't found this turn."""
        last = self._last
        if not self.carry or last is None:
            return []
        if not self._later_turn(board):
            return []
        return [a for a, _ in last.top(len(last)) if board.consistent(a) and a not in found][:self.carry]

    def real_rate(self) -> float:
        """``judge_real_rate`` for this puzzle's category (a dict falls back to its "*" entry, then 0.9)."""
        rate = self.judge_real_rate
        if rate is None:
            rate = getattr(self.judge, "real_rates", None)
        if rate is None:
            return 0.9
        if isinstance(rate, dict):
            return rate.get(self.clue.category, rate.get("*", 0.9))
        return rate

    def _judged(self, post: AnswerPosterior, info: dict) -> tuple[AnswerPosterior, dict]:
        """``post`` reweighted by the judge, plus what the judge said (stored on the turn)."""
        from .judge import apply_judge

        t0 = time.perf_counter()
        asked = list(post.answers) + ([normalize(self.truth)] if self.truth else [])
        scores = self.judge.judge(asked, self.clue)
        if "coverage" in info:  # sampling estimated how much of the model's probability the found answers hold
            prior = 1 - info["coverage"]
        else:  # fewer hidden letters = fewer answers the search could have missed
            board = post.board
            n_letters = sum(c == HIDDEN or c in LETTERS for c in board.template)
            prior = self.judge_prior_unknown * board.n_hidden / max(n_letters, 1)
        before = dict(post.top(len(post)))
        new, res = apply_judge(post, scores, weight=self.judge_weight, prior_unknown=prior,
                               real_rate=self.real_rate(), unknown_letters=self.unknown_letters)
        out = {"name": getattr(self.judge, "name", type(self.judge).__name__), "scores": res.scores,
               "before": before, "prior_unknown": res.prior_unknown, "unknown": res.unknown,
               "real_rate": self.real_rate(),
               "seconds": time.perf_counter() - t0}
        if self.truth:
            out["truth_score"] = scores.get(normalize(self.truth))
        return new, out

    def prompt(self, board: Board) -> str:
        called = self._reveals(self.examples, board, self.example_reveal, self.reveal_seed)
        return self._build_prompt(self.clue, board, self.examples, style=self.style, show_board=self.show_board,
                                  example_called=called, show_called=self.show_called)

    def __call__(self, board: Board) -> tuple[AnswerPosterior, dict]:
        t0 = time.perf_counter()
        prompt = self.prompt(board)
        beam = self.solver.beam_search(prompt, board, beams=self.beams, keep=self.keep,
                                       max_per_first_word=self.max_per_first_word, case=self.case,
                                       expand=self.expand)
        carried = self._carried(board, set(beam.answer_logweights()))
        if carried:  # add them as if the search had found them: same score, the model's log P
            beam.paths += [self.solver.score(prompt, board, self._styled(a, self.style), self.case)
                           for a in carried]
        filled, fill_info = [], None
        if self.fill_slots is not None:  # also scored whole: the model's full log P
            t_fill = time.perf_counter()
            opts = dict(self.fill_slots)
            refill = opts.pop("refill", "changed")
            if refill not in ("changed", "always"):
                raise ValueError(f"fill_slots refill must be 'changed' or 'always', got {refill!r}")
            if not self._later_turn(board):
                self._filled = set()  # a new game: nothing filled yet
            skip = self._filled if refill == "changed" else frozenset()
            fill = self.solver.fill_slots(prompt, board, beam, case=self.case, skip=skip, **opts)
            self._filled |= fill.done
            beam.paths += fill.paths
            filled = [p.answer for p in fill.paths]
            fill_info = {"slots": fill.slots, "skipped": fill.skipped, "top_logprob": fill.top_logprob,
                         "seconds": time.perf_counter() - t_fill}
        logw = beam.answer_logweights()
        info: dict = {"beam": beam, "found_mass": sum(math.exp(w) for w in logw.values()),
                      "carried": carried, "filled": filled, "fill": fill_info}
        if logw:
            best = max(logw, key=logw.get)
            info["top_logprob"] = logw[best]
        if self.n_samples:
            sample = self.solver.sample(prompt, board, n=self.n_samples, temperature=self.temperature,
                                        seed=self.sample_seed, case=self.case)
            post, hybrid = hybrid_posterior(beam, sample)
            info.update(sample=sample, coverage=hybrid.coverage, outside_ess=hybrid.outside_ess)
        else:
            post = beam.posterior()
        if self.truth:
            info["truth_logprob"] = self.solver.score(prompt, board, self._styled(self.truth, self.style),
                                                      self.case).logprob
        if self.judge is not None and not post.empty:
            post, info["judge"] = self._judged(post, info)
        info["seconds"] = time.perf_counter() - t0
        self._last = post
        return post, info


# --- one game as a row ------------------------------------------------------------------------------

def game_summary(game: Game, sure: float = 0.9) -> dict:
    """One game as a flat row: how many letters, how many missed, and when it got it right.

    ``found``: letters called before the truth first became a candidate. ``right``: letters called
    before the turn from which the top answer was right for good (``Game.right_from``); both count
    letters, not turns, since a wrong guess is a turn with no letter. ``hidden_at_right``: letters
    still hidden then.
    ``sure_wrong``: turns at least ``sure`` sure of a wrong answer. ``won``: solved by guessing
    (with ``solve_at``); ``wrong_guesses``: how many guesses missed. ``None`` = never.
    """
    right = game.right_from()
    found = game.first_found()
    return {
        "answer": game.answer,
        "letters": len(game.calls),
        "misses": game.misses,
        "won": game.won_by_guess is not None,
        "wrong_guesses": len(game.wrong_guesses),
        "found": game.turns[found].letters_before if found is not None else None,
        "right": game.turns[right].letters_before if right is not None else None,
        "hidden_at_right": game.turns[right].n_hidden if right is not None else None,
        "sure_wrong": len(game.wrong_and_sure(sure)),
        "seconds": sum(t.info.get("seconds", 0.0) for t in game.turns),
    }
