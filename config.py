"""Every setting the solver has, in one place. Edit a value, re-run — nothing else to change.

``explore.py`` and ``play.py`` both read this file, so a change here applies to both. Run

    python config.py

to print the current settings and check your setup (is torch installed? is the model downloaded?
is the answer judge's key set?). Each section below says what the setting does and what the
interesting alternatives are; the defaults are the ones that worked best in practice.
"""

from __future__ import annotations

import os

# =====================================================================================
# 1. THE MODEL
# =====================================================================================
# Any causal language model on Hugging Face. Use a BASE model, not an "-Instruct" / chat one: the
# prompt is a plain list of solved puzzles for it to continue, not a conversation.
#
#   Qwen/Qwen2.5-0.5B   ~1 GB download, fastest, noticeably worse at long phrases
#   Qwen/Qwen2.5-1.5B   ~3 GB, the default: the smallest one that solves real puzzles
#   Qwen/Qwen2.5-7B     ~15 GB and ~30 GB of RAM in float32; clearly better if you have the room
#
# The first run downloads it (see docs/setup.md). Nothing else needs changing when you swap it.
MODEL = "Qwen/Qwen2.5-1.5B"

# None = CUDA if you have it, else Apple Silicon ("mps"), else CPU. Force it with "cpu"/"cuda"/"mps".
DEVICE = None

# =====================================================================================
# 2. THE PUZZLES AND THE PROMPT
# =====================================================================================
# None = the bundled data/puzzles.json. Point it at your own file in the same shape to use yours
# (see wof/puzzles.py for the format).
PUZZLES_FILE = None

# How many solved puzzles go in the prompt as worked examples, before the one being solved. They
# come from the example pool, which never contains a playable answer, so an example can't give the
# answer away. 0 = no examples: the model then has to work out what the task even is, and does much
# worse. More than ~8 mostly costs time.
N_EXAMPLES = 5

# Prefer examples from the puzzle's own category, so the model sees how this kind of clue is answered.
SAME_CATEGORY = True

# How the answers are written in the examples: "title" ("Burt Lancaster"), "upper", "lower".
# Title case is how a model usually sees names and phrases, and the search then holds the answer to
# the same style, which matters a lot: see CASE below.
ANSWER_STYLE = "title"

# Put the board ("Board: ____ _________") in the prompt. The search enforces the board either way;
# showing it lets the model plan ahead, though it counts underscores poorly.
SHOW_BOARD = True

# What the EXAMPLE boards show, so the model has seen a half-revealed board before:
#   "random"  each example at its own point in a game (the first 0-12 letters of E T A O I N ...
#             called). The default: it did best in a head-to-head comparison.
#   "same"    every example shows exactly the letters called on the real board so far
#   "none"    every example board blank
EXAMPLE_REVEAL = "random"

# Add "Called: A E T (not in puzzle: B)" under a board, so the model is told which letters are out
# and not only which are showing.
SHOW_CALLED = True

# =====================================================================================
# 3. THE SEARCH
# =====================================================================================
# Partial answers kept at each board position. More = more candidates found, slower, more memory.
# 16 is the sweet spot on a laptop; 64 finds the truth earlier but was ~4x slower.
BEAMS = 16

# How many next tokens each partial answer tries. None = BEAMS. Raising this is cheap (the model
# only runs on the answers actually kept), and 64 with no first-word cap was the best setting found.
EXPAND = 64

# At most this many beams per position may share a first word, so one likely opening can't spend
# every beam on its endings. None = no cap, which is the better pairing with a generous EXPAND.
MAX_PER_FIRST_WORD = None

# Finished answers to keep from one search.
KEEP = 64

# Write answers only in ANSWER_STYLE. Leave this as is. Set to None and, once letters are showing,
# the model copies the board line in capitals ("ZEEPING THE ZAIN THING") because copying is so
# cheap for it — those junk answers then crowd the real ones out of the beams.
CASE = ANSWER_STYLE

# =====================================================================================
# 4. PLAYING A PUZZLE
# =====================================================================================
# Which letter to call:
#   "likely"               most likely to be in the puzzle (the show's play: a miss loses your turn)
#   "money"                most expected occurrences (the wheel pays per occurrence)
#   "info"                 most expected information about the answer
#   "decisive"             the letter that best separates the leading answers (fastest to solve,
#                          since here a miss costs no more than a hit)
#   "likely_then_decisive" the default: "likely" until a few answers hold most of the probability
#                          AND a quarter of the board is showing, then "decisive"
POLICY = "likely_then_decisive"

# "any", "consonant" or "vowel" (falls back to any letter once that kind runs out).
KIND = "any"

# Re-score this many of last turn's best answers that still fit and this turn's search missed, so
# an answer found once isn't lost to a later search's bad luck. 0 = every turn starts from scratch.
CARRY = 10

# Once every other word of the phrase is settled, try every word of a word list in the one open
# board word and score each whole answer. This is what finds "A WISH COME TRUE" when the
# left-to-right search only ever writes "A Fish Come True". False = off (`play.py --no-fill`), which
# is faster but leaves the solver stuck on puzzles of that shape.
FILL_SLOTS = True

# It is the most expensive thing in a turn, because every fill is a whole answer through the model.
# These two bound it: at most FILL_MAX_WORDS x FILL_CONTEXTS answers are scored per board word, and
# only one board word is filled per turn. Measured on a laptop CPU: 300 fills is ~20s with the 0.5B
# model, ~2.5 minutes with the 1.5B one. Halve FILL_MAX_WORDS if a turn feels too slow — the words
# are tried most-common-first, so what you lose is the rarer end of the list.
FILL_MAX_WORDS = 300
# Rest-of-the-phrase variants to fill into (the best answers that differ outside the open word). 1 =
# only the leading phrasing, which is usually the only one that matters; 2 doubles the cost.
FILL_CONTEXTS = 1
# How settled "the rest of the phrase is settled" means: every OTHER board word with a hidden letter
# needs its most likely word to hold this much of the candidates before the open one is filled.
# Raise it to make filling fire later and less often (early on the candidates are mostly junk
# anyway, so a fill there rarely pays for itself); 0 = fill on every turn it can.
FILL_MIN_CONFIDENCE = 0.8

# Solve instead of calling a letter once the solve signal is at or below this (lower = surer).
# Roughly: 0.01 means the top answer is ~89% likely, 0.005 ~95%. None = never guess, just keep
# calling letters until the board is full (useful for watching the whole trajectory).
SOLVE_AT = 0.01

# ... and only if the top answer itself is at least this likely. This is what stops the solver
# guessing an answer the judge doubts, when the letters look certain but the phrase doesn't.
SOLVE_MIN_TOP = 0.9

# What "sure enough" means: "expected_brier" is the solver's own estimate and uses nothing it
# couldn't know. "brier" measures against the true answer, i.e. it peeks: a cheating upper bound.
SOLVE_SIGNAL = "expected_brier"

# Sampling draws per turn, to estimate how much of the model's probability the beams found
# (wof/adjust.py). Each draw costs about one beam, and small numbers are noisy. 0 = beams only.
GAME_SAMPLES = 0

# =====================================================================================
# 5. THE ANSWER JUDGE (optional — everything works without it)
# =====================================================================================
# A small base model can end up confident in nonsense that happens to fit the board ("What You
# Give Me Treats!"), simply because nothing better was found. The judge asks a second model one
# yes/no question per candidate — "is this real, familiar English that fits the category?" — and
# reweights the answers, moving probability to an explicit "none of these" when every candidate is
# junk. The solver then keeps calling letters instead of confidently guessing garbage.
#
# It needs a TypeSafe API key in the TYPESAFE_API_KEY environment variable. With no key, the solver
# plays without it and says so once. See docs/setup.md for how to get one and what it costs.
USE_JUDGE = True

# The judge model, pinned on purpose: "jev-latest" moves with every release, which would quietly
# shift your results.
JUDGE_MODEL = "jev-1.14.0"

# How much the judge counts: P(answer) x judge score ** JUDGE_WEIGHT. 0 = ignore its scores (but
# keep the "none of these" share).
JUDGE_WEIGHT = 1.0

# P(the answer is none of the candidates) before the judge looks, on an empty board. It shrinks as
# letters are revealed. With GAME_SAMPLES > 0, the measured 1 - coverage is used instead.
JUDGE_PRIOR_UNKNOWN = 0.1

# How highly the judge rates a genuinely REAL answer, which is what "none of these" is weighed
# against. None = the judge's own measured rates (wof/judge.py's REAL_RATES). A number overrides it;
# re-measure with wof.judge.calibrate_real_rate after changing the model or a question's wording.
JUDGE_REAL_RATE = None


# =====================================================================================
# HELPERS — the two scripts build everything through these, so nothing is configured twice
# =====================================================================================

def puzzles():
    """The bundled puzzle set (``wof.puzzles.Puzzles``): ``.play`` and ``.examples``."""
    from wof.puzzles import load_puzzles

    return load_puzzles(PUZZLES_FILE)


def load_model(quiet: bool = False):
    """The language model, loaded once. Downloads it on the first run (see docs/setup.md)."""
    from wof.search import LLMSolver

    if not quiet:
        print(f"loading {MODEL} ...", flush=True)
    solver = LLMSolver(MODEL, device=DEVICE)
    if not quiet:
        print(f"{solver.model_name} on {solver.device}: {len(solver.token_text):,} of its tokens can appear "
              f"inside an answer, {len(solver.end_ids)} can end one")
    return solver


def make_judge(quiet: bool = False):
    """The answer judge, or ``None`` if it's switched off or has no API key."""
    from wof import judge as judge_mod

    if not USE_JUDGE:
        return None
    if not os.environ.get(judge_mod.API_KEY_ENV):
        if not quiet:
            print(f"note: USE_JUDGE is on but {judge_mod.API_KEY_ENV} isn't set, so there is no answer judge.\n"
                  f"      Everything still works. See docs/setup.md to set it up, or put USE_JUDGE = False\n"
                  f"      in config.py to stop asking.")
        return None
    return judge_mod.JevJudge(JUDGE_MODEL)


def letter_policy():
    """The letter policy named by ``POLICY``."""
    from wof.policy import POLICIES

    if POLICY not in POLICIES:
        raise ValueError(f"POLICY must be one of {sorted(POLICIES)}, got {POLICY!r}")
    return POLICIES[POLICY]


def prompt_examples(puzzle, pool=None, seed: int = 0):
    """The solved puzzles to show the model before ``puzzle`` (``N_EXAMPLES``, ``SAME_CATEGORY``)."""
    from wof.puzzles import pick_examples

    if not N_EXAMPLES:
        return []
    pool = pool if pool is not None else puzzles().examples
    return pick_examples(pool, puzzle, n=N_EXAMPLES, same_category=SAME_CATEGORY, seed=seed)


def fill_vocab(pool):
    """The word list ``FILL_SLOTS`` fills single board words from (the example answers' words)."""
    from wof.slots import build_vocab

    return build_vocab(pool)


def turn_solver(solver, puzzle, examples, *, judge=None, vocab=None, truth: str | None = None):
    """An ``LLMTurnSolver`` wired up from this file: what ``play_letters`` calls each turn.

    ``truth``: the real answer, scored directly each turn so you can see how likely the model
    rates it even when the search misses it. It never changes a call.
    """
    from wof.calling import LLMTurnSolver

    fill = None
    if FILL_SLOTS:
        if vocab is None:
            raise ValueError("FILL_SLOTS is on, so turn_solver needs vocab=config.fill_vocab(examples_pool)")
        fill = dict(vocab=vocab, min_confidence=FILL_MIN_CONFIDENCE, max_words=FILL_MAX_WORDS,
                    contexts=FILL_CONTEXTS, keep=8, max_slots=1, refill="changed", batch=32)
    return LLMTurnSolver(
        solver, puzzle.clue, examples,
        beams=BEAMS, keep=KEEP, max_per_first_word=MAX_PER_FIRST_WORD, expand=EXPAND, case=CASE,
        n_samples=GAME_SAMPLES, style=ANSWER_STYLE, show_board=SHOW_BOARD,
        example_reveal=EXAMPLE_REVEAL, show_called=SHOW_CALLED, carry=CARRY, fill_slots=fill,
        truth=truth, judge=judge, judge_weight=JUDGE_WEIGHT, judge_prior_unknown=JUDGE_PRIOR_UNKNOWN,
        judge_real_rate=JUDGE_REAL_RATE,
    )


def play_kwargs():
    """The ``wof.calling.play_letters`` settings from section 4."""
    return dict(policy=letter_policy(), kind=KIND, solve_at=SOLVE_AT, solve_signal=SOLVE_SIGNAL,
                min_top=SOLVE_MIN_TOP)


# =====================================================================================
# python config.py — print the settings and check the setup
# =====================================================================================

def status() -> None:
    import importlib.util
    import sys
    from pathlib import Path

    from wof import judge as judge_mod

    def line(label, value, note=""):
        print(f"  {label:<22} {value}" + (f"   {note}" if note else ""))

    print("\n── SETUP " + "─" * 71)
    line("python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
         "3.10+ needed")
    have_torch = importlib.util.find_spec("torch") is not None
    have_tf = importlib.util.find_spec("transformers") is not None
    if have_torch and have_tf:
        import torch
        where = ("cuda (GPU)" if torch.cuda.is_available()
                 else "mps (Apple Silicon)" if torch.backends.mps.is_available() else "cpu")
        line("torch / transformers", "installed", f"will run on: {DEVICE or where}")
    else:
        line("torch / transformers", "MISSING", "pip install -r requirements.txt")
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cached = sorted(home.glob(f"**/models--{MODEL.replace('/', '--')}"))
    line("model", MODEL, "already downloaded" if cached else "will download on first use (~1-15 GB)")
    key = bool(os.environ.get(judge_mod.API_KEY_ENV))
    line("answer judge", "on" if (USE_JUDGE and key) else "off",
         "" if (USE_JUDGE and key) else
         (f"no {judge_mod.API_KEY_ENV} set (optional; see docs/setup.md)" if USE_JUDGE
          else "USE_JUDGE = False in config.py"))
    try:
        ps = puzzles()
        cats = ps.categories()
        line("puzzles", f"{len(ps.play)} to play, {len(ps.examples)} as prompt examples",
             f"{len(cats)} categories")
    except Exception as e:  # a broken or missing puzzle file should say so here, not mid-game
        line("puzzles", "FAILED", str(e))

    print("\n── SETTINGS " + "─" * 68)
    for name in ("MODEL", "DEVICE", "N_EXAMPLES", "SAME_CATEGORY", "ANSWER_STYLE", "SHOW_BOARD",
                 "EXAMPLE_REVEAL", "SHOW_CALLED", "BEAMS", "EXPAND", "MAX_PER_FIRST_WORD", "KEEP",
                 "CASE", "POLICY", "KIND", "CARRY", "FILL_SLOTS", "FILL_MAX_WORDS", "FILL_CONTEXTS", "FILL_MIN_CONFIDENCE",
                 "SOLVE_AT", "SOLVE_MIN_TOP",
                 "SOLVE_SIGNAL", "GAME_SAMPLES", "USE_JUDGE", "JUDGE_MODEL", "JUDGE_WEIGHT",
                 "JUDGE_PRIOR_UNKNOWN", "JUDGE_REAL_RATE"):
        print(f"  {name:<22} {globals()[name]!r}")
    print("\nedit config.py to change any of these, then:  python play.py   (or open explore.py)\n")


if __name__ == "__main__":
    status()
