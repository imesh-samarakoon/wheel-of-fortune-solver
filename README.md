# Wheel of Fortune solver

A solver that puts a **probability on every letter** of a Wheel of Fortune board, decides which one
to call, and knows when to stop calling and guess the answer. It runs on a small open-weight
language model on your own machine.

The interesting part is what it takes to get there. A language model predicts **tokens**
(`" Burt"`, `"aster"`), never single letters, so "what letter is in slot 7?" is not a question it
can answer. So the solver never asks. It makes the model write **whole answers that fit the board**,
keeps the model's own probability for each, and adds them up:

```
P(letter is in the puzzle) = total probability of the candidate answers that contain it
```

Everything else — which letter to call, how sure it is, when to guess — falls out of that one sum.

```
── TURN 3   __A__ _A__ ──────────────────────────────────────────────────────────
called so far: A E R   (not in puzzle: E R)   7 letters hidden
58 candidate answers, entropy 3.06 bits   (4.8s)
most likely  : BLACK JACK  29.8%  (wrong)
true answer  : rank 2, 26.8%   (scored directly, the model rates it 0.901x as likely as the top answer)
top answers  : SMALL TALK 26.8% · BLACK CATS 8.1% · BLACK HAND 7.6% · BLANK WALL 6.6%
P(in puzzle) : L 100%  K 98.6%  B 71.7%  C 63.7%  S 43.7%  T 36.7%  M 31.9%  J 29.8%
solve signal : 0.083  (lower = surer; it guesses at or below the threshold)
=> calls L  (it said 100.0% in puzzle, expected 1.78x)  ...  appears 3x
```

One turn later it solved it: `SMALL TALK`, four letters called, two of them misses. (That run
used the smaller 0.5B model, hence the 4-second turn.)

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python config.py          # check the setup; nothing to sign up for
python play.py            # play a puzzle and watch the solver work
```

The first run downloads the model (~3 GB, once) — there is no account or token involved.

**How long a game takes.** On an ordinary CPU with the defaults, a short puzzle is about a minute
(`BREAK THE ICE` took 62 seconds and one letter) and a long phrase is several minutes. For something
quicker, set `MODEL = "Qwen/Qwen2.5-0.5B"` in `config.py` — three times faster, and wrong in more
interesting ways — or pass `play.py --no-fill`. [`docs/setup.md`](docs/setup.md) has the measured
numbers and every other knob.

## The two things to do with this

**Watch it think, step by step** — open [`explore.py`](explore.py) and run it cell by cell (VS Code
or Cursor: "Run Cell"; or just `python explore.py`). One puzzle, every stage printed with real
numbers: the prompt, the model's next-token predictions and how little of them legally fit the
board, the search growing partial answers, each candidate's probability token by token, the
normalization, the letter probabilities, where the true answer actually sits, how much of the
model's distribution the search missed, and finally the whole puzzle played out.

**Play puzzles** — [`play.py`](play.py), including your own:

```bash
python play.py                                      # one of the 101 bundled puzzles, at random
python play.py 42                                   # puzzle 42
python play.py "DOUBLE DUTCH"                       # by part of its answer
python play.py --list                               # all 101, as boards, no spoilers
python play.py --new                                # type in your own, at the prompt
python play.py --answer "A WISH COME TRUE" --category PHRASE
python play.py --called RSTLNE                      # start from a board that already shows R S T L N E
python play.py --batch 10 --quiet                   # ten puzzles, one line each
```

Give it any answer and category you like. The solver is told the category and the board, never the
answer — the answer is only used to reveal letters and to score what it believed afterwards.

## Everything is in config.py

[`config.py`](config.py) is the only place with settings: which Hugging Face model, how wide the
search, which letter-calling policy, when to guess, whether the answer judge is on. Each one says
what it does and what the interesting alternatives are. Both scripts read it, so a change applies to
both. `python config.py` prints the current settings and checks your setup.

Good first things to change: `MODEL` (`Qwen/Qwen2.5-0.5B` is three times faster and visibly dumber),
`POLICY` (how it picks a letter), `SOLVE_AT` (how sure it has to be before guessing), and `CASE`
(set it to `None` to watch the model start copying the board in capitals).

## What's in the repo

```
config.py            every setting, plus a setup check
explore.py           the walkthrough: one puzzle, every step printed
play.py              play a bundled or your own puzzle
data/puzzles.json    101 puzzles to play + 208 kept aside as prompt examples
data/words.txt       43,653 common English words, for filling a single open board word
docs/how-it-works.md the design, and what went wrong before it worked
docs/setup.md        Python, Hugging Face, and the optional answer judge
wof/
  board.py           what's showing, what's been called, what still fits
  search.py          the model, searched token by token under the board's constraint (the core)
  posterior.py       candidate answers -> per-slot and per-letter probabilities
  judge.py           a second model's opinion: real English, or board-fitting junk?
  slots.py           filling one open word from a word list
  policy.py          which letter to call
  calling.py         play a whole puzzle, one letter at a time
  adjust.py          how much of the model's probability the search actually found
  puzzles.py         the puzzle file and the prompt examples
  show.py            every printout
tests/               85 tests, no model needed
```

## How it works, in six steps

1. **Prompt** the model with a few solved puzzles, then this puzzle's category and board.
2. **Filter its tokens**: at each board position, only tokens whose characters could legally be
   written there. On an empty board that is typically 2-5% of the model's probability.
3. **Search** them for the most likely whole answers that fill the board exactly — comparing partial
   answers only at the same board position, which turns out to matter enormously.
4. **Normalize** over the answers found: that is `P(answer | clue, board)`.
5. **Marginalize** to letters, and let a policy pick one.
6. **Call it, reveal it, and run the model again** on the new board. The search now can't use a
   letter that was called and missed, and the prompt shows the revealed ones.

The things that make it actually work — why the search compares answers by board position, why the
answer is held to one letter case, why coverage is the main failure mode, what the answer judge is
for, and how the open word gets filled — are in [`docs/how-it-works.md`](docs/how-it-works.md),
with the measurements that led to each one.

## Honest caveats

- **The bundled puzzles are hand-written** in the show's style, not the show's own archive. They are
  ordinary English phrases, places, people and the wordplay categories, which is what the solver
  needs; they are not a benchmark. Swap in your own file (`config.PUZZLES_FILE`).
- **A model trained on the web has read these phrases.** Recognizing a phrase is not the same as
  solving a puzzle. Treat a good result as "the model knows this phrase and the search found it",
  not as evidence about the show.
- **Small models are wrong in interesting ways**, which is half the fun. With
  `THE MAIN THING THE MAIN THING` showing, the 0.5B model still rates `REEPING` and `WEEPING` above
  `KEEPING`.
- **Nothing here is tuned per puzzle.** Every setting is global, and the defaults were chosen on a
  small number of puzzles.

## The answer judge (optional)

A small model can end up confident in nonsense that happens to fit the board —
`"What You Give Me Treats!"` at 63%. The optional judge asks a second model one yes/no question per
candidate ("is this real, familiar English that fits the category?") and moves probability to an
explicit *"the answer is none of these"* when they are all junk, so the solver keeps calling letters
instead of guessing garbage.

It needs a [TypeSafe](https://typesafe.ai) API key in `TYPESAFE_API_KEY`. **Without it
everything still works** — `play.py` prints one note and plays on, or set `USE_JUDGE = False`.
See [`docs/setup.md`](docs/setup.md#3-typesafe-the-optional-answer-judge).

## Tests

```bash
python -m pytest -q
```

No model, no network, no API key: the board logic, the posterior, the policies, the judge's maths and
its HTTP handling (mocked), the word-filling pass, and the bundled puzzle file.

## Licence

MIT. See [LICENSE](LICENSE).

`data/words.txt` is derived from the English Speller Database (SCOWL), copyright 2000-2026 Kevin
Atkinson, and is redistributed under its own permission notice, which is reproduced at the top of
that file. The puzzles in `data/puzzles.json` are written for this repo.
