# How the solver works

A language model predicts **tokens** — `" Burt"`, `"aster"`, `" THE"` — never single letters. So
"what letter is in slot 7?" is not a question it can answer. Its next-token distribution tells you
about the *next* token, left to right, and nothing about slot 7 given the rest of the board.

The whole design follows from working around that.

## 1. Letter probabilities come from whole answers

Never ask about letters. Make the model write **whole answers that fit the board**, keep the
model's own probability for each, and marginalize:

```
board + category
   │
   ├─► search the model under the board's constraint ──► candidate answers, each with log P
   │
   └─► AnswerPosterior:  P(letter at slot i)   = Σ P(answer) · [answer[i] = letter]
                         P(letter anywhere)    = Σ P(answer) · [letter is in a blank]
                         E[occurrences], entropy, information gain, the posterior after a call
```

Everything a player cares about is a sum over the candidate set, computed exactly
(`wof/posterior.py`). It also makes "what does calling K teach us?" trivial: keep the candidates
consistent with the reveal and renormalize.

This is why the solver needs an **open-weight** model, not a chat API. It needs two things no
hosted chat endpoint offers: constrained decoding (only tokens whose characters fit the board) and
scoring an arbitrary string (`log P(answer | prompt)` for an answer you hand it).

## 2. The board is used twice, in two different ways

- It **decides which answers exist.** At each step we precompute which of the model's ~150,000
  tokens could be written at that board position: a blank takes any *uncalled* letter (either case),
  a revealed letter takes itself, punctuation must match, and nothing may run past the end.
- It is **not** used to score the answers. Each answer's score is the model's plain
  `log P(answer | prompt)` — the product of its tokens' probabilities, plus `log P(stop)` once the
  board is full. Renormalizing that over the board-fitting answers is Bayes' rule,
  `P(answer | clue, board)`. No re-ranking, no hand-tuned features.

Run `explore.py` step 2 and watch the fit share: on an empty board, typically **2-5% of the model's
probability** sits on tokens that could legally be written. The other 95%+ is discarded.

## 3. The search compares answers at the same board position

Plain beam search ranks every live beam together after each token. That is wrong here, because a
partial answer's log P only ever falls as it grows: `" Keeping"` (8 characters, its cost paid)
loses to `" Me"` (3 characters, most of its cost still ahead). On a real puzzle it was cut on the
very first step, by 0.4 nats, and the surviving fragments grew into junk — `"Meehahs Hae Hahs"` at
log P −53, while the true `"KEEPING THE MAIN THING THE MAIN THING"` scored −36.5 and was never
found at all.

So the search is **position-synchronous**: only partial answers that have written the same number of
board characters are ever compared, `BEAMS` are kept at each position, and answers that are further
ahead wait for the others to catch up. Rows at different positions share one batched KV cache (a
waiting row is fed a masked padding token and each row gets its own position ids).

The other half of the same fix: **hold the answer to one letter case.** Once letters are revealed,
the prompt's `Board:` line is nearly the answer in capitals, and copying it is so cheap for the
model that all-caps strings with arbitrary letters in the blanks fill the beams
(`"ZEEPING THE ZAIN THING"`). Writing answers only in the examples' title case makes copying
impossible. (`config.CASE`; set it to `None` to watch this happen.)

Measured on full games (0.5B, CPU):

| puzzle | search | letters called (misses) | truth first a candidate |
|---|---|---|---|
| KEEPING THE MAIN THING… | ranked by token, either case, 32 beams | 24 (14) | turn 23 |
| KEEPING THE MAIN THING… | by position, title case, 16 beams | 18 (8) | turn 8 |
| A STITCH IN TIME SAVES NINE | ranked by token | 13 (3) | turn 7 |
| A STITCH IN TIME SAVES NINE | by position | 10 (0) | turn 3 |

Position search with 12 beams beat token search with **192** beams, in half the time. The
comparison rule was the problem, not the beam count.

## 4. Coverage is the main failure mode

The posterior is only over the answers the search found. If the true answer isn't among them, every
number is confidently wrong — and the search finds a few dozen answers out of an astronomical space.
`explore.py` step 8 measures it by sampling: on an empty board, 64 beams held roughly **5%** of the
model's board-fitting probability. Renormalizing over them treats the other 95% as impossible, so
whatever the top few answers agree on goes to ~100%.

Three things help, and the project uses the last two:

- **Sampling** (`wof/adjust.py`) estimates what the search missed, but needs a lot of draws to be
  anything but noisy, and most sampled answers are junk that merely fits the board. It is a
  diagnostic here, not part of a game (`config.GAME_SAMPLES = 0`).
- **The answer judge** (§5) puts an explicit "the answer is none of these" share in the posterior.
- **Filling one word** (§6) looks for the answer the left-to-right search structurally cannot reach.

## 5. The answer judge: is this real English, or board-fitting junk?

A small base model can end up confident in nonsense simply because nothing better was found —
`"What You Give Me Treats!"` at 63%. The judge (`wof/judge.py`) asks a second model one yes/no
question per candidate ("is this a real, familiar piece of English that fits the category?") and
treats the answer as a likelihood:

```
w(candidate i)    = (1 − u) · p_i · q_i ^ weight
w(none of these)  = u · real_rate ^ weight
```

where `p_i` is the search's probability, `q_i` the judge's score, `u` the prior chance the answer is
none of the candidates, and `real_rate` how highly the judge rates a genuinely real answer. The
point is the last line: if **every** candidate is nonsense, their weights all collapse and the
probability moves to "none of these" rather than to the least-bad nonsense. Paired with
`config.SOLVE_MIN_TOP`, the solver then keeps calling letters instead of confidently guessing junk.
Nothing is ever deleted, so a judge mistake on a pun or an obscure name costs probability, not the
answer.

Two details that were learned the hard way:

- **`real_rate` has to be measured, not guessed.** The first guess, 0.9, was far too high: the judge
  rated the true `"Keeping The Main Thing The Main Thing"` 54% real, so every candidate looked weak
  next to "none of these" and that share ran to 36-53% on early turns. Measured over 200 real
  answers it is ~0.73 (`wof.judge.REAL_RATES`, re-measurable with `calibrate_real_rate`).
- **What "none of these" looks like letter by letter matters.** It defaults to looking like the
  candidates the search found, so the judge can only change a call by reweighting candidates. An
  earlier version used English letter frequencies instead, which ignore the board (a blank in
  `THIN_` is treated like any other) and flipped a real game's turn-2 call to a worse letter.

Wordplay categories need their own question. Asked the generic one, real Before & After answers
rated ~22% and Same Name ~17%, because they genuinely aren't phrases anyone says. Each gets a
question explaining the wordplay (`judge.CATEGORY_RULES`), and Same Name answers are split in code
first — `"STRAWBERRY & PAPER JAM"` becomes `"Strawberry Jam"` + `"Paper Jam"`, each asked about
separately, because the judge rated `"Strawberry & Paper Fan"` as real as the truth.

The judge is **optional**. Everything works without it; the solver is simply more willing to believe
a confident-looking piece of nonsense.

## 6. Filling one word: the answer the search cannot reach

`"A WISH COME TRUE"` is the case this exists for. With `COME TRUE` settled, the search still writes
`"A Fish Come True"`, because `" A W"` is an unlikely *start* and the search works left to right —
even though the model rates the whole phrase ~80x higher. Recombining words from other candidates
can't help either, since no partial answer ever wrote "Wish".

So: when every *other* word of the phrase is settled (its most likely word holds
`min_confidence` of the candidates), take the one open board word, put every word from a word list
that fits its pattern into it, and **score each whole answer**. The late payoff then counts, and the
candidate comes from the word list rather than from the model's left-to-right ranking.

This is the most expensive part of a turn — each fill is a whole answer through the model, and a
turn that fires can score a few thousand of them — so only the turns where the rest of the phrase is
settled pay it, one board word at a time, skipping words that haven't changed since they were last
filled.

## 7. Which letter to call

Given a posterior, `wof/policy.py` answers different questions:

| policy | maximizes | when it's right |
|---|---|---|
| `likely` | P(letter is in the puzzle) | on the show: a miss hands the wheel to an opponent |
| `money` | E[occurrences] | the wheel pays per occurrence |
| `info` | expected bits learned | generic uncertainty reduction |
| `decisive` | P(the top answer is right after the call) | reaching the answer in the fewest turns |
| `likely_then_decisive` | `likely`, then `decisive` | the default |

`decisive` exists because of a specific trap. With `"STRAWBERRY & PAPER FAN"` and `"... PAPER JAM"`
in front, `likely` called B, Y, W and S — all certain to be there, all in the same places in every
candidate, and all teaching nothing about which answer it was. `decisive` calls the letter that
separates them. But early on, when the answer may not be among the candidates at all, revealing
sure letters is still worth it, because it gives the model more board to search with. Hence the
switch: `likely` until a few answers hold most of the probability **and** a quarter of the board is
showing. Both conditions matter — on a near-empty board a confident-looking top 4 means nothing
(the beams hold ~5% of the mass), and switching early cost a real game several turns.

## 8. Solving instead of calling

Each turn the solver either calls a letter or guesses the answer. It guesses once its **solve
signal** — the Brier score it expects for its own letter probabilities, mean `p(1−p)`, which needs
no answer key — drops to `SOLVE_AT`, and only if the top answer itself is at least `SOLVE_MIN_TOP`
likely. A wrong guess is ruled out of every later posterior and costs the turn; the board hasn't
changed, so the next turn reuses the same posterior minus that answer rather than re-running the
model.

## 9. What this is not

- **Not a measurement of how good the model is at the real show.** The bundled puzzles are
  hand-written in the show's style (see `data/puzzles.json`), and any model trained on the web has
  read the phrases they use. A model recognizing a phrase is not the same as solving a puzzle.
- **Not tuned per puzzle.** Every setting in `config.py` is global. The defaults are the ones that
  worked best across the puzzles they were tried on, which is a small number of puzzles.
- **Small models get things wrong in interesting ways.** With `THE MAIN THING THE MAIN THING`
  showing, the 0.5B model still rates `REEPING`, `SEEPING` and `WEEPING` above `KEEPING`. That is
  the model's own preference, not a bug in the search — and watching it is half the point.
