# Setup

Three things, in order of how much they matter:

1. **Python and the packages** — required.
2. **Hugging Face** (the language model) — required, but there is nothing to sign up for.
3. **TypeSafe** (the optional answer judge) — needs an API key. Skip it and everything works.

At any point, `python config.py` prints what it can see and what is missing.

---

## 1. Python and the packages

Python 3.10 or newer.

```bash
git clone https://github.com/imesh-samarakoon/wheel-of-fortune-solver
cd wheel-of-fortune-solver
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

That pulls in PyTorch, which is a large download (a few hundred MB on CPU). Then:

```bash
python -m pytest -q     # 85 tests, no model needed, a fraction of a second
python config.py        # what's set and what's missing
```

**On a GPU.** `pip install torch` gives you the CPU build on some platforms. If you have an NVIDIA
card and want to use it, install the CUDA build first, following the selector at
<https://pytorch.org/get-started/locally/>, then `pip install -r requirements.txt` for the rest.
Apple Silicon needs nothing special — PyTorch uses the GPU through `mps` automatically.

---

## 2. Hugging Face: the language model

**There is nothing to sign up for.** The default model (`Qwen/Qwen2.5-1.5B`) is openly downloadable,
and `transformers` fetches it the first time you run the solver. You do not need an account, a token,
or the `huggingface-cli`.

What happens on that first run:

- it downloads **~3 GB** for the 1.5B default,
- into `~/.cache/huggingface/` (override with the `HF_HOME` environment variable),
- once. Every later run loads it from there, offline.

### Choosing a model

Any **base** causal language model works. Use a base model, not an `-Instruct` or chat one: the
prompt is a plain list of solved puzzles for the model to continue, not a conversation, and an
instruct-tuned model will try to talk to you instead.

| `config.MODEL` | download | notes |
|---|---|---|
| `Qwen/Qwen2.5-0.5B` | ~1 GB | fastest; noticeably worse on long phrases, which makes it fun to watch |
| `Qwen/Qwen2.5-1.5B` | ~3 GB | the default: the smallest one that solves real puzzles |
| `Qwen/Qwen2.5-7B` | ~15 GB | clearly better, needs ~30 GB of RAM in float32 |
| `meta-llama/Llama-3.2-1B` | ~2.5 GB | works, but **gated**: see below |

Change `MODEL` in `config.py`. Nothing else needs changing — the vocabulary scan that works out
which tokens can appear inside an answer runs per model at load time.

### How long it takes

Measured on an ordinary cloud CPU, no GPU, default settings:

| | 0.5B | 1.5B |
|---|---|---|
| one search (one turn) | ~4-6 s | ~15 s |
| a turn that also fills a word | | +24 s for 296 fills |
| a whole short puzzle, `--no-fill` | 21-38 s (three games) | |
| a whole short puzzle | | `BREAK THE ICE`: **53 s**, solved after one letter |

A long phrase takes several minutes, because it needs more turns and each turn's search is over a
longer board. A GPU is several times faster.

If a game feels too slow, in order of effect:

```python
MODEL = "Qwen/Qwen2.5-0.5B"   # ~3x faster, and worse in ways that are interesting to watch
FILL_MAX_WORDS = 150          # halves the expensive pass; or FILL_SLOTS = False to skip it
BEAMS = 8                     # halves the search, finds fewer candidates
```

or `play.py --no-fill` for a one-off.

### Gated models (only if you pick one)

A few models — Llama, Gemma — need you to accept a licence on their Hugging Face page first. Then:

```bash
pip install huggingface_hub
huggingface-cli login        # paste a token from huggingface.co/settings/tokens ("read" is enough)
```

The token is stored in `~/.cache/huggingface/token` by Hugging Face itself. **Nothing in this repo
reads or stores it**, and you don't need one at all for the Qwen defaults.

### If the download fails

- **Behind a proxy or firewall**: `transformers` honours `HTTPS_PROXY`.
- **No internet on the machine that runs it**: download on another machine, copy
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B` across, and set `HF_HOME` to match.
- **Out of disk**: point `HF_HOME` at a bigger drive, or use the 0.5B model.

---

## 3. TypeSafe: the optional answer judge

### What it does, and why you might want it

The search model only ever judges an answer token by token, so a string like
`"What You Give Me Treats!"` can fit the board, score well enough, and become the top answer simply
because nothing better was found. The **answer judge** asks a second model one yes/no question per
candidate — *"is this a real, familiar piece of English that fits the category?"* — and reweights
the candidates, moving probability to an explicit **"none of these"** share when they are all junk.
The solver then keeps calling letters instead of confidently guessing garbage.

The judge is [TypeSafe](https://typesafe.ai)'s **Jev**, a decision model that answers a typed
question with a probability instead of generating text, which is exactly the shape of this question
and lets one request carry one question per candidate.

`docs/how-it-works.md` §5 has the maths and what went wrong before it was calibrated.

### Running without it

Nothing breaks. With no key, `play.py` and `explore.py` print one note and play on. To stop being
asked, set `USE_JUDGE = False` in `config.py`, or pass `play.py --no-judge`.

What you give up: on puzzles where the model can't find the real answer, the solver stays confident
in whatever junk it did find, guesses it, and gets it wrong.

### Turning it on

1. Make an account at <https://typesafe.ai> and create an API key in its console.
2. Put the key in the `TYPESAFE_API_KEY` environment variable:

   ```bash
   export TYPESAFE_API_KEY=...          # macOS / Linux, add it to ~/.zshrc or ~/.bashrc to keep it
   ```
   ```powershell
   $env:TYPESAFE_API_KEY = "..."        # Windows PowerShell, this session only
   setx TYPESAFE_API_KEY "..."          # Windows, permanently
   ```
3. Check it:

   ```bash
   python config.py                      # "answer judge  on"
   ```

**Never put the key in `config.py`** or any other file in the repo. It is read from the environment
when the first request is made, and nothing here writes it anywhere.

### What it costs, and the cache

One request per turn, carrying one ~40-token question per candidate — so a game is a few dozen
requests of a few dozen questions. Every answer is cached on disk in `cache/judge_cache.json`
(git-ignored), keyed by model, question version, clue and answer, so replaying a puzzle, or the
candidates carried over between turns, cost nothing. Delete that file to re-ask everything.

### Using a different judge

`wof/judge.py` defines the `AnswerJudge` protocol: one method, `judge(answers, clue) ->
{answer: P(real)}`. Anything that implements it works — `FixedJudge` (hand-set scores, used by the
tests) is the second implementation and shows how little there is to it. If you write one around
another model, re-measure `real_rate` for it with `calibrate_real_rate`: the "none of these" share
is weighed against how highly *your* judge rates a genuinely real answer, and getting it wrong is
the one way to make the judge actively harmful (see how-it-works §5).

### If the judge errors

- `no TypeSafe API key: set the TYPESAFE_API_KEY environment variable` — the variable isn't visible
  to this shell. Re-run `export`, or restart the terminal after `setx` on Windows.
- `HTTP 401` — the key is wrong or revoked.
- `HTTP 429` — rate limited. It retries with backoff on its own; if it keeps happening, lower
  `JUDGE_WEIGHT` traffic by playing fewer puzzles at once, or play without the judge.
- Anything else, with the response body quoted — the API said something unexpected. `JUDGE_MODEL` in
  `config.py` is pinned to a specific version on purpose; a model name that no longer exists fails
  here.
