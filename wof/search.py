"""Candidate answers from an open-weight LLM, searched token by token under the board's constraint.

How it works (``explore.py`` walks through each step with real output):

1. **Prompt.** A few solved puzzles as examples, then the target's category / hint / year /
   category context (and, by default, its board), ending in ``Answer:``.
2. **Which tokens fit the board.** The answer the model writes is ``" " + board`` long. For every
   character position we precompute which vocabulary tokens could start there: a hidden ``_``
   takes any *uncalled* letter (either case), a revealed letter takes itself (either case),
   punctuation and spaces must match exactly, and nothing may run past the end. The model may
   write ``Burt Lancaster`` or ``BURT LANCASTER``; both fit ``____ _________`` and count as the
   same answer. (Normal case matters: Qwen spells " Burt Lancaster" in 3 tokens it knows well and
   " BURT LANCASTER" in 5 awkward ones.)
3. **Search.** At each step the model gives a probability for every one of its ~150k tokens. We
   keep the model's own (unrenormalized) log-probability of each token and simply skip tokens
   that don't fit. Once an answer fills the board, it must end: we add log P(newline), the
   model's probability that the answer stops there.
   - ``beam_search``: keep the ``beams`` highest-probability partial answers at every step.
     Gives the model's top guesses; their probabilities, renormalized, are the posterior.
   - ``sample``: draw answers at random from the board-fitting tokens and reweight each by
     (model probability) / (probability of drawing it). Noisier, but estimates the whole
     distribution rather than just its top, and also estimates the total probability mass the
     model puts on board-fitting answers, which says how much of it the beams covered.
4. ``AnswerPosterior`` turns the scored answers into letter probabilities (``wof/posterior.py``).
5. ``fill_slots`` is a last pass for the answer the left-to-right search can't reach: once every
   other word of the phrase is settled, try every word of a word list in the one open word and
   score each whole answer.

Because the score is always the model's full log P(answer | prompt), no rescoring is needed:
the probability of a finished answer is the product of its tokens' probabilities (plus the stop).
Renormalizing over board-fitting answers is Bayes' rule, P(answer | clue, board).

One approximation: a string can be written as different token sequences (" Bu"+"rt" vs
" B"+"urt"). Beam search finds whichever paths it finds and sums duplicates; it does not
enumerate every spelling, so probabilities are slightly understated (usually negligibly).
"""

from __future__ import annotations

import copy
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field

import random

from .board import BOARD_PUNCTUATION, FREQUENCY_ORDER, HIDDEN, LETTERS, Board, normalize
from .posterior import AnswerPosterior, Candidate
from .puzzles import Clue, Puzzle

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B"  # base (not instruct) model; "Qwen/Qwen2.5-0.5B" is ~3x faster, a bit dumber

PROMPT_HEADER = (
    "Wheel of Fortune puzzles. Each one lists its category, any hint, the year it aired and the "
    "board (each _ is a hidden letter), then the solution."
)
# Characters a token may contain inside an answer: letters (either case) and board punctuation.
ANSWER_CHARS = set(LETTERS) | set(LETTERS.lower()) | set(BOARD_PUNCTUATION)


# --- text helpers --------------------------------------------------------------------------------

def title_case(answer: str) -> str:
    """ "THE WAY TO A MAN'S HEART" -> "The Way To A Man's Heart" (hyphen parts capitalized too)."""
    def word(w: str) -> str:
        return "-".join(part[:1].upper() + part[1:].lower() for part in w.split("-"))
    return " ".join(word(w) for w in answer.split(" "))


def styled(answer: str, style: str) -> str:
    """How an answer is written in the prompt's examples: 'title' (default), 'upper' or 'lower'."""
    answer = normalize(answer)
    return {"title": title_case, "upper": str.upper, "lower": str.lower}[style](answer)


def show_token(s: str) -> str:
    """A token made readable: leading/trailing spaces as '␣', newlines as '⏎'."""
    return "[" + s.replace(" ", "␣").replace("\n", "⏎") + "]"


CASES = (None, "title", "upper", "lower")


def letter_cases(template: str, case: str | None) -> list[str | None]:
    """For each position of ``template``, the case a letter there must be written in.

    ``None`` = either case (the default: the model may write "Burt Lancaster" or "BURT LANCASTER").
    "title" / "upper" / "lower" lock it to that style, exactly as ``styled()`` writes the examples:
    in title case a letter is uppercase when it starts a word or a hyphen-separated part
    ("Man's", "Jack-In-The-Box"), lowercase otherwise. Non-letter positions get ``None``.

    Why lock it: once letters are revealed, the prompt's board line ("_EEPING THE _AIN THING")
    is nearly the answer in capitals, and copying it is so cheap for the model that all-caps
    answers with arbitrary letters in the blanks ("ZEEPING THE ZAIN THING") crowd the beams out.
    Locked to the examples' title case, copying the board is impossible and every beam is spent
    on answers written the way the model writes real phrases.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}, got {case!r}")
    out: list[str | None] = []
    for i, t in enumerate(template):
        if t != HIDDEN and t not in LETTERS:
            out.append(None)
        elif case is None:
            out.append(None)
        elif case == "upper":
            out.append("U")
        elif case == "lower":
            out.append("L")
        else:  # title: first character of a word or of a hyphen-separated part
            out.append("U" if i == 0 or template[i - 1] in " -" else "L")
    return out


def _char_class(t: str, allowed: str, want: str | None = None) -> str:
    """Regex for what can sit at a board position: hidden -> uncalled letters, else itself.

    ``want``: "U" / "L" restricts a letter to that case; ``None`` allows either.
    """
    if t == HIDDEN:
        if not allowed:
            return "(?!)"
        return {"U": f"[{allowed}]", "L": f"[{allowed.lower()}]"}.get(want, f"[{allowed}{allowed.lower()}]")
    if t in LETTERS:
        return {"U": t, "L": t.lower()}.get(want, f"[{t}{t.lower()}]")
    return re.escape(t)


def token_fits(template: str, allowed: frozenset[str] | str, pos: int, s: str,
               case: str | None = None) -> bool:
    """Can token text ``s`` be written starting at ``template[pos]``? (letters in ``case``, see letter_cases)"""
    if pos + len(s) > len(template):
        return False
    cases = letter_cases(template, case)
    for i, (t, c) in enumerate(zip(template[pos:], s), start=pos):
        if t == HIDDEN:
            if c.upper() not in allowed or c.upper() not in LETTERS:
                return False
        elif t in LETTERS:
            if c.upper() != t:
                return False
        elif c != t:
            return False
        if cases[i] == "U" and c != c.upper() or cases[i] == "L" and c != c.lower():
            return False
    return True


def prefix_ok(template: str, allowed: frozenset[str] | str, text: str, case: str | None = None) -> bool:
    """Can ``text`` be the start of an answer to ``template``?"""
    return token_fits(template, allowed, 0, text, case)


REVEAL_MODES = ("none", "same", "random")


def example_reveals(examples: list[Puzzle], board: Board | None, mode: str = "none",
                    seed: int = 0, max_called: int = 12) -> list[frozenset[str]]:
    """Which letters each example's board shows as called (every occurrence revealed).

    - ``"none"``: every example board is blank, as on an empty board.
    - ``"same"``: the letters called so far on the target ``board``, so each example looks like
      the target does right now ("these letters are showing; write the whole answer").
    - ``"random"``: each example gets its own point in a game: the first 0..``max_called`` letters
      in English-frequency order (E, T, A, O, I, N, ...), roughly what a contestant calls, so some
      land and some miss. Seeded by ``seed`` and the example, so a prompt is reproducible.

    Why: with blank example boards the model never sees a half-revealed board next to its answer,
    so it has to work out on its own what "_EEPING THE _AIN THING" asks for.
    """
    if mode not in REVEAL_MODES:
        raise ValueError(f"mode must be one of {REVEAL_MODES}, got {mode!r}")
    if mode == "none":
        return [frozenset()] * len(examples)
    if mode == "same":
        called = board.called if board is not None else frozenset()
        return [called] * len(examples)
    out = []
    for ex in examples:
        rng = random.Random(f"{seed}|{ex.answer}")
        out.append(frozenset(FREQUENCY_ORDER[:rng.randint(0, max_called)]))
    return out


def called_line(board: Board) -> str:
    """ "Called: A E T (not in puzzle: B)" for a board with letters called, "" otherwise."""
    if not board.called:
        return ""
    missed = sorted(board.absent)
    return f"Called: {' '.join(sorted(board.called))}" + (f" (not in puzzle: {' '.join(missed)})" if missed else "")


def build_prompt(clue: Clue, board: Board | None = None, examples: list[Puzzle] = (),
                 style: str = "title", show_board: bool = True,
                 example_called: list[frozenset[str]] | None = None, show_called: bool = False) -> str:
    """The text the model continues. Ends in "Answer:"; the model writes " <answer>\\n".

    ``show_board``: include the board ("Board: ____ _________") for the examples and the target.
    The board constraint is enforced during the search either way; showing it lets the model plan
    ahead (it knows the next word must be 9 letters before it starts it). Note that the model is
    weak at counting underscores, so this helps less than you'd hope.

    ``example_called``: for each example, the letters its board shows as called (see
    ``example_reveals``); default none, i.e. blank example boards.
    ``show_called``: add a "Called: A E T (not in puzzle: B)" line under any board with letters
    called, so the model is told which letters are out, not only which are showing.
    """
    def block(c: Clue, b: Board | None, answer: str | None) -> list[str]:
        lines = [f"Category: {c.category}"]
        if c.hint:
            lines.append(f"Hint: {c.hint}")
        if c.year:
            lines.append(f"Year: {c.year}")
        if c.context:
            lines.append(f"Context: {c.context}")
        if show_board and b is not None:
            lines.append(f"Board: {b.template}")
            if show_called and called_line(b):
                lines.append(called_line(b))
        lines.append("Answer:" + (f" {styled(answer, style)}" if answer is not None else ""))
        return lines

    example_called = example_called or [frozenset()] * len(examples)
    if len(example_called) != len(examples):
        raise ValueError("example_called needs one set of letters per example")
    lines = [PROMPT_HEADER, ""]
    for ex, called in zip(examples, example_called):
        lines += block(ex.clue, ex.board(set(called)), ex.answer) + [""]
    lines += block(clue, board, None)
    return "\n".join(lines)


# --- results -------------------------------------------------------------------------------------

@dataclass
class TokenStep:
    text: str          # the token's characters
    logprob: float     # the model's log P(this token | everything before), NOT renormalized
    fit_logmass: float  # log of the total probability of all tokens that fit the board at this step

    @property
    def prob(self) -> float:
        return math.exp(self.logprob)


@dataclass
class Path:
    """One answer the search produced, with how the model got there token by token."""

    text: str                     # as the model wrote it, e.g. "Burt Lancaster"
    steps: list[TokenStep]
    end_logprob: float = 0.0      # log P(the answer stops here: a newline next)
    q_logprob: float = 0.0        # sampling only: log P(the sampler drew exactly this path)

    @property
    def answer(self) -> str:
        return normalize(self.text)

    @property
    def logprob(self) -> float:
        """log P(answer | prompt) along this path: every token plus the stop."""
        return sum(s.logprob for s in self.steps) + self.end_logprob

    @property
    def log_weight(self) -> float:
        """Sampling only: log [ P(path) / P(drawing it) ], its importance weight."""
        return self.logprob - self.q_logprob


@dataclass
class TokenTable:
    """The model's next-token prediction at one point: its top tokens and which of them fit."""

    prefix: str
    rows: list[tuple[str, float, bool]]  # (token text, P, fits the board)
    fit_mass: float                      # total P of every token that fits the board
    top_fitting: list[tuple[str, float]]  # the most likely tokens among those that fit
    ends_here: bool = False              # the prefix already fills the board: only a newline fits


@dataclass
class SearchResult:
    method: str
    prompt: str
    board: Board
    paths: list[Path]
    trace: list[dict] = field(default_factory=list)  # beam search: the beams at each step
    n_draws: int = 0  # sampling only: draws attempted, incl. dead ends that produced no answer (weight 0)
    # beam search: for each word of the board (by index), every word the search wrote there -> the best
    # log P of a partial answer that had just written it. ``LLMSolver.fill_slots`` draws on these.
    words: list[dict[str, float]] = field(default_factory=list)

    def answer_logweights(self) -> dict[str, float]:
        """Each distinct answer's log-weight, spellings/tokenizations of the same answer summed.

        beam:   log P(answer | prompt)
        sample: log of the summed importance weights of every draw of that answer
        """
        out: dict[str, list[float]] = defaultdict(list)
        for p in self.paths:
            out[p.answer].append(p.logprob if self.method == "beam" else p.log_weight)
        return {a: _logsumexp(v) for a, v in out.items()}

    def posterior(self) -> AnswerPosterior:
        return AnswerPosterior(self.board, [Candidate(a, w, f"llm-{self.method}")
                                            for a, w in self.answer_logweights().items()])

    def fit_mass_estimate(self) -> float:
        """Sampling only: estimated total P(model writes SOME board-fitting answer | prompt)."""
        if not self.paths:
            return 0.0
        n = self.n_draws or len(self.paths)
        return math.exp(_logsumexp([p.log_weight for p in self.paths]) - math.log(n))

    def effective_sample_size(self) -> float:
        """Sampling only: how many equally-weighted draws the weighted draws are worth."""
        lw = [p.log_weight for p in self.paths]
        if not lw:
            return 0.0
        return math.exp(2 * _logsumexp(lw) - _logsumexp([2 * x for x in lw]))


@dataclass
class FillResult:
    """What ``LLMSolver.fill_slots`` added, and per open board word what it tried.

    ``slots``: one dict per board word filled: ``word`` (index), ``pattern`` (as shown on the
    board), ``confidence`` (the weakest other hidden word's top share), ``locked`` ((index, word,
    share) for every other hidden word), ``contexts`` (the answers whose other words were used),
    ``tried`` (new answers scored), ``best`` ((answer, log P) of the best fill, kept or not) and
    ``added`` (the fills that made it into ``paths``). ``top_logprob``: the search's best answer
    before filling, to compare against. ``done``: a key per (board word, its pattern, context)
    filled this time, to pass back as ``skip`` on the next turn. ``skipped``: open board words left
    alone this time, each ``{"word", "pattern", "reason"}`` with reason ``"unchanged"`` (filled
    the same way on an earlier turn) or ``"max_slots"`` (this turn's quota was used up).
    """

    paths: list[Path]
    slots: list[dict]
    top_logprob: float = float("nan")
    done: set = field(default_factory=set)
    skipped: list = field(default_factory=list)


def board_word(board: Board, i: int) -> str:
    """Board word ``i`` as shown (``"_O_E"``)."""
    return board.template.split(" ")[i]


def first_word(text: str) -> str:
    """The first word of an answer-so-far, case-folded ("" before any letter): " Kfc Man" -> "KFC"."""
    words = text.split()
    return normalize(words[0]) if words else ""


def open_word(text: str, template: str) -> str | None:
    """The first word of ``text`` that the board doesn't already give away, case-folded.

    ``template`` is the board as the model writes it (``" " + board.template``), aligned with
    ``text`` character by character. A word whose letters are all revealed (or a word of only
    punctuation) is the same in every answer, so it can't tell answers apart; this skips those and
    returns the first word with a hidden slot, as far as ``text`` has written it. ``None`` if
    ``text`` hasn't reached such a word yet. On an empty board this is just ``first_word``.
    """
    start = 0
    for word in template.split(" "):
        end = start + len(word)
        if HIDDEN in word:
            if len(text) <= start:
                return None
            return normalize(text[start:end])
        start = end + 1
    return None


def _logsumexp(xs: list[float]) -> float:
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return m
    return m + math.log(sum(math.exp(x - m) for x in xs))


def word_spans(template: str) -> list[tuple[int, int]]:
    """Each board word's (start, end) in ``template`` = ``" " + board.template`` (the answer as written)."""
    spans, start = [], 1
    for word in template[1:].split(" "):
        spans.append((start, start + len(word)))
        start += len(word) + 1
    return spans


def word_index(spans: list[tuple[int, int]], reach: int) -> int:
    """Which word (of ``word_spans``) a text ``reach`` characters long is in (a space counts with the next word)."""
    for i, (_, end) in enumerate(spans):
        if reach <= end:
            return i
    return len(spans) - 1


class _WordBank:
    """Every word a search writes in each board word, with the best partial-answer log P that wrote it."""

    def __init__(self, template: str):
        self.spans = word_spans(template)
        self.words: list[dict[str, float]] = [{} for _ in self.spans]

    def add(self, text: str, n_before: int, score: float) -> None:
        """``text`` just grew from ``n_before`` characters: record any word it completed."""
        for i, (start, end) in enumerate(self.spans):
            if n_before < end <= len(text):
                w = text[start:end]
                if score > self.words[i].get(w, -math.inf):
                    self.words[i][w] = score


# --- one shared copy of the prompt in the search's KV cache ----------------------------------------

_SHARED_PREFIX_LAYER = None


def _shared_prefix_layer_class():
    """A ``transformers`` cache layer that keeps the prompt once and only the answer tokens per row.

    The search keeps hundreds of partial answers alive at once, and an ordinary cache stores the
    whole prompt again for every one of them: with Qwen2.5-1.5B (~57 KB per token in float32) and a
    ~600-token prompt that's ~34 MB per row, ~10 GB at 64 beams (~300 rows), twice that while rows
    are reordered. This layer stores the prompt's keys/values once (``[1, heads, prompt, dim]``)
    and per row only what the answer added. When a layer runs, it joins the shared prompt
    (broadcast, not copied) to each row's answer part and hands that to attention; the joined copy
    exists for one layer at a time and is freed straight after. Same numbers as the plain cache.
    """
    global _SHARED_PREFIX_LAYER
    if _SHARED_PREFIX_LAYER is None:
        import torch
        from transformers.cache_utils import DynamicLayer

        class SharedPrefixLayer(DynamicLayer):
            def __init__(self, prefix_keys, prefix_values):
                super().__init__()
                self.prefix_keys, self.prefix_values = prefix_keys, prefix_values
                # The rows' own part starts empty but 4-D, so reordering rows (index_select on dim 0)
                # works before anything has been added to it.
                self.keys, self.values = prefix_keys[:, :, :0], prefix_values[:, :, :0]
                self.device = prefix_keys.device
                self.is_initialized = True

            def update(self, key_states, value_states, *args, **kwargs):
                self.keys = torch.cat([self.keys, key_states], dim=-2)
                self.values = torch.cat([self.values, value_states], dim=-2)
                rows = self.keys.shape[0]
                return (torch.cat([self.prefix_keys.expand(rows, -1, -1, -1), self.keys], dim=-2),
                        torch.cat([self.prefix_values.expand(rows, -1, -1, -1), self.values], dim=-2))

            def get_seq_length(self) -> int:
                return self.prefix_keys.shape[-2] + self.keys.shape[-2]

        _SHARED_PREFIX_LAYER = SharedPrefixLayer
    return _SHARED_PREFIX_LAYER


def share_prompt_cache(cache) -> bool:
    """Turn a prompt's KV cache into one whose rows share a single copy of the prompt, in place.

    Returns ``False`` (and changes nothing) if this ``transformers`` version's cache isn't the
    per-layer ``DynamicLayer`` kind, so the caller keeps the plain cache: same results, more memory.
    """
    layers = getattr(cache, "layers", None)
    if not layers:
        return False
    try:
        from transformers.cache_utils import DynamicLayer
        cls = _shared_prefix_layer_class()
    except ImportError:
        return False
    if not all(type(layer) is DynamicLayer and layer.keys is not None and layer.keys.dim() == 4
               for layer in layers):
        return False
    for i, layer in enumerate(layers):
        layers[i] = cls(layer.keys, layer.values)
    return True


# --- the model -----------------------------------------------------------------------------------

class LLMSolver:
    """An open-weight causal LM plus everything needed to search it under a board constraint."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if dtype is None:  # float32 on CPU/Apple (accurate log-probs, bf16 is slow on many CPUs); native on CUDA
            dtype = "auto" if device == "cuda" else torch.float32
        self.model_name, self.device = model_name, device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        self._load_vocab()
        self._legal_cache: dict = {}
        # beam search, sampling and score_many: keep one copy of the prompt in the KV cache for every row, not
        # one each (``share_prompt_cache``). False = the plain cache, for comparison; same results, more memory.
        self.share_prompt = True

    # vocabulary ----------------------------------------------------------------------------------

    def _load_vocab(self) -> None:
        """Every token's text, split into those usable inside an answer and those that end one.

        A lone token decodes without its leading space in some tokenizers, so each one is decoded
        after a fixed sentinel token and the sentinel's text is cut off.
        """
        sentinel = self.tok.encode("A", add_special_tokens=False)
        base = self.tok.decode(sentinel)
        n = len(self.tok)
        texts = self.tok.batch_decode([sentinel + [i] for i in range(n)])
        self.token_text: dict[int, str] = {}
        end_ids = set()
        for i, t in enumerate(texts):
            s = t[len(base):]
            if not s:
                continue
            if s.startswith("\n"):
                end_ids.add(i)
            elif set(s) <= ANSWER_CHARS:
                self.token_text[i] = s
        if self.tok.eos_token_id is not None:
            end_ids.add(self.tok.eos_token_id)
        self.end_ids = self.torch.tensor(sorted(end_ids), device=self.device)
        # Group answer tokens by length, as one newline-joined string per length, for fast regex scans.
        by_len: dict[int, list[int]] = defaultdict(list)
        for i, s in self.token_text.items():
            by_len[len(s)].append(i)
        self._by_len = {}  # length -> ({character offset in joined: token id}, joined)
        for length, ids in by_len.items():
            joined = "\n".join(self.token_text[i] for i in ids)
            self._by_len[length] = ({k * (length + 1): i for k, i in enumerate(ids)}, joined)

    def legal_ids(self, board: Board, case: str | None = None) -> list:
        """For each character position of " " + board, the ids of tokens that can start there.

        ``case``: None = letters in either case; "title" / "upper" / "lower" = only that style
        (``letter_cases``).
        """
        torch = self.torch
        template = " " + board.template
        key = (template, board.called, case)
        if key in self._legal_cache:
            return self._legal_cache[key]
        allowed = "".join(c for c in LETTERS if c not in board.called)
        classes = [_char_class(t, allowed, w) for t, w in zip(template, letter_cases(template, case))]
        out = []
        for pos in range(len(template)):
            ids = []
            for length, (at_offset, joined) in self._by_len.items():
                if pos + length > len(template):
                    continue
                pattern = re.compile("^" + "".join(classes[pos:pos + length]) + "$", re.M)
                ids.extend(at_offset[m.start()] for m in pattern.finditer(joined))
            out.append(torch.tensor(sorted(ids), dtype=torch.long, device=self.device))
        self._legal_cache[key] = out
        return out

    # model calls ---------------------------------------------------------------------------------

    def _encode(self, text: str):
        return self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)

    def _cache_kw(self) -> str:
        """transformers >= 5.24 renamed the KV-cache argument from ``past_key_values`` to ``cache``."""
        import inspect

        return "cache" if "cache" in inspect.signature(self.model.forward).parameters else "past_key_values"

    @staticmethod
    def _cache_of(out):
        return getattr(out, "cache", None) or out.past_key_values

    def _start(self, prompt: str):
        """Run the prompt once: (log-probs of the first answer token [1, vocab], KV cache)."""
        with self.torch.no_grad():
            out = self.model(self._encode(prompt), use_cache=True)
        return self.torch.log_softmax(out.logits[:, -1].float(), dim=-1), self._cache_of(out)

    def _step(self, token_ids, cache):
        """Feed one token per row: (log-probs of the next token [rows, vocab], cache)."""
        with self.torch.no_grad():
            out = self.model(token_ids[:, None], use_cache=True, **{self._cache_kw(): cache})
        return self.torch.log_softmax(out.logits[:, -1].float(), dim=-1), self._cache_of(out)

    def _end_logprob(self, logp_row) -> float:
        return float(self.torch.logsumexp(logp_row[self.end_ids], dim=0))

    # one prediction, in full ---------------------------------------------------------------------

    def next_tokens(self, prompt: str, board: Board, prefix: str = "", top: int = 15,
                    case: str | None = None) -> TokenTable:
        """What the model predicts right after ``prompt`` + " " + ``prefix`` (prefix = answer so far).

        Shows the model's raw top tokens (most of which usually DON'T fit the board), how much of
        its probability falls on tokens that do fit, and the top fitting ones.
        """
        torch = self.torch
        template = " " + board.template
        text = " " + prefix if prefix else ""
        if not prefix_ok(template, frozenset(c for c in LETTERS if c not in board.called), text, case):
            raise ValueError(f"prefix {prefix!r} doesn't fit the board {board.template!r}")
        ids = self._encode(prompt)
        if text:
            ids = torch.cat([ids, self._encode(text)], dim=1)
        with torch.no_grad():
            logp = torch.log_softmax(self.model(ids).logits[0, -1].float(), dim=-1)
        pos = len(text)
        if pos == len(template):
            legal = self.end_ids
        else:
            legal = self.legal_ids(board, case)[pos]
        legal_set = set(legal.tolist())
        vals, idx = logp.topk(top)
        rows = [(self.tok.decode([i]), math.exp(v), i in legal_set) for v, i in zip(vals.tolist(), idx.tolist())]
        lv, li = logp[legal].topk(min(top, len(legal)))
        top_fit = [(self.token_text.get(int(legal[j]), self.tok.decode([int(legal[j])])), math.exp(v))
                   for v, j in zip(lv.tolist(), li.tolist())]
        return TokenTable(prefix, rows, float(torch.logsumexp(logp[legal], 0).exp()), top_fit,
                          ends_here=pos == len(template))

    def score(self, prompt: str, board: Board, text: str, case: str | None = None) -> Path:
        """Force the model to write ``text`` (e.g. the true answer): its probability, token by token.

        ``case`` only changes each step's fit mass (which tokens count as fitting), not the score.
        """
        torch = self.torch
        p_ids = self._encode(prompt)
        a_ids = self._encode(" " + text)
        ids = torch.cat([p_ids, a_ids], dim=1)
        with torch.no_grad():
            logp = torch.log_softmax(self.model(ids).logits[0].float(), dim=-1)
        n_prompt = p_ids.shape[1]
        return self._forced_path(text, a_ids[0].tolist(), logp[n_prompt - 1:], self.legal_ids(board, case))

    def _forced_path(self, text: str, token_ids: list[int], logp, legal) -> Path:
        """``text`` written as ``token_ids``; ``logp[k]`` = log-probs of token k (``logp[len]`` = after the last)."""
        torch = self.torch
        steps, pos = [], 0
        for k, tid in enumerate(token_ids):
            row = logp[k]
            s = self.token_text.get(tid) or self.tok.decode([tid])
            fit = float(torch.logsumexp(row[legal[pos]], 0)) if pos < len(legal) and len(legal[pos]) else -math.inf
            steps.append(TokenStep(s, float(row[tid]), fit))
            pos += len(s)
        return Path(text, steps, end_logprob=self._end_logprob(logp[len(token_ids)]))

    def score_many(self, prompt: str, board: Board, texts: list[str], case: str | None = None,
                   batch: int = 16, prefixes: list[str] | None = None) -> list[Path]:
        """``score`` for many answers at once: the prompt runs once, the answers in batches after it.

        Same numbers as calling ``score`` on each (up to float rounding).

        ``prefixes``: optionally one per text, a start that ``" " + text`` begins with and that
        other texts share (``fill_slots`` passes the words before the one it swaps, e.g. " A" for
        every " A ____ Come True"). Each distinct prefix then runs through the model once, after the
        prompt, and only the rest of each answer runs per answer: same numbers, less work. A text
        whose tokens don't begin with its prefix's tokens (the tokenizer merged across the cut) is
        scored the plain way, so this can never change a score.
        """
        torch = self.torch
        if not texts:
            return []
        legal = self.legal_ids(board, case)
        logp0, cache0 = self._start(prompt)
        enc = [self._encode(" " + t)[0].tolist() for t in texts]
        # Group the texts by the prefix they'll actually share ("" = none: start right after the prompt)
        prefix_ids: dict[str, list[int]] = {}
        groups: dict[str, list[int]] = defaultdict(list)
        for k, e in enumerate(enc):
            p = prefixes[k] if prefixes else ""
            if p and p not in prefix_ids:
                prefix_ids[p] = self._encode(p)[0].tolist()
            ids = prefix_ids.get(p, [])
            ok = p and (" " + texts[k]).startswith(p) and len(e) > len(ids) and e[:len(ids)] == ids
            groups[p if ok else ""].append(k)
        out_paths: list[Path | None] = [None] * len(texts)
        for p, ks in groups.items():
            if not p:
                start_rows, cache_start, m = logp0, cache0, 0
            else:
                ids = prefix_ids[p]
                m = len(ids)
                cache_start = copy.deepcopy(cache0)
                with torch.no_grad():
                    o = self.model(torch.tensor([ids], device=self.device), use_cache=True,
                                   **{self._cache_kw(): cache_start})
                cache_start = self._cache_of(o)
                # row k = log-probs of token k: the prompt's last row, then one per prefix token
                start_rows = torch.cat([logp0, torch.log_softmax(o.logits[0].float(), dim=-1)], dim=0)
            for at in range(0, len(ks), batch):
                chunk = ks[at:at + batch]
                rest = [enc[k][m:] for k in chunk]
                T = max(len(r) for r in rest)
                filler = int(self.end_ids[0])
                tokens = torch.tensor([r + [filler] * (T - len(r)) for r in rest], device=self.device)
                cache = copy.deepcopy(cache_start)
                if self.share_prompt:
                    share_prompt_cache(cache)
                cache.batch_repeat_interleave(len(chunk))
                with torch.no_grad():  # right padding: a real token never attends to the padding after it
                    out = self.model(tokens, use_cache=True, **{self._cache_kw(): cache})
                logp = torch.log_softmax(out.logits.float(), dim=-1)
                for b, (k, r) in enumerate(zip(chunk, rest)):
                    rows = torch.cat([start_rows, logp[b, :len(r)]], dim=0)
                    out_paths[k] = self._forced_path(texts[k], enc[k], rows, legal)
        return out_paths

    def fill_slots(self, prompt: str, board: Board, result: SearchResult, vocab, *,
                   min_confidence: float = 0.8, contexts: int = 2, max_words: int = 300, keep: int = 8,
                   case: str | None = None, batch: int = 16, exclude: set[str] = frozenset(),
                   max_slots: int | None = None, skip=frozenset(), share_prefix: bool = True) -> FillResult:
        """Fill one open board word with every word that fits it, holding the rest of the phrase fixed.

        For "A WISH COME TRUE" the search finds "A Fish Come True" / "A City Come True": "COME TRUE"
        is settled, word 2 isn't, and "Wish" never comes up because " A W" is an unlikely *start*
        and the search works left to right. Scored whole, the truth beats both by ~80x. So:

        - per-word confidence: ``slots.word_marginals`` over the answers found (``result``);
        - every board word with a hidden letter whose OTHER hidden words are all at
          ``min_confidence`` or more is open (``slots.open_slots``; the answers can be sure of the
          open word itself, e.g. "FISH" at 99%, and it's still worth checking);
        - into it go the words that fit it, written in the case of the word they replace: the most
          common ones from ``vocab`` (``slots.slot_words``, e.g. ``slots.build_vocab(examples)``)
          first, then the words the search itself wrote there (``result.words``) to fill the rest of
          the budget. ``max_words`` is that whole budget, so the work per slot is bounded: the
          search's own word bank can run to hundreds of words on a wide search, and scoring every
          one of them is what used to make a single turn take minutes;
        - the rest of the phrase comes from the best ``contexts`` answers that differ outside the
          open word (usually one: "A City Come True" and "A Fish Come True" are the same context);
        - every new answer that fits the board is scored whole (``score_many``: the model's full
          log P, the same footing as the search's own answers), and the best ``keep`` that score
          at least as well as the worst answer ``result`` kept are returned.

        This only changes where the solver looks, never what it believes: the new answers join
        the search's answers, nothing is removed, so if the "settled" words were wrong, the old
        answers are still there. ``exclude``: answers (normalized) to leave out, e.g. ones already
        added some other way. ``FillResult.slots`` says what was tried, for the turn printout.

        Speed (every fill is a whole answer through the model, so a turn can mean ~1,000 of them):

        - ``max_words`` and ``contexts`` bound it directly: at most ``max_words x contexts`` answers
          are scored per board word filled. A rough measurement on a laptop CPU, batch 32: ~300
          fills is ~20s with a 0.5B model and ~2.5 minutes with a 1.5B one.
        - ``max_slots``: fill at most this many board words this call, the least settled first
          (lowest share for its own most likely word). ``None`` = every open word.
        - ``skip``: keys from earlier turns' ``FillResult.done``. A (board word, its pattern on the
          board, context) already filled is left alone: nothing about it changed, and what it found
          then is still among the answers if it was any good (``LLMTurnSolver`` carries last turn's
          best answers forward). A new letter in that word or a different context fills it again.
        - ``share_prefix``: score the words before the open one once per context rather than once
          per fill (``score_many(prefixes=)``). Same numbers either way.
        """
        from .slots import match_case, open_slots, slot_words, top_word, word_marginals

        weights = result.answer_logweights()
        if not weights or not result.paths:
            return FillResult([], [])
        top = max(weights.values())
        marginals = word_marginals({a: math.exp(w - top) for a, w in weights.items()}, board)
        slots = open_slots(marginals, board, min_confidence)
        if not slots:
            return FillResult([], [])
        template = " " + board.template
        spans = word_spans(template)
        floor = min(p.logprob for p in result.paths)
        ranked = sorted(result.paths, key=lambda p: -p.logprob)
        seen = set(weights) | set(exclude)
        variants: dict[str, tuple[int, str, str]] = {}  # normalized -> (slot, text to score, shared prefix)
        report = []
        own = {i: top_word(marginals[i])[1] for i, _ in slots}
        done: set = set()
        skipped: list[dict] = []
        n_filled = 0
        for i, conf in sorted(slots, key=lambda s: own[s[0]]):  # least settled word first
            pattern = board_word(board, i)
            if max_slots is not None and n_filled >= max_slots:
                skipped.append({"word": i, "pattern": pattern, "reason": "max_slots"})
                continue
            a, b = spans[i]
            picked, keys = [], set()
            for p in ranked:  # the best answers that differ outside the open word
                text = " " + p.text.lstrip(" ")
                if len(text) != len(template):
                    continue
                key = normalize(text[:a] + "_" * (b - a) + text[b:])
                if key not in keys:
                    keys.add(key)
                    picked.append((text, key))
                if len(picked) >= contexts:
                    break
            fresh = [(text, key) for text, key in picked if (i, pattern, key) not in skip]
            if picked and not fresh:
                skipped.append({"word": i, "pattern": pattern, "reason": "unchanged"})
                continue
            words = slot_words(board, i, vocab, max_words)
            searched = result.words[i] if i < len(result.words) else {}  # words the search wrote here
            have = set(words)
            for w in slot_words(board, i, {normalize(w): s for w, s in searched.items()}):
                if len(words) >= max_words:
                    break  # ``max_words`` is the whole budget for this slot: see the docstring
                if w not in have:
                    words.append(w)
                    have.add(w)
            n_tried = 0
            for text, key in fresh:
                done.add((i, pattern, key))
                old = text[a:b]
                for w in words:
                    t = text[:a] + match_case(w, old) + text[b:]
                    norm = normalize(t)
                    if norm in seen or norm in variants or not board.consistent(norm):
                        continue
                    variants[norm] = (i, t[1:], t[:a - 1] if share_prefix else "")
                    n_tried += 1
            n_filled += n_tried > 0
            locked = [(j, *top_word(marginals[j])) for j in range(len(marginals))
                      if j != i and HIDDEN in board_word(board, j)]
            report.append({"word": i, "pattern": pattern, "confidence": conf, "locked": locked,
                           "contexts": [t[1:] for t, _ in fresh], "tried": n_tried, "best": None, "added": []})
        if not variants:
            return FillResult([], report, done=done, skipped=skipped)
        texts = [t for _, t, _ in variants.values()]
        scored = self.score_many(prompt, board, texts, case=case, batch=batch,
                                 prefixes=[p for _, _, p in variants.values()])
        by_slot: dict[int, list[Path]] = defaultdict(list)
        for (i, _, _), path in zip(variants.values(), scored):
            by_slot[i].append(path)
        good = sorted((p for p in scored if p.logprob >= floor), key=lambda p: -p.logprob)[:keep]
        kept = {id(p) for p in good}
        for r in report:
            paths = sorted(by_slot.get(r["word"], []), key=lambda p: -p.logprob)
            if paths:
                r["best"] = (paths[0].answer, paths[0].logprob)
            r["added"] = [p.answer for p in paths if id(p) in kept]
        return FillResult(good, report, top_logprob=top, done=done, skipped=skipped)

    # search --------------------------------------------------------------------------------------

    def beam_search(self, prompt: str, board: Board, beams: int = 16, keep: int | None = None,
                    trace_top: int = 8, max_per_first_word: int | None = None,
                    case: str | None = None, expand: int | None = None) -> SearchResult:
        """The most likely board-fitting answers, found by beam search under the board constraint.

        The search is **position-synchronous**: only partial answers that have written the same
        number of board characters are ever compared, and at most ``beams`` are kept at each
        position. Every step extends the partial answers that are furthest behind; the ones ahead
        wait for the others to catch up. That matters because a partial answer's log P only ever
        goes down as it grows, so comparing " Keeping" (8 characters, its cost paid) with " Me"
        (3 characters, most of its cost still to come) is unfair to the longer one. Ranking every
        live beam together instead, as plain beam search does, threw out whole-word starts in
        favour of fragments that turned into junk ("Meehahs Hae Hahs", log P -53, while the true
        "Keeping The Main Thing The Main Thing" scored -36.5 and was never found at all).

        A beam that fills the board is finished by adding log P(stop). Returns up to ``keep``
        (default ``beams``) finished answers, best first.

        ``max_per_first_word``: at most this many beams per position may share a first word
        (case-folded, e.g. "KFC" in " KFC Manager" and " Kfc Colonel"), so one likely first word
        can't take most of the beams with its endings. "First word" means the first word that
        still has a hidden letter (``open_word``): once letters are called, a fully revealed first
        word (often just "A" or "THE") is shared by every answer, and capping on it would cap the
        whole search. Scores are unchanged; the cap only changes which answers get found.
        ``None`` = no cap, which is the better setting with a generous ``expand``.

        ``case``: write the answer only in this style ("title" / "upper" / "lower", see
        ``letter_cases``); ``None`` = either case. Use the examples' style ("title") once the board
        shows revealed letters, or the model copies the board line in capitals.

        ``expand``: how many next tokens each partial answer tries (its likeliest board-fitting
        ones); ``None`` = ``beams``. The extensions still compete for the ``beams`` places at their
        position, so this only lets a lower-ranked token that leads somewhere likely through. Cheap:
        the model only runs on the partial answers kept, so more tokens tried costs little.

        Everything shares one batched KV cache, holding the prompt once for every row
        (``share_prompt_cache``; ``LLMSolver.share_prompt``). A waiting row is fed a padding token
        that the attention mask hides, and each row's real tokens get their own position ids, so
        rows with different numbers of tokens still run in one batch (the same trick as
        left-padded batch generation).

        The result's ``words`` lists every word the search wrote in each board word, which
        ``fill_slots`` draws on.
        """
        torch = self.torch
        keep = keep or beams
        template = " " + board.template
        L = len(template)
        legal = self.legal_ids(board, case)
        logp0, cache = self._start(prompt)
        n_prompt = cache.get_seq_length()
        if self.share_prompt:
            share_prompt_cache(cache)  # one copy of the prompt for every row (see the function)
        filler = int(self.end_ids[0])
        # One row per partial answer. logp: its next-token log-probs (from its last real token).
        rows: list[dict] = [{"score": 0.0, "text": "", "steps": [], "n": 0}]
        logp = logp0
        mask = torch.ones(1, n_prompt, dtype=torch.long, device=self.device)
        finished: list[Path] = []
        trace: list[dict] = []
        capped_total = 0
        bank = _WordBank(template)
        while rows:
            p_min = min(len(r["text"]) for r in rows)
            # Extend every row at p_min; the rest wait.
            cands = []  # (score, text, steps, parent row, token id or None)
            for i, r in enumerate(rows):
                if len(r["text"]) != p_min:
                    cands.append((r["score"], r["text"], r["steps"], i, None))
                    continue
                ids = legal[p_min]
                if len(ids) == 0:
                    continue
                row = logp[i, ids]
                fit = float(torch.logsumexp(row, 0))
                vals, j = row.topk(min(expand or beams, len(ids)))
                for v, jj in zip(vals.tolist(), j.tolist()):
                    tid = int(ids[jj])
                    s = self.token_text[tid]
                    text = r["text"] + s
                    cands.append((r["score"] + v, text, r["steps"] + [TokenStep(s, v, fit)], i, tid))
                    bank.add(text, p_min, r["score"] + v)
            # Keep the best `beams` at each position (first-word cap applied within each position).
            cands.sort(key=lambda c: -c[0])
            per_pos: dict[int, int] = defaultdict(int)
            per_word: dict[tuple[int, str], int] = defaultdict(int)
            kept, capped = [], 0
            for c in cands:
                pos = len(c[1])
                if per_pos[pos] >= beams:
                    continue
                if max_per_first_word:
                    key = open_word(c[1], template)
                    if key is not None and per_word[(pos, key)] >= max_per_first_word:
                        capped += 1
                        continue
                    per_word[(pos, key)] += 1
                per_pos[pos] += 1
                kept.append(c)
            capped_total += capped
            at_min = sorted((c for c in kept if c[4] is not None), key=lambda c: -c[0])
            trace.append({
                "step": len(trace),
                "position": p_min,
                "live": [(c[1], c[0]) for c in at_min[:trace_top]],
                "n_live": len(kept),
                "finished": [(p.text, p.logprob) for p in sorted(finished, key=lambda p: -p.logprob)[:trace_top]],
                "n_finished": len(finished),
                "capped": capped,
            })
            if not kept:
                break
            # One batched forward: new rows feed their token, waiting rows a masked filler.
            parents = torch.tensor([c[3] for c in kept], device=self.device)
            cache.reorder_cache(parents)
            mask = mask[parents]
            new = torch.tensor([c[4] is not None for c in kept], device=self.device)
            mask = torch.cat([mask, new.long()[:, None]], dim=1)
            n_real = [rows[c[3]]["n"] for c in kept]
            tokens = torch.tensor([c[4] if c[4] is not None else filler for c in kept], device=self.device)
            pos_ids = torch.tensor([n_prompt + n for n in n_real], device=self.device)[:, None]
            with torch.no_grad():
                out = self.model(tokens[:, None], attention_mask=mask, position_ids=pos_ids, use_cache=True,
                                 **{self._cache_kw(): cache})
            step_logp = torch.log_softmax(out.logits[:, -1].float(), dim=-1)
            cache = self._cache_of(out)
            old_logp = logp[parents]
            logp = torch.where(new[:, None], step_logp, old_logp)
            rows = [{"score": c[0], "text": c[1], "steps": c[2], "n": n + (c[4] is not None)}
                    for c, n in zip(kept, n_real)]
            # Rows that just filled the board are finished: add log P(stop), drop them.
            done = [i for i, r in enumerate(rows) if len(r["text"]) == L]
            for i in done:
                finished.append(Path(rows[i]["text"], rows[i]["steps"], end_logprob=self._end_logprob(logp[i])))
            if done:
                stay = [i for i in range(len(rows)) if len(rows[i]["text"]) < L]
                rows = [rows[i] for i in stay]
                if stay:
                    idx = torch.tensor(stay, device=self.device)
                    cache.reorder_cache(idx)
                    mask, logp = mask[idx], logp[idx]
        finished.sort(key=lambda p: -p.logprob)
        return SearchResult("beam", prompt, board, finished[:keep], trace, words=bank.words)

    def sample(self, prompt: str, board: Board, n: int = 200, batch: int = 50,
               temperature: float = 1.0, seed: int = 0, case: str | None = None) -> SearchResult:
        """``n`` answers drawn at random, each token from the board-fitting tokens only.

        Draw probabilities are the model's (over fitting tokens only, renormalized, then sharpened
        or flattened by ``temperature``). Each draw records both its true log P and its log P of
        being drawn, so ``SearchResult.posterior()`` can importance-weight it back to the model's
        actual distribution over board-fitting answers. ``case`` as in ``beam_search``.
        """
        torch = self.torch
        gen = torch.Generator(device="cpu").manual_seed(seed)
        legal = self.legal_ids(board, case)
        L = len(" " + board.template)
        logp0, cache0 = self._start(prompt)
        filler = int(self.end_ids[0])
        paths: list[Path] = []
        for start in range(0, n, batch):
            bs = min(batch, n - start)
            cache = copy.deepcopy(cache0)
            if self.share_prompt:
                share_prompt_cache(cache)
            cache.batch_repeat_interleave(bs)
            logp = logp0.expand(bs, -1)
            texts, steps, q, done = [""] * bs, [[] for _ in range(bs)], [0.0] * bs, [False] * bs
            for _ in range(L + 1):
                nxt = []
                for r in range(bs):
                    if done[r]:
                        nxt.append(filler)
                        continue
                    pos = len(texts[r])
                    if pos == L:
                        paths.append(Path(texts[r], steps[r], self._end_logprob(logp[r]), q[r]))
                        done[r] = True
                        nxt.append(filler)
                        continue
                    ids = legal[pos]
                    if len(ids) == 0:  # dead end: no token fits (rare); drop the draw
                        done[r] = True
                        nxt.append(filler)
                        continue
                    row = logp[r, ids]
                    qrow = torch.log_softmax(row / temperature, 0)
                    j = int(torch.multinomial(qrow.exp().cpu(), 1, generator=gen))
                    tid = int(ids[j])
                    s = self.token_text[tid]
                    steps[r].append(TokenStep(s, float(row[j]), float(torch.logsumexp(row, 0))))
                    q[r] += float(qrow[j])
                    texts[r] += s
                    nxt.append(tid)
                if all(done):
                    break
                logp, cache = self._step(torch.tensor(nxt, device=self.device), cache)
        return SearchResult("sample", prompt, board, paths, n_draws=n)
