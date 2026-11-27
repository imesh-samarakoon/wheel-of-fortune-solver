"""A second opinion on each candidate: is it real English, or nonsense that happens to fit the board?

The search model (a small base LLM) only judges an answer token by token, so a string like
"What You Give Me Treats!" can fit the board, score well enough, and end up the top answer
simply because nothing better was found. An **answer judge** asks a separate model one yes/no
question per candidate ("is this a real, familiar piece of English that fits the category?")
and uses the answer to reweight the posterior.

The default judge is **Jev** (TypeSafe AI's "System One" decision model): it answers typed
questions with a probability instead of generating text, and one request can carry one
question per candidate. ``JevJudge`` calls its HTTP API directly (no SDK); the key is read
from the ``TYPESAFE_API_KEY`` environment variable and never written anywhere.
Answers are cached on disk (``cache/judge_cache.json``, git-ignored), keyed by model,
question version, clue and answer, so a game's carried-over candidates and re-runs of the same
puzzle cost nothing. ``FixedJudge`` stands in for it in tests and offline experiments.

How the judge's answer changes the posterior (``apply_judge``)
---------------------------------------------------------------
Each candidate's judge score q (P(yes)) is treated as a likelihood, tempered by ``weight``:

    w(candidate i)    = (1 - u) x p_i x q_i ** weight
    w(none of these)  = u x real_rate ** weight

and everything is renormalized. ``p_i`` is the search's own probability, ``u`` the prior chance
that the answer is none of the candidates (``prior_unknown``; ``1 - coverage`` when sampling
estimated it), and ``real_rate`` how highly the judge typically rates a real puzzle answer.

So the judge doesn't just reshuffle the candidates. If every candidate is nonsense, their
weights all collapse and most of the probability moves to "none of these" instead of to the
least-bad nonsense: the solver stops being 63% sure of "What You Give Me Treats!", and
``play_letters(min_top=)`` holds back any guess until the top answer itself is likely enough, so
it keeps calling letters. When a real phrase is among the candidates it keeps (and gains) its share.
Nothing is ever deleted outright: a judge mistake on an odd puzzle (a pun, an obscure name)
costs probability, not the answer.

**What "none of these" looks like, letter by letter** (``unknown_letters``). By default
(``candidate_letters``) it has the letters of the candidates as the search found them, before
the judge: the judge then changes which letter is called only by reweighting the candidates,
never through the stand-in. The first version used English letter frequencies over the hidden
slots (``english_letters``, still available); those ignore the board (a slot in "THIN_" is
treated like any other), pushed common letters up, and flipped a real game's call from H to A
on turn 2 ("Keeping The Main Thing The Main Thing"), sending it down a slower path.

**Calibrating ``real_rate``** (``calibrate_real_rate``): ask the judge about a sample of real
answers from past puzzles and use the average of q ** weight. The first guess, 0.9, was too
high: Jev rated "Keeping The Main Thing The Main Thing" 54% real (and junk 5-30%), so every
candidate looked weak next to "none of these" and that share ran to 36-53% on early turns.

``weight = 0`` turns the judge off (q ** 0 = 1) apart from the "none of these" share.

**The judge is optional.** Everything works without it (``config.USE_JUDGE = False``): the solver
is just more willing to believe a confident-looking piece of nonsense. ``docs/setup.md`` covers
what it costs and how to turn it on.

**Leakage.** A model trained on the web has likely read the phrases these puzzles use, and may
rate a candidate highly because it recognizes it rather than because it is ordinary English. The
question never mentions the show, but keep it in mind when the judge looks unreasonably good.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .board import LETTERS, normalize
from .posterior import AnswerPosterior, Candidate, Unknown
from .puzzles import Clue

API_KEY_ENV = "TYPESAFE_API_KEY"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.14.0"  # pinned: "jev-latest" moves on every release, which would shift results
QUESTION_VERSION = 1          # bump whenever the wording below changes (it is part of the cache key)
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "cache" / "judge_cache.json"

# How highly Jev (DEFAULT_MODEL) rates REAL answers, measured with ``calibrate_real_rate`` over 200
# real puzzle answers (weight 1), so a game needn't re-measure it. "*" is the fallback; a category
# gets its own entry only if it was measured separately. The wordplay categories are deliberately
# NOT in here: asked the generic question they rated ~0.2, which is why they get their own questions
# (``CATEGORY_RULES``). Re-measure (``calibrate_real_rate``) after changing the model or any wording.
REAL_RATES: dict[str, float] = {"*": 0.73, "PHRASE": 0.74}


class JudgeError(RuntimeError):
    pass


class AnswerJudge(Protocol):
    """Anything that rates candidate answers: {normalized answer: P(real, familiar English)}."""

    name: str

    def judge(self, answers: Sequence[str], clue: Clue) -> dict[str, float]: ...


# --- the question ----------------------------------------------------------------------------------

def _title(answer: str) -> str:
    from .search import title_case  # the examples' style; search only imports torch with a model loaded

    return title_case(normalize(answer))


@dataclass(frozen=True)
class CategoryRule:
    """A wordplay category's own question: its answers aren't familiar phrases, so the generic one fails.

    "Chariots Of Fire Extinguishers" (Before & After) is a real answer but not a phrase anyone says;
    asked the generic question, Jev rated it 9% (Before & After answers averaged 22%, Same Name 17%,
    Same Letter 28%, vs. ~73% for everything else). Each rule explains the wordplay and asks whether
    the answer is a valid instance of it. ``version`` goes into the cache key (bump it on any
    wording change; other categories' cached answers stay valid).
    """

    note: str          # added to the state: how the category works, with an example
    instructions: str  # the yes/no statement; "{answer}" = the candidate, title case
    true: str
    false: str
    version: str
    # Optional: spell the answer out in code instead of asking the judge to. Returns the possible
    # readings, each a list of the full phrases it stands for (e.g. "STRAWBERRY & PAPER JAM" ->
    # [["STRAWBERRY JAM", "PAPER JAM"]]); ``[]`` = it doesn't parse, ask the question above instead.
    split: Callable[[str], list[list[str]]] | None = None


def same_name_readings(answer: str) -> list[list[str]]:
    """The phrases a "Same Name" answer stands for, one list per way of reading it.

    The parts are joined by "&" (or "AND" when there's no "&"; a comma also separates parts in a
    list of three or more). The shared word is written once, either at the end of the last part
    ("STRAWBERRY & PAPER JAM" -> STRAWBERRY JAM, PAPER JAM) or at the start of the first
    ("FRENCH TOAST & POODLE" -> FRENCH TOAST, FRENCH POODLE). When both parts have several words
    both readings are possible, so both are returned; a reading whose shared side is a single
    word ("SALT & PEPPER") isn't one. The judge was asked about the whole answer before, and rated
    "Strawberry & Paper Fan" (53%) as real as "Strawberry & Paper Jam" (52%): it didn't do the
    split itself, so it's done here and each phrase is asked about on its own.
    """
    text = normalize(answer)
    sep = r"\s*&\s*" if "&" in text else r"\s+AND\s+"
    parts = [p.strip(" ,") for chunk in re.split(sep, text) for p in chunk.split(",")]
    parts = [p.split() for p in parts if p.strip()]
    if len(parts) < 2:
        return []
    readings = []
    if len(parts[-1]) >= 2:  # shared word at the end of the last part
        shared = parts[-1][-1]
        readings.append([" ".join(p + [shared]) for p in parts[:-1]] + [" ".join(parts[-1])])
    if len(parts[0]) >= 2:   # shared word at the start of the first part
        shared = parts[0][0]
        readings.append([" ".join(parts[0])] + [" ".join([shared] + p) for p in parts[1:]])
    return readings


PHRASE_VERSION = "p1"  # the per-phrase question below (part of its cache key)


def phrase_question(phrase: str) -> dict:
    """The question about one spelled-out phrase (a part of a split answer, plus the shared word)."""
    return {
        "type": "noul",
        "instructions": (f'"{_title(phrase)}" is a real, familiar English phrase, name, title or thing: '
                         "something people actually say or name."),
        "criteria": {
            "true": "A real, commonly used phrase, compound word, name, title or thing.",
            "false": "Nonsense, or words that don't go together (e.g. 'Strawberry Fan', 'Paper Hat Box').",
        },
    }


def phrase_state(clue: Clue) -> str:
    parts = ["Phrases built from the parts of a word puzzle's answer. Each question asks about one "
             "phrase on its own."]
    if clue.hint:
        parts.append(f"Hint shown with the puzzle: {clue.hint}.")
    return " ".join(parts)


def combine_readings(readings: list[list[str]], scores: dict[str, float]) -> float:
    """The answer's score from its phrases' scores: the best reading, each as good as its weakest phrase.

    Every phrase has to be real for the answer to be valid, so a reading counts as its weakest
    phrase (min, not product: a two-phrase answer shouldn't score lower than a one-phrase one just
    for having more parts, which would make it look worse than "none of these").
    """
    return max(min(scores[normalize(p)] for p in reading) for reading in readings)


CATEGORY_RULES: dict[str, CategoryRule] = {
    "BEFORE & AFTER": CategoryRule(
        note=('A "Before & After" answer joins two real, familiar phrases that overlap on a shared word: the '
              "first phrase ends with that word, the second starts with it, and the word is written only once. "
              'E.g. "Chariots Of Fire Extinguishers" = "Chariots Of Fire" + "Fire Extinguishers". The joined '
              "answer is not itself a familiar phrase; what matters is whether it splits into two real ones."),
        instructions=('"{answer}" splits at one shared word into two real, familiar English phrases, names or '
                      "titles (the first ending with that word, the second starting with it), as a valid "
                      '"Before & After" answer does.'),
        true="Both parts are real, commonly used phrases, names, titles or things, overlapping on one shared word.",
        false=("It can't be split that way, or a part is nonsense, unnatural, or a familiar phrase with a word "
               "swapped for another."),
        version="ba1",
    ),
    "SAME NAME": CategoryRule(
        note=('A "Same Name" answer lists two or more things, joined by "&", that share one word, written only '
              'once (usually at the end). E.g. "Catering & Internal Revenue Service" = "Catering Service" + '
              '"Internal Revenue Service". The answer as written is not itself a familiar phrase.'),
        instructions=('"{answer}" is a valid "Same Name" answer: adding the shared word (written once) to each '
                      'part joined by "&" gives real, familiar English phrases, names or things.'),
        true="Every part makes a real, commonly used phrase, name or thing with the shared word.",
        false="Some part doesn't make a real phrase with the shared word, or the answer is nonsense or garbled.",
        version="sn1",
        split=same_name_readings,
    ),
    "SAME LETTER": CategoryRule(
        note=('In a "Same Letter" answer the main words all start with the same letter, e.g. "Pretty Pink '
              'Petunias". It is often a list or a made-up but sensible combination, not a familiar saying.'),
        instructions=('"{answer}" is a sensible, natural English phrase or list of real things whose main '
                      "words all start with the same letter."),
        true="Real words, sensibly combined, the main words sharing a first letter.",
        false="Nonsense, garbled or ungrammatical words, or the main words don't share a first letter.",
        version="sl1",
    ),
}


def category_rule(clue: Clue) -> CategoryRule | None:
    return CATEGORY_RULES.get(clue.category.strip().upper())


def judge_state(clue: Clue) -> str:
    """What the judge reads before every question: the puzzle's category and any hint."""
    parts = [f'Candidate answers to a word puzzle. Category: "{clue.category}".']
    rule = category_rule(clue)
    if rule:
        parts.append(rule.note)
    if clue.context:
        parts.append(f"Category notes: {clue.context}.")
    if clue.hint:
        parts.append(f"Hint shown with the puzzle: {clue.hint}.")
    parts.append("Each question asks about one candidate on its own.")
    return " ".join(parts)


def judge_question(answer: str, clue: Clue) -> dict:
    """One Jev yes/no ("noul") question about one candidate answer (a wordplay category's own, if it has one)."""
    rule = category_rule(clue)
    if rule:
        return {"type": "noul", "instructions": rule.instructions.format(answer=_title(answer)),
                "criteria": {"true": rule.true, "false": rule.false}}
    return {
        "type": "noul",
        "instructions": (f'"{_title(answer)}" is a real, familiar piece of English that fits the category '
                         f'"{clue.category}": something people actually say or name, written the way it is '
                         "normally written."),
        "criteria": {
            "true": ("A real, commonly used phrase, saying, expression, title, name, place or thing that fits "
                     "the category, worded the way people actually use it."),
            "false": ("Nonsense, garbled or ungrammatical words, an unnatural word sequence, or a familiar phrase "
                      "with a word swapped for another (e.g. 'Better Late Than Ever' for 'Better Late Than "
                      "Never')."),
        },
    }


# --- judges -----------------------------------------------------------------------------------------

@dataclass
class FixedJudge:
    """A judge with hand-set scores (tests, offline experiments). Unlisted answers get ``default``."""

    scores: dict[str, float]
    default: float = 0.5
    name: str = "fixed"
    asked: list[list[str]] = field(default_factory=list)

    def judge(self, answers: Sequence[str], clue: Clue) -> dict[str, float]:
        answers = [normalize(a) for a in answers]
        self.asked.append(answers)
        table = {normalize(k): v for k, v in self.scores.items()}
        return {a: table.get(a, self.default) for a in answers}


@dataclass
class JudgeStats:
    requests: int = 0
    questions: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    seconds: float = 0.0


class JevJudge:
    """TypeSafe's Jev, one yes/no question per candidate, batched into as few requests as possible.

    ``api_key``: the key itself; default = the ``TYPESAFE_API_KEY`` environment
    variable (read when the first request is made, so building a judge never fails).
    ``cache_path``: a JSON file of past answers (``None`` = no disk cache, memory only).
    ``batch``: questions per request (they share the request's token budget; ~40 tokens each).
    ``max_retries``: retries on 429 / 5xx / network errors, waiting ``Retry-After`` or backing off.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, api_key: str | None = None, api_key_env: str = API_KEY_ENV,
                 endpoint: str = ENDPOINT, cache_path: str | Path | None = DEFAULT_CACHE, batch: int = 50,
                 timeout: float = 30.0, max_retries: int = 4):
        self.model, self.endpoint, self.batch = model, endpoint, batch
        self.timeout, self.max_retries = timeout, max_retries
        self._api_key, self.api_key_env = api_key, api_key_env
        self.cache_path = Path(cache_path) if cache_path else None
        self.name = f"Jev ({model})"
        # measured for the pinned model only; another model needs its own calibration
        self.real_rates: dict[str, float] | None = dict(REAL_RATES) if model == DEFAULT_MODEL else None
        self.stats = JudgeStats()
        self._cache: dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            self._cache = json.loads(self.cache_path.read_text())

    def _key(self, answer: str, clue: Clue) -> str:
        rule = category_rule(clue)
        version = f"q{QUESTION_VERSION}" + (f"+{rule.version}" if rule else "")
        return "|".join([self.model, version, clue.category, clue.context, clue.hint, answer])

    def _phrase_key(self, phrase: str, clue: Clue) -> str:
        return "|".join([self.model, f"q{QUESTION_VERSION}+{PHRASE_VERSION}", "", "", clue.hint, phrase])

    def judge(self, answers: Sequence[str], clue: Clue) -> dict[str, float]:
        """{answer: P(real)}. A category with a ``split`` (Same Name) is asked about each spelled-out
        phrase instead (``combine_readings``); answers that don't split get the whole-answer question."""
        answers = list(dict.fromkeys(normalize(a) for a in answers))
        rule = category_rule(clue)
        readings = {a: rule.split(a) for a in answers} if rule and rule.split else {}
        whole = [a for a in answers if not readings.get(a)]
        phrases = list(dict.fromkeys(normalize(p) for a in answers for r in readings.get(a, []) for p in r))
        out = self._cached_or_ask(whole, clue, self._key, lambda chunk: self._ask(chunk, clue))
        if phrases:
            got = self._cached_or_ask(phrases, clue, self._phrase_key,
                                      lambda chunk: self._ask(chunk, clue, phrase_state(clue), phrase_question))
            out.update({a: combine_readings(readings[a], got) for a in answers if readings.get(a)})
        return {a: out[a] for a in answers}

    def _cached_or_ask(self, items: list[str], clue: Clue, key: Callable[[str, Clue], str],
                       ask: Callable[[list[str]], dict[str, float]]) -> dict[str, float]:
        out = {a: self._cache[key(a, clue)] for a in items if key(a, clue) in self._cache}
        self.stats.cache_hits += len(out)
        todo = [a for a in items if a not in out]
        for start in range(0, len(todo), self.batch):
            scores = ask(todo[start:start + self.batch])
            for a, q in scores.items():
                self._cache[key(a, clue)] = q
            out.update(scores)
        if todo:
            self._save()
        return out

    def _ask(self, answers: list[str], clue: Clue, state: str | None = None,
             question: Callable[[str], dict] | None = None) -> dict[str, float]:
        ids = {f"a{i}": a for i, a in enumerate(answers)}
        ask = question or (lambda a: judge_question(a, clue))
        body = {"model": self.model, "state": state if state is not None else judge_state(clue),
                "questions": {k: ask(a) for k, a in ids.items()}}
        reply = self._post(body)
        try:
            got = reply["answers"]
            scores = {a: float(got[k]["noul"]) for k, a in ids.items()}
        except (KeyError, TypeError, ValueError) as e:
            raise JudgeError(f"unexpected Jev response shape: {json.dumps(reply)[:500]}") from e
        self.stats.questions += len(answers)
        self.stats.input_tokens += int(reply.get("usage", {}).get("input_tokens", 0) or 0)
        return scores

    def _post(self, body: dict) -> dict:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise JudgeError(f"no TypeSafe API key: set the {self.api_key_env} environment variable")
        data = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.stats.requests += 1
                    self.stats.seconds += time.perf_counter() - t0
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code != 429 and e.code < 500 or attempt == self.max_retries:
                    raise JudgeError(f"Jev request failed: HTTP {e.code}: {detail}") from e
                wait = _retry_after(e.headers.get("Retry-After"), attempt)
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.max_retries:
                    raise JudgeError(f"Jev request failed: {e}") from e
                wait = _retry_after(None, attempt)
            time.sleep(wait)
        raise AssertionError("unreachable")

    def _save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._cache, indent=0, sort_keys=True))
        tmp.replace(self.cache_path)


def _retry_after(header: str | None, attempt: int) -> float:
    try:
        return max(float(header), 0.0) if header else 2.0 ** attempt
    except ValueError:
        return 2.0 ** attempt


# --- what an answer outside the candidates looks like ----------------------------------------------

# Relative frequency of each letter in English text (Lewand), for the "none of these" stand-in.
ENGLISH_FREQ = {
    "A": 8.167, "B": 1.492, "C": 2.782, "D": 4.253, "E": 12.702, "F": 2.228, "G": 2.015, "H": 6.094,
    "I": 6.966, "J": 0.153, "K": 0.772, "L": 4.025, "M": 2.406, "N": 6.749, "O": 7.507, "P": 1.929,
    "Q": 0.095, "R": 5.987, "S": 6.327, "T": 9.056, "U": 2.758, "V": 0.978, "W": 2.360, "X": 0.150,
    "Y": 1.974, "Z": 0.074,
}

UnknownLetters = Callable[[AnswerPosterior], tuple[dict[str, float], dict[str, float]]]


def candidate_letters(post: AnswerPosterior) -> tuple[dict[str, float], dict[str, float]]:
    """(P(letter in the hidden part), E[# times]) of the candidates themselves, before the judge.

    The default stand-in for "none of these": an answer the search missed is assumed to look,
    letter by letter, like the ones it found. So the judge can't change which letter is called
    through the stand-in, only by reweighting the candidates.
    """
    return post.presence(), post.expected_count()


def english_letters(post: AnswerPosterior) -> tuple[dict[str, float], dict[str, float]]:
    """(P(letter in the hidden part), E[# times]) for an answer we know nothing about.

    Each hidden slot is an independent draw from English letter frequencies restricted to the
    uncalled letters (a called letter can't be hiding). It ignores the board's words (a slot in
    "THIN_" is treated like any other), so it pushes common letters up; not the default any more.
    """
    board = post.board
    freq = {c: ENGLISH_FREQ[c] for c in LETTERS if c not in board.called}
    z = sum(freq.values()) or 1.0
    k = board.n_hidden
    presence = {c: 1 - (1 - f / z) ** k for c, f in freq.items()}
    counts = {c: k * f / z for c, f in freq.items()}
    return presence, counts


# --- reweighting the posterior ---------------------------------------------------------------------

@dataclass
class JudgeResult:
    scores: dict[str, float]  # judge score per candidate answer
    prior_unknown: float      # P(none of these) before the judge
    unknown: float            # ... and after


def apply_judge(post: AnswerPosterior, scores: dict[str, float], *, weight: float = 1.0,
                prior_unknown: float = 0.1, real_rate: float = 0.9,
                unknown_letters: UnknownLetters = candidate_letters) -> tuple[AnswerPosterior, JudgeResult]:
    """The posterior reweighted by the judge's scores, with an explicit "none of these" share.

    See the module docstring for the formula. ``post``'s own unknown share, if it has one, is
    treated as part of the candidates' prior (it's re-derived here from ``prior_unknown``).
    Candidates missing from ``scores`` count as ``real_rate`` (no evidence either way).
    ``unknown_letters`` is handed ``post`` as it was before the judge.
    """
    if post.empty:
        return post, JudgeResult({}, prior_unknown, post.unknown_mass)
    u = min(max(prior_unknown, 0.0), 1.0)
    z = sum(post.probs)
    eps = 1e-6
    weights = []
    for a, p in zip(post.answers, post.probs):
        q = min(max(scores.get(a, real_rate), eps), 1.0)
        weights.append((1 - u) * (p / z) * q ** weight)
    w_unknown = u * max(real_rate, eps) ** weight
    total = sum(weights) + w_unknown
    mass = w_unknown / total if total > 0 else 1.0
    presence, counts = unknown_letters(post)
    cands = [Candidate(a, math.log(w), s) for a, w, s in zip(post.answers, weights, post.sources) if w > 0]
    new = AnswerPosterior(post.board, cands, Unknown(mass, presence, counts))
    return new, JudgeResult({a: scores.get(a, real_rate) for a in post.answers}, u, new.unknown_mass)


# --- calibrating real_rate -------------------------------------------------------------------------

def calibrate_real_rate(judge: AnswerJudge, puzzles: Sequence, n: int = 100, weight: float = 1.0,
                        seed: int = 0) -> tuple[float, list[tuple[str, float]]]:
    """Measure how highly ``judge`` rates REAL answers: ``(real_rate, [(answer, score), ...])``.

    ``real_rate`` is E[q ** weight] ** (1 / weight) over ``n`` real answers drawn from ``puzzles``
    (``Puzzle`` objects, e.g. the example pool) -- the value that makes "none of these" weigh what a
    genuinely real answer typically scores. Run this after changing the judge model or any question
    wording, and put the result in ``REAL_RATES`` (or pass it as ``judge_real_rate``). Answers
    sharing a clue go in one request, and every answer is cached, so re-running costs nothing.

    The scores come back sorted, lowest first, so the tail of the list is the judge's own misses:
    real answers it rated as nonsense. Those are worth reading before trusting it.
    """
    import random

    pool = list(puzzles)
    picked = random.Random(seed).sample(pool, min(n, len(pool)))
    groups: dict[tuple[str, str, str], tuple[Clue, list[str]]] = {}
    for p in picked:
        groups.setdefault((p.clue.category, p.clue.context, p.clue.hint), (p.clue, []))[1].append(normalize(p.answer))
    scores: list[tuple[str, float]] = []
    for clue, answers in groups.values():
        got = judge.judge(answers, clue)
        scores += [(a, got[a]) for a in answers]
    qs = [q for _, q in scores]
    if not qs:
        return math.nan, scores
    if weight == 0:
        return 1.0, scores
    rate = (sum(max(q, 1e-6) ** weight for q in qs) / len(qs)) ** (1 / weight)
    return rate, sorted(scores, key=lambda t: t[1])
