"""Printouts that show every step of how the solver gets from a board to letter probabilities.

Used by ``explore.py`` and ``play.py``. Nothing here calls the model: each function just formats
a result that ``wof.search`` or ``wof.calling`` already produced, so re-running one with different
display settings costs nothing.
"""

from __future__ import annotations

import math

from .board import LETTERS, Board, normalize
from .search import Path, SearchResult, TokenTable, show_token
from .posterior import AnswerPosterior
from .puzzles import Puzzle


def _rule(title: str = "") -> None:
    print(f"\n── {title} " + "─" * max(0, 76 - len(title)) if title else "─" * 80)


def print_puzzle(puzzle: Puzzle, board: Board, reveal_answer: bool = False) -> None:
    """The puzzle as the solver sees it: everything except the answer, which it never gets."""
    c = puzzle.clue
    _rule("THE PUZZLE" + (f"  ({puzzle.id})" if puzzle.id else ""))
    print(f"category : {c.category}" + (f"   (aired {c.year})" if c.year else ""))
    if c.hint:
        print(f"hint     : {c.hint}")
    if c.context:
        print(f"context  : {c.context}")
    print(f"board    : {board.template}     ({board.n_hidden} hidden letters, "
          f"{len(board.words)} word{'s' * (len(board.words) != 1)})")
    if board.called:
        print(f"called   : {' '.join(sorted(board.called))}   (not in puzzle: {' '.join(sorted(board.absent)) or '-'})")
    print(f"solution : {puzzle.answer if reveal_answer else '(hidden)'}   <- never shown to the model")


def print_prompt(prompt: str) -> None:
    _rule(f"PROMPT  ({len(prompt)} characters) — the model continues this text after 'Answer:'")
    print(prompt)


def print_next_tokens(t: TokenTable) -> None:
    where = f"after 'Answer: {t.prefix}'" if t.prefix else "right after 'Answer:'"
    _rule(f"NEXT-TOKEN PREDICTION {where}")
    if t.ends_here:
        print("The answer already fills the board, so the only token that fits is a newline (stop).")
    print("The model's own top tokens (over its whole vocabulary). 'fits' = can be written on the board here:")
    print(f"  {'rank':>4}  {'token':<20} {'P':>8}  fits")
    for k, (s, p, ok) in enumerate(t.rows, 1):
        print(f"  {k:>4}  {show_token(s):<20} {p:>8.2%}  {'yes' if ok else ' - '}")
    print(f"\nTotal probability on tokens that fit the board: {t.fit_mass:.1%}. "
          f"The other {1 - t.fit_mass:.1%} is thrown away (wrong length, called letter, ...).")
    print("Top tokens among those that fit (P is still the model's own, not renormalized):")
    print("  " + "   ".join(f"{show_token(s)} {p:.2%}" for s, p in t.top_fitting[:10]))


def print_beam_trace(res: SearchResult, every: int = 1, top: int = 5) -> None:
    by_position = bool(res.trace) and "position" in res.trace[0]
    how = ("the partial answers at the lowest position get one more token, the rest wait" if by_position
           else "one token added per step")
    _rule(f"BEAM SEARCH TRACE  ({len(res.trace)} steps, {how})")
    if by_position:
        print("Position-synchronous: at each step the answers that have written the fewest board characters get\n"
              "one more token; answers are only ever compared with others at the same position. 'live' = every\n"
              "partial answer kept, at any position; the lines below are the ones just extended.")
    print("log P = the model's log-probability of the text so far (closer to 0 = more likely; -2.3 = 10%,\n"
          "-4.6 = 1%, -6.9 = 0.1%). 'grow' = still filling the board; 'full' = just filled it, finished on\n"
          "the next step once log P(stop here) is added; 'done' = finished. The same answer can show up twice\n"
          "when the model can spell it with different tokens ([␣Aff][leck] vs [␣Affleck]); those are summed.")
    L = len(" " + res.board.template)
    for t in res.trace[::every]:
        capped = f", {t['capped']} extensions skipped (first word already had its cap)" if t.get("capped") else ""
        at = f" (extending position {t['position']} of {L})" if by_position else ""
        print(f"\nstep {t['step']:>2}{at}: {t['n_live']} live, {t['n_finished']} finished{capped}")
        for text, lp in t["live"][:top]:
            print(f"   {'full' if len(text) == L else 'grow'}  {lp:8.2f}  {text!r}")
        for text, lp in t["finished"][:min(3, top)]:
            print(f"   done  {lp:8.2f}  {text!r}")


def path_tokens(p: Path) -> str:
    """ [ B] -4.21 · [urt] -0.05 · [ Lancaster] -0.30 · stop -0.02 """
    parts = [f"{show_token(s.text)} {s.logprob:.2f}" for s in p.steps]
    return " · ".join(parts + [f"stop {p.end_logprob:.2f}"])


def print_candidates(res: SearchResult, n: int = 15, token_detail: int = 5) -> None:
    """Every answer found, how the model scored it token by token, and the normalization."""
    post = res.posterior()
    logw = res.answer_logweights()
    total = sum(math.exp(w) for w in logw.values())
    by_answer: dict[str, Path] = {}
    for p in sorted(res.paths, key=lambda p: -p.logprob):
        by_answer.setdefault(p.answer, p)
    ranked = post.top(len(post))
    if res.method == "beam":
        _rule(f"CANDIDATE ANSWERS  ({len(post)} distinct, from {len(res.paths)} finished beams)")
        print("P(answer | prompt) = product of its tokens' probabilities (= exp of the summed log P), stop included.\n"
              "Normalizing over just these candidates gives P(answer | clue, board) — assuming the truth is\n"
              f"among them. These {len(post)} answers together hold {total:.4%} of the model's probability.\n")
        print(f"  {'#':>3}  {'log P':>7}  {'P(answer|prompt)':>16}  {'normalized':>10}  answer")
        for k, (a, p) in enumerate(ranked[:n], 1):
            print(f"  {k:>3}  {logw[a]:7.2f}  {math.exp(logw[a]):16.3e}  {p:10.2%}  {a}")
    else:
        _rule(f"SAMPLED ANSWERS  ({len(res.paths)} draws, {len(post)} distinct)")
        print("Each draw's weight = P(model writes it) / P(the sampler drew it). Answers drawn from low-probability\n"
              "regions get small weights; the normalized weight is the estimated P(answer | clue, board).\n"
              f"Effective sample size {res.effective_sample_size():.1f} of {len(res.paths)} "
              "(small = a few draws carry all the weight, so the estimate is noisy).\n")
        counts: dict[str, int] = {}
        for p in res.paths:
            counts[p.answer] = counts.get(p.answer, 0) + 1
        print(f"  {'#':>3}  {'draws':>5}  {'log P':>7}  {'normalized':>10}  answer")
        for k, (a, p) in enumerate(ranked[:n], 1):
            print(f"  {k:>3}  {counts[a]:>5}  {by_answer[a].logprob:7.2f}  {p:10.2%}  {a}")
    if len(post) > n:
        print(f"  ... and {len(post) - n} more")
    if token_detail:
        print(f"\nToken by token for the top {min(token_detail, len(ranked))} (each token's own log P | everything before it):")
        for a, _ in ranked[:token_detail]:
            print(f"  {a}\n      {path_tokens(by_answer[a])}")


def print_letter_probs(post: AnswerPosterior, truth: str | None = None) -> dict[str, float]:
    """P(letter is in the puzzle) for every uncalled letter, and how well those did if ``truth`` is known.

    Returns the scores (Brier, log loss) when ``truth`` is given.
    """
    pres = post.presence()
    cnt = post.expected_count()
    hidden_truth = {truth[i] for i in post.board.hidden_positions} if truth else set()
    _rule("LETTER PROBABILITIES  P(letter is in the puzzle) = sum of P(answer) over answers containing it")
    print(f"  {'letter':<6} {'P(in puzzle)':>12}  {'E[# times]':>10}  " + ("in truth" if truth else ""))
    for c, p in pres.items():
        bar = "█" * round(p * 20)
        mark = ("  yes" if c in hidden_truth else "   - ") if truth else ""
        print(f"  {c:<6} {p:>12.1%}  {cnt[c]:>10.2f}{mark}  {bar}")
    if not truth:
        return {}
    eps = 1e-6
    brier = sum((p - (c in hidden_truth)) ** 2 for c, p in pres.items()) / len(pres)
    logloss = -sum(math.log(max(eps, p if c in hidden_truth else 1 - p)) for c, p in pres.items()) / len(pres)
    print(f"\nScore vs. the truth over {len(pres)} letters: Brier {brier:.3f} (0 = perfect, 0.25 = always saying 50%), "
          f"log loss {logloss:.3f}")
    return {"brier": brier, "logloss": logloss}


def print_letter_breakdown(post: AnswerPosterior, letter: str, n: int = 10) -> None:
    """Which candidates put ``letter`` in the puzzle, and how much each contributes to its probability."""
    letter = letter.upper()
    hidden = post.board.hidden_positions
    rows = [(a, p) for a, p in post.top(len(post)) if any(a[i] == letter for i in hidden)]
    total = sum(p for _, p in rows)
    _rule(f"WHERE P({letter} in puzzle) = {total:.1%} COMES FROM  ({len(rows)} of {len(post)} candidates contain {letter})")
    run = 0.0
    for a, p in rows[:n]:
        run += p
        marked = "".join(c.lower() if (i in hidden and c == letter) else c for i, c in enumerate(a))
        print(f"  {p:7.2%}  (running {run:6.1%})  {marked}")
    if len(rows) > n:
        print(f"  ... {len(rows) - n} more, together {total - run:.2%}")


def print_truth_check(truth: Path, res: SearchResult) -> None:
    """Where the true answer sits: its own log P (forced) vs. what the search kept."""
    post = res.posterior()
    _rule("WHERE IS THE TRUE ANSWER?")
    print(f"truth    : {truth.answer}")
    print(f"tokens   : {path_tokens(truth)}")
    print(f"log P    : {truth.logprob:.2f}  (P = {math.exp(truth.logprob):.3e})")
    fit = [s.fit_logmass for s in truth.steps]
    if fit:
        print("fit mass : at each of its steps, the share of the model's probability on board-fitting tokens: "
              + ", ".join(f"{math.exp(x):.0%}" for x in fit))
    ranked = [a for a, _ in post.top(len(post))]
    if truth.answer in ranked:
        k = ranked.index(truth.answer) + 1
        print(f"found    : yes, rank {k} of {len(ranked)}, normalized P = {post.prob_of(truth.answer):.2%}")
    else:
        worst = min(p.logprob for p in res.paths) if res.paths else float("nan")
        best = max(p.logprob for p in res.paths) if res.paths else float("nan")
        print(f"found    : NO. The kept candidates' log P runs {best:.2f} (best) to {worst:.2f} (worst).")
        if truth.logprob > worst:
            print("           The truth scores better than some kept answers, so the search pruned it early: a prefix\n"
                  "           of it looked unlikely before its later tokens paid off. More beams would catch it.")
        else:
            print("           The truth scores worse than everything kept: the model genuinely prefers the others.")
        print("           Every letter probability above is computed as if the truth were impossible.")


def fill_lines(fill: dict | None) -> list[str]:
    """One line per board word ``LLMSolver.fill_slots`` filled on a turn (``info["fill"]``)."""
    if not fill:
        return []
    out = []
    for r in fill["slots"]:
        settled = ", ".join(f"{w} {p:.0%}" for _, w, p in r["locked"]) or "nothing else hidden"
        line = (f"word {r['word'] + 1} {r['pattern']} with the rest settled ({settled}): "
                f"{r['tried']} fill{'s' * (r['tried'] != 1)} scored")
        if r["best"]:
            answer, logprob = r["best"]
            ratio = math.exp(logprob - fill["top_logprob"])
            line += f", best {answer} ({ratio:.3g}x the search's top answer)"
        line += f", {len(r['added'])} added" if r["added"] else ", none good enough to add"
        out.append(line)
    why = {"unchanged": "unchanged since an earlier turn", "max_slots": "one word per turn (max_slots)",
           "too many words fit": "too many words still fit it to be worth a model pass each "
                                 "(FILL_MAX_WORDS); another revealed letter will cut that down",
           "no word fits": "nothing in the word list or the search fits it"}
    for r in fill.get("skipped", []):
        out.append(f"word {r['word'] + 1} {r['pattern']}: not filled, {why.get(r['reason'], r['reason'])}")
    if out:
        out[-1] += f"   ({fill['seconds']:.1f}s)"
    return out


def print_filled(fill, res: SearchResult, truth: str | None = None, n: int = 10) -> None:
    """``LLMSolver.fill_slots``' result: which board words it filled, and the new answers it kept."""
    _rule("FILLED WORDS (one open word filled from the word list, the rest of the phrase held fixed)")
    if not fill.slots:
        print("no board word qualified: every one has another hidden word below min_confidence "
              "(lower it to fill anyway)")
        return
    for r in fill.slots:
        settled = ", ".join(f"{w} {p:.0%}" for _, w, p in r["locked"]) or "nothing else hidden"
        print(f"word {r['word'] + 1}  {r['pattern']}   settled: {settled}   contexts: {' | '.join(r['contexts'])}")
        print(f"  {r['tried']} new answers scored" + (f", best {r['best'][0]} ({r['best'][1]:.2f})" if r["best"] else ""))
    best = max((p.logprob for p in res.paths), default=float("nan"))
    worst = min((p.logprob for p in res.paths), default=float("nan"))
    print(f"the search's own answers run {best:.2f} (best) to {worst:.2f} (worst); "
          f"{len(fill.paths)} fill{'s' * (len(fill.paths) != 1)} kept (within that range or better)")
    for p in fill.paths[:n]:
        mark = "   <- the true answer" if truth and p.answer == normalize(truth) else ""
        beats = "  (beats every answer the search found)" if p.logprob > best else ""
        print(f"  {p.logprob:8.2f}  {p.answer}{beats}{mark}")
    if truth:
        t = normalize(truth)
        found = any(p.answer == t for p in res.paths)
        made = any(p.answer == t for p in fill.paths)
        print("true answer: " + ("already found by the search" if found else
                                 "FOUND by filling a word" if made else "not found either way"))


def _share(x: float) -> str:
    """A share as a percentage that stays readable when it is tiny ("0.0057%", not "0%")."""
    x = min(max(x, 0.0), 1.0)
    for floor, fmt in ((0.095, "{:.0%}"), (0.0095, "{:.1%}"), (0.00095, "{:.2%}")):
        if x >= floor:
            return fmt.format(x)
    return f"{x:.4%}" if x >= 1e-6 else ("0%" if x == 0 else "under 0.0001%")


def print_coverage(beam: SearchResult, sample: SearchResult) -> None:
    """How much of the model's board-fitting probability the beam candidates hold (estimated by sampling)."""
    found = sum(math.exp(w) for w in beam.answer_logweights().values())
    z = sample.fit_mass_estimate()
    _rule("COVERAGE  how much of the model's board-fitting probability did the beams find?")
    print(f"beam candidates' total P(answer | prompt)            : {found:.3e}")
    print(f"sampling estimate of P(any board-fitting answer)     : {z:.3e}  (ESS {sample.effective_sample_size():.1f})")
    if z > 0:
        print(f"=> the beams cover roughly {_share(found / z)} of it, so renormalizing over them alone "
              f"is\n   overconfident. With few effective samples this is itself a noisy estimate.")


def compare_letters(columns: dict[str, AnswerPosterior | dict[str, float]], board: Board,
                    truth: str | None = None) -> None:
    """Side by side P(letter in puzzle) from several sources, plus each one's score if ``truth`` is known.

    Each column is a posterior (its ``presence()`` is used) or a ready {letter: P} dict
    (a baseline or a blend). Rows are sorted by the first column.
    """
    from .adjust import letter_scores

    names = list(columns)
    pres = {k: (v.presence() if isinstance(v, AnswerPosterior) else v) for k, v in columns.items()}
    hidden_truth = {truth[i] for i in board.hidden_positions} if truth else set()
    letters = [c for c in LETTERS if c not in board.called]
    letters.sort(key=lambda c: -pres[names[0]].get(c, 0))
    width = max(12, *(len(n) + 2 for n in names))
    _rule("P(letter in puzzle), side by side")
    print("  letter " + "".join(f"{n:>{width}}" for n in names) + ("   in truth" if truth else ""))
    for c in letters:
        mark = ("   yes" if c in hidden_truth else "    - ") if truth else ""
        print(f"  {c:<6} " + "".join(f"{pres[n].get(c, 0):>{width}.1%}" for n in names) + mark)
    if truth:
        scores = {n: letter_scores({c: pres[n].get(c, 0.0) for c in letters}, board, truth) for n in names}
        print("  " + "-" * (7 + width * len(names)))
        print("  Brier  " + "".join(f"{scores[n]['brier']:>{width}.3f}" for n in names))
        print("  logloss" + "".join(f"{scores[n]['logloss']:>{width}.3f}" for n in names))
        print("  (lower = better; Brier 0.25 = saying 50% for every letter)")


def print_hybrid(post: AnswerPosterior, info, n: int = 15) -> None:
    """Change 2: beam answers (exact) + sampled answers the beams missed (estimated)."""
    _rule("BEAMS + SAMPLES  every answer the beams found, plus sampled answers they missed")
    print(f"beam answers' total P(answer | prompt) (exact)      : {info.found_mass:.3e}")
    print(f"estimated P of every answer the beams missed       : {info.missed_mass:.3e}"
          f"   from {info.n_outside} of {info.n_draws} draws ({info.n_outside_answers} distinct answers,"
          f" effective sample size {info.outside_ess:.1f})")
    print(f"=> the beams cover about {_share(info.coverage)} of the model's board-fitting probability")
    if info.outside_ess and info.outside_ess < 10:
        print("   (few effective samples: the 'missed' part rests on a handful of draws; more samples sharpen it)")
    src = dict(zip(post.answers, post.sources))
    print(f"\n  {'#':>3}  {'normalized':>10}  {'from':<6}  answer")
    for k, (a, p) in enumerate(post.top(n), 1):
        print(f"  {k:>3}  {p:10.2%}  {src[a]:<6}  {a}")
    if len(post) > n:
        print(f"  ... and {len(post) - n} more")
    beam_share = sum(p for a, p in zip(post.answers, post.probs) if src[a] == "beam")
    print(f"\nbeam answers now hold {_share(beam_share)} of the normalized probability, sampled ones "
          f"{_share(1 - beam_share)} (before: 100% / 0%)")
    if info.outside_ess and info.outside_ess < 10:
        print("Read this as a warning about the beams, not as better letter probabilities: a couple of\n"
              "lucky draws can take over the estimate, and most sampled answers are junk that merely\n"
              "fits the board. The solver's own answer to thin coverage is the judge and fill_slots.")


# --- calling letters one at a time (wof.calling) ---------------------------------------------------

def _pct(p: float) -> str:
    return "  -  " if p is None or math.isnan(p) else f"{p:5.1%}" if p < 0.9995 else " 100%"


def print_turn(turn, n_answers: int = 5, n_letters: int = 8) -> None:
    """One turn of a game: the board, the solver's top answers, its letter probabilities, the call."""
    b = turn.board
    missed = " ".join(sorted(b.absent)) or "-"
    _rule(f"TURN {turn.number}   {b.template}")
    if b.called:
        print(f"called so far: {' '.join(sorted(b.called))}   (not in puzzle: {missed})   "
              f"{b.n_hidden} letter{'s' * (b.n_hidden != 1)} hidden")
    extra = []
    if "reused" in turn.info:
        extra.append("no new model run: same board as the last turn, minus the wrong guess")
    if "seconds" in turn.info:
        extra.append(f"{turn.info['seconds']:.1f}s")
    if turn.info.get("carried"):
        extra.append(f"{len(turn.info['carried'])} carried over from the last turn")
    if turn.info.get("filled"):
        extra.append(f"{len(turn.info['filled'])} found by filling a word")
    if "coverage" in turn.info:
        extra.append(f"beams cover ~{turn.info['coverage']:.0%} of the model's probability")
    print(f"{turn.n_candidates} candidate answers, entropy {max(turn.entropy_bits, 0.0):.2f} bits"
          + (f"   ({', '.join(extra)})" if extra else ""))
    if turn.top:
        mark = "RIGHT" if turn.top_correct else "wrong"
        print(f"most likely  : {turn.top_answer}  {_pct(turn.p_top).strip()}  ({mark})")
    else:
        print("most likely  : (no candidates found)")
    where = f"rank {turn.truth_rank}, {turn.p_truth:.1%}" if turn.truth_rank else "not among the candidates"
    if turn.truth_rank != 1 and "truth_logprob" in turn.info and "top_logprob" in turn.info:
        ratio = turn.info["truth_logprob"] - turn.info["top_logprob"]
        where += f"   (scored directly, the model rates it {math.exp(ratio):.3g}x as likely as the top answer)"
    print(f"true answer  : {where}")
    for line in fill_lines(turn.info.get("fill")):
        print("filled word  : " + line)
    if len(turn.top) > 1:
        print("top answers  : " + " · ".join(f"{a} {p:.1%}" for a, p in turn.top[1:n_answers]))
    judge = turn.info.get("judge")
    if judge:
        truth = judge.get("truth_score")
        print(f"judge        : {judge['name']}  P(none of these) {judge['prior_unknown']:.0%} -> {judge['unknown']:.0%}"
              f"   (real_rate {judge.get('real_rate', math.nan):.2f})"
              f"   ({judge['seconds']:.1f}s)" + (f"   true answer judged {truth:.0%} real" if truth is not None else ""))
        rows = [(a, judge["scores"].get(a), judge["before"].get(a, 0.0), p) for a, p in turn.top[:n_answers]]
        print("judged real  : " + " · ".join(f"{a} {q:.0%} ({b:.1%} -> {p:.1%})" for a, q, b, p in rows
                                             if q is not None))
    letters = list(turn.presence.items())[:n_letters]
    if letters and not b.solved:
        print("P(in puzzle) : " + "  ".join(f"{c} {_pct(p).strip()}" for c, p in letters))
    if not math.isnan(turn.signal) and not b.solved:
        print(f"solve signal : {turn.signal:.3f}  (lower = surer; it guesses at or below the threshold)")
    if turn.held_back:
        print(f"=> the solve signal is low enough to guess, but the top answer is only {turn.p_top:.0%} likely "
              f"(P(none of these) {turn.p_unknown:.0%}): calls a letter instead")
    if turn.guess:
        verdict = ("RIGHT, it wins" if turn.guess_correct
                   else "WRONG: ruled out for the rest of the game; no letter this turn")
        print(f"=> solves instead of calling a letter: {turn.guess}  ...  {verdict}")
    if turn.called:
        result = f"appears {turn.count}x" if turn.hit else "NOT in the puzzle"
        print(f"=> calls {turn.called}  (it said {turn.p_called:.1%} in puzzle, expected {turn.expected_called:.2f}x)"
              f"  ...  {result}")


def print_game(game) -> None:
    """The whole game as one table, then when it got it right and how sure it was along the way."""
    _rule(f"GAME SUMMARY   {game.answer}   ({len(game.calls)} letters called: {game.hits} in the puzzle, "
          f"{game.misses} not; {len(game.guesses)} solve attempt(s); stopped: {game.stopped})")
    width = max(len(game.answer), 12)
    judged = any("judge" in t.info or t.p_unknown > 0 for t in game.turns)
    print("  signal = the solve signal (lower = surer); guess = solved instead of calling a letter: WIN or wrong"
          + ("\n  none = P(the answer is none of the candidates), after the answer judge" if judged else ""))
    print(f"  {'turn':>4}  {'board':<{width}}  {'cands':>5}  {'most likely answer':<{width}}  {'P(top)':>6}  "
          + (f"{'none':>6}  " if judged else "")
          + f"{'right':>5}  {'P(truth)':>8}  {'rank':>4}  {'Brier':>5}  {'signal':>6}  {'guess':>5}   call")
    for t in game.turns:
        call = ""
        if t.called:
            call = f"{t.called} ({t.p_called:.0%}) " + (f"x{t.count}" if t.hit else "miss")
        print(f"  {t.number:>4}  {t.board.template:<{width}}  {t.n_candidates:>5}  {(t.top_answer or '-'):<{width}}  "
              f"{_pct(t.p_top):>6}  "
              + (f"{_pct(t.p_unknown if not t.board.solved else math.nan):>6}  " if judged else "")
              + f"{'yes' if t.top_correct else '-':>5}  {_pct(t.p_truth):>8}  "
              f"{(t.truth_rank or '-'):>4}  {t.scores.get('brier', math.nan):5.3f}  {t.signal:6.3f}  "
              f"{('WIN' if t.guess_correct else 'wrong') if t.guess else '':>5}   {call}")
    print()
    won = game.won_by_guess
    if won is not None or game.guesses:
        print("solved by guessing               : "
              + ("no" if won is None else
                 f"turn {won.number} ({won.letters_before} letters called, {won.n_hidden} letters still hidden)"))
        print(f"wrong guesses                    : {len(game.wrong_guesses)}"
              + (f" ({', '.join(game.wrong_guesses)})" if game.wrong_guesses else ""))
    found = game.first_found()
    print("truth first among the candidates : "
          + ("never" if found is None else f"turn {found} ({game.turns[found].letters_before} letters called)"))
    right = game.right_from()
    print("top answer right for good from   : "
          + ("never" if right is None else
             f"turn {right} ({game.turns[right].letters_before} letters called, "
             f"{game.turns[right].n_hidden} letters still hidden)"))
    for m in game.milestones():
        if m.turn is None:
            print(f"first >= {m.level:.0%} sure on one answer : never")
        else:
            print(f"first >= {m.level:.0%} sure on one answer : turn {m.turn} ({m.hidden_left} letters still hidden): "
                  f"{m.answer}  ({'right' if m.correct else 'WRONG'})")
    sure_wrong = game.wrong_and_sure(0.9)
    if sure_wrong:
        print(f"turns >= 90% sure of a WRONG answer: {', '.join(str(t.number) for t in sure_wrong)}")
    p, hit = game.call_calibration()
    if not math.isnan(p):
        print(f"letters it called: it gave them {p:.0%} on average to be in the puzzle; {hit:.0%} were")


def print_letter_history(game, letters: str | None = None) -> None:
    """P(letter in puzzle) for each letter, turn by turn: how every letter's probability moved.

    A cell is blank once the letter has been called; the call itself shows as [x2] (found twice)
    or [ - ] (not in the puzzle). Letters default to every letter that ever had at least 5%.
    """
    turns = game.turns
    if letters is None:
        letters = "".join(c for c in LETTERS if any(t.presence.get(c, 0) >= 0.05 for t in turns)
                          or c in game.letters_called)
    truth_letters = set(game.answer)
    _rule("P(LETTER IN PUZZLE), TURN BY TURN   [xN] = called here and found N times, [ - ] = called, not there")
    print("  letter  in truth  " + "".join(f"{t.number:>6}" for t in turns))
    for c in letters:
        cells = []
        for t in turns:
            if t.called == c:
                cells.append(f"  [x{t.count}]" if t.hit else "  [ - ]")
            elif c in t.board.called:
                cells.append(" " * 6)
            else:
                cells.append(f"{_pct(t.presence.get(c, 0.0)):>6}")
        print(f"  {c:<6}  {'yes' if c in truth_letters else ' - ':>8}  " + "".join(cells))
