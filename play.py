"""Play a puzzle and watch the solver work through it, one letter at a time.

    python play.py                          a puzzle at random from the bundled 100
    python play.py 42                       puzzle 42
    python play.py "DOUBLE DUTCH"           the first bundled puzzle whose answer contains this
    python play.py --list                   every bundled puzzle, as a board (no spoilers)
    python play.py --new                    type in your own puzzle
    python play.py --answer "A WISH COME TRUE" --category PHRASE
    python play.py --called RSTLNE          start from a board that already shows R S T L N E
    python play.py --batch 10               play ten puzzles and print one row each

Each turn the solver re-runs the model on the current board, prints what it believes, and either
calls a letter or solves. Settings (model, search width, which letter policy, whether the answer
judge is on) all live in config.py — `python config.py` prints them and checks your setup.
"""

from __future__ import annotations

import argparse
import sys
import time

import config


def check_install() -> None:
    """Fail early and helpfully, rather than with an ImportError mid-game."""
    import importlib.util

    missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
    if missing:
        sys.exit(f"{' and '.join(missing)} not installed, and the solver needs a language model.\n"
                 f"  pip install -r requirements.txt\n"
                 f"Then run `python config.py` to check the rest of the setup.")


def list_puzzles(puzzles, answers: bool = False) -> None:
    print(f"\n{len(puzzles.play)} puzzles. Play one with:  python play.py <number>\n")
    width = max(len(p.clue.category) for p in puzzles.play)
    for i, p in enumerate(puzzles.play, 1):
        board = p.board().template
        hint = f"   hint: {p.clue.hint}" if p.clue.hint else ""
        print(f"  {i:>3}.  {p.clue.category:<{width}}  {p.answer if answers else board}"
              f"   ({sum(c.isalpha() for c in p.answer)} letters){hint}")
    print(f"\n{puzzles.note}\n")


def ask_for_puzzle():
    """--new: ask for a category and an answer at the prompt."""
    from wof.puzzles import from_text

    print("\nType in your own puzzle. The solver is told the category, never the answer.\n")
    cats = list(config.puzzles().categories())
    print("categories the bundled examples cover (anything else is fine too, the model just gets")
    print(f"fewer worked examples of it): {', '.join(cats)}\n")
    category = input("category  (e.g. PHRASE): ").strip() or "PHRASE"
    answer = input("answer    (e.g. A WISH COME TRUE): ").strip()
    hint = input("hint      (optional, Enter to skip): ").strip()
    return from_text(answer, category, hint)


def play_one(puzzle, solver, puzzles, args, judge, vocab, label: str = "") -> None:
    from wof import show
    from wof.calling import play_letters

    board = puzzle.board(args.called)
    if not args.quiet:
        show.print_puzzle(puzzle, board, reveal_answer=True)
    examples = config.prompt_examples(puzzle, puzzles.examples)
    look = config.turn_solver(solver, puzzle, examples, judge=judge, vocab=vocab,
                              truth=puzzle.answer if args.score_truth else None)
    t0 = time.time()
    game = play_letters(puzzle.answer, look, start=board,
                        on_turn=None if args.quiet else show.print_turn, **config.play_kwargs())
    if args.quiet:
        from wof.calling import game_summary
        r = game_summary(game)
        right = "never" if r["right"] is None else f"after {r['right']} letters"
        print(f"  {label}{puzzle.answer:<32} [{puzzle.clue.category}] "
              f"{r['letters']} letters, {r['misses']} missed, right {right}, "
              f"{'solved by guessing' if r['won'] else 'no winning guess'}, "
              f"{r['wrong_guesses']} wrong guess(es)  ({time.time() - t0:.0f}s)", flush=True)
    else:
        show.print_game(game)
        if judge is not None and hasattr(judge, "stats"):
            s = judge.stats
            print(f"\nanswer judge: {s.questions} question(s) asked, {s.cache_hits} answered from the "
                  f"cache, {s.requests} request(s), {s.seconds:.1f}s waiting")
        print(f"\ntotal {time.time() - t0:.0f}s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("puzzle", nargs="?", default=None,
                    help="a bundled puzzle: its number, or part of its answer. Default: one at random")
    ap.add_argument("--list", action="store_true", help="list the bundled puzzles and exit")
    ap.add_argument("--answers", action="store_true", help="with --list, show the answers too")
    ap.add_argument("--new", action="store_true", help="type in your own puzzle")
    ap.add_argument("--answer", help="your own puzzle's answer (with --category)")
    ap.add_argument("--category", default="PHRASE", help="your own puzzle's category (default PHRASE)")
    ap.add_argument("--hint", default="", help="your own puzzle's hint, if it has one")
    ap.add_argument("--called", default="", metavar="LETTERS",
                    help="letters already called before the solver starts, e.g. RSTLNE")
    ap.add_argument("--batch", type=int, metavar="N",
                    help="play N bundled puzzles in a row and print one line each")
    ap.add_argument("--seed", type=int, default=0, help="which random puzzle / batch you get")
    ap.add_argument("--no-judge", action="store_true", help="play without the answer judge")
    ap.add_argument("--no-fill", action="store_true", help="skip the fill-one-word pass (faster)")
    ap.add_argument("--quiet", action="store_true", help="one line per game instead of every turn")
    ap.add_argument("--no-score-truth", dest="score_truth", action="store_false",
                    help="don't also score the true answer each turn (it never changes a call)")
    args = ap.parse_args(argv)

    puzzles = config.puzzles()
    if args.list:
        list_puzzles(puzzles, args.answers)
        return
    if args.no_judge:
        config.USE_JUDGE = False
    if args.no_fill:
        config.FILL_SLOTS = False
    check_install()

    if args.new:
        chosen = [ask_for_puzzle()]
    elif args.answer:
        from wof.puzzles import from_text
        chosen = [from_text(args.answer, args.category, args.hint)]
    elif args.batch:
        import random
        order = list(puzzles.play)
        random.Random(args.seed).shuffle(order)
        chosen = order[:args.batch]
    else:
        chosen = [puzzles.get(args.puzzle if args.puzzle is not None else "random", seed=args.seed)]

    solver = config.load_model()
    judge = config.make_judge()
    vocab = config.fill_vocab(puzzles.examples) if config.FILL_SLOTS else None
    if args.batch:
        print(f"\nplaying {len(chosen)} puzzles with {config.POLICY}, "
              f"{'judge on' if judge else 'no judge'}:\n")
        args.quiet = True
    for k, puzzle in enumerate(chosen, 1):
        label = f"{k:>3}/{len(chosen)}  " if len(chosen) > 1 else ""
        play_one(puzzle, solver, puzzles, args, judge, vocab, label)


if __name__ == "__main__":
    main()
