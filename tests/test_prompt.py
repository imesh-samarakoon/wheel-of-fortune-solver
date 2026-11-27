"""The prompt the model is given, and the bookkeeping on a search result. No model needed."""

import math

import pytest

from wof.board import Board
from wof.puzzles import Clue, Puzzle
from wof.search import (Path, SearchResult, TokenStep, build_prompt, example_reveals, styled,
                        title_case)


def test_prompt_contents():
    board = Board.from_answer("TOM SELLECK", {"T"})
    ex = Puzzle("BURT LANCASTER", Clue("PERSON", year="1983"))
    p = build_prompt(Clue("PERSON", hint="H", year="1983", context="C"), board, [ex])
    assert "Answer: Burt Lancaster\n" in p
    assert "Board: ____ _________" in p
    assert p.endswith("Category: PERSON\nHint: H\nYear: 1983\nContext: C\nBoard: T__ _______\nAnswer:")
    assert "Board:" not in build_prompt(Clue("PERSON"), board, show_board=False)


def test_the_prompt_never_shows_the_target_answer():
    """The target block ends at "Answer:" — the model has to write the rest."""
    board = Board.from_answer("A PIECE OF CAKE")
    p = build_prompt(Clue("PHRASE"), board, [Puzzle("BITE THE BULLET", Clue("PHRASE"))])
    assert "A PIECE OF CAKE" not in p.upper()
    assert p.count("Answer:") == 2 and p.endswith("Answer:")


def test_title_case():
    assert title_case("THE WAY TO A MAN'S HEART") == "The Way To A Man's Heart"
    assert styled("full-length mirror", "title") == "Full-Length Mirror"
    assert styled("Kenny Rogers", "upper") == "KENNY ROGERS"


def test_search_result_merges_spellings_and_weights():
    board = Board.from_answer("CAT")
    step = lambda s, lp: TokenStep(s, lp, 0.0)
    paths = [
        Path(" Cat", [step(" Cat", math.log(0.2))]),
        Path(" CAT", [step(" C", math.log(0.1)), step("AT", math.log(0.5))]),  # same answer, 0.05
        Path(" Dog", [step(" Dog", math.log(0.25))]),
    ]
    res = SearchResult("beam", "", board, paths)
    w = res.answer_logweights()
    assert math.exp(w["CAT"]) == pytest.approx(0.25)
    post = res.posterior()
    assert post.prob_of("CAT") == pytest.approx(0.5)
    assert post.presence()["A"] == pytest.approx(0.5)

    # sampling: weight = P / q
    s = SearchResult("sample", "", board, [Path(" Cat", [step(" Cat", math.log(0.2))], q_logprob=math.log(0.8)),
                                          Path(" Dog", [step(" Dog", math.log(0.2))], q_logprob=math.log(0.2))])
    assert s.posterior().prob_of("DOG") == pytest.approx(0.8)  # weights 0.25 vs 1.0
    assert s.fit_mass_estimate() == pytest.approx(0.625)


def test_example_boards_can_show_revealed_letters():
    examples = [Puzzle("BITE THE BULLET", Clue("PHRASE")), Puzzle("BREAK A LEG", Clue("PHRASE"))]
    target = Board.from_answer("HIT THE ROAD", {"T", "E", "Z"})
    assert example_reveals(examples, target, "none") == [frozenset(), frozenset()]
    assert example_reveals(examples, target, "same") == [target.called] * 2
    rnd = example_reveals(examples, target, "random", seed=1)
    assert rnd == example_reveals(examples, target, "random", seed=1)  # reproducible
    assert all(set(c) <= set("ETAOINSRHLDC") for c in rnd)  # frequency order, at most 12 letters

    prompt = build_prompt(Clue("PHRASE"), target, examples, example_called=example_reveals(examples, target, "same"),
                          show_called=True)
    assert "Board: __TE T_E ____ET\nCalled: E T Z (not in puzzle: Z)\nAnswer: Bite The Bullet" in prompt
    assert "Board: __E__ _ _E_\nCalled: E T Z (not in puzzle: T Z)\nAnswer: Break A Leg" in prompt
    assert prompt.endswith("Board: __T T_E ____\nCalled: E T Z (not in puzzle: Z)\nAnswer:")
    blank = build_prompt(Clue("PHRASE"), target, examples)  # defaults: blank examples, no Called line
    assert "Board: ____ ___ ______" in blank and "Called:" not in blank
