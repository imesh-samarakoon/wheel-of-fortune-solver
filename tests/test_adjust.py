"""Corrections for answers the beam search missed: beams + samples, dataset baseline, blend. No model."""

import math

import pytest

from wof.adjust import hybrid_posterior, letter_scores
from wof.board import Board
from wof.search import Path, SearchResult, TokenStep, first_word, open_word


def _path(text, p, q=None):
    return Path(text, [TokenStep(text, math.log(p), 0.0)], q_logprob=math.log(q) if q else 0.0)


def test_first_word_is_case_folded():
    assert first_word(" Kfc Man") == "KFC"
    assert first_word(" KFC") == "KFC"
    assert first_word("") == ""


def test_open_word_skips_words_the_board_gives_away():
    empty = " " + Board.from_answer("A STITCH IN TIME").template
    assert open_word(" Kfc Man", " " + Board.from_answer("KFC MANAGER").template) == "KFC"  # empty board: first word
    assert open_word(" A Sti", empty) == "A"
    called = " " + Board.from_answer("A STITCH IN TIME", {"A"}).template  # " A ______ __ ____"
    assert open_word(" A", called) is None  # hasn't reached a word with a hidden letter
    assert open_word(" A Sti", called) == "STI"
    assert open_word(" A Stitch In", called) == "STITCH"


def test_hybrid_adds_missed_answers_at_their_estimated_weight():
    board = Board.from_answer("CAT")
    beam = SearchResult("beam", "", board, [_path(" Cat", 0.02), _path(" Bat", 0.01)])
    # 4 draws (1 dead end): CAT again (already found: ignored), HAT twice, a dead end
    draws = [_path(" Cat", 0.02, 0.5), _path(" Hat", 0.03, 0.2), _path(" Hat", 0.03, 0.2)]
    sample = SearchResult("sample", "", board, draws, n_draws=4)
    post, info = hybrid_posterior(beam, sample)
    # HAT's weight = (0.15 + 0.15) / 4 draws = 0.075
    assert info.found_mass == pytest.approx(0.03)
    assert info.missed_mass == pytest.approx(0.075)
    assert info.coverage == pytest.approx(0.03 / 0.105)
    assert post.prob_of("HAT") == pytest.approx(0.075 / 0.105)
    assert post.prob_of("CAT") == pytest.approx(0.02 / 0.105)
    assert dict(zip(post.answers, post.sources)) == {"CAT": "beam", "BAT": "beam", "HAT": "sample"}
    assert info.n_outside == 2 and info.n_outside_answers == 1


def test_letter_scores_grade_the_letter_probabilities():
    board = Board.from_answer("CAT", {"C"})          # "C__": A and T hidden, C already called
    perfect = {"A": 1.0, "T": 1.0, "B": 0.0, "Z": 0.0}
    assert letter_scores(perfect, board, "CAT")["brier"] == pytest.approx(0.0)
    hedged = {c: 0.5 for c in perfect}
    assert letter_scores(hedged, board, "CAT")["brier"] == pytest.approx(0.25)
    backwards = {"A": 0.0, "T": 0.0, "B": 1.0, "Z": 1.0}
    assert letter_scores(backwards, board, "CAT")["brier"] == pytest.approx(1.0)
    assert letter_scores(hedged, board, "CAT")["logloss"] == pytest.approx(-math.log(0.5))


def test_letter_cases_follow_the_examples_style():
    from wof.search import letter_cases, styled, token_fits

    template = " " + Board.from_answer("JACK-IN-THE-BOX MAN'S").template  # " ____-__-___-___ ___'_"
    title = letter_cases(template, "title")
    assert "".join("u" if c == "U" else "l" if c == "L" else "." for c in title) == ".ulll.ul.ull.ull.ull.l"
    allowed = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    text = " " + styled("JACK-IN-THE-BOX MAN'S", "title")  # what the examples look like
    assert token_fits(template, allowed, 0, text, "title")
    assert not token_fits(template, allowed, 0, text.upper(), "title")  # copying in capitals is out
    assert token_fits(template, allowed, 0, text.upper(), None)  # either case without a lock
    revealed = " " + Board.from_answer("THE CAT", {"T"}).template  # " T__ __T"
    assert token_fits(revealed, allowed - {"T"}, 0, " The Cat", "title")
    assert not token_fits(revealed, allowed - {"T"}, 0, " THE CAT", "title")
    assert not token_fits(revealed, allowed - {"T"}, 0, " the cat", "title")
