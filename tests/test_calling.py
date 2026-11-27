"""Calling letters one at a time (wof.calling), with a fixed candidate list standing in for the model."""

import math

import pytest

from wof.board import Board
from wof.calling import Game, choose_letter, play_letters
from wof.policy import most_info
from wof.posterior import AnswerPosterior, Candidate

# A model that "knows" these answers with these weights, whatever the board.
WEIGHTS = {"CAT": 0.5, "BAT": 0.3, "HAT": 0.15, "RAT": 0.05}


def fake_solver(weights=WEIGHTS, info=None):
    calls = []

    def look(board: Board):
        calls.append(board.template)
        post = AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in weights.items()])
        return (post, dict(info)) if info is not None else post

    look.calls = calls
    return look


def test_plays_until_the_board_is_full_and_counts_each_reveal():
    look = fake_solver()
    game = play_letters("HAT", look)
    assert game.stopped == "solved"
    # A and T are in every candidate (100%); T and A tie on expected count, English order picks T first
    assert game.letters_called[:2] == "TA"
    assert [t.count for t in game.calls[:2]] == [1, 1]
    # Then C (50%) misses, B (30% of the rest... renormalized) misses, then H
    assert game.letters_called[2:] == "CBH"
    assert [t.hit for t in game.calls] == [True, True, False, False, True]
    assert (game.hits, game.misses) == (3, 2)
    last = game.turns[-1]
    assert last.board.template == "HAT" and last.called is None
    assert last.p_top == 1.0 and last.top_correct
    # The model is never asked about the full board
    assert "HAT" not in look.calls and len(look.calls) == len(game.turns) - 1


def test_turns_track_the_top_answer_and_the_truth():
    game = play_letters("HAT", fake_solver())
    first = game.turns[0]
    assert first.top_answer == "CAT" and first.p_top == pytest.approx(0.5)
    assert first.truth_rank == 3 and first.p_truth == pytest.approx(0.15)
    assert first.p_called == pytest.approx(1.0)  # T: in every candidate
    after_c_miss = game.turns[3]  # T, A, C called: C wasn't there
    assert after_c_miss.board.absent == {"C"}
    assert after_c_miss.top_answer == "BAT" and after_c_miss.p_top == pytest.approx(0.6)
    assert after_c_miss.p_truth == pytest.approx(0.3)
    assert game.first_found() == 0
    assert game.right_from() == 4  # once B missed too, HAT is the top answer for good


def test_milestones_report_when_and_whether_it_was_right():
    game = play_letters("HAT", fake_solver())
    m50, m90, m99 = game.milestones()
    assert (m50.turn, m50.answer, m50.correct) == (0, "CAT", False)
    # 75% on HAT before the H is called; 90% only once the board is full
    assert game.turns[4].p_top == pytest.approx(0.75)
    assert (m90.turn, m90.answer, m90.correct, m90.hidden_left) == (5, "HAT", True, 0)
    assert game.wrong_and_sure(0.9) == []


def test_stop_at_ends_the_game_once_confident():
    game = play_letters("CAT", fake_solver(), stop_at=0.5)
    assert game.stopped == "confident" and len(game.turns) == 1 and not game.calls


def test_truth_outside_the_candidates():
    game = play_letters("MAT", fake_solver())
    assert game.first_found() is None or game.turns[game.first_found()].board.solved
    assert game.turns[0].truth_rank is None and game.turns[0].p_truth == 0
    assert game.turns[-1].top_answer == "MAT"  # the full board, no model needed
    assert game.stopped == "solved"


def test_empty_posterior_falls_back_to_letter_frequency():
    game = play_letters("ZZZ", fake_solver({"CAT": 1.0}), max_letters=3)
    # T (in CAT, the only candidate) misses; then no candidate fits, so English frequency order: E, A
    assert game.letters_called == "TEA"
    assert game.stopped == "max letters"


def test_choose_letter_respects_kind_and_falls_back():
    post = AnswerPosterior(Board.from_answer("CAT"), [Candidate(a, math.log(w)) for a, w in WEIGHTS.items()])
    assert choose_letter(post, kind="vowel") == "A"
    assert choose_letter(post, kind="consonant") == "T"
    assert choose_letter(post, most_info) in {"C", "B", "H", "R"}
    vowels_done = AnswerPosterior(Board.from_answer("CAT", set("AEIOU")), [Candidate("CAT", 0.0)])
    assert choose_letter(vowels_done, kind="vowel") in {"C", "T"}  # no vowels left: any letter


def test_info_and_start_board():
    start = Board.from_answer("HAT", {"T", "Z"})
    game = play_letters("HAT", fake_solver(info={"seconds": 0.1}), start=start)
    assert game.turns[0].board == start and game.turns[0].info == {"seconds": 0.1}
    assert "T" not in game.letters_called and game.turns[0].number == 0
    with pytest.raises(ValueError):
        play_letters("HAT", fake_solver(), start=Board.from_answer("CAT", {"C"}))


def test_call_calibration():
    game = play_letters("HAT", fake_solver())
    p, hit = game.call_calibration()
    assert hit == pytest.approx(3 / 5)
    assert 0 < p <= 1
    assert isinstance(game, Game)


def test_a_turn_either_solves_or_calls_a_letter():
    from wof.calling import expected_brier

    look = fake_solver()
    game = play_letters("HAT", look, solve_at=0.05)
    # Turn 0: CAT 50% on top, expected Brier = (.25 + .21 + .1275 + .0475) / 26 letters = 0.024: solve with CAT, wrong
    t0 = game.turns[0]
    assert t0.signal == pytest.approx(expected_brier(t0.presence)) == pytest.approx(0.635 / 26)
    assert (t0.guess, t0.guess_correct, t0.called) == ("CAT", False, None)  # no letter on a solve turn
    # Turn 1: same board, CAT gone, BAT on top: solve again, wrong again; the model isn't asked again
    t1 = game.turns[1]
    assert t1.board == t0.board and "reused" in t1.info
    assert "CAT" not in dict(t1.top) and (t1.guess, t1.guess_correct, t1.called) == ("BAT", False, None)
    # Turn 2: HAT 75%: solve, right, game over. Won without calling a single letter
    t2 = game.turns[2]
    assert (t2.guess, t2.guess_correct, t2.called) == ("HAT", True, None)
    assert game.stopped == "guessed" and game.won_by_guess is t2
    assert game.letters_called == "" and game.wrong_guesses == ["CAT", "BAT"]
    assert look.calls == ["___"]  # one model run for three turns


def test_after_a_wrong_guess_it_goes_back_to_calling_letters():
    look = fake_solver({"CAT": 0.9, "HAT": 0.05, "BAT": 0.05})
    game = play_letters("HAT", look, solve_at=0.008)
    # Turn 0: CAT 90%, expected Brier (.09 + .0475 + .0475) / 26 = 0.0071: solve with CAT, wrong
    assert (game.turns[0].guess, game.turns[0].called) == ("CAT", None)
    # Turn 1: HAT vs BAT 50/50, expected Brier 0.5 / 26 = 0.019 > 0.008: call a letter instead
    t1 = game.turns[1]
    assert t1.guess is None and t1.called == "T" and t1.letters_before == 0
    assert game.letters_called == "TAH" and game.stopped == "solved"  # then fills the board
    assert [t.letters_before for t in game.turns] == [0, 0, 1, 2, 3]
    assert look.calls == ["___", "__T", "_AT"]  # turn 1 reused turn 0's search


def test_no_guess_above_the_threshold_and_never_on_a_full_board():
    game = play_letters("HAT", fake_solver(), solve_at=0.001)
    assert not any(t.guess for t in game.turns[:-1])  # never sure enough
    assert game.stopped == "solved" and game.turns[-1].guess is None  # the full board isn't a "guess"
    assert game.letters_called == "TACBH"  # same game as without solve_at


def test_brier_signal_peeks_at_the_truth():
    game = play_letters("HAT", fake_solver(), solve_at=0.05, solve_signal="brier")
    assert game.turns[0].signal == pytest.approx(game.turns[0].scores["brier"])
    with pytest.raises(ValueError):
        play_letters("HAT", fake_solver(), solve_at=0.05, solve_signal="p_top")


def test_game_summary_reports_guesses():
    from wof.calling import game_summary

    row = game_summary(play_letters("HAT", fake_solver(), solve_at=0.05))
    assert (row["won"], row["wrong_guesses"], row["letters"], row["found"], row["right"]) == (True, 2, 0, 0, 0)
    row = game_summary(play_letters("HAT", fake_solver({"CAT": 0.9, "HAT": 0.05, "BAT": 0.05}), solve_at=0.008))
    # HAT is on top from turn 1 on, before any letter: "right" counts letters, not turns
    assert (row["won"], row["wrong_guesses"], row["letters"], row["right"]) == (False, 1, 3, 0)


# --- splitting the leading answers instead of calling sure letters ----------------------------------

# Turn 11 of a real game: STRAWBERRY & PAPER _A_, the true answer JAM second.
LATE = {"STRAWBERRY & PAPER FAN": 0.429, "STRAWBERRY & PAPER JAM": 0.242, "STRAWBERRY & PAPER MAN": 0.081,
        "STRAWBERRY & PAPER DAM": 0.071, "STRAWBERRY & PAPER OAK": 0.037}


def _late_post(called="ABELPRTWY"):
    board = Board.from_answer("STRAWBERRY & PAPER JAM", called)
    return AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in LATE.items()])


def test_decisive_skips_letters_every_leading_answer_shares():
    from wof.policy import most_decisive, most_likely

    post = _late_post()
    assert choose_letter(post, most_likely) == "S"  # 100% there, same place in every candidate: teaches nothing
    scores = most_decisive(post)
    p_top = max(LATE.values()) / sum(LATE.values())
    assert scores["S"] == pytest.approx(p_top)      # can't change which answer it would guess
    # M splits FAN / JAM+DAM / MAN three ways: after it, the guess is right with P(FAN)+P(JAM)+P(MAN)
    assert choose_letter(post, most_decisive) == "M"
    assert scores["M"] == pytest.approx((0.429 + 0.242 + 0.081) / sum(LATE.values()))


def test_switching_policy_waits_for_the_leaders():
    from wof.policy import switch_when_close, most_decisive, most_likely, top_share
    from wof.posterior import Unknown

    sure = _late_post()
    policy = switch_when_close(most_likely, most_decisive, top=4, share=0.5)
    assert top_share(sure) > 0.5 and choose_letter(sure, policy) == "M"
    # the same candidates under a big "none of these" share: keep revealing sure letters
    doubtful = AnswerPosterior(sure.board, [Candidate(a, math.log(w)) for a, w in LATE.items()], Unknown(0.8, {}, {}))
    assert top_share(doubtful) < 0.5 and choose_letter(doubtful, policy) == "S"


def test_decisive_game_solves_in_fewer_turns():
    from wof.policy import most_decisive, most_likely

    look = fake_solver(LATE)
    start = Board.from_answer("STRAWBERRY & PAPER JAM", "ABELPRTWY")
    greedy = play_letters("STRAWBERRY & PAPER JAM", look, most_likely, start=start, solve_at=0.0)
    decisive = play_letters("STRAWBERRY & PAPER JAM", look, most_decisive, start=start, solve_at=0.0)
    assert len(decisive.turns) < len(greedy.turns)


def test_switching_waits_until_a_quarter_of_the_board_shows():
    """THE SPEED OF LIGHT, turn 0: the top 4 held ~70% of an empty board's probability, so it switched to
    decisive and called A (which split BOARD from THING) and missed; E would have hit 3x."""
    from wof.policy import revealed_share, switch_when_close
    from wof.posterior import Unknown

    early = {"THE BOARD IS EMPTY": 45.2, "THE THING IS ALIVE": 13.3, "THE BOARD IS BLANK": 6.3,
             "THE THING IN SPACE": 4.8, "THE BOARD IS CLEAR": 3.0}
    empty = Board.from_answer("THE SPEED OF LIGHT", "")
    post = AnswerPosterior(empty, [Candidate(a, math.log(w)) for a, w in early.items()], Unknown(0.16, {}, {}))
    assert revealed_share(empty) == 0
    assert choose_letter(post, switch_when_close(min_revealed=0)) == "A"  # the old behaviour
    assert choose_letter(post, switch_when_close(min_revealed=0.25)) == "E"
    # a quarter showing: the late board of the Strawberry game switches as before
    late = _late_post()
    assert revealed_share(late.board) >= 0.25 and choose_letter(late, switch_when_close(min_revealed=0.25)) == "M"
