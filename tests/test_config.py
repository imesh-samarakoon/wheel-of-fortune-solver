"""config.py is the user-facing surface, so check that its settings really reach the solver."""

import pytest

import config
from wof.board import Board
from wof.judge import FixedJudge
from wof.policy import most_likely
from wof.puzzles import from_text
from wof.search import Path, SearchResult, TokenStep


class StubSolver:
    """Stands in for LLMSolver: one fixed search result, and scoring from a table."""

    model_name = "stub"

    def __init__(self, found):
        self.found = list(found)

    def beam_search(self, prompt, board, **kwargs):
        self.kwargs = kwargs
        paths = [Path(t, [TokenStep(" " + t, s, 0.0)]) for t, s in self.found if board.consistent(t)]
        return SearchResult("beam", prompt, board, paths)

    def score(self, prompt, board, text, case=None):
        return Path(text, [TokenStep(" " + text, -1.0, 0.0)])

    def score_many(self, prompt, board, texts, **kwargs):
        return [self.score(prompt, board, t) for t in texts]


def test_the_search_settings_reach_beam_search(monkeypatch):
    monkeypatch.setattr(config, "BEAMS", 7)
    monkeypatch.setattr(config, "EXPAND", 23)
    monkeypatch.setattr(config, "MAX_PER_FIRST_WORD", 3)
    monkeypatch.setattr(config, "FILL_SLOTS", False)
    puzzle = from_text("SMALL TALK", "PHRASE")
    solver = StubSolver([("Small Talk", -2.0)])
    look = config.turn_solver(solver, puzzle, [])
    look(puzzle.board())
    assert solver.kwargs["beams"] == 7
    assert solver.kwargs["expand"] == 23
    assert solver.kwargs["max_per_first_word"] == 3
    assert solver.kwargs["case"] == config.ANSWER_STYLE  # answers written in the examples' style


def test_fill_slots_needs_a_vocab_when_it_is_on(monkeypatch):
    monkeypatch.setattr(config, "FILL_SLOTS", True)
    with pytest.raises(ValueError, match="vocab"):
        config.turn_solver(StubSolver([]), from_text("SMALL TALK", "PHRASE"), [])


def test_the_judge_reaches_the_posterior(monkeypatch):
    """A judge that calls every candidate nonsense should move the probability to "none of these"."""
    monkeypatch.setattr(config, "FILL_SLOTS", False)
    monkeypatch.setattr(config, "CARRY", 0)
    puzzle = from_text("SMALL TALK", "PHRASE")
    solver = StubSolver([("Small Talk", -2.0), ("Smarl Tajk", -2.1)])
    plain, _ = config.turn_solver(solver, puzzle, [])(puzzle.board())
    assert plain.unknown_mass == 0.0

    judge = FixedJudge({}, default=0.02)  # "none of these are real English"
    judged, _ = config.turn_solver(solver, puzzle, [], judge=judge)(puzzle.board())
    assert judged.unknown_mass > 0.5
    assert judge.asked  # it really was asked


def test_make_judge_is_off_without_a_key(monkeypatch, capsys):
    from wof import judge as judge_mod

    monkeypatch.delenv(judge_mod.API_KEY_ENV, raising=False)
    monkeypatch.setattr(config, "USE_JUDGE", True)
    assert config.make_judge() is None
    assert judge_mod.API_KEY_ENV in capsys.readouterr().out  # says what to set, once
    monkeypatch.setattr(config, "USE_JUDGE", False)
    assert config.make_judge(quiet=True) is None


def test_play_kwargs_and_the_policy_name(monkeypatch):
    monkeypatch.setattr(config, "POLICY", "likely")
    kw = config.play_kwargs()
    assert kw["policy"] is most_likely
    assert kw["solve_at"] == config.SOLVE_AT and kw["min_top"] == config.SOLVE_MIN_TOP
    monkeypatch.setattr(config, "POLICY", "nonsense")
    with pytest.raises(ValueError, match="POLICY"):
        config.letter_policy()


def test_the_bundled_puzzles_and_their_word_list_load():
    ps = config.puzzles()
    vocab = config.fill_vocab(ps.examples)
    assert len(vocab) > 100
    board = Board.from_answer("A WISH COME TRUE", set("AELOR"))
    assert board.template == "A ____ _O_E _R_E"
