import math

import pytest

from wof.board import Board, normalize
from wof.search import prefix_ok
from wof.policy import most_info, most_likely
from wof.posterior import AnswerPosterior, Candidate


def test_normalize_keeps_punctuation_and_strips_accents():
    assert normalize("  crème  brûlée ") == "CREME BRULEE"
    assert normalize("rock ’n’ roll") == "ROCK 'N' ROLL"


def test_board_reveals_every_occurrence_and_keeps_punctuation():
    b = Board.from_answer("Piece of Cake", set("RSTLNE"))
    assert b.template == "__E_E __ ___E"
    assert b.n_hidden == 8  # P I C O F C A K
    assert b.absent == frozenset({"R", "S", "T", "L", "N"})


def test_consistent_rejects_called_letters_in_hidden_slots():
    b = Board.from_answer("CAT", {"C", "O"})  # O called and absent
    assert b.template == "C__"
    assert b.consistent("CAT")
    assert not b.consistent("COT")  # O was called and isn't there
    assert not b.consistent("CCT")  # C was called: every C is already showing
    assert not b.consistent("CATS")


def test_regex_with_every_letter_called():
    b = Board("A_", frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    assert not b.consistent("AB")


def test_reveal_positions_matches_reveal():
    b = Board.from_answer("BANANA")
    assert b.reveal_positions("A", [1, 3, 5]) == b.reveal("A", "BANANA")
    with pytest.raises(ValueError):
        b.reveal("A", "BANANA").reveal_positions("N", [1])


def _post():
    b = Board.from_answer("CAT", {"A"})
    cands = [Candidate("CAT", math.log(0.5)), Candidate("BAT", math.log(0.3)),
             Candidate("HAT", math.log(0.2)), Candidate("DOG", 0.0)]  # DOG doesn't fit
    return AnswerPosterior(b, cands)


def test_posterior_filters_and_normalizes():
    p = _post()
    assert len(p) == 3
    assert sum(p.probs) == pytest.approx(1.0)
    assert p.prob_of("cat") == pytest.approx(0.5)


def test_slot_and_presence_marginals():
    p = _post()
    m = p.slot_marginals()
    assert set(m) == {0, 2}
    assert m[0] == pytest.approx({"C": 0.5, "B": 0.3, "H": 0.2})
    assert m[2] == pytest.approx({"T": 1.0})
    pres = p.presence()
    assert pres["T"] == pytest.approx(1.0) and pres["C"] == pytest.approx(0.5) and pres["Z"] == 0
    assert "A" not in pres  # already called


def test_information_gain_prefers_the_disambiguating_letter():
    p = _post()
    gain = most_info(p)
    # T is certain, so calling it tells us nothing; C/B/H each split the set.
    assert gain["T"] == pytest.approx(0.0, abs=1e-12)
    assert max(gain, key=gain.get) in {"C", "B", "H"}
    assert most_likely(p)["T"] == pytest.approx(1.0)


def test_after_call_updates_posterior():
    p = _post().after_call("C", [])  # C not in puzzle
    assert p.prob_of("CAT") == 0
    assert p.prob_of("BAT") == pytest.approx(0.6)


def test_llm_prefix_logic_without_a_model():
    allowed = frozenset("ABCDFGHIJKMOPQUVWXYZ")  # RSTLNE called
    tmpl = " __E_E"
    assert prefix_ok(tmpl, allowed, " PIE")
    assert not prefix_ok(tmpl, allowed, " PRE")  # R was called
    assert not prefix_ok(tmpl, allowed, " PIECES")  # too long
    assert prefix_ok(tmpl, allowed, " Piece")  # letters match the board case-insensitively
    assert not prefix_ok(tmpl, allowed, " Pi-")  # punctuation must match exactly
