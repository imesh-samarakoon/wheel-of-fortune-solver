"""The answer judge (wof.candidates.judge): the "none of these" share, the reweighting, and JevJudge's HTTP
handling (faked: no real request is ever made)."""

import io
import json
import math
import urllib.error
import urllib.request

import pytest

from wof.board import Board
from wof.calling import LLMTurnSolver, play_letters, without
from wof import judge as J
from wof.posterior import AnswerPosterior, Candidate, Unknown
from wof.puzzles import Clue

CLUE = Clue("Phrase")
BOARD = Board.from_answer("DROP AND GIVE ME TWENTY!", "DANGIVEMY")  # D___ AND GIVE ME __EN_Y!
SEARCH = {"DROP AND GIVE ME PLENTY!": 0.614, "DROP AND GIVE ME TWENTY!": 0.309, "DROP AND GIVE ME FRENZY!": 0.077}
JUDGED = {"DROP AND GIVE ME PLENTY!": 0.2, "DROP AND GIVE ME TWENTY!": 0.95, "DROP AND GIVE ME FRENZY!": 0.1}


def _post(weights=SEARCH, board=BOARD, unknown=None):
    return AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in weights.items()], unknown)


# --- the unknown share on AnswerPosterior --------------------------------------------------------

def test_unknown_share_scales_candidates_and_mixes_letters():
    board = Board.from_answer("CAT", "A")
    plain = AnswerPosterior(board, [Candidate("CAT", 0.0), Candidate("BAT", 0.0)])
    u = Unknown(0.2, {"C": 0.0, "B": 0.0, "T": 0.0, "Z": 1.0}, {"Z": 2.0})
    post = AnswerPosterior(board, [Candidate("CAT", 0.0), Candidate("BAT", 0.0)], u)
    assert sum(post.probs) == pytest.approx(0.8)
    assert post.prob_of("CAT") == pytest.approx(0.4)
    assert post.unknown_mass == 0.2
    pres = post.presence()
    assert pres["T"] == pytest.approx(0.8) and pres["C"] == pytest.approx(0.4) and pres["Z"] == pytest.approx(0.2)
    assert post.expected_count()["Z"] == pytest.approx(0.4)
    assert post.entropy() > plain.entropy()
    # "what if I call this letter" stays over the candidates
    assert sum(post.outcomes("C").values()) == pytest.approx(1.0)
    assert post.information_gain()["C"] == pytest.approx(plain.information_gain()["C"])


def test_no_candidates_means_all_unknown():
    post = AnswerPosterior(Board.from_answer("CAT"), [], Unknown(0.3, {"E": 0.5}, {"E": 0.5}))
    assert post.empty and post.unknown_mass == 1.0 and post.presence()["E"] == 0.5


def test_without_renormalizes_the_unknown_share():
    post = _post(unknown=Unknown(0.1, {}, {}))
    out = without(post, {"DROP AND GIVE ME PLENTY!"})
    # remaining candidates had 0.9 * (0.309 + 0.077); unknown 0.1
    left = 0.9 * (0.309 + 0.077)
    assert out.unknown_mass == pytest.approx(0.1 / (0.1 + left))
    assert sum(out.probs) + out.unknown_mass == pytest.approx(1.0)


# --- reweighting ---------------------------------------------------------------------------------

def test_a_real_phrase_overtakes_a_nonsense_rival():
    post = _post()
    assert post.top(1)[0][0] == "DROP AND GIVE ME PLENTY!"
    new, res = J.apply_judge(post, JUDGED, prior_unknown=0.1, real_rate=0.9)
    assert new.top(1)[0][0] == "DROP AND GIVE ME TWENTY!"
    assert new.prob_of("DROP AND GIVE ME TWENTY!") > 0.5
    assert sum(new.probs) + new.unknown_mass == pytest.approx(1.0)
    assert res.prior_unknown == 0.1 and res.unknown == pytest.approx(new.unknown_mass)


def test_all_nonsense_moves_probability_to_none_of_these():
    board = Board.from_answer("DROP AND GIVE ME TWENTY!", "E")
    junk = {"WHAT YOU GIVE ME TREATS!": 0.632, "JUST ADD SOME LE CHEMIN!": 0.3, "HAVE FUN WITH MY FAMILY!": 0.068}
    post = AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in junk.items() if board.consistent(a)])
    new, _ = J.apply_judge(post, {a: 0.03 for a in junk}, prior_unknown=0.1, real_rate=0.9)
    assert new.unknown_mass > 0.6
    assert new.top(1)[0][1] < 0.25  # was 63%+
    # by default "none of these" has the candidates' own letters, so equal scores change no letter at all
    for c, p in post.presence().items():
        assert new.presence()[c] == pytest.approx(p)
    # the old English-frequency stand-in: no uncalled letter stays near-certain
    eng, _ = J.apply_judge(post, {a: 0.03 for a in junk}, prior_unknown=0.1, real_rate=0.9,
                           unknown_letters=J.english_letters)
    assert max(eng.presence().values()) < 0.95


def test_the_judge_changes_letters_only_by_reweighting():
    # Turn 2 of the real "Keeping The Main Thing The Main Thing" game. Without a judge, H/A/I/N are all
    # certain and H (expected most often) is called. The English-frequency stand-in made every one of
    # them uncertain and flipped the call to A. The default stand-in keeps them certain; the call only
    # moves (to I) because the judge rates "Baby Twins" (no H) above the "Fish" answers: reweighting.
    from wof.calling import choose_letter

    board = Board.from_answer("KEEPING THE MAIN THING THE MAIN THING", "ET")
    before = {"FEEDING THE FISH TANKS THE FISH TANKS": 47.9, "FEEDING THE BABY TWINS THE BABY TWINS": 29.7,
              "FEEDING THE FISH TAILS THE FISH TAILS": 11.6, "FEEDING THE FISH TALLY THE FISH TALLY": 5.5,
              "FEELING THE WIND TODAY THE WIND TODAY": 2.3}
    q = {"FEEDING THE FISH TANKS THE FISH TANKS": .09, "FEEDING THE BABY TWINS THE BABY TWINS": .18,
         "FEEDING THE FISH TAILS THE FISH TAILS": .10, "FEEDING THE FISH TALLY THE FISH TALLY": .08,
         "FEELING THE WIND TODAY THE WIND TODAY": .19}
    post = AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in before.items()])
    assert choose_letter(post) == "H"
    judged, _ = J.apply_judge(post, q, prior_unknown=0.1 * 23 / 30)
    assert all(judged.presence()[c] == pytest.approx(1.0) for c in "HAIN")
    assert choose_letter(judged) == "I"
    english, _ = J.apply_judge(post, q, prior_unknown=0.1 * 23 / 30, unknown_letters=J.english_letters)
    assert all(english.presence()[c] < 0.98 for c in "HAIN")
    assert choose_letter(english) == "A"


def test_weight_zero_and_no_prior_unknown_changes_nothing():
    post = _post()
    new, _ = J.apply_judge(post, JUDGED, weight=0.0, prior_unknown=0.0)
    for a in SEARCH:
        assert new.prob_of(a) == pytest.approx(post.prob_of(a))
    assert new.unknown_mass == 0.0


def test_unscored_candidates_count_as_real_rate():
    post = _post()
    new, res = J.apply_judge(post, {}, prior_unknown=0.0, real_rate=0.9)
    assert res.scores == {a: 0.9 for a in SEARCH}
    for a in SEARCH:
        assert new.prob_of(a) == pytest.approx(post.prob_of(a))


def test_english_letters_skips_called_letters_and_stays_uncertain():
    presence, counts = J.english_letters(_post())
    assert not set(presence) & BOARD.called
    assert all(0 < p < 1 for p in presence.values())
    assert sum(counts.values()) == pytest.approx(BOARD.n_hidden)
    assert presence["T"] > presence["Q"]


# --- in a game -----------------------------------------------------------------------------------

def test_judged_game_does_not_guess_the_nonsense_answer():
    weights = {"QAT": 0.9, "CAT": 0.1}
    fixed = J.FixedJudge({"QAT": 0.01, "CAT": 0.95})

    def look(board):
        post = AnswerPosterior(board, [Candidate(a, math.log(w)) for a, w in weights.items()])
        return J.apply_judge(post, fixed.judge(post.answers, CLUE), prior_unknown=0.05)[0]

    game = play_letters("CAT", look, solve_at=0.2)
    assert game.wrong_guesses == []
    assert game.turns[0].top_answer == "CAT"


def test_turn_solver_judges_candidates_and_the_truth():
    fixed = J.FixedJudge(JUDGED, default=0.5)
    look = LLMTurnSolver(None, CLUE, judge=fixed, judge_prior_unknown=0.1, truth="Drop and give me twenty!")
    new, info = look._judged(_post(), {})
    assert new.top(1)[0][0] == "DROP AND GIVE ME TWENTY!"
    # the prior shrinks with the share of letters still hidden: 6 of 19 here
    assert info["truth_score"] == 0.95 and info["prior_unknown"] == pytest.approx(0.1 * 6 / 19)
    assert info["before"]["DROP AND GIVE ME PLENTY!"] == pytest.approx(0.614)
    # sampling's coverage estimate, when there is one, replaces the fixed prior
    _, info = look._judged(_post(), {"coverage": 0.7})
    assert info["prior_unknown"] == pytest.approx(0.3)


# --- JevJudge's HTTP handling (urlopen faked) ----------------------------------------------------

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen(sent, fail_first=0):
    state = {"fails": fail_first}

    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        sent.append((req, body))
        if state["fails"]:
            state["fails"] -= 1
            raise urllib.error.HTTPError(req.full_url, 429, "slow down", {"Retry-After": "0"}, io.BytesIO(b"{}"))
        answers = {k: {"type": "noul", "noul": 0.9 if "Twenty" in q["instructions"] else 0.1}
                   for k, q in body["questions"].items()}
        return _Resp(json.dumps({"model": body["model"], "answers": answers,
                                 "usage": {"input_tokens": 100, "output_tokens": 0}}).encode())

    return urlopen


def test_jev_request_shape_batching_and_cache(monkeypatch, tmp_path):
    sent = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(sent))
    cache = tmp_path / "cache.json"
    jev = J.JevJudge(api_key="test-key", cache_path=cache, batch=2)
    scores = jev.judge(list(SEARCH), CLUE)
    assert scores["DROP AND GIVE ME TWENTY!"] == 0.9 and scores["DROP AND GIVE ME PLENTY!"] == 0.1
    assert len(sent) == 2  # 3 questions, 2 per request
    req, body = sent[0]
    assert req.full_url == J.ENDPOINT and req.get_header("Authorization") == "Bearer test-key"
    assert body["model"] == J.DEFAULT_MODEL and "Phrase" in body["state"]
    q = next(iter(body["questions"].values()))
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert "Wheel" not in json.dumps(body)  # never name the show (leakage)
    assert jev.stats.questions == 3 and jev.stats.input_tokens == 200
    # cached: a fresh judge reading the same file makes no request
    again = J.JevJudge(api_key="test-key", cache_path=cache)
    assert again.judge(list(SEARCH), CLUE) == scores and len(sent) == 2
    assert again.stats.cache_hits == 3
    # a different category is a different question
    again.judge(["DROP AND GIVE ME TWENTY!"], Clue("Thing"))
    assert len(sent) == 3


def test_jev_retries_on_429(monkeypatch):
    sent = []
    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen(sent, fail_first=2))
    jev = J.JevJudge(api_key="k", cache_path=None)
    assert jev.judge(["DROP AND GIVE ME TWENTY!"], CLUE)["DROP AND GIVE ME TWENTY!"] == 0.9
    assert len(sent) == 3


def test_jev_needs_a_key(monkeypatch):
    monkeypatch.delenv(J.API_KEY_ENV, raising=False)
    jev = J.JevJudge(cache_path=None)
    with pytest.raises(J.JudgeError, match=J.API_KEY_ENV):
        jev.judge(["CAT"], CLUE)


def test_min_top_holds_back_a_guess_the_judge_doubts():
    def look(board):
        post = AnswerPosterior(board, [Candidate("CAT", 0.0)])
        # the judge doubts CAT on the empty board only: certain letters, but CAT is just ~60% likely
        return J.apply_judge(post, {"CAT": 0.1 if board.n_hidden == 3 else 0.9}, prior_unknown=0.1)[0]

    held = play_letters("CAT", look, solve_at=0.01, min_top=0.9)
    first = held.turns[0]
    assert first.signal <= 0.01 and first.p_top < 0.9
    assert first.held_back and first.guess is None and first.called
    assert held.won_by_guess is not None and held.won_by_guess.number == 1
    free = play_letters("CAT", look, solve_at=0.01)
    assert free.won_by_guess.number == 0 and not free.turns[0].held_back


def test_min_top_doesnt_delay_an_answer_the_judge_agrees_with():
    def look(board):
        post = AnswerPosterior(board, [Candidate("CAT", 0.0)])
        return J.apply_judge(post, {"CAT": 0.9}, prior_unknown=0.05, real_rate=0.9)[0]

    game = play_letters("CAT", look, solve_at=0.01, min_top=0.9)
    assert game.won_by_guess.number == 0


# --- calibration ---------------------------------------------------------------------------------

def _puzzle(answer, category):
    from wof.puzzles import Puzzle

    return Puzzle(answer, Clue(category), 40, "1", "R1", "R", "", "")


def test_before_and_after_gets_its_own_question():
    clue = Clue("BEFORE & AFTER")
    q = J.judge_question("CHARIOTS OF FIRE EXTINGUISHERS", clue)
    assert q["type"] == "noul" and '"Chariots Of Fire Extinguishers" splits at one shared word' in q["instructions"]
    assert "shared word" in q["criteria"]["true"]
    state = J.judge_state(clue)
    assert '"Chariots Of Fire" + "Fire Extinguishers"' in state and "Wheel" not in state
    assert J.category_rule(Clue("Before & After ")) is J.CATEGORY_RULES["BEFORE & AFTER"]  # case/space-insensitive
    # other categories keep the generic question
    assert "real, familiar piece of English" in J.judge_question("DROP AND GIVE ME TWENTY!", CLUE)["instructions"]
    assert J.category_rule(CLUE) is None


def test_cache_keys_keep_generic_answers_and_version_the_rules():
    jev = J.JevJudge(api_key="k", cache_path=None)
    # unchanged from the first version, so answers already cached for ordinary categories still count
    assert jev._key("DROP AND GIVE ME TWENTY!", CLUE) == "jev-1.14.0|q1|Phrase|||DROP AND GIVE ME TWENTY!"
    assert jev._key("SHOE RACK OF LAMB", Clue("BEFORE & AFTER")).startswith("jev-1.14.0|q1+ba1|")
    assert "|q1+sn1|" in jev._key("CATERING & INTERNAL REVENUE SERVICE", Clue("SAME NAME"))


def test_turn_solver_uses_the_judges_measured_rates_by_default():
    jev = J.JevJudge(api_key="k", cache_path=None)
    assert jev.real_rates == J.REAL_RATES
    assert LLMTurnSolver(None, Clue("PHRASE"), judge=jev).real_rate() == J.REAL_RATES["PHRASE"]
    assert LLMTurnSolver(None, Clue("THING"), judge=jev).real_rate() == J.REAL_RATES["*"]
    assert LLMTurnSolver(None, Clue("PHRASE"), judge=jev, judge_real_rate=0.5).real_rate() == 0.5
    # another model has no measured rates, and a stand-in judge none at all: 0.9
    assert J.JevJudge("jev-latest", api_key="k", cache_path=None).real_rates is None
    assert LLMTurnSolver(None, CLUE, judge=J.FixedJudge({})).real_rate() == 0.9


def test_same_name_readings():
    assert J.same_name_readings("STRAWBERRY & PAPER JAM") == [["STRAWBERRY JAM", "PAPER JAM"]]
    assert J.same_name_readings("CATERING & INTERNAL REVENUE SERVICE") == [
        ["CATERING SERVICE", "INTERNAL REVENUE SERVICE"]]
    assert J.same_name_readings("FRENCH TOAST & POODLE") == [["FRENCH TOAST", "FRENCH POODLE"]]
    assert J.same_name_readings("APPLE, PUMPKIN & POT PIE") == [["APPLE PIE", "PUMPKIN PIE", "POT PIE"]]
    assert J.same_name_readings("BLACKBOARD AND BLACK BOX") == [["BLACKBOARD BOX", "BLACK BOX"]]
    # both parts several words: either side could hold the shared word
    assert J.same_name_readings("ICE CREAM & BIRTHDAY CAKE") == [
        ["ICE CREAM CAKE", "BIRTHDAY CAKE"], ["ICE CREAM", "ICE BIRTHDAY CAKE"]]
    assert J.same_name_readings("SALT & PEPPER") == [] and J.same_name_readings("PAPER JAM") == []


def _phrase_urlopen(sent, real):
    def urlopen(req, timeout=None):
        body = json.loads(req.data)
        sent.append(body)
        answers = {k: {"type": "noul", "noul": 0.9 if any(f'"{r}"' in q["instructions"] for r in real) else 0.05}
                   for k, q in body["questions"].items()}
        return _Resp(json.dumps({"model": body["model"], "answers": answers, "usage": {}}).encode())

    return urlopen


def test_same_name_asks_about_each_phrase_and_keeps_the_weakest(monkeypatch, tmp_path):
    sent = []
    real = {"Strawberry Jam", "Paper Jam", "Paper Fan", "Pepper"}
    monkeypatch.setattr(urllib.request, "urlopen", _phrase_urlopen(sent, real))
    clue = Clue("SAME NAME")
    jev = J.JevJudge(api_key="k", cache_path=tmp_path / "c.json")
    got = jev.judge(["STRAWBERRY & PAPER JAM", "STRAWBERRY & PAPER FAN", "SALT & PEPPER"], clue)
    assert got["STRAWBERRY & PAPER JAM"] == 0.9   # both phrases real
    assert got["STRAWBERRY & PAPER FAN"] == 0.05  # "Strawberry Fan" isn't, so the answer isn't either
    assert got["SALT & PEPPER"] == 0.05           # doesn't split: the whole-answer Same Name question
    phrase_asks = [q for b in sent if "Phrases built" in b["state"] for q in b["questions"].values()]
    assert len(phrase_asks) == 4  # STRAWBERRY JAM, PAPER JAM, STRAWBERRY FAN, PAPER FAN: each once
    assert all("Same Name" not in b["state"] for b in sent if "Phrases built" in b["state"])
    # a later turn's candidate reuses the cached phrases: only "Jam" with a new first part is asked
    n = len(sent)
    jev.judge(["RASPBERRY & PAPER JAM"], clue)
    assert len(sent) == n + 1 and len(sent[-1]["questions"]) == 1


def test_combine_readings_takes_the_best_reading():
    readings = [["ICE CREAM CAKE", "BIRTHDAY CAKE"], ["ICE CREAM", "ICE BIRTHDAY CAKE"]]
    scores = {"ICE CREAM CAKE": 0.8, "BIRTHDAY CAKE": 0.9, "ICE CREAM": 0.95, "ICE BIRTHDAY CAKE": 0.1}
    assert J.combine_readings(readings, scores) == 0.8
