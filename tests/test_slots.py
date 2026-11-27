"""Filling one board word at a time (wof.slots + LLMSolver.fill_slots), without a model.

The game this was built for: "A WISH COME TRUE" (Thing). From turn 4 on, "COME TRUE" was in every
leading answer but the search kept writing "A Fish Come True" / "A City Come True", because
" A W" is an unlikely start; scored whole, the truth was ~80x the top answer.
"""

import math

from wof.board import Board
from wof.calling import LLMTurnSolver
from wof.search import LLMSolver, Path, SearchResult, TokenStep
from wof.slots import build_vocab, match_case, open_slots, slot_words, top_word, word_marginals
from wof.show import fill_lines, print_filled
from wof.puzzles import Clue

TRUTH = "A WISH COME TRUE"
TURN5 = Board.from_answer(TRUTH, set("AELOR"))  # A ____ _O_E _R_E


def _path(text, logprob):
    return Path(text, [TokenStep(" " + text, logprob, 0.0)])


class FakeSolver(LLMSolver):
    """Scores answers from a table (unknown answers: very unlikely); the "search" is a fixed result."""

    def __init__(self, table, found=()):
        self.table = table
        self.found = list(found)
        self.scored: list[list[str]] = []

    def score_many(self, prompt, board, texts, case=None, batch=16, prefixes=None):
        self.scored.append(list(texts))
        return [_path(t, self.table.get(t.upper(), -100.0)) for t in texts]

    def score(self, prompt, board, text, case=None):
        return self.score_many(prompt, board, [text])[0]

    def beam_search(self, prompt, board, **kwargs):
        return SearchResult("beam", prompt, board, [_path(t, s) for t, s in self.found if board.consistent(t)])


# A word list like the example pool's: "WISH" is in it, as are plenty of other 4-letter words.
VOCAB = build_vocab(["WISH YOU WERE HERE", "MAKE A WISH", "FISH AND CHIPS", "CITY LIGHTS", "JUST IN TIME",
                     "THIS IS IT", "SHIP SHAPE", "WITH FLYING COLORS", "A DOG'S LIFE", "COME AND GET IT"])


def test_word_marginals_and_top_word():
    m = word_marginals({"A CITY COME TRUE": 0.55, "A FISH COME TRUE": 0.44, "A KIDS COME TRUE": 0.01,
                        "NOT ON THE BOARD": 5.0}, TURN5)
    assert math.isclose(m[1]["CITY"], 0.55) and math.isclose(m[1]["FISH"], 0.44)
    assert math.isclose(m[2]["COME"], 1.0) and math.isclose(m[3]["TRUE"], 1.0)  # the off-board answer is ignored
    assert top_word(m[1]) == ("CITY", 0.55) and top_word({}) == ("", 0.0)
    assert word_marginals({}, TURN5) == [{}, {}, {}, {}]


def test_open_slots_needs_the_rest_of_the_phrase_settled():
    m = word_marginals({"A CITY COME TRUE": 0.55, "A FISH COME TRUE": 0.45}, TURN5)
    # word 2 (index 1): COME and TRUE are certain. COME/TRUE: word 2 is only 55%, so they stay closed.
    assert open_slots(m, TURN5, 0.8) == [(1, 1.0)]
    # a confident but suspicious word is still opened: "FISH" at 99% doesn't close it
    m = word_marginals({"A FISH COME TRUE": 0.99, "A CITY COME TRUE": 0.01}, TURN5)
    assert [i for i, _ in open_slots(m, TURN5, 0.8)] == [1, 2, 3]
    # turn 3 ("A WILD ROSE VINE" 50%): nothing is settled, nothing is filled
    turn3 = Board.from_answer(TRUTH, set("AEO"))
    m = word_marginals({"A WILD ROSE VINE": 0.5, "A BIRD COME TRUE": 0.3, "A GIRL GONE BLUE": 0.2}, turn3)
    assert open_slots(m, turn3, 0.8) == []
    # the last hidden word has nothing else to wait for
    last = Board.from_answer(TRUTH, set("AECMORTU"))  # A ____ COME TRUE
    assert open_slots(word_marginals({"A FISH COME TRUE": 1.0}, last), last, 0.99) == [(1, 1.0)]


def test_build_vocab_and_slot_words():
    vocab = build_vocab(["MAKE A WISH", "WISH YOU WERE HERE", "A DOG'S LIFE"])
    # a word in two answers outranks one in a single answer, and both outrank the word list
    assert vocab["WISH"] == vocab["A"] > vocab["DOG'S"] > 1
    # word 2 of TURN5 is 4 letters, none of A E L O R
    words = slot_words(TURN5, 1, VOCAB)
    assert "WISH" in words and "FISH" in words and "CITY" in words and "JUST" in words
    assert "COME" not in words and "HERE" not in words and "LIFE" not in words  # called letters can't be hidden
    assert words.index("WISH") == 0  # most common first (2 answers)
    assert slot_words(TURN5, 1, VOCAB, limit=2) == words[:2]
    assert slot_words(TURN5, 2, VOCAB) == ["COME"]  # _O_E


def test_a_word_list_file_fills_in_what_the_answers_miss(tmp_path):
    """The point of the word list: "WISH" never appears in an answer, so it has to come from there."""
    answers = ["A FISH OUT OF WATER"]
    assert "WISH" not in build_vocab(answers)
    path = tmp_path / "words.txt"
    path.write_text("# a comment, and a blank line\n\nwish\nfish\nnot a word\n", encoding="utf-8")
    vocab = build_vocab(answers, path)
    assert "WISH" in vocab and "NOT A WORD" not in vocab  # one word per line, uppercased
    assert vocab["FISH"] > vocab["WISH"]  # a word the answers use outranks one only in the list


def test_the_bundled_word_list_is_usable():
    import config

    vocab = build_vocab([], config.FILL_WORDS_FILE)
    assert len(vocab) > 20_000
    assert {"WISH", "TALK", "PETUNIAS", "ZEBRA"} <= set(vocab)
    assert not any(c.islower() for w in vocab for c in w)


def test_match_case():
    assert match_case("WISH", "Fish") == "Wish"
    assert match_case("WISH", "FISH") == "WISH"
    assert match_case("WISH", "fish") == "wish"
    assert match_case("I", "A") == "I"
    assert match_case("WELL-TO-DO", "Rags-To-Riches") == "Well-To-Do"


def test_fill_slots_finds_a_wish_come_true():
    found = [_path("A City Come True", -10.0), _path("A Fish Come True", -10.2), _path("A City Come True", -10.5)]
    res = SearchResult("beam", "p", TURN5, found, words=[{"A": -1.0}, {"City": -6.0, "Fish": -6.1, "Kids": -7.0},
                                                         {"Come": -8.0}, {"True": -10.0}])
    solver = FakeSolver({TRUTH: -5.6, "A KIDS COME TRUE": -9.9, "A JUST COME TRUE": -30.0})
    fill = solver.fill_slots("p", TURN5, res, VOCAB, min_confidence=0.8, keep=8)
    assert fill.paths[0].answer == TRUTH and fill.paths[0].text == "A Wish Come True"  # written like the others
    assert {p.answer for p in fill.paths} == {TRUTH, "A KIDS COME TRUE"}  # "Kids": a word the search wrote
    assert all(p.logprob >= -10.5 for p in fill.paths)  # only as good as the search's worst or better
    every = [t.upper() for batch in solver.scored for t in batch]
    assert len(every) == len(set(every)) and all(TURN5.consistent(a) for a in every)
    assert "A CITY COME TRUE" not in every and "A FISH COME TRUE" not in every  # never re-scores what was found
    assert len(fill.slots) == 1  # only word 2: COME TRUE settled, word 2 isn't
    r = fill.slots[0]
    assert r["word"] == 1 and r["pattern"] == "____" and r["best"] == (TRUTH, -5.6)
    assert r["locked"] == [(2, "COME", 1.0), (3, "TRUE", 1.0)]
    assert r["contexts"] == ["A City Come True"]  # one context: City/Fish differ only in the open word
    assert math.isclose(fill.top_logprob, math.log(math.exp(-10.0) + math.exp(-10.5)))
    lines = fill_lines({"slots": fill.slots, "top_logprob": fill.top_logprob, "seconds": 1.0})
    assert "word 2 ____" in lines[0] and "COME 100%" in lines[0] and "best A WISH COME TRUE" in lines[0]


def test_fill_slots_keeps_only_the_best():
    found = [_path("A City Come True", -10.0)]
    res = SearchResult("beam", "p", TURN5, found)
    table = {TRUTH: -5.6, "A FISH COME TRUE": -6.0, "A KIDS COME TRUE": -7.0}
    fill = FakeSolver(table).fill_slots("p", TURN5, res, VOCAB, keep=2)
    assert [p.answer for p in fill.paths] == [TRUTH, "A FISH COME TRUE"]
    assert fill.slots[0]["added"] == [TRUTH, "A FISH COME TRUE"]


def test_fill_slots_does_nothing_when_nothing_is_settled():
    turn3 = Board.from_answer(TRUTH, set("AEO"))
    found = [_path("A Wild Rose Vine", -9.0), _path("A Bird Come True", -9.2), _path("A Girl Gone Blue", -9.5)]
    solver = FakeSolver({TRUTH: -5.0})
    fill = solver.fill_slots("p", turn3, SearchResult("beam", "p", turn3, found), VOCAB)
    assert fill.paths == [] and fill.slots == [] and solver.scored == []
    assert solver.fill_slots("p", turn3, SearchResult("beam", "p", turn3, []), VOCAB).paths == []


def test_turn_solver_adds_the_fills(capsys):
    solver = FakeSolver({TRUTH: -5.6}, found=[("A City Come True", -10.0), ("A Fish Come True", -10.2)])
    look = LLMTurnSolver(solver, Clue("Thing"), fill_slots=dict(vocab=VOCAB, min_confidence=0.8), truth=TRUTH)
    post, info = look(TURN5)
    assert post.top(1)[0][0] == TRUTH and post.top(1)[0][1] > 0.95
    assert info["filled"] == [TRUTH] and info["fill"]["slots"][0]["word"] == 1
    # off by default
    post, info = LLMTurnSolver(solver, Clue("Thing"))(TURN5)
    assert post.prob_of(TRUTH) == 0.0 and info["fill"] is None and info["filled"] == []


def test_print_filled(capsys):
    found = [_path("A City Come True", -10.0)]
    res = SearchResult("beam", "p", TURN5, found)
    fill = FakeSolver({TRUTH: -5.6}).fill_slots("p", TURN5, res, VOCAB)
    print_filled(fill, res, TRUTH)
    out = capsys.readouterr().out
    assert "FOUND by filling a word" in out and "beats every answer" in out and "settled: COME 100%" in out
    turn3 = Board.from_answer(TRUTH, set("AEO"))
    res3 = SearchResult("beam", "p", turn3, [_path("A Wild Rose Vine", -9.0), _path("A Bird Come True", -9.1)])
    print_filled(FakeSolver({}).fill_slots("p", turn3, res3, VOCAB), res3, TRUTH)
    assert "no board word qualified" in capsys.readouterr().out
