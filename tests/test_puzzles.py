"""The puzzle file: loading it, picking examples, and the bundled data itself."""

import json

import pytest

from wof.judge import CATEGORY_RULES
from wof.puzzles import DATA_FILE, from_text, load_puzzles, pick_examples

FILE = {
    "note": "test data",
    "category_notes": {"RHYME TIME": "The words rhyme: LEGAL EAGLE."},
    "puzzles": [{"answer": "A piece of cake", "category": "phrase"},
                {"answer": "DOUBLE TROUBLE", "category": "RHYME TIME"},
                {"answer": "COSTA RICA", "category": "ON THE MAP", "hint": "Central America"}],
    "examples": [{"answer": "BITE THE BULLET", "category": "PHRASE"},
                 {"answer": "CALL IT A DAY", "category": "PHRASE"},
                 {"answer": "PRIME TIME", "category": "RHYME TIME"}],
}


@pytest.fixture
def puzzles(tmp_path):
    path = tmp_path / "puzzles.json"
    path.write_text(json.dumps(FILE))
    return load_puzzles(path)


def test_loads_normalizes_and_applies_category_notes(puzzles):
    assert len(puzzles.play) == 3 and len(puzzles.examples) == 3
    first = puzzles.get(1)
    assert first.answer == "A PIECE OF CAKE" and first.clue.category == "PHRASE"
    assert first.board().template == "_ _____ __ ____"
    rhyme = puzzles.find("double")
    assert rhyme.clue.context.startswith("The words rhyme")  # from category_notes, written once
    assert puzzles.find("COSTA").clue.hint == "Central America"
    assert puzzles.categories() == {"PHRASE": 1, "ON THE MAP": 1, "RHYME TIME": 1}


def test_get_by_number_name_or_random(puzzles):
    assert puzzles.get("2").answer == "DOUBLE TROUBLE"
    assert puzzles.get(2) == puzzles.get("2")
    assert puzzles.get("random", seed=3) in puzzles.play
    with pytest.raises(IndexError):
        puzzles.get(99)
    with pytest.raises(KeyError):
        puzzles.find("NOT A PUZZLE")


def test_an_example_answer_cannot_be_played(puzzles):
    with pytest.raises(KeyError, match="example pool"):
        puzzles.find("BITE THE BULLET")


def test_overlapping_groups_are_refused(tmp_path):
    bad = {**FILE, "examples": FILE["examples"] + [{"answer": "COSTA RICA", "category": "ON THE MAP"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="both groups"):
        load_puzzles(path)


def test_pick_examples_prefers_the_category_and_is_deterministic(puzzles):
    target = puzzles.find("DOUBLE TROUBLE")
    ex = pick_examples(puzzles.examples, target, n=2)
    assert len(ex) == 2
    assert ex[0].clue.category == "RHYME TIME" or ex[1].clue.category == "RHYME TIME"
    assert [e.answer for e in ex] == [e.answer for e in pick_examples(puzzles.examples, target, n=2)]
    assert all(e.answer != target.answer for e in ex)
    assert pick_examples(puzzles.examples, target, n=99) == pick_examples(puzzles.examples, target, n=3)


def test_from_text():
    p = from_text(" a wish come true ", "phrase", hint="hi")
    assert p.answer == "A WISH COME TRUE" and p.clue.category == "PHRASE" and p.clue.hint == "hi"
    assert p.board("AEIOU").template == "A _I__ _O_E __UE"
    with pytest.raises(ValueError):
        from_text("", "PHRASE")
    with pytest.raises(ValueError):
        from_text("123 -- 456", "PHRASE")  # nothing to guess


def test_a_board_shows_everything_that_is_not_a_letter():
    """Digits and punctuation are never hidden, and the search has to be able to write them."""
    assert from_text("ROCK 'N' ROLL 24/7", "PHRASE").board("O").template == "_O__ '_' _O__ 24/7"
    assert from_text("50% OFF SALE", "PHRASE").board().template == "50% ___ ____"
    assert from_text("$1,000 A DAY!", "PHRASE").board("A").template == "$1,000 A _A_!"


def test_from_text_refuses_characters_no_board_can_show():
    for bad in ("CAFE\u2605 NIGHT",   # not ASCII, so no token can write it
                "TOP_SECRET"):        # "_" is how a hidden slot is written
        with pytest.raises(ValueError, match="can't write"):
            from_text(bad, "PHRASE")


# --- the bundled file ----------------------------------------------------------------------------

def test_the_bundled_file_loads():
    ps = load_puzzles()
    assert len(ps.play) >= 50 and len(ps.examples) >= 50
    assert all(any(c.isalpha() for c in p.answer) for p in ps.play + ps.examples)
    assert len({p.answer for p in ps.play}) == len(ps.play)       # no repeats
    assert len({p.answer for p in ps.examples}) == len(ps.examples)
    assert not {p.answer for p in ps.play} & {p.answer for p in ps.examples}


def test_nothing_the_model_always_sees_contains_an_answer():
    """A category note and the judge's category explanations go into EVERY prompt for that category.

    An answer quoted in one of them would be handed to the solver (or to the judge) for free on the
    puzzle it belongs to, which would quietly flatter every result.
    """
    ps = load_puzzles()
    answers = {p.answer for p in ps.play} | {p.answer for p in ps.examples}
    notes = {p.clue.context for p in ps.play + ps.examples if p.clue.context}
    notes |= {rule.note for rule in CATEGORY_RULES.values()}
    notes |= {rule.instructions for rule in CATEGORY_RULES.values()}
    for note in notes:
        for answer in answers:
            assert answer not in note.upper(), f"{answer!r} is quoted in: {note[:80]}..."


def test_the_bundled_file_is_one_puzzle_per_line():
    """Kept readable on purpose: adding a puzzle should be a one-line diff."""
    text = DATA_FILE.read_text()
    assert text.count('{"answer":') == len(load_puzzles().play) + len(load_puzzles().examples)
    json.loads(text)
