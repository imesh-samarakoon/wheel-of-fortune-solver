# %% [markdown]
# # How the solver works, one step at a time
#
# Run this file cell by cell and watch a real puzzle go from a blank board to a letter to call.
# Every `# %%` line starts a cell: in VS Code or Cursor click "Run Cell" above it (or Shift+Enter)
# and the output appears in the Interactive Window. In JupyterLab, `jupytext explore.py --to
# notebook` first. Plain `python explore.py` also works — it just prints everything in one go.
#
# **The settings live in `config.py`**, not here. This file only has display knobs (how many rows
# to print). Change a real setting there, re-run from the top.
#
# ## The idea
#
# A language model predicts **tokens** (`" Burt"`, `"aster"`), never single letters. So we never
# ask it "what letter is in slot 7?" — it has no way to answer that. Instead:
#
# 1. **Prompt** it with a few solved puzzles, then this puzzle's category and board.
# 2. At every step it gives a probability for each of its ~150,000 tokens. **Keep only the tokens
#    that fit the board** (right length, no letter that's already been called and missed,
#    punctuation in place) and throw the rest away.
# 3. **Search** those tokens for the most likely whole answers that fill the board exactly. Each
#    one comes with the model's own probability P(answer | prompt).
# 4. **Normalize** over the answers found: that is P(answer | clue, board), Bayes' rule.
# 5. **Marginalize** to letters: P(letter in puzzle) = total probability of the answers containing
#    it. Now every letter has a number, and a policy can pick one.
#
# Steps 6-8 are about the thing that goes wrong: the search only ever finds a few dozen answers out
# of an enormous space, and if the true answer isn't among them, every number above is confidently
# wrong.

# %% Setup — imports and the puzzle set (nothing to tune)
import config
from wof import show

puzzles = config.puzzles()
print(f"{len(puzzles.play)} puzzles to play, {len(puzzles.examples)} in the example pool")
print("categories:", ", ".join(f"{c} ({n})" for c, n in puzzles.categories().items()))

# %% PICK A PUZZLE
WHICH = 1            # its number in data/puzzles.json (1-100), part of its answer, or "random"
CALLED = ""          # letters already called, e.g. "RSTLNE". Every occurrence is revealed
SHOW_ANSWER = True   # False = play along without spoilers (the truth checks below then go quiet)

puzzle = puzzles.get(WHICH)
board = puzzle.board(CALLED)
truth = puzzle.answer if SHOW_ANSWER else None
show.print_puzzle(puzzle, board, reveal_answer=SHOW_ANSWER)

# %% THE MODEL (downloads it the first time: see docs/setup.md)
solver = globals().get("solver")        # re-running this cell only reloads it if config.MODEL changed
if solver is None or solver.model_name != config.MODEL:
    solver = config.load_model()

# %% STEP 1: THE PROMPT — what the model is actually asked
# Worked examples come from the example pool, which never contains a playable answer, so they
# cannot leak this one. Settings: config.N_EXAMPLES, SAME_CATEGORY, ANSWER_STYLE, SHOW_BOARD,
# EXAMPLE_REVEAL, SHOW_CALLED.
from wof.search import build_prompt, example_reveals, styled

examples = config.prompt_examples(puzzle, puzzles.examples)
prompt = build_prompt(puzzle.clue, board, examples, style=config.ANSWER_STYLE,
                     show_board=config.SHOW_BOARD, show_called=config.SHOW_CALLED,
                     example_called=example_reveals(examples, board, config.EXAMPLE_REVEAL))
show.print_prompt(prompt)

# %% STEP 2: THE FIRST TOKEN — and how little of it fits the board
# The model's favourite next tokens are mostly unusable: wrong length for the first word, a letter
# that's been called and missed, a space where the board has a letter. The "fits the board" share
# is what the search is allowed to work with — on an empty board it is often a few percent.
TOP_TOKENS = 15

show.print_next_tokens(solver.next_tokens(prompt, board, top=TOP_TOKENS, case=config.CASE))

# %% ... and after a start of your choosing (the answer so far, as the model would write it)
PREFIX = styled(puzzle.answer, config.ANSWER_STYLE).split(" ")[0] if SHOW_ANSWER else ""

show.print_next_tokens(solver.next_tokens(prompt, board, prefix=PREFIX, top=TOP_TOKENS, case=config.CASE))

# %% STEP 3: THE SEARCH — grow partial answers until they fill the board
# Only partial answers that have written the same number of board characters compete with each
# other, and at most config.BEAMS survive at each position. That matters: a partial answer's
# probability only ever falls as it grows, so ranking " Keeping" (8 characters, its cost paid)
# against " Me" (3 characters, most of its cost still ahead) throws away the good one.
TRACE_EVERY = 2   # print every Nth step
TRACE_TOP = 4     # partial answers shown per step

beam = solver.beam_search(prompt, board, beams=config.BEAMS, keep=config.KEEP,
                          max_per_first_word=config.MAX_PER_FIRST_WORD, expand=config.EXPAND,
                          case=config.CASE)
show.print_beam_trace(beam, every=TRACE_EVERY, top=TRACE_TOP)

# %% STEP 4: THE CANDIDATE ANSWERS — and the normalization
# Each answer's score is the model's full log P(answer | prompt): its tokens' probabilities plus
# the probability that the answer stops there. No rescoring is needed. Dividing by the total over
# the answers found gives P(answer | clue, board) — IF the true answer is one of them. Look at how
# small the raw probabilities are: these few answers hold a sliver of the model's distribution.
N_SHOWN = 15
TOKEN_DETAIL = 3   # how many to break down token by token

show.print_candidates(beam, n=N_SHOWN, token_detail=TOKEN_DETAIL)

# %% STEP 5: LETTER PROBABILITIES — the thing a player can act on
post = beam.posterior()
print(post.render_slots(k=3))  # the board with the 3 likeliest letters under each blank
show.print_letter_probs(post, truth=truth)

# %% ... where one letter's number comes from
LETTER = "R"   # any letter that hasn't been called
N_SHOWN = 10

show.print_letter_breakdown(post, LETTER, n=N_SHOWN)

# %% STEP 6: WHERE IS THE TRUE ANSWER?
# The model is forced to write the real answer so we can read off its probability directly, even
# when the search never found it. A truth that scores WELL but was never found is a search
# failure, not a model failure — and that is the common case.
if SHOW_ANSWER:
    true_path = solver.score(prompt, board, styled(puzzle.answer, config.ANSWER_STYLE), config.CASE)
    show.print_truth_check(true_path, beam)

# %% STEP 7: FILL ONE WORD — the answer a left-to-right search cannot reach
# "A WISH COME TRUE" is the case this exists for. Once "COME TRUE" is settled, the search still
# writes "A Fish Come True", because " A W" is an unlikely *start* even though the whole phrase is
# ~80x likelier. So: when every other word of the phrase is settled, try every word of a word list
# in the one open word and score each whole answer. The late payoff then counts.
# (config.FILL_SLOTS turns this on during a game; here it runs regardless so you can see it.)
vocab = config.fill_vocab(puzzles.examples)
print(f"word list: {len(vocab):,} distinct words from {len(puzzles.examples)} example answers "
      f"(a bigger example pool makes this stronger)")
fill = solver.fill_slots(prompt, board, beam, vocab, case=config.CASE, keep=8,
                         min_confidence=config.FILL_MIN_CONFIDENCE, max_words=config.FILL_MAX_WORDS,
                         contexts=config.FILL_CONTEXTS)
show.print_filled(fill, beam, truth)

# %% STEP 8 (slower): HOW MUCH DID THE SEARCH MISS?
# Sampling draws answers at random from the board-fitting tokens and weights each draw by
# P(model writes it) / P(the sampler drew it). That is an unbiased estimate of the probability
# OUTSIDE the answers the search found, so it measures what share of the model's distribution the
# search actually covered. On an empty board that share is tiny — well under 1% here, and about 5%
# with 64 beams in the original measurement. THAT is why the letter probabilities above are
# overconfident: Step 5 divides by the few answers found as if nothing else were possible.
#
# Read the table below as a warning about the search, not as better numbers. Watch the effective
# sample size: a couple of lucky draws usually dominate the estimate, and most sampled answers are
# junk that merely fits the board. Pushing the samples up until this is trustworthy costs more than
# it is worth, which is why a real game leaves sampling off (config.GAME_SAMPLES = 0) and handles
# "the answer may be none of these" with the answer judge instead.
N_SAMPLES = 150   # each draw costs about one beam, so this is ~10 searches' worth
from wof.adjust import hybrid_posterior

sample = solver.sample(prompt, board, n=N_SAMPLES, case=config.CASE)
show.print_coverage(beam, sample)
hybrid, info = hybrid_posterior(beam, sample)
show.print_hybrid(hybrid, info, n=10)
show.compare_letters({"search only": post, "search + samples": hybrid}, board, truth=truth)

# %% STEP 9: PLAY IT OUT, ONE LETTER AT A TIME
# Each turn: run the model again on the current board (the prompt now shows the revealed letters,
# and the search can no longer use a letter that's been called and missed), record what it believes,
# then either call the best letter or solve. A wrong solve is ruled out for the rest of the game.
# Everything here comes from config.py: POLICY, CARRY, FILL_SLOTS, SOLVE_AT, SOLVE_MIN_TOP, the
# answer judge.
from wof.calling import play_letters

judge = config.make_judge()
look = config.turn_solver(solver, puzzle, examples, judge=judge, vocab=vocab, truth=truth)
game = play_letters(puzzle.answer, look, start=board, on_turn=show.print_turn, **config.play_kwargs())

# %% ... the whole game in one table
show.print_game(game)

# %% ... every letter's probability, turn by turn
show.print_letter_history(game)

# %% ... one turn in full (by default the turn the answer first came out on top)
TURN = game.right_from() or 0

show.print_turn(game.turns[TURN], n_answers=10, n_letters=26)
