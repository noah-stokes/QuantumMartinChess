# Full-chess training data

50,000 unique legal positions in the **265-bit square-board format**, with
the same `mate-bands-v1` scalar targets as the rook-endgame dataset.
The [QNN exports](qnn/README.md) are plain JSON arrays of `input`/`output` pairs.

| Split | Positions |
| --- | ---: |
| Training | 40,000 |
| Validation | 5,000 |
| Test | 5,000 |

Every split has the same advantage distribution:

| Target category | All positions | Percentage |
| --- | ---: | ---: |
| Black forced mate | 5,000 | 10% |
| Black large advantage | 7,500 | 15% |
| Black moderate advantage | 7,500 | 15% |
| Near-equal | 10,000 | 20% |
| White moderate advantage | 7,500 | 15% |
| White large advantage | 7,500 | 15% |
| White forced mate | 5,000 | 10% |

Near-equal targets have magnitude at most `0.2`, moderate advantages have
magnitude above `0.2` through `0.6`, and large ordinary advantages have
magnitude above `0.6` through `0.75`. Forced mates have magnitude above
`0.75`; actual checkmates alone receive exactly `-1` or `1`.

Positions come from normal starting boards, one of twenty legal opening
prefixes, and mixed-strength Stockfish/random legal play. Each sampled
position has a color-swapped, vertically reflected partner with a negated
target. Whole games stay in one split. Placement families, ignoring turn
and state and grouping color mirrors, are sampled only once across all
splits. Games continue through promotion; promotion choices use weights
7/1/1/1 for queen/rook/bishop/knight, giving all underpromotion types coverage.

The selected positions come from 1,239 games. There are 4,974 examples after
a promotion has occurred, including 3,850 with a promoted piece still on
the board. Queen, rook, bishop, and knight promotions are represented in
each split. White and Black each have 25,000 positions with their side to move.

Stockfish 19 uses 10,000 nodes per position, one thread per worker, 32 MiB
hash, no tablebases, and cleared search state between positions. The last
exact search score supplies the label. These are engine estimates and
engine-reported mate distances, rather than theoretical proofs or shortest
mate guarantees. Random play and explicit quotas give a training distribution
that differs from human games. Position labels omit repetition history and
move clocks, as does the input encoding.

`train.jsonl.gz`, `validation.jsonl.gz`, and `test.jsonl.gz` retain FENs,
265-bit inputs, raw centipawn/mate evaluations, mapped targets, source game
and ply, promotion history, and search depth/node counts. `games.jsonl.gz`
stores complete legal move sequences for the contributing games. The source
and QNN manifests record checksums, counts, coverage, and validation results.
Every accepted source position is checked against a replay from the normal
initial board; every input is decoded and re-encoded before export.
The manifests' `phase_counts` are material-based summaries: at least 28
pieces is called opening, 13–27 middlegame, and at most 12 endgame. They
describe piece counts, rather than determining the actual phase of a game.

Regenerate from the repository root:

```sh
.venv/bin/python -m training.generate_standard
```

On macOS, the default command downloads and verifies a pinned Stockfish
release. An existing executable can be supplied with `--engine PATH` and
`--engine-sha256 VERIFIED_SHA256`. Completed games and position analyses
are cached in `.cache/standard/`; rerun the same command to resume. Changing
generation settings or code requires a separate `--cache-dir`. Use
`--positions 200 --output-dir PATH --cache-dir OTHER_PATH` for a small pilot.

The existing QNN model and loader accept 35-bit rook-endgame inputs.
Training a model on this dataset requires a model and loader for 265 inputs.
