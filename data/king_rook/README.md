# Rook-endgame source data

These are the source positions and engine evaluations used to create the
[QNN training files](qnn/README.md). Use those JSON exports to train the
current model; this directory keeps the full records for inspection and
regeneration.

## Positions

The dataset contains 50,000 unique legal positions:

| Material | Positions |
| --- | ---: |
| King and rook on each side (`KRvKR`) | 40,000 |
| White king and rook versus Black king (`KRvK`) | 4,000 |
| White king versus Black king and rook (`KvKR`) | 4,000 |
| Bare kings (`KvK`) | 2,000 |

This mix covers equal material, rook advantages for either color, and draws
with no rooks. Positions are random placements checked for legality, rather
than positions from played games. Checks and hanging rooks are included.

Each position has a color-swapped, vertically reflected partner with a
negated score. Both turns are equally represented. There are no castling
rights or en passant targets; halfmove clocks start at zero and there is no
repetition history.

The split is roughly 80% training, 10% validation, and 10% test. Rotations,
reflections, color swaps, and turn variants of a placement share a split,
so equivalent boards cannot leak into held-out evaluation.

## Files and records

| File | Contents |
| --- | --- |
| `train.jsonl.gz` | 39,956 source records, including training weights. |
| `validation.jsonl.gz` | 4,922 source records. |
| `test.jsonl.gz` | 5,122 source records. |
| `manifest.json` | Generation settings, checksums, counts, score distributions, and source-score baselines. |
| `pilot.json` | Statistics and timing estimates for the first 2,000 positions, which are also included in the full dataset. |

Each compressed file contains one JSON object per line. The main fields are:

| Field | Meaning |
| --- | --- |
| `fen` | Readable board position. |
| `bitstring`, `rook_bitstring` | The same board encoded in 231 bits and 35 bits. See [board representations](../../README.md#board-representations). |
| `raw_engine_evaluation` | White-perspective centipawn or mate evaluation; `null` for terminal draws. Checkmate records include an explicit `winner`. |
| `target_score` | Original score mapping, described below. |
| `material_class` | One of the four material groups above. |
| `family_id` | Placement family used to keep related positions in one split. |
| `source_fen`, `augmentation` | The analyzed position and whether this record is its color-swapped mirror. |
| `termination` | `checkmate`, `stalemate`, `insufficient_material`, or `null`. |
| `engine_depth`, `engine_nodes` | Search depth and node count at the last exact score reported. |
| `sampling_weight` | Training-only weight that gives each occupied source-score bin equal total weight. |

The source `target_score` uses `tanh(white_centipawns / 400)`, with mates at
`+1` or `-1` and terminal draws at zero. Positive always favors White.
The QNN converter reads the raw evaluations to produce its newer scores,
which distinguish mate distances. Source targets, weights, and baseline
errors belong to the original mapping; use the [QNN manifest](qnn/manifest.json)
for current training and evaluation. Source weights can be used for weighted
sampling or weighted loss, but should not be applied twice. Validation and
test records have no weights and are evaluated unweighted.

## Regenerating

Run from the repository root:

```sh
.venv/bin/python -m training.generate_rook_endgame
.venv/bin/python -m training.translate_qnn_data
```

The generator uses Stockfish 19 with 10,000 nodes per analyzed position,
one thread, 64 MiB hash, and no tablebases. Terminal positions use chess
rules directly. Search state is cleared between positions; completed
analyses are cached in SQLite under `.cache/rook_endgame/`, so rerunning
the same command resumes work.

On macOS, the generator downloads and checksum-verifies a pinned Stockfish
release. On other platforms, pass `--engine /path/to/stockfish` and
`--engine-sha256 VERIFIED_BINARY_SHA256`. Use a separate `--cache-dir` when
changing the engine or analysis settings.
