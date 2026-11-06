# Rook-endgame training data

`train.jsonl.gz`, `validation.jsonl.gz`, and `test.jsonl.gz` contain 50,000
unique legal positions in total: 40,000 with one rook on each side, 4,000
with only White's rook, 4,000 with only Black's rook, and 2,000 with bare kings.
Both turns are equally represented. Positions have no castling rights or
en passant, and start with a zero halfmove clock and no repetition history.

The dataset is a first evaluation experiment. It does not establish that
learning rook endgames will transfer to other pieces. Positions are sampled
uniformly from distinct-square placements conditioned on legality, rather
than from human games. Random placements can contain checks, hanging rooks,
and tactical captures. These are intentionally retained.

## Inputs and targets

Each line is a JSON object with these fields:

| Field | Meaning |
| --- | --- |
| `bitstring` | Existing 231-bit standard encoding, including turn and game state |
| `rook_bitstring` | Compact 35-bit `PositionType.ROOK_ENDGAME` encoding |
| `target_score` | Continuous White-perspective evaluation in `[-1, 1]` |
| `raw_engine_evaluation` | White-perspective `{"kind":"cp","value":...}` or `{"kind":"mate","value":...,"unit":"moves"}`; `null` for terminal draws |
| `fen` | Position for inspection, not a model input |
| `material_class` | `KRvKR`, `KRvK`, `KvKR`, or `KvK` |
| `family_id` | Canonical piece placement used to prevent split leakage |
| `source_fen` | Position actually analyzed or checked for terminal rules |
| `augmentation` | `identity` or `color_swap_vertical` |
| `termination` | `checkmate`, `stalemate`, `insufficient_material`, or `null` |
| `engine_depth`, `engine_nodes` | Depth/nodes at the last exact score reported by the search; not necessarily the full node budget |
| `sampling_weight` | Training-only inverse-frequency score-bin weight |

Only a bitstring goes into the QNN. The scalar `target_score` is the answer
it learns to predict; all other fields are labels, inspection data, or
training metadata. Keep bitstrings as strings to preserve leading zeros.
Use the 231-bit format if later curriculum stages must keep the same input
layout. Choosing the compact format does not promise automatic weight
transfer when the input layout changes.

Positive scores favor White; negative scores favor Black, independent of
whose turn it is. For finite engine evaluations the fixed transformation is:

```python
target_score = math.tanh(white_centipawns / 400.0)
```

For example, `+200` centipawns maps to approximately `+0.4621`. These values
are bounded engine evaluations, **not calibrated win probabilities**.
Engine-reported mates and checkmates use the winning side's endpoint.
Checkmate's raw mate value is zero with an explicit `winner` field to avoid
signed-zero ambiguity. Stalemate and insufficient material use zero.
Nonterminal theoretical draws retain their engine evaluations. No tablebase
outcome overrides the continuous training target. Mate distance is not a
separate training objective in this dataset.

The same transformation should be used when adding pieces. Do not normalize
each material class separately merely to stretch its score distribution.

## Loading

From the repository root:

```python
import gzip
import json
from pathlib import Path

directory = Path("data/king_rook")
with gzip.open(directory / "train.jsonl.gz", "rt") as stream:
    for line in stream:
        example = json.loads(line)
        bits = example["bitstring"]
        target = example["target_score"]
        weight = example["sampling_weight"]
        # Send bits and target to your black-box training adapter.
```

Training weights are `N_train / (occupied_bins * examples_in_bin)`, giving
each occupied score bin equal total weight. Use them either for weighted
sampling or for a weighted loss; applying both would double-correct the
distribution. Weights cannot create coverage in empty bins. Validation and
test files intentionally omit weights and should be evaluated unweighted.

The bins are `[-1,-0.8)`, `[-0.8,-0.2)`, `[-0.2,0.2]`, `(0.2,0.8]`, and
`(0.8,1]`. The manifest reports every bin, bins below 1% coverage, exact
endpoints, and scores with absolute value at least `0.95`.

## Reproducing and resuming

```sh
.venv/bin/python generate_training_data.py
```

The generator downloads the official macOS Stockfish 19 universal release
into `.cache/rook_endgame/stockfish/`, verifies the pinned release SHA-256,
and only then extracts and runs it. For another platform, provide an
independently verified Stockfish 19 executable:

```sh
.venv/bin/python generate_training_data.py \
  --engine /path/to/stockfish \
  --engine-sha256 YOUR_VERIFIED_BINARY_SHA256
```

Analysis uses one thread, 64 MiB hash, 10,000 nodes, no engine tablebases,
and cleared hash/history between positions. The last exact search score
is used; upper/lower bound scores are ignored. Every analyzed position's
result is committed to SQLite. Rerunning the same command recreates the
deterministic sample and reuses completed analyses. A changed engine or
analysis configuration requires a separate `--cache-dir`.

The first 2,000 generated positions form the pilot, recorded in `pilot.json`
and included in the final dataset. Its cold-run estimate is approximate;
cache reuse can reduce runtime. `generation.log` records the delivered run.

Color-swapped, vertically reflected partners are added with exactly negated
scores, without a second engine analysis. All eight board symmetries,
color swaps, and turn variants of a placement share a split family. A seeded
SHA-256 family hash assigns approximately 80%/10%/10% to the three splits.
This geometric equivalence is for pawnless positions without castling.

Gzip timestamps and row order are deterministic, so reruns with the same
configuration and cached labels produce identical dataset file checksums.
Runtime/cache statistics in the manifest can differ between runs. The
manifest records code and engine checksums, Python/chess versions, settings,
counts, score distributions, encoding validation, and held-out baseline MSE.

## Interpreting results

Compare a trained model's unweighted held-out MSE with both manifest
baselines: always predicting zero, and predicting the training-set mean
for each material class. Beating a material-only baseline is evidence of
learning distinctions within this dataset, not proof of full-chess skill.
Keep the split files fixed when comparing experiments.

The compact layout is White king `[6]`, White rook `[captured][6]`, Black
king `[6]`, Black rook `[captured][6]`, then turn `[1]`, castling `[4]`, and
en passant `[4]`. Captured rooks have zero coordinates. The existing standard
and 28-bit encodings retain their formats and validation rules.

Engine source: [Stockfish 19 release](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19).
The engine is cached locally, not included in the dataset files; the official
archive includes its source/license information. Dataset generation needs
only Python and the existing `chess` dependency, without Qiskit or a QNN.
