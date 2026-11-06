# QNN training data

These files pair a 35-bit rook-endgame position with the score the QNN
should learn to predict. They contain the same 50,000 positions as the
[source dataset](../README.md), with just the input and target retained.

See the [QNN prototype guide](../../../qnns/rook_endgame/README.md) to train the evaluator, make
predictions, and generate circuit diagrams.

## Files

| File | Records | Use |
| --- | ---: | --- |
| `train.json` | 40,000 | Fit the model. |
| `validation.json` | 5,000 | Tune the model. |
| `test.json` | 5,000 | Measure final performance. |

`manifest.json` records the score mapping, file checksums, distributions,
and validation/test baselines. Equivalent boards—including rotations,
reflections, color swaps, and turn variants—stay in the same split.

## Distribution-preserving split

The QNN exports use a deterministic, family-grouped 80%/10%/10% split with
seed 42. Families are stratified by material and the five score bands below,
then whole families are moved to balance the row totals. There is no
oversampling, discarded position, changed target, or family leakage.

“Near-equal” means a target in `[-0.2, +0.2]`; it does not necessarily mean
the position is a theoretical draw. The advantage columns include mates.

| Set | Black advantage | Near-equal | White advantage |
| --- | ---: | ---: | ---: |
| All 50,000 positions | 19.482% | 61.036% | 19.482% |
| Training | 19.4825% | 61.035% | 19.4825% |
| Validation | 19.480% | 61.040% | 19.480% |
| Test | 19.480% | 61.040% | 19.480% |

Every split is within 0.004 percentage points of the overall three-way
distribution. The manifest also reports the finer mate-band distributions,
material counts, family checks, and deviations from the global distribution.

The original engine-record files keep their historical split assignments.
Only these derived QNN exports have been regrouped. Models trained before
this change need a fresh training run before assessment on the new held-out
sets; their old training positions may now be in validation or test.

## Record format

Each file is a plain JSON array. Every record has exactly two fields:

```json
{
  "input": "00000000000010100100110010000000000",
  "output": 0.6063275578555084
}
```

- `input`: the 35-bit `PositionType.ROOK_ENDGAME` encoding of both kings,
  up to one rook per color, and game state. Keep it as a string to preserve
  leading zeros. See the [encoding explanation](../../../README.md#board-representations).
- `output`: a score in `[-1, 1]`. Positive favors White and negative favors
  Black, regardless of whose turn it is. This is a position evaluation,
  not a win probability or a move label.

## Target scores

The mapping is called `mate-bands-v1`. It reserves the outer score bands
for forced mates, keeping them separate from ordinary advantages:

| Score range | Meaning |
| --- | --- |
| `[-1, -0.75)` | Forced mate for Black. |
| `[-0.75, +0.75]` | Ordinary engine evaluations and draws. |
| `(+0.75, +1]` | Forced mate for White. |

Ordinary evaluations use `0.75 * tanh(white_centipawns / 400)`.
Nonterminal mates use `sign(m) * (0.75 + 0.25 * 0.9 ** abs(m))`, where
`m` is the signed engine-reported mate distance in moves, positive for White.
This gives shorter mates scores closer to the winning side's endpoint:

| White-winning position | Target |
| --- | ---: |
| Mate in 10 moves | `0.837170` |
| Mate in 5 moves | `0.897623` |
| Mate in 1 move | `0.975000` |
| Black already checkmated | `1.000000` |

Black wins receive symmetric negative values. Stalemate and insufficient
material receive zero; other positions keep their scaled engine evaluations.
Mate distances come from the engine search and are not guaranteed shortest.

## Loading

From the repository root:

```python
import json
from pathlib import Path

records = json.loads(Path("data/king_rook/qnn/train.json").read_text())
inputs = [record["input"] for record in records]
targets = [record["output"] for record in records]
```

## Rebuilding and evaluating

```sh
.venv/bin/python -m training.translate_qnn_data --stratify --seed 42
```

The converter checks the source-file checksums and calculates scores from
raw evaluations and terminal metadata, without running Stockfish. It keeps
the complete set of input/target pairs, assigns whole families to the QNN
splits, and sorts each output by input bitstring. The original source files
are unchanged. Rebuilding existing stratified exports without `--stratify`
also remembers the method and seed from their manifest. A new output
directory requires `--stratify` to choose this method instead of preserving
the original source memberships.

Compare unweighted validation/test mean squared error (MSE) with the
baselines in this directory's `manifest.json`: predicting zero and predicting
the training mean for each material group. The parent directory's targets,
weights, and baseline errors use the older score mapping.
