# QNN training data

These files pair a 35-bit rook-endgame position with the score the QNN
should learn to predict. They contain the same 50,000 positions as the
[source dataset](../README.md), with just the input and target retained.

## Files

| File | Records | Use |
| --- | ---: | --- |
| `train.json` | 39,956 | Fit the model. |
| `validation.json` | 4,922 | Tune the model. |
| `test.json` | 5,122 | Measure final performance. |

`manifest.json` records the score mapping, file checksums, distributions,
and validation/test baselines. Equivalent boards—including rotations,
reflections, color swaps, and turn variants—stay in the same split.

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
.venv/bin/python translate_qnn_data.py
```

The converter checks the source-file checksums and calculates scores from
raw evaluations and terminal metadata, without running Stockfish. It keeps
the source inputs, order, and splits, and writes these exports and their
manifest. The source files are unchanged.

Compare unweighted validation/test mean squared error (MSE) with the
baselines in this directory's `manifest.json`: predicting zero and predicting
the training mean for each material group. The parent directory's targets,
weights, and baseline errors use the older score mapping.
