# QNN inputs and targets

Each file is a plain JSON array. Every record has exactly two fields:

```json
{
  "input": "00000000000010100100110010000000000",
  "output": 0.6063275578555084
}
```

`input` is a 35-character binary string. `output` is the target score in
`[-1, 1]`, with positive values favoring White and negative values favoring
Black. Keep the input as a string to preserve leading zeros.

| File | Records |
| --- | ---: |
| `train.json` | 39,956 |
| `validation.json` | 4,922 |
| `test.json` | 5,122 |

Load from the repository root:

```python
import json
from pathlib import Path

records = json.loads(Path("data/king_rook/qnn/train.json").read_text())
inputs = [record["input"] for record in records]
targets = [record["output"] for record in records]
```

## Mate-aware target scores

These exports use mapping version `mate-bands-v1`. They preserve position
order, split assignments, and input bitstrings from the original records,
and replace the old targets with the following White-perspective scores:

| Band | Meaning |
| --- | --- |
| `[-1, -0.75)` | Forced mates for Black |
| `[-0.75, +0.75]` | Ordinary centipawn evaluations and draws |
| `(+0.75, +1]` | Forced mates for White |

Ordinary evaluations use `0.75 * tanh(white_centipawns / 400)`. Nonterminal
mates use `sign(m) * (0.75 + 0.25 * 0.9 ** abs(m))`, where `m` is the signed
engine-reported mate distance in **moves**, from White's perspective.
The scale and decay are fixed rather than fitted to this dataset's longest
mate. For extremely long future distances, a floating-point rounding guard
keeps mate scores strictly outside the ordinary band.

| White-winning position | Target |
| --- | ---: |
| Black already checkmated | `1.000000` |
| Mate in 1 move | `0.975000` |
| Mate in 5 moves | `0.897623` |
| Mate in 10 moves | `0.837170` |

Black-winning positions receive symmetric negative scores. Actual checkmate
uses the explicit winner because a zero mate distance carries no sign.
Stalemate and insufficient material use zero. Nonterminal theoretical draws
keep their scaled engine evaluations.

Only 36 of the current 50,000 records receive exact endpoint targets:
already-checkmated boards. The other records comprise 46,898 centipawn
evaluations, 1,066 nonterminal mates, and 2,000 terminal draws. These targets
encourage progress toward shorter mates; engine-reported distances are not
guaranteed globally shortest, and learned predictions do not guarantee a
correct mating policy.

## Reproducing the translation

From the repository root:

```sh
.venv/bin/python translate_qnn_data.py
```

The converter reads the original compressed rich records, verifies their
source checksums, and calculates targets from raw evaluations and terminal
metadata. It validates all three splits before replacing these JSON exports.
No Stockfish analysis is run. The original records, generator, cache, and
original manifest stay unchanged. Rerunning the command is deterministic.

The conversion's own `manifest.json` records the mapping, source/output
checksums, score distributions, validation, and recalculated unweighted
validation/test baselines. Use those baselines for these targets. The parent
directory's manifest, original rich-record targets, and original sampling
weights refer to the previous mapping; their baseline errors cannot be
compared directly with the new ones as evidence of model improvement.
