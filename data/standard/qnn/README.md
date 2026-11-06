# 265-bit QNN examples

`train.json`, `validation.json`, and `test.json` contain 40,000, 5,000,
and 5,000 records, respectively. Each record has exactly these fields:

```python
{"input": "...265 binary characters...", "output": 0.3465878679450073}
```

Keep `input` as a string to preserve leading zeros. Its layout is:

```text
64 squares × 4 bits [256] | Turn [1] | Castling KQkq [4] | En passant file [4]
```

Squares run `a1, b1, ... h1, a2, ... h8`. Square code `0` means empty;
codes `1–6` mean White pawn, knight, bishop, rook, queen, king; codes `7–12`
mean the same pieces for Black. Codes `13–15` are invalid. Promoted pieces
use their current type, so promotions and underpromotions need no extra bits.

Turn is White `0` / Black `1`. Castling flags are in `KQkq` order.
En passant is `0000` for none, or file a–h as integers `1–8`; the side to
move determines the rank. The encoding retains a double-push target even
when no opposing pawn can capture. Decode with
`bitstring_to_board(bits, PositionType.SQUARE_BOARD)`.

`output` is one White-perspective position score, positive for White and
negative for Black, in `[-1, 1]`. The mapping matches the existing dataset:

- Ordinary evaluation: `0.75 * tanh(white_centipawns / 400)`.
- Nonterminal forced mate: `sign(m) * (0.75 + 0.25 * 0.9 ** abs(m))`.
- Actual checkmate: `+1` for White winning, `-1` for Black winning.
- Stalemate and insufficient material: `0`.

Ordinary scores lie within `[-0.75, 0.75]`; the outer bands are reserved
for forced mates. The score is an evaluation, rather than a win probability.

Load without quantum dependencies:

```python
import json
from pathlib import Path

records = json.loads(Path("data/standard/qnn/train.json").read_text())
inputs = [[int(bit) for bit in row["input"]] for row in records]  # [40000, 265]
targets = [row["output"] for row in records]                    # [40000]
```

Each split has 20% near-equal, 30% moderate advantages, 30% large
advantages, and 20% mates. Advantages and mates are balanced by color.
All examples are unweighted. The [source guide](../README.md) explains
sampling, game isolation, raw records, reproducibility, and limitations.
`manifest.json` contains exact distributions, checksums, coverage, and
unweighted validation/test baselines. The current 35-bit QNN model and
loader require adaptation before using these inputs.
