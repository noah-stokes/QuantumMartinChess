# QNN inputs and targets

Each file is a plain JSON array. Every record has exactly two fields:

```json
{
  "input": "00000000000010100100110010000000000",
  "output": 0.8084367438073445
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

These are simplified copies of the original split files. Position order,
split assignments, bitstrings, and target values are preserved. The dataset
generator and original records are unchanged.
