# QuantumMartinChess

The [QNN prototype guide](qnns/rook_endgame/README.md) explains how to train the 35-bit rook-endgame
evaluator, save a model, make predictions, and inspect circuit diagrams.

The goal is to build a quantum neural network (QNN) that evaluates chess
positions, then use it as the evaluation function in a chess engine. The
current work covers board encodings, rook-endgame data, and
[full-chess training data with promotions](data/standard/README.md).

## Project layout

```text
data/                 Training datasets and their documentation
training/             Data generators, score conversion, and split helpers
qnns/
  rook_endgame/       Current 35-bit QNN: model, training CLI, data loader, diagrams
tools/                Board encodings, legal moves, and the chess engine
tests/                All automated tests
examples/             Small Qiskit setup examples
runs/                 Saved models, metrics, and plots (ignored by Git)
```

Keep each new QNN in its own subfolder under `qnns/`. The Qiskit and Torch
implementations are backends of the same rook-endgame model and live together.

Run commands from the repository root using Python 3.12 and the project environment:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m qnns.rook_endgame train
.venv/bin/python -m unittest discover -s tests -v
```

To generate training data or convert the rook-endgame source records:

```sh
.venv/bin/python -m training.generate_rook_endgame
.venv/bin/python -m training.translate_qnn_data
.venv/bin/python -m training.generate_standard
```

## Choose a move with a QNN

[tools/qnn_engine.py](tools/qnn_engine.py) applies every legal move returned by
`tools/legal_moves.py`, encodes each resulting board, and scores it with the supplied
QNN. Scores are from White's perspective: **+1 favors White, -1 favors Black**.
White picks the highest score; Black picks the lowest. This looks one move
ahead, without searching the opponent's replies.

```python
from functools import partial

from tools.board_bitstrings import board_to_bitstring
from tools.position_types import PositionType
from tools.qnn_engine import choose_move
from qnns.rook_endgame.model import load_checkpoint

model, _ = load_checkpoint("runs/qnn/my-first-run/best.pt")
kind = PositionType.ROOK_ENDGAME
encode = partial(board_to_bitstring, position_type=kind)
move = choose_move(
    "4k3/8/8/8/8/8/8/5K1R w - - 0 1",
    model.predict,
    encode,
    position_type=kind,
)
print(move)  # UCI move, such as h1h7
```

Pass a FEN string or a `chess.Board`. `qnn` can be any callable returning one
finite score in `[-1, 1]`; `encode` can return any input it needs, including a
bitstring, array, or tensor. To change the QNN or its input size, replace those
two callables. The move picker has no Qiskit or PyTorch dependency. Use
`PositionType.SQUARE_BOARD` with the built-in encoder for full chess, including
promotions; custom encoders can use the default unrestricted piece set.

The function returns just the UCI move string, or `None` for checkmate or
stalemate. Tied scores keep the first legal move. Supplied boards and move
histories are preserved.

For the current saved 35-bit QNN, the command prints only the selected move
(`0000` when there are no legal moves):

```sh
.venv/bin/python -m tools.qnn_engine --checkpoint runs/qnn/my-first-run/best.pt \
  --fen "4k3/8/8/8/8/8/8/5K1R w - - 0 1"
```

`--position-type` selects another built-in encoding when using a model that
accepts it. The current `PositionQNN` requires the default `rook-endgame` encoding.

## Board representations

| Format | Pieces represented | Size |
| --- | --- | --- |
| `PositionType.SQUARE_BOARD` | All 64 squares, including promoted pieces | **265 bits** |
| `PositionType.STANDARD` | Slots for all 32 original pieces | **231 bits** |
| `PositionType.ROOK_ENDGAME` | Both kings and at most one rook per color | **35 bits** |

The **265-bit format** uses four bits per square from `a1` through `h8`,
then nine state bits: turn, `KQkq` castling rights, and en passant file.
Square codes are empty `0`, White pawn/knight/bishop/rook/queen/king `1–6`,
and Black pawn/knight/bishop/rook/queen/king `7–12`. Codes `13–15` are
invalid. Promotions and underpromotions are represented by the resulting
piece type. Use `board_to_bitstring(board, PositionType.SQUARE_BOARD)` and
the same type when decoding. This is the format of the new full-chess data.

The original 231-bit format reserves a slot for every original piece, including captured
pieces. The 35-bit format keeps only the four king/rook slots, making the
input smaller for rook-endgame training. Either or both rooks can be missing;
both kings must be present. These original slot formats do not support promotions.

The original slot formats use these rules:

- A square takes **6 bits**, numbered from `a1 = 0` to `h8 = 63`.
- Each king uses just its square. Every other piece adds a **captured flag**:
  `0` for present, `1` for captured. A captured piece is stored as `1000000`.
- The final **9 bits** store whose turn it is (White `0`, Black `1`), four
  castling rights, and a four-bit en passant file. Move history and clocks
  are omitted.

The original 231-bit format uses **222 piece bits + 9 state bits**. The rook-endgame
format uses **26 piece bits + 9 state bits**, in this order:

```text
White king [6] | White rook [7] | Black king [6] | Black rook [7] | State [9]
```

Fixed slots keep the input length and each field's meaning consistent after
captures. The current rook-endgame dataset has no castling rights or en
passant, so those eight state bits are always zero.

`board_to_bitstring(board)` defaults to 231 bits. Pass
`PositionType.ROOK_ENDGAME` to get 35 bits, and supply the same type when
decoding. Keep bitstrings as strings so leading zeros are preserved.

## Important files

| File | What it does |
| --- | --- |
| [qnns/rook_endgame/cli.py](qnns/rook_endgame/cli.py) | Trains, evaluates, and runs the current QNN with `python -m qnns.rook_endgame`. |
| [qnns/rook_endgame/model.py](qnns/rook_endgame/model.py) | Defines the model, its Qiskit and Torch backends, and checkpoint loading. |
| [qnns/rook_endgame/data.py](qnns/rook_endgame/data.py) | Loads training data, chooses subsets, and calculates evaluation metrics. |
| [qnns/rook_endgame/diagrams.py](qnns/rook_endgame/diagrams.py) | Draws model circuits and training plots. |
| [tools/legal_moves.py](tools/legal_moves.py) | Takes a FEN and returns legal moves for the side to move using `python-chess`, in UCI notation such as `e2e4`. |
| [tools/qnn_engine.py](tools/qnn_engine.py) | Scores every legal successor with a passed QNN and encoder, then returns the best UCI move for the side to move. |
| [tools/board_bitstrings.py](tools/board_bitstrings.py) | Converts boards to bitstrings and back. Can also apply supplied legal moves and encode each resulting board. |
| [tools/position_types.py](tools/position_types.py) | Defines the supported piece sets, their fixed slot order, and encoding lengths. |
| [training/generate_rook_endgame.py](training/generate_rook_endgame.py) | Samples legal rook-endgame positions, labels them with Stockfish or terminal rules, caches results, and creates the data splits. |
| [training/generate_standard.py](training/generate_standard.py) | Generates balanced 265-bit full-chess examples from legal games, including promotions, with the same reserved mate-band targets. |
| [training/translate_qnn_data.py](training/translate_qnn_data.py) | Converts the source records into `input`/`output` JSON pairs, with scores that distinguish ordinary evaluations from forced mates. |
| [examples/qiskit_example.py](examples/qiskit_example.py) | Runs a small Bell-pair simulation to check the local Qiskit setup. |
| `tests/test_*.py` | Checks board encoding, legal moves, data generation, and score conversion. |

The [QNN data guide](data/king_rook/qnn/README.md) explains the 50,000 training,
validation, and test examples, their input layout, and target scores. The
[source data guide](data/king_rook/README.md) covers generation and the richer
records containing FENs, raw evaluations, and metadata.
