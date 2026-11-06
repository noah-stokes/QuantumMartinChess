# QuantumMartinChess

Use Python 3.12 and the project's `.venv`. Dependencies include Qiskit with
visualization support and Qiskit Aer for local simulation. No IBM account or
API key is needed to run locally.

To create a new environment with [uv](https://docs.astral.sh/uv/):

```sh
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the environment already installed in this checkout, activate it and run
the example:

```sh
source .venv/bin/activate
python qiskit_example.py
```

The example creates a Bell pair and simulates 1,024 measurements. The output
contains only `00` and `11`, with roughly half of the measurements in each.
Select `.venv/bin/python` as your IDE's Python interpreter.

Setup references: [Qiskit installation](https://quantum.cloud.ibm.com/docs/en/guides/install-qiskit)
and [Qiskit Aer](https://qiskit.github.io/qiskit-aer/getting_started.html).

`board_bitstrings.py` encodes a supplied `chess.Board` in one call. Use
`PositionType.STANDARD` for **231 binary characters** (the default), or
`PositionType.KING_ROOK_VS_KING` for **28 binary characters**:

```python
import chess
from board_bitstrings import board_to_bitstring
from position_types import PositionType

board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w - - 0 1")
bits = board_to_bitstring(board, PositionType.KING_ROOK_VS_KING)
assert bits == "0001000000000111100000000000"
```

The return value is the complete bitstring, including game state. The type
argument also accepts keyword syntax, `position_type=PositionType.KING_ROOK_VS_KING`.
Omitting it returns the standard 231-bit encoding, including for sparse boards.

The reduced format contains only these slots, in this order:

| Bits (zero based, inclusive) | Contents |
| --- | --- |
| 0–5 | White king position |
| 6 | White rook captured flag |
| 7–12 | White rook position |
| 13–18 | Black king position |
| 19 | Turn: White `0`, Black `1` |
| 20–23 | Castling rights: White kingside, White queenside, Black kingside, Black queenside |
| 24–27 | En passant file (always `0000` in this piece set) |

Both kings are required. The White rook may be alive or captured; a captured
rook is encoded as `1000000`, leaving both kings on the board. Extra pieces
raise `ValueError`. Normal chess legality and White castling rules still apply.
Start reduced games from a board containing only this material; its history
root must also fit the selected piece set.

The same optional type argument is accepted by `bitstring_to_board`,
`decode_bitstring`, `get_legal_moves`, and `moves_to_bitstrings`. Supply the type
when decoding; the string contains no type header and is not auto-detected.
Decoded slot identities are retained for their selected format; selecting
another format reseeds slots from the available history root.

`legal_moves.py` lists legal UCI moves for a FEN. To encode every legal successor
from the standard starting position:

```sh
python3 -m pip install -r requirements.txt
python3 legal_moves.py | python3 board_bitstrings.py
```

For another position, pass the same quoted FEN to both scripts.

Both CLIs accept `--position-type standard|king-rook-vs-king`. Reduced move
generation requires an explicit FEN:

```sh
python3 legal_moves.py "4k3/8/8/8/8/8/8/R3K3 w - - 0 1" --position-type king-rook-vs-king | \
  python3 board_bitstrings.py "4k3/8/8/8/8/8/8/R3K3 w - - 0 1" --position-type king-rook-vs-king
```

To decode a reduced string, use
`python3 board_bitstrings.py --decode BITSTRING --position-type king-rook-vs-king`.

The standard fixed slot order is White first, then Black. Within each color:

1. King, queen
2. Rooks originally on a/h files
3. Bishops originally on c/f files
4. Knights originally on b/g files
5. Pawns originally on a through h files

Each king occupies six position bits. Every other piece occupies
`[captured flag][six position bits]`, where `0` means alive and `1` means
captured. Captured pieces always occupy `1000000`. Squares are numbered
`a1 = 0` through `h8 = 63`; an alive piece on a1 therefore has position
`000000` and flag `0`.

| Bits (zero based, inclusive) | Contents |
| --- | --- |
| 0–110 | White's 16 piece slots |
| 111–221 | Black's 16 piece slots |
| 222 | Turn: White `0`, Black `1` |
| 223–226 | Castling rights: White kingside, White queenside, Black kingside, Black queenside |
| 227–230 | En passant file: none `0000`, a–h `0001`–`1000` |

En passant targets are on rank 6 when White is to move, rank 3 when Black
is to move. The target is stored after every double pawn push, even when
there is no legal en passant capture. Halfmove and fullmove clocks are omitted.

`board_to_bitstring(board)` tracks original slots through the board's move
history, including captures, castling, and en passant captures. Retain the
`chess.Board` and its move stack when encoding successive game positions.

A standalone FEN has no original-piece identities. The history root is
assigned deterministic slots, preferring pieces on their home squares and
then assigning remaining pieces by square index. Bishops use their original
square color. These assignments stay fixed through subsequent moves. Loading
each successive position from a separate FEN cannot preserve historical IDs.

Promotions are unsupported because the layout contains no promoted-piece
type bits. Promotion moves, promoted pieces marked with `~` in FEN, excess
pieces of one type, and same-color bishop pairs are rejected with `ValueError`.
Ordinary FEN omits promotion history, so some past promotions cannot be detected
when original material has also been captured. Use a board with complete move
history or FEN with promotion markers when that distinction matters.

`decode_bitstring(bits)` retains the square-to-piece dictionary interface;
the CLI equivalent is `python3 board_bitstrings.py --decode BITSTRING`.
Use `bitstring_to_board(bits)` to read the complete placement and game state:

```python
from board_bitstrings import board_to_bitstring, bitstring_to_board
import chess

board = chess.Board()
board.push_uci("e2e4")
bits = board_to_bitstring(board)
restored = bitstring_to_board(bits)
assert restored.turn == chess.BLACK
assert restored.ep_square == chess.E3
```

Decoding validates captured coordinates, overlapping pieces, game state, and
board validity. The restored `chess.Board` retains the encoded slot identities,
including through moves, undo, and copies. Loading a new FEN, manually editing
pieces, or clearing its move stack resets those identities. Historical moves and
clocks are unavailable. The format supports standard chess; Chess960 is rejected.

Run the regression suite with `python3 -m unittest -v`.

## Rook-endgame training data

The generated [rook-endgame dataset](data/king_rook/README.md) contains 50,000
positions with continuous White-perspective scores in `[-1, 1]`. It includes
opposing rooks, single-rook advantages for both colors, and bare kings. Each
record contains the existing 231-bit input and a compact 35-bit input.

Use `PositionType.ROOK_ENDGAME` (CLI `--position-type rook-endgame`) for the
35-bit format: White king `[6]`, White rook `[7]`, Black king `[6]`, Black
rook `[7]`, and the existing nine state bits. Either rook may be captured.
This format also supports castling, although the initial dataset samples
positions without castling rights. Existing formats are unchanged.

Generate or resume the dataset with:

```sh
.venv/bin/python generate_training_data.py
```

See the dataset guide for loading examples, engine verification, pilot
statistics, split isolation, sampling weights, and generalization limits.
