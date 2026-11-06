# QuantumMartinChess

The goal is to build a quantum neural network (QNN) that evaluates chess
positions, then use it as the evaluation function in a chess engine. The
current work covers board encodings and rook-endgame training data.

## Board representations

| Format | Pieces represented | Size |
| --- | --- | --- |
| `PositionType.STANDARD` | Slots for all 32 original pieces | **231 bits** |
| `PositionType.ROOK_ENDGAME` | Both kings and at most one rook per color | **35 bits** |

The full format reserves a slot for every original piece, including captured
pieces. The 35-bit format keeps only the four king/rook slots, making the
input smaller for rook-endgame training. Either or both rooks can be missing;
both kings must be present. The full format does not support promotions.

Both formats use the same rules:

- A square takes **6 bits**, numbered from `a1 = 0` to `h8 = 63`.
- Each king uses just its square. Every other piece adds a **captured flag**:
  `0` for present, `1` for captured. A captured piece is stored as `1000000`.
- The final **9 bits** store whose turn it is (White `0`, Black `1`), four
  castling rights, and a four-bit en passant file. Move history and clocks
  are omitted.

The full format uses **222 piece bits + 9 state bits**. The rook-endgame
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
| [legal_moves.py](legal_moves.py) | Takes a FEN and returns legal moves for the side to move using `python-chess`, in UCI notation such as `e2e4`. |
| [board_bitstrings.py](board_bitstrings.py) | Converts boards to bitstrings and back. Can also apply supplied legal moves and encode each resulting board. |
| [position_types.py](position_types.py) | Defines the supported piece sets, their fixed slot order, and encoding lengths. |
| [generate_training_data.py](generate_training_data.py) | Samples legal rook-endgame positions, labels them with Stockfish or terminal rules, caches results, and creates the data splits. |
| [translate_qnn_data.py](translate_qnn_data.py) | Converts the source records into `input`/`output` JSON pairs, with scores that distinguish ordinary evaluations from forced mates. |
| [qiskit_example.py](qiskit_example.py) | Runs a small Bell-pair simulation to check the local Qiskit setup. |
| `test_*.py` | Checks board encoding, legal moves, data generation, and score conversion. |

The [QNN data guide](data/king_rook/qnn/README.md) explains the 50,000 training,
validation, and test examples, their input layout, and target scores. The
[source data guide](data/king_rook/README.md) covers generation and the richer
records containing FENs, raw evaluations, and metadata.
