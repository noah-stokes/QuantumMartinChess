"""List legal moves for a FEN position using python-chess."""

import argparse
import json

import chess

from tools.position_types import PositionType, _validate_piece_set


def get_legal_moves(
    fen: str, position_type: PositionType = PositionType.STANDARD,
) -> list[str]:
    """Return legal moves for the side to move in UCI notation (e.g. e2e4)."""
    board = chess.Board(fen)
    _validate_piece_set(board, position_type)
    if not board.is_valid():
        raise ValueError("FEN describes an invalid chess position")
    return [move.uci() for move in board.legal_moves]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fen",
        nargs="?",
        help="FEN position in quotes (standard default: starting position)",
    )
    parser.add_argument(
        "--position-type", choices=[kind.value for kind in PositionType],
        default=PositionType.STANDARD.value, help="Allowed piece set (default: standard)",
    )
    args = parser.parse_args()
    position_type = PositionType(args.position_type)
    if args.fen is None and position_type not in (PositionType.STANDARD, PositionType.SQUARE_BOARD):
        parser.error("An explicit FEN is required for reduced position types")
    try:
        fen = chess.STARTING_FEN if args.fen is None else args.fen
        moves = get_legal_moves(fen, position_type)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(moves))
