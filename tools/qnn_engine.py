"""Pick a move by scoring every legal successor with a supplied QNN."""

import argparse
from functools import partial
import math
from pathlib import Path
from typing import Callable, TypeVar

import chess

from tools.legal_moves import get_legal_moves
from tools.position_types import PositionType


Input = TypeVar("Input")


def choose_move(
    position: str | chess.Board,
    qnn: Callable[[Input], float],
    encode: Callable[[chess.Board], Input],
    *,
    position_type: PositionType = PositionType.SQUARE_BOARD,
) -> str | None:
    """Return the best UCI move, or None when there are no legal moves.

    ``encode`` converts each board AFTER its candidate move to whatever input
    ``qnn`` accepts. No input length or model library is assumed. The QNN must
    return one White-perspective score in [-1, 1]: positive favors White,
    negative favors Black. White maximizes that score and Black minimizes it,
    using the ORIGINAL side to move, since each successor has the opposite turn.

    Every legal move is evaluated once. Ties keep the first move returned by
    legal_moves.py. Supplied boards and their move histories are preserved.
    This is a one-ply selection, without a search of the opponent's replies.
    """
    board = chess.Board(position) if isinstance(position, str) else position
    if board.chess960:
        raise ValueError("Chess960 positions are unsupported")
    moves = get_legal_moves(board.fen(), position_type)
    best_move = None
    best_score = -math.inf if board.turn == chess.WHITE else math.inf

    for uci in moves:
        successor = board.copy()
        successor.push(chess.Move.from_uci(uci))
        output = qnn(encode(successor))
        try:
            score = float(output)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"QNN score for {uci} must be a number in [-1, 1]") from error
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError(f"QNN score for {uci} must be finite and in [-1, 1]")
        if (board.turn == chess.WHITE and score > best_score
                or board.turn == chess.BLACK and score < best_score):
            best_move, best_score = uci, score

    return best_move


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--fen", required=True, help="Position to choose a move for")
    cli.add_argument("--checkpoint", type=Path, required=True, help="Saved local QNN checkpoint")
    cli.add_argument(
        "--position-type", choices=[kind.value for kind in PositionType],
        default=PositionType.ROOK_ENDGAME.value,
        help="Input encoding; must match the model (default: rook-endgame, 35 bits)",
    )
    args = cli.parse_args(argv)
    # Only the checkpoint CLI depends on the current QNN implementation.
    from tools.board_bitstrings import board_to_bitstring
    from qnns.rook_endgame.model import load_checkpoint

    try:
        model, _ = load_checkpoint(args.checkpoint)
        position_type = PositionType(args.position_type)
        encode = partial(board_to_bitstring, position_type=position_type)
        move = choose_move(args.fen, model.predict, encode, position_type=position_type)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        cli.error(str(error))
    print(move if move is not None else "0000")


if __name__ == "__main__":
    main()
