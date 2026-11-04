"""Store chess positions with fixed original-piece slots.

Standard positions use 231 bits. KING_ROOK_VS_KING positions use 28 bits:
White king [6], White rook [captured][6], Black king [6], game state [9].

White's slots precede Black's. Each standard color uses this order: king, queen,
a/h rooks, c/f bishops, b/g knights, then a-h pawns (by original square).
Kings use six position bits. Every other slot uses a captured bit followed
by six position bits; captured slots are always 1000000. Squares use
python-chess indices: a1=0 through h8=63.

The final nine bits are turn (White=0), castling rights (KQkq), and the
en passant file (none=0, a-h=1-8). Clocks are not stored.

Move history preserves slot identities. A standalone FEN cannot identify
original pieces: its slots are seeded deterministically, preferring home
squares, then square order (bishops retain their original square color).
Retain the Board's move stack when encoding successive positions. Known
promotions are unsupported; an unmarked FEN can hide past promotions.

Encode legal moves: python3 legal_moves.py | python3 board_bitstrings.py
Decode one board:  python3 board_bitstrings.py --decode BITSTRING
"""

import argparse
import json
import sys
from typing import Optional, Union

import chess

from position_types import PositionType, _validate_piece_set, _validate_position_type


# Keep the original constants as aliases for the default standard format.
PIECE_SLOTS = PositionType.STANDARD.piece_slots
PIECE_DATA_LENGTH = PositionType.STANDARD.piece_data_length
BITSTRING_LENGTH = PositionType.STANDARD.bitstring_length
_CASTLING_ROOKS = (chess.H1, chess.A1, chess.H8, chess.A8)


class _SlotBoard(chess.Board):
    """A decoded Board whose history root retains the encoded piece IDs."""

    _slot_seed: Optional[tuple[Optional[int], ...]] = None
    _slot_position_type = PositionType.STANDARD

    def clear_stack(self) -> None:
        super().clear_stack()
        self._slot_seed = None
        self._slot_position_type = PositionType.STANDARD

    def copy(self, *, stack: Union[bool, int] = True) -> "_SlotBoard":
        copied = super().copy(stack=stack)
        if self._slot_seed is not None:
            copied._slot_position_type = self._slot_position_type
            if len(copied.move_stack) == len(self.move_stack):
                copied._slot_seed = self._slot_seed
            else:
                # A partial copy starts later in the history, so seed it with
                # the IDs at that new root rather than the original root.
                new_root = super().copy()
                new_root._slot_seed = self._slot_seed
                new_root._slot_position_type = self._slot_position_type
                for _ in copied.move_stack:
                    new_root.pop()
                copied._slot_seed = tuple(_tracked_slot_positions(new_root, self._slot_position_type))
        return copied

    def root(self) -> "_SlotBoard":
        root = super().root()
        root._slot_seed = self._slot_seed
        root._slot_position_type = self._slot_position_type
        return root


def _validate_board(board: chess.Board, position_type: PositionType) -> None:
    _validate_piece_set(board, position_type)
    if board.chess960:
        raise ValueError("Chess960 positions are unsupported")
    if board.promoted:
        raise ValueError(
            f"Promoted positions are unsupported by the {position_type.bitstring_length}-bit format"
        )
    if not board.is_valid():
        raise ValueError("FEN describes an invalid chess position")


def _square_color(square: int) -> int:
    return (chess.square_file(square) + chess.square_rank(square)) % 2


def _initial_slot_positions(board: chess.Board, position_type: PositionType) -> list[Optional[int]]:
    """Seed a history root once; FEN itself contains no original-piece IDs."""
    _validate_board(board, position_type)
    piece_slots = position_type.piece_slots
    positions: list[Optional[int]] = [None] * len(piece_slots)
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            indices = [
                index for index, (slot_color, slot_type, _) in enumerate(piece_slots)
                if (slot_color, slot_type) == (color, piece_type)
            ]
            squares = set(board.pieces(piece_type, color))
            if len(squares) > len(indices):
                raise ValueError("Promoted positions are unsupported: too many pieces of one type")

            for index in indices:
                home = piece_slots[index][2]
                if home in squares:
                    positions[index] = home
                    squares.remove(home)
            for square in sorted(squares):
                candidates = [
                    index for index in indices if positions[index] is None
                    and (piece_type != chess.BISHOP
                         or _square_color(square) == _square_color(piece_slots[index][2]))
                ]
                if not candidates:
                    raise ValueError("Promoted positions are unsupported: bishops share a square color")
                positions[candidates[0]] = square
    return positions


def _tracked_slot_positions(board: chess.Board, position_type: PositionType) -> list[Optional[int]]:
    """Replay moves from the root so captures never renumber surviving slots."""
    replay = board.root()
    if (
        isinstance(board, _SlotBoard)
        and board._slot_seed is not None
        and board._slot_position_type == position_type
    ):
        _validate_board(replay, position_type)
        positions = list(board._slot_seed)
    else:
        positions = _initial_slot_positions(replay, position_type)
    for move in board.move_stack:
        if move.promotion:
            raise ValueError("Promotion moves are unsupported by this encoding")
        if not move:  # python-chess null moves change turn and clear en passant.
            replay.push(move)
            continue
        if move not in replay.legal_moves:
            raise ValueError("Board contains an illegal move in its history")
        moving_index = positions.index(move.from_square)
        captured_square = move.to_square
        if replay.is_en_passant(move):
            captured_square += -8 if replay.turn == chess.WHITE else 8
        if captured_square in positions:
            positions[positions.index(captured_square)] = None
        if replay.is_castling(move):
            rank = 0 if replay.turn == chess.WHITE else 7
            kingside = chess.square_file(move.to_square) > chess.square_file(move.from_square)
            rook_from = chess.square(7 if kingside else 0, rank)
            rook_to = chess.square(5 if kingside else 3, rank)
            positions[positions.index(rook_from)] = rook_to
        positions[moving_index] = move.to_square
        replay.push(move)
    return positions


def board_to_bitstring(
    board: chess.Board, position_type: PositionType = PositionType.STANDARD,
) -> str:
    """Return the complete position: 231 standard bits or 28 KING_ROOK_VS_KING bits."""
    _validate_board(board, position_type)
    positions = _tracked_slot_positions(board, position_type)
    chunks = []
    for (_, piece_type, _), square in zip(position_type.piece_slots, positions):
        if piece_type != chess.KING:
            chunks.append("1" if square is None else "0")
        chunks.append(format(square if square is not None else 0, "06b"))
    chunks.append("0" if board.turn == chess.WHITE else "1")
    chunks.extend(
        "1" if has_right(color) else "0"
        for color in (chess.WHITE, chess.BLACK)
        for has_right in (board.has_kingside_castling_rights, board.has_queenside_castling_rights)
    )
    ep_file = 0 if board.ep_square is None else chess.square_file(board.ep_square) + 1
    chunks.append(format(ep_file, "04b"))
    return "".join(chunks)


def bitstring_to_board(
    bitstring: str, position_type: PositionType = PositionType.STANDARD,
) -> chess.Board:
    """Restore placement, state, and slot IDs; historical moves/clocks are omitted."""
    _validate_position_type(position_type)
    if (
        not isinstance(bitstring, str)
        or len(bitstring) != position_type.bitstring_length
        or set(bitstring) - {"0", "1"}
    ):
        raise ValueError(f"Expected exactly {position_type.bitstring_length} bits (only 0 and 1)")

    board = _SlotBoard(None)
    positions: list[Optional[int]] = []
    offset = 0
    for color, piece_type, home in position_type.piece_slots:
        captured = False
        if piece_type != chess.KING:
            captured = bitstring[offset] == "1"
            offset += 1
        square = int(bitstring[offset:offset + 6], 2)
        offset += 6
        if captured:
            if square:
                raise ValueError("Captured pieces must have zero position bits")
            positions.append(None)
            continue
        if board.piece_at(square) is not None:
            raise ValueError(f"Multiple pieces occupy {chess.square_name(square)}")
        if piece_type == chess.BISHOP and _square_color(square) != _square_color(home):
            raise ValueError("A bishop cannot change its original square color")
        board.set_piece_at(square, chess.Piece(piece_type, color))
        positions.append(square)

    board.turn = chess.WHITE if bitstring[offset] == "0" else chess.BLACK
    for flag, rook_square in zip(bitstring[offset + 1:offset + 5], _CASTLING_ROOKS):
        if flag == "1":
            board.castling_rights |= chess.BB_SQUARES[rook_square]
    ep_file = int(bitstring[offset + 5:offset + 9], 2)
    if ep_file > 8:
        raise ValueError("En passant must be 0000 (none) or 0001-1000 (files a-h)")
    if ep_file:
        board.ep_square = chess.square(ep_file - 1, 5 if board.turn == chess.WHITE else 2)
    _validate_board(board, position_type)
    board._slot_seed = tuple(positions)
    board._slot_position_type = position_type
    return board


def decode_bitstring(
    bitstring: str, position_type: PositionType = PositionType.STANDARD,
) -> dict[str, str]:
    """Return occupied squares and named pieces, e.g. {'b1': 'white_knight'}.

    The complete encoding is validated. Use bitstring_to_board() to read the
    additional game state as well as piece placement.
    """
    board = bitstring_to_board(bitstring, position_type)
    locations = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            color = "white" if piece.color == chess.WHITE else "black"
            locations[chess.square_name(square)] = f"{color}_{chess.piece_name(piece.piece_type)}"
    return locations


def moves_to_bitstrings(
    fen: str, moves: list[str], position_type: PositionType = PositionType.STANDARD,
) -> dict[str, str]:
    """Return {UCI move: resulting board bitstring} for legal_moves.py output."""
    if not isinstance(moves, list) or any(not isinstance(m, str) for m in moves):
        raise ValueError("Expected a JSON list of UCI move strings")
    board = chess.Board(fen)
    board_to_bitstring(board, position_type)  # Validate even for an empty move list.

    results = {}
    for uci in moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Illegal move for this FEN: {uci}")
        after = board.copy()
        after.push(move)
        results[uci] = board_to_bitstring(after, position_type)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "fen",
        nargs="?",
        help="The same FEN passed to legal_moves.py (standard default: starting position)",
    )
    parser.add_argument(
        "--position-type", choices=[kind.value for kind in PositionType],
        default=PositionType.STANDARD.value, help="Piece set and binary format (default: standard)",
    )
    parser.add_argument("--decode", metavar="BITSTRING", help="Show piece locations")
    args = parser.parse_args()
    position_type = PositionType(args.position_type)
    try:
        if args.decode is not None:
            result = decode_bitstring(args.decode, position_type)
        else:
            if args.fen is None and position_type != PositionType.STANDARD:
                parser.error("An explicit FEN is required for king-rook-vs-king")
            if sys.stdin.isatty():
                parser.error("Pipe the JSON output from legal_moves.py into this script")
            fen = chess.STARTING_FEN if args.fen is None else args.fen
            result = moves_to_bitstrings(fen, json.load(sys.stdin), position_type)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
