"""Supported piece sets and their fixed binary-encoding slots."""

from enum import Enum

import chess


class PositionType(Enum):
    STANDARD = "standard"
    KING_ROOK_VS_KING = "king-rook-vs-king"

    @property
    def piece_slots(self) -> tuple[tuple[bool, int, int], ...]:
        """Return (color, piece type, original square) for each encoding slot."""
        return _PIECE_SLOTS[self]

    @property
    def piece_data_length(self) -> int:
        return sum(6 if piece_type == chess.KING else 7 for _, piece_type, _ in self.piece_slots)

    @property
    def bitstring_length(self) -> int:
        return self.piece_data_length + 9


# Slots never depend on the current square order or the number of survivors.
_WHITE_SLOTS = (
    (chess.KING, chess.E1),
    (chess.QUEEN, chess.D1),
    (chess.ROOK, chess.A1), (chess.ROOK, chess.H1),
    (chess.BISHOP, chess.C1), (chess.BISHOP, chess.F1),
    (chess.KNIGHT, chess.B1), (chess.KNIGHT, chess.G1),
) + tuple((chess.PAWN, square) for square in range(chess.A2, chess.H2 + 1))
_PIECE_SLOTS = {
    PositionType.STANDARD: tuple(
        (color, piece_type, square if color == chess.WHITE else chess.square_mirror(square))
        for color in (chess.WHITE, chess.BLACK)
        for piece_type, square in _WHITE_SLOTS
    ),
    PositionType.KING_ROOK_VS_KING: (
        (chess.WHITE, chess.KING, chess.E1),
        (chess.WHITE, chess.ROOK, chess.A1),
        (chess.BLACK, chess.KING, chess.E8),
    ),
}


def _validate_position_type(position_type: PositionType) -> None:
    if not isinstance(position_type, PositionType):
        raise ValueError("Expected a PositionType enum value")


def _validate_piece_set(board: chess.Board, position_type: PositionType) -> None:
    """Restrict reduced positions without changing standard legal-move support."""
    _validate_position_type(position_type)
    if position_type == PositionType.STANDARD:
        return
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in chess.PIECE_TYPES:
            allowed_count = sum(
                (slot_color, slot_type) == (color, piece_type)
                for slot_color, slot_type, _ in position_type.piece_slots
            )
            if len(board.pieces(piece_type, color)) > allowed_count:
                raise ValueError(
                    "king-rook-vs-king allows only a White king, at most one White rook, "
                    "and a Black king"
                )
