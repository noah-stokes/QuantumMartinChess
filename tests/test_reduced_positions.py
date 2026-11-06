"""Single-call encoding and game transitions for king and rook versus king."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

import chess

from tools.board_bitstrings import (
    BITSTRING_LENGTH,
    board_to_bitstring,
    bitstring_to_board,
    decode_bitstring,
    moves_to_bitstrings,
)
from tools.legal_moves import get_legal_moves
from tools.position_types import PositionType


REDUCED = PositionType.KING_ROOK_VS_KING
REDUCED_FEN = "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"


class ReducedPositionTests(unittest.TestCase):
    def assert_round_trip(self, board: chess.Board) -> str:
        bits = board_to_bitstring(board, REDUCED)
        self.assertEqual(len(bits), 28)
        self.assertLessEqual(set(bits), {"0", "1"})
        restored = bitstring_to_board(bits, REDUCED)
        self.assertEqual(restored.board_fen(), board.board_fen())
        self.assertEqual(restored.turn, board.turn)
        self.assertEqual(restored.castling_rights, board.castling_rights)
        self.assertEqual(restored.ep_square, board.ep_square)
        self.assertEqual(set(restored.legal_moves), set(board.legal_moves))
        self.assertEqual(board_to_bitstring(restored, REDUCED), bits)
        return bits

    def test_single_call_returns_complete_literal_28_bit_string(self):
        board = chess.Board(REDUCED_FEN)
        bits = board_to_bitstring(board, PositionType.KING_ROOK_VS_KING)
        self.assertIsInstance(bits, str)
        self.assertEqual(bits, "000100" "0000000" "111100" "000000000")
        self.assertEqual(len(bits), 28)
        self.assertEqual(board_to_bitstring(board, position_type=REDUCED), bits)
        self.assertEqual(REDUCED.bitstring_length, 28)
        self.assertEqual(REDUCED.piece_data_length, 19)

    def test_default_and_explicit_standard_still_use_231_bits(self):
        board = chess.Board()
        self.assertEqual(BITSTRING_LENGTH, 231)
        self.assertEqual(len(board_to_bitstring(board)), 231)
        self.assertEqual(board_to_bitstring(board), board_to_bitstring(board, PositionType.STANDARD))

    def test_rook_and_king_movement_turn_undo_and_decoded_copies(self):
        board = chess.Board(REDUCED_FEN)
        initial_bits = self.assert_round_trip(board)
        restored = bitstring_to_board(initial_bits, REDUCED)
        for move in ("a1a4", "e8f7", "e1f2", "f7e8", "a4h4"):
            board.push_uci(move)
            restored.push_uci(move)
            bits = self.assert_round_trip(board)
            self.assertEqual(bits[6], "0")
            self.assertEqual(bits[7:13], format(next(iter(board.pieces(chess.ROOK, chess.WHITE))), "06b"))
            self.assertEqual(bits[19], "0" if board.turn == chess.WHITE else "1")
            self.assertEqual(bits[24:], "0000")
            self.assertEqual(board_to_bitstring(restored, REDUCED), bits)
            for stack in (True, False, 1, 2):
                with self.subTest(move=move, stack=stack):
                    copied = restored.copy(stack=stack)
                    self.assertEqual(board_to_bitstring(copied, REDUCED), bits)
                    self.assert_round_trip(copied)
            self.assertEqual(board_to_bitstring(restored.root(), REDUCED), initial_bits)
        for _ in range(5):
            board.pop()
            restored.pop()
            self.assertEqual(board_to_bitstring(restored, REDUCED), self.assert_round_trip(board))

    def test_rook_capture_zeroes_coordinates_and_allows_only_kings(self):
        board = chess.Board("1k6/R7/8/8/8/8/8/4K3 b - - 0 1")
        restored = bitstring_to_board(self.assert_round_trip(board), REDUCED)
        board.push_uci("b8a7")
        restored.push_uci("b8a7")
        bits = self.assert_round_trip(board)
        self.assertEqual(bits, "000100" "1000000" "110000" "000000000")
        self.assertEqual(board_to_bitstring(restored, REDUCED), bits)
        self.assertEqual(
            decode_bitstring(bits, REDUCED), {"e1": "white_king", "a7": "black_king"},
        )
        board.push_uci("e1f2")
        restored.push_uci("e1f2")
        self.assertEqual(board_to_bitstring(restored, REDUCED), self.assert_round_trip(board))

    def test_both_white_castles_update_king_rook_and_rights(self):
        for row, right, move, king, rook, state in (
            ("4K2R", "K", "e1g1", chess.G1, chess.F1, "010000000"),
            ("R3K3", "Q", "e1c1", chess.C1, chess.D1, "001000000"),
        ):
            with self.subTest(move=move):
                fen = f"4k3/8/8/8/8/8/8/{row} w {right} - 0 1"
                board = chess.Board(fen)
                bits = self.assert_round_trip(board)
                self.assertEqual(bits[19:], state)
                restored = bitstring_to_board(bits, REDUCED)
                self.assertIn(move, get_legal_moves(fen, REDUCED))
                expected = moves_to_bitstrings(fen, [move], REDUCED)[move]
                board.push_uci(move)
                restored.push_uci(move)
                bits = self.assert_round_trip(board)
                self.assertEqual(bits[:6], format(king, "06b"))
                self.assertEqual(bits[6:13], "0" + format(rook, "06b"))
                self.assertEqual(bits[19:], "100000000")
                self.assertEqual(board_to_bitstring(restored, REDUCED), bits)
                self.assertEqual(bits, expected)

    def test_moving_king_or_rook_and_returning_does_not_restore_rights(self):
        for white_moves in (("a1a2", "a2a1"), ("e1f1", "f1e1")):
            board = chess.Board("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1")
            for move in (white_moves[0], "e8e7", white_moves[1], "e7e8"):
                board.push_uci(move)
            self.assertEqual(self.assert_round_trip(board)[20:24], "0000")

    def test_capture_of_castling_rook_removes_rights(self):
        board = chess.Board("8/8/8/8/8/8/1k6/R3K3 b Q - 0 1")
        board.push_uci("b2a1")
        bits = self.assert_round_trip(board)
        self.assertEqual(bits[6:13], "1000000")
        self.assertEqual(bits[20:24], "0000")

    def test_reduced_legal_move_pipeline_returns_independent_successors(self):
        moves = get_legal_moves(REDUCED_FEN, position_type=REDUCED)
        self.assertEqual(set(moves), {move.uci() for move in chess.Board(REDUCED_FEN).legal_moves})
        successors = moves_to_bitstrings(REDUCED_FEN, moves, position_type=REDUCED)
        self.assertEqual(set(successors), set(moves))
        for move, bits in successors.items():
            board = chess.Board(REDUCED_FEN)
            board.push_uci(move)
            self.assertEqual(bits, self.assert_round_trip(board))

    def test_switching_selected_format_reseeds_instead_of_reusing_wrong_slots(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        reduced = bitstring_to_board(board_to_bitstring(board, REDUCED), REDUCED)
        standard = bitstring_to_board(board_to_bitstring(board))
        for move in ("h1h4", "e8e7", "e1f2"):
            board.push_uci(move)
            reduced.push_uci(move)
            standard.push_uci(move)
            self.assertEqual(board_to_bitstring(reduced), board_to_bitstring(board))
            self.assertEqual(board_to_bitstring(standard, REDUCED), board_to_bitstring(board, REDUCED))
            # Changing formats reseeds at the available history root. Dropping
            # history means both boards now seed at the current position.
            self.assertEqual(
                board_to_bitstring(reduced.copy(stack=False)),
                board_to_bitstring(board.copy(stack=False)),
            )

    def test_reduced_cli_pipeline_and_decode(self):
        directory = Path(__file__).resolve().parents[1]
        arguments = [REDUCED_FEN, "--position-type", "king-rook-vs-king"]
        legal = subprocess.run(
            [sys.executable, "-m", "tools.legal_moves", *arguments],
            cwd=directory, check=True, capture_output=True, text=True,
        )
        encoded = subprocess.run(
            [sys.executable, "-m", "tools.board_bitstrings", *arguments],
            cwd=directory, input=legal.stdout, check=True, capture_output=True, text=True,
        )
        positions = json.loads(encoded.stdout)
        self.assertEqual(set(positions), set(json.loads(legal.stdout)))
        self.assertTrue(all(len(bits) == 28 for bits in positions.values()))
        decoded = subprocess.run(
            [sys.executable, "-m", "tools.board_bitstrings",
             "--decode", positions["a1a4"], "--position-type", "king-rook-vs-king"],
            cwd=directory, check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(decoded.stdout)["a4"], "white_rook")


class ReducedValidationTests(unittest.TestCase):
    def test_rejects_extra_material_in_all_position_input_functions(self):
        for fen in (
            chess.STARTING_FEN,
            "4k3/8/8/8/8/8/8/R3K2R w - - 0 1",
            "r3k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/P7/R3K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/RN2K3 w - - 0 1",
        ):
            for call in (
                lambda: board_to_bitstring(chess.Board(fen), REDUCED),
                lambda: get_legal_moves(fen, REDUCED),
                lambda: moves_to_bitstrings(fen, [], REDUCED),
            ):
                with self.subTest(fen=fen), self.assertRaisesRegex(ValueError, "allows only"):
                    call()

    def test_rejects_missing_kings(self):
        for fen in ("8/8/8/8/8/8/8/R3K3 w - - 0 1", "4k3/8/8/8/8/8/8/R7 w - - 0 1"):
            with self.subTest(fen=fen), self.assertRaisesRegex(ValueError, "invalid"):
                board_to_bitstring(chess.Board(fen), REDUCED)

    def test_rejects_marked_promoted_rook(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R~3K3 w - - 0 1")
        with self.assertRaisesRegex(ValueError, "Promoted"):
            board_to_bitstring(board, REDUCED)

    def test_rejects_wrong_length_type_and_nonbinary_input(self):
        for value in (None, 28, "", "0" * 27, "0" * 29, "0" * 27 + "x", "0" * 231):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "28 bits"):
                decode_bitstring(value, REDUCED)
        with self.assertRaisesRegex(ValueError, "231 bits"):
            bitstring_to_board(board_to_bitstring(chess.Board(REDUCED_FEN), REDUCED))

    def test_rejects_nonzero_captured_coordinates_and_overlaps(self):
        bits = board_to_bitstring(chess.Board(REDUCED_FEN), REDUCED)
        with self.assertRaisesRegex(ValueError, "zero position"):
            bitstring_to_board(bits[:6] + "1000001" + bits[13:], REDUCED)
        with self.assertRaisesRegex(ValueError, "Multiple pieces"):
            bitstring_to_board(bits[:13] + bits[:6] + bits[19:], REDUCED)

    def test_rejects_impossible_castling_and_any_en_passant(self):
        bits = board_to_bitstring(chess.Board(REDUCED_FEN), REDUCED)
        with self.assertRaisesRegex(ValueError, "invalid"):
            bitstring_to_board(bits[:20] + "0010" + bits[24:], REDUCED)
        for ep_file in range(1, 16):
            with self.subTest(ep_file=ep_file), self.assertRaises(ValueError):
                bitstring_to_board(bits[:24] + format(ep_file, "04b"), REDUCED)

    def test_rejects_non_enum_position_type(self):
        for value in (None, "standard", "king-rook-vs-king", 1):
            for call in (
                lambda: board_to_bitstring(chess.Board(REDUCED_FEN), value),
                lambda: bitstring_to_board("0" * 28, value),
                lambda: decode_bitstring("0" * 28, value),
                lambda: moves_to_bitstrings(REDUCED_FEN, [], value),
                lambda: get_legal_moves(REDUCED_FEN, value),
            ):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "PositionType"):
                    call()

    def test_reduced_clis_require_fen_and_reject_unknown_type(self):
        directory = Path(__file__).resolve().parents[1]
        for module in ("tools.legal_moves", "tools.board_bitstrings"):
            for position_type, message in (("king-rook-vs-king", "explicit FEN"), ("unknown", "invalid choice")):
                with self.subTest(module=module, position_type=position_type):
                    result = subprocess.run(
                        [sys.executable, "-m", module, "--position-type", position_type],
                        cwd=directory, input="[]", capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)


if __name__ == "__main__":
    unittest.main()
