"""Regression tests for the 231-bit layout and legal game transitions."""

import json
from pathlib import Path
import random
import subprocess
import sys
import unittest

import chess

from board_bitstrings import (
    BITSTRING_LENGTH,
    board_to_bitstring,
    bitstring_to_board,
    decode_bitstring,
    moves_to_bitstrings,
)
from legal_moves import get_legal_moves


def play(*moves: str, fen: str = chess.STARTING_FEN) -> chess.Board:
    board = chess.Board(fen)
    for move in moves:
        board.push_uci(move)
    return board


class EncodingTests(unittest.TestCase):
    def assert_round_trip(self, board: chess.Board) -> str:
        encoded = board_to_bitstring(board)
        self.assertEqual(len(encoded), 231)
        self.assertLessEqual(set(encoded), {"0", "1"})
        restored = bitstring_to_board(encoded)
        self.assertEqual(restored.board_fen(), board.board_fen())
        self.assertEqual(restored.turn, board.turn)
        self.assertEqual(restored.castling_rights, board.castling_rights)
        self.assertEqual(restored.ep_square, board.ep_square)
        self.assertEqual(set(restored.legal_moves), set(board.legal_moves))
        self.assertEqual(board_to_bitstring(restored), encoded)
        return encoded

    def test_starting_layout_matches_literal_piece_slots(self):
        # Six bits per king; each following seven-bit slot has its own flag.
        expected_white = (
            "000100" "0000011" "0000000" "0000111"
            "0000010" "0000101" "0000001" "0000110"
            "0001000" "0001001" "0001010" "0001011"
            "0001100" "0001101" "0001110" "0001111"
        )
        expected_black = (
            "111100" "0111011" "0111000" "0111111"
            "0111010" "0111101" "0111001" "0111110"
            "0110000" "0110001" "0110010" "0110011"
            "0110100" "0110101" "0110110" "0110111"
        )
        self.assertEqual(BITSTRING_LENGTH, 231)
        self.assertEqual(
            self.assert_round_trip(chess.Board()),
            expected_white + expected_black + "011110000",
        )

    def test_capture_zeroes_only_the_captured_original_slot(self):
        board = play("e2e4", "d7d5", "e4d5")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[83:90], "0" + format(chess.D5, "06b"))
        self.assertEqual(encoded[187:194], "1000000")  # Black d-pawn.
        self.assertEqual(encoded[194:201], "0" + format(chess.E7, "06b"))
        board.push_uci("d8d5")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[83:90], "1000000")
        self.assertEqual(encoded[117:124], "0" + format(chess.D5, "06b"))

    def test_rook_capture_keeps_surviving_rook_in_second_slot(self):
        board = play("a8a1", fen="r3k3/8/8/8/8/8/8/R3K2R b KQ - 0 1")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[13:20], "1000000")
        self.assertEqual(encoded[20:27], "0000111")
        self.assertEqual(encoded[223:227], "1000")

    def test_identical_pieces_keep_slots_when_square_order_changes(self):
        board = play(
            "b1c3", "e8e7", "g1e2", "e7e8", "c3b5", "e8e7", "e2c3",
            fen="4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1",
        )
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[41:48], "0" + format(chess.B5, "06b"))
        self.assertEqual(encoded[48:55], "0" + format(chess.C3, "06b"))
        self.assertEqual(board_to_bitstring(board.copy()), encoded)
        board.pop()
        self.assert_round_trip(board)

    def test_all_castles_move_both_original_slots_and_remove_rights(self):
        for turn, move, king_slice, rook_slice, king_square, rook_square, rights in (
            ("w", "e1g1", slice(0, 6), slice(20, 27), chess.G1, chess.F1, "0011"),
            ("w", "e1c1", slice(0, 6), slice(13, 20), chess.C1, chess.D1, "0011"),
            ("b", "e8g8", slice(111, 117), slice(131, 138), chess.G8, chess.F8, "1100"),
            ("b", "e8c8", slice(111, 117), slice(124, 131), chess.C8, chess.D8, "1100"),
        ):
            with self.subTest(move=move):
                board = play(move, fen=f"r3k2r/8/8/8/8/8/8/R3K2R {turn} KQkq - 0 1")
                encoded = self.assert_round_trip(board)
                self.assertEqual(encoded[king_slice], format(king_square, "06b"))
                self.assertEqual(encoded[rook_slice], "0" + format(rook_square, "06b"))
                self.assertEqual(encoded[223:227], rights)

    def test_returning_king_and_rook_do_not_restore_castling_rights(self):
        board = play(
            "a1a2", "e8e7", "a2a1", "e7e8",
            fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        )
        self.assertEqual(self.assert_round_trip(board)[223:227], "1000")

    def test_all_en_passant_files_and_both_inferred_ranks(self):
        for index, file in enumerate("abcdefgh", start=1):
            for setup, move, rank in (
                ((), f"{file}2{file}4", 2),
                (("g1f3",), f"{file}7{file}5", 5),
            ):
                with self.subTest(move=move):
                    board = play(*setup, move)
                    encoded = self.assert_round_trip(board)
                    self.assertEqual(encoded[227:], format(index, "04b"))
                    self.assertEqual(
                        bitstring_to_board(encoded).ep_square, chess.square(index - 1, rank)
                    )

    def test_en_passant_expires_after_other_move(self):
        board = play("e2e4", "g8f6")
        self.assertEqual(self.assert_round_trip(board)[227:], "0000")

    def test_white_en_passant_capture_zeroes_bypassed_pawn(self):
        board = play("e2e4", "a7a6", "e4e5", "d7d5", "e5d6")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[83:90], "0" + format(chess.D6, "06b"))
        self.assertEqual(encoded[187:194], "1000000")
        self.assertEqual(encoded[227:], "0000")

    def test_black_en_passant_capture_zeroes_bypassed_pawn(self):
        board = play("d2d4", "e7e5", "d4d5", "e5e4", "f2f4", "e4f3")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[194:201], "0" + format(chess.F3, "06b"))
        self.assertEqual(encoded[90:97], "1000000")
        self.assertEqual(encoded[227:], "0000")

    def test_sparse_fen_has_captured_slots_and_home_square_preferences(self):
        board = chess.Board("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[6:20], "1000000" * 2)
        self.assertEqual(encoded[20:27], "0000111")
        self.assertEqual(encoded[27:111], "1000000" * 12)
        self.assertEqual(encoded[117:222], "1000000" * 15)

    def test_bare_fen_bishop_is_assigned_by_original_square_color(self):
        board = chess.Board("4k3/8/8/8/8/6B1/8/4K3 w - - 0 1")
        encoded = self.assert_round_trip(board)
        self.assertEqual(encoded[27:34], "0" + format(chess.G3, "06b"))
        self.assertEqual(encoded[34:41], "1000000")

    def test_decode_preserves_existing_square_to_piece_interface(self):
        decoded = decode_bitstring(board_to_bitstring(chess.Board()))
        self.assertEqual(len(decoded), 32)
        self.assertEqual(decoded["a1"], "white_rook")
        self.assertEqual(decoded["h8"], "black_rook")
        self.assertEqual(decoded["e8"], "black_king")

    def test_decoded_board_retains_ids_through_moves_undo_and_copies(self):
        original = play(
            "b1c3", "e8e7", "g1e2", "e7e8", "c3b5", "e8e7", "e2c3",
            fen="4k3/8/8/8/8/8/8/1N2K1N1 w - - 0 1",
        )
        bits = board_to_bitstring(original)
        restored = bitstring_to_board(bits)
        self.assertEqual(board_to_bitstring(restored.root()), bits)
        for move in ("e7e8", "b5a3", "e8e7", "c3e2"):
            original.push_uci(move)
            restored.push_uci(move)
            expected = board_to_bitstring(original)
            self.assertEqual(board_to_bitstring(restored), expected)
            for stack in (True, False, 1, 2):
                with self.subTest(move=move, stack=stack):
                    self.assertEqual(board_to_bitstring(restored.copy(stack=stack)), expected)
            self.assertEqual(board_to_bitstring(restored.root()), bits)
        for _ in range(4):
            original.pop()
            restored.pop()
            self.assertEqual(board_to_bitstring(restored), board_to_bitstring(original))

    def test_decoded_board_manual_edits_reset_identity_metadata(self):
        board = bitstring_to_board(board_to_bitstring(chess.Board()))
        board.set_fen("4k3/8/8/8/8/8/8/4K2R w K - 0 1")
        self.assert_round_trip(board)
        board.remove_piece_at(chess.H1)
        board.castling_rights = 0
        self.assertEqual(self.assert_round_trip(board)[20:27], "1000000")

    def test_legal_move_pipeline_returns_independent_successors(self):
        moves = get_legal_moves(chess.STARTING_FEN)
        encoded = moves_to_bitstrings(chess.STARTING_FEN, moves)
        self.assertEqual(len(encoded), 20)
        for move in moves:
            self.assertEqual(encoded[move], self.assert_round_trip(play(move)))

    def test_random_legal_games_preserve_decoded_state(self):
        generator = random.Random(231)
        for _ in range(3):
            board = chess.Board()
            for _ in range(80):
                self.assert_round_trip(board)
                moves = [move for move in board.legal_moves if not move.promotion]
                if not moves:
                    break
                board.push(generator.choice(moves))

    def test_cli_pipeline_and_decode(self):
        directory = Path(__file__).resolve().parent
        legal = subprocess.run(
            [sys.executable, str(directory / "legal_moves.py")],
            check=True, capture_output=True, text=True,
        )
        encoded = subprocess.run(
            [sys.executable, str(directory / "board_bitstrings.py")],
            input=legal.stdout, check=True, capture_output=True, text=True,
        )
        positions = json.loads(encoded.stdout)
        self.assertEqual(len(positions), 20)
        self.assertTrue(all(len(bits) == 231 for bits in positions.values()))
        decoded = subprocess.run(
            [sys.executable, str(directory / "board_bitstrings.py"), "--decode", positions["e2e4"]],
            check=True, capture_output=True, text=True,
        )
        self.assertEqual(json.loads(decoded.stdout)["e4"], "white_pawn")


class ValidationTests(unittest.TestCase):
    def test_rejects_bad_length_type_and_non_binary_characters(self):
        for value in (None, 231, "", "0" * 230, "0" * 232, "0" * 230 + "x"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "231 bits"):
                decode_bitstring(value)

    def test_rejects_nonzero_captured_position(self):
        encoded = board_to_bitstring(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        malformed = encoded[:6] + "1000001" + encoded[13:]
        with self.assertRaisesRegex(ValueError, "zero position"):
            decode_bitstring(malformed)

    def test_rejects_overlapping_pieces(self):
        encoded = board_to_bitstring(chess.Board())
        malformed = encoded[:6] + "0000100" + encoded[13:]
        with self.assertRaisesRegex(ValueError, "Multiple pieces"):
            decode_bitstring(malformed)

    def test_rejects_undefined_en_passant_file_codes(self):
        encoded = board_to_bitstring(chess.Board())
        for code in range(9, 16):
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, "En passant"):
                decode_bitstring(encoded[:227] + format(code, "04b"))

    def test_rejects_inconsistent_game_state(self):
        encoded = board_to_bitstring(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        for state in ("010000000", "000000001"):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "invalid"):
                decode_bitstring(encoded[:222] + state)

    def test_rejects_invalid_boards_and_chess960(self):
        with self.assertRaisesRegex(ValueError, "invalid"):
            board_to_bitstring(chess.Board(None))
        with self.assertRaisesRegex(ValueError, "Chess960"):
            board_to_bitstring(chess.Board(chess960=True))

    def test_rejects_known_promoted_fens_even_without_requested_moves(self):
        for fen in (
            "4k3/8/8/8/8/8/8/Q~3K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/Q2QK3 w - - 0 1",
            "4k3/8/8/8/8/4B3/8/2B1K3 w - - 0 1",
        ):
            with self.subTest(fen=fen), self.assertRaisesRegex(ValueError, "[Pp]romot"):
                moves_to_bitstrings(fen, [])

    def test_rejects_all_promotion_moves(self):
        fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        for piece in "qrbn":
            with self.subTest(piece=piece), self.assertRaisesRegex(ValueError, "[Pp]romot"):
                moves_to_bitstrings(fen, ["a7a8" + piece])

    def test_rejects_promotion_history_after_promoted_piece_is_captured(self):
        board = play("a7a8q", "b8a8", fen="1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        self.assertFalse(board.promoted)
        with self.assertRaisesRegex(ValueError, "Promotion"):
            board_to_bitstring(board)

    def test_rejects_invalid_move_lists_and_illegal_moves(self):
        for moves in ("e2e4", [3], ["e2e5"], ["bad"]):
            with self.subTest(moves=moves), self.assertRaises(ValueError):
                moves_to_bitstrings(chess.STARTING_FEN, moves)


if __name__ == "__main__":
    unittest.main()
