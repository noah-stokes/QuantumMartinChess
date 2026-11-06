"""Move selection, successor inputs, and independence from the QNN layout."""

from functools import partial
import unittest
from unittest.mock import patch

import chess

from tools.board_bitstrings import board_to_bitstring, moves_to_bitstrings
from tools.legal_moves import get_legal_moves
from tools.position_types import PositionType
from tools.qnn_engine import choose_move


ROOK_FEN = "4k3/8/8/8/8/8/8/5K1R w - - 0 1"


class MoveSelectionTests(unittest.TestCase):
    def test_each_successor_is_scored_and_original_turn_selects_best(self):
        for turn, expected in ((chess.WHITE, "e2e4"), (chess.BLACK, "d7d5")):
            with self.subTest(turn=turn):
                board = chess.Board()
                board.turn = turn
                original = board.fen()
                seen = []

                def evaluate(after):
                    seen.append(after)
                    self.assertEqual(after.turn, not turn)
                    move = after.peek().uci()
                    if move == "e2e4":
                        return 1.0
                    if move == "d7d5":
                        return -1.0
                    return 0.0

                with patch("tools.qnn_engine.get_legal_moves", wraps=get_legal_moves) as legal:
                    selected = choose_move(board, evaluate, lambda after: after)
                self.assertEqual(selected, expected)
                legal.assert_called_once_with(original, PositionType.SQUARE_BOARD)
                moves = get_legal_moves(original)
                self.assertEqual([after.peek().uci() for after in seen], moves)
                self.assertEqual(len({id(after) for after in seen}), len(moves))
                for after, uci in zip(seen, moves):
                    expected_board = board.copy()
                    expected_board.push_uci(uci)
                    self.assertEqual(after.fen(), expected_board.fen())
                self.assertEqual(board.fen(), original)
                self.assertEqual(board.move_stack, [])

    def test_ties_keep_first_legal_move(self):
        moves = get_legal_moves(chess.STARTING_FEN)
        for score in (-1.0, 0.0, 1.0):
            with self.subTest(score=score):
                self.assertEqual(
                    choose_move(chess.STARTING_FEN, lambda _: score, lambda after: after),
                    moves[0],
                )

    def test_checkmate_and_stalemate_do_not_call_qnn_or_encoder(self):
        for fen in ("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1",
                    "7k/5K2/6Q1/8/8/8/8/8 b - - 0 1"):
            with self.subTest(fen=fen):
                def unexpected(_):
                    self.fail("A position without legal moves must not be evaluated")

                self.assertIsNone(choose_move(fen, unexpected, unexpected))

    def test_encoder_accepts_different_bitstring_lengths_and_nonbinary_inputs(self):
        for kind in PositionType:
            with self.subTest(kind=kind):
                seen = []

                def evaluate(bits):
                    self.assertEqual(len(bits), kind.bitstring_length)
                    seen.append(bits)
                    return 0.0

                moves = get_legal_moves(ROOK_FEN, kind)
                choose_move(ROOK_FEN, evaluate,
                            partial(board_to_bitstring, position_type=kind), position_type=kind)
                self.assertEqual(seen, list(moves_to_bitstrings(ROOK_FEN, moves, kind).values()))

        def tuple_model(features):
            self.assertIsInstance(features, tuple)
            return .8 if features[0] == "e2e4" else -.2

        self.assertEqual(choose_move(chess.STARTING_FEN, tuple_model,
                                     lambda after: (after.peek().uci(), after.turn)), "e2e4")

    def test_promotions_castling_and_en_passant_reach_encoder(self):
        for fen, expected in (
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8n"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1n"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
        ):
            with self.subTest(move=expected):
                before = chess.Board(fen)
                target = before.copy()
                target.push_uci(expected)
                encode = partial(board_to_bitstring, position_type=PositionType.SQUARE_BOARD)
                target_bits = encode(target)
                best_score = 1.0 if before.turn == chess.WHITE else -1.0
                self.assertEqual(choose_move(fen, lambda bits: best_score if bits == target_bits else 0,
                                            encode), expected)

    def test_board_history_and_slot_identity_survive(self):
        board = chess.Board("4k3/8/8/8/8/8/8/R3K2R w - - 0 1")
        for move in ("a1a4", "e8e7", "h1h2", "e7e8"):
            board.push_uci(move)
        original = board.copy()
        expected = []
        for uci in get_legal_moves(board.fen()):
            after = board.copy()
            after.push_uci(uci)
            expected.append(board_to_bitstring(after))
        seen = []

        def evaluate(bits):
            seen.append(bits)
            return 0.0

        choose_move(board, evaluate, board_to_bitstring)
        self.assertEqual(seen, expected)
        self.assertEqual(board.fen(), original.fen())
        self.assertEqual(board.move_stack, original.move_stack)

    def test_invalid_scores_fail_instead_of_silently_selecting_a_move(self):
        for score in (float("nan"), float("inf"), -float("inf"), 1.01, -1.01, None, [0.0]):
            with self.subTest(score=score), self.assertRaisesRegex(ValueError, "QNN score"):
                choose_move(ROOK_FEN, lambda _: score, lambda after: after)

    def test_invalid_positions_and_reduced_piece_sets_are_rejected(self):
        for position in ("bad FEN", "8/8/8/8/8/8/8/8 w - - 0 1",
                         chess.Board(chess960=True)):
            with self.subTest(position=position), self.assertRaises(ValueError):
                choose_move(position, lambda _: 0.0, lambda after: after)
        with self.assertRaisesRegex(ValueError, "allows only"):
            choose_move(chess.STARTING_FEN, lambda _: 0.0, lambda after: after,
                        position_type=PositionType.ROOK_ENDGAME)


if __name__ == "__main__":
    unittest.main()
