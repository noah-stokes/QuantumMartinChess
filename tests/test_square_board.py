"""Promotion-aware square encoding and full-chess dataset contracts."""

import random
import unittest

import chess

from tools.board_bitstrings import board_to_bitstring, bitstring_to_board, moves_to_bitstrings
from training.generate_standard import (
    FULL, decode, encode, make_pair, mirror_bits, placement_family, quotas,
    replay_game, score_class, validate_dataset,
)
from training.translate_qnn_data import translate_score


class SquareBoardTests(unittest.TestCase):
    def assert_round_trip(self, board):
        bits = board_to_bitstring(board, FULL)
        self.assertEqual(len(bits), 265)
        restored = bitstring_to_board(bits, FULL)
        self.assertEqual(restored.board_fen(), board.board_fen())
        self.assertEqual(restored.turn, board.turn)
        self.assertEqual(restored.castling_rights, board.castling_rights)
        self.assertEqual(restored.ep_square, board.ep_square)
        self.assertEqual(set(restored.legal_moves), set(board.legal_moves))
        self.assertEqual(encode(restored), bits)
        self.assertEqual(mirror_bits(mirror_bits(bits)), bits)
        self.assertEqual(decode(mirror_bits(bits)).board_fen(), board.mirror().board_fen())
        return bits

    def test_literal_starting_layout_and_state(self):
        codes = [4, 2, 3, 5, 6, 3, 2, 4] + [1] * 8 + [0] * 32
        codes += [7] * 8 + [10, 8, 9, 11, 12, 9, 8, 10]
        expected = "".join(format(code, "04b") for code in codes) + "011110000"
        self.assertEqual(self.assert_round_trip(chess.Board()), expected)
        self.assertEqual(FULL.piece_data_length, 256)
        self.assertEqual(FULL.bitstring_length, 265)

    def test_all_promotion_types_for_both_colors(self):
        for fen, start, end in (
            ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7", "a8"),
            ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2", "a1"),
        ):
            for piece in "qrbn":
                with self.subTest(color=fen, piece=piece):
                    board = chess.Board(fen)
                    uci = start + end + piece
                    board.push_uci(uci)
                    bits = self.assert_round_trip(board)
                    self.assertEqual(moves_to_bitstrings(fen, [uci], FULL)[uci], bits)
                    fresh = chess.Board(board.fen())
                    self.assertEqual(encode(fresh), bits)

    def test_extra_queens_same_color_bishops_and_promotion_capture(self):
        for fen in (
            "4k3/8/8/8/8/8/8/Q2QK3 w - - 0 1",
            "4k3/8/8/8/8/4B3/8/2B1K3 w - - 0 1",
            "4k3/8/8/8/8/8/8/Q~3K3 w - - 0 1",
        ):
            self.assert_round_trip(chess.Board(fen))
        board = chess.Board("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1")
        for move in ("a7a8q", "b8a8"):
            board.push_uci(move)
            self.assert_round_trip(board)

    def test_castling_both_colors_and_sides(self):
        for uci in ("e1g1", "e1c1", "e8g8", "e8c8"):
            board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
            board.turn = uci.startswith("e1")
            before = self.assert_round_trip(board)
            self.assertEqual(mirror_bits(before)[257:261], before[259:261] + before[257:259])
            board.push_uci(uci)
            self.assert_round_trip(board)

    def test_en_passant_and_non_capturable_double_push(self):
        for fen, uci in (
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"),
            (chess.STARTING_FEN, "e2e4"),
        ):
            board = chess.Board(fen)
            self.assert_round_trip(board)
            board.push_uci(uci)
            bits = self.assert_round_trip(board)
            if uci == "e2e4":
                self.assertEqual(bits[261:], "0101")

    def test_random_legal_games_include_promotions(self):
        rng = random.Random(265)
        promotions = 0
        for _ in range(8):
            board = chess.Board()
            for _ in range(220):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                promotions += bool(move.promotion)
                board.push(move)
                self.assert_round_trip(board)
        self.assertGreater(promotions, 0)

    def test_rejects_bad_codes_lengths_and_state(self):
        bits = encode(chess.Board())
        for code in (13, 14, 15):
            with self.assertRaisesRegex(ValueError, "undefined"):
                decode(format(code, "04b") + bits[4:])
        for corrupt in (None, 265, bits[:-1], bits + "0", "x" + bits[1:]):
            with self.assertRaisesRegex(ValueError, "265 bits"):
                decode(corrupt)
        for code in range(9, 16):
            with self.assertRaisesRegex(ValueError, "En passant"):
                decode(bits[:261] + format(code, "04b"))
        bare = encode(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        with self.assertRaisesRegex(ValueError, "invalid"):
            decode(bare[:257] + "1000" + bare[261:])


class StandardDatasetTests(unittest.TestCase):
    def test_quotas_and_boundaries(self):
        q = quotas(50000)
        self.assertEqual({s: sum(c.values()) * 2 for s, c in q.items()},
                         {"train": 40000, "validation": 5000, "test": 5000})
        self.assertEqual(q["validation"], {"near_equal": 500, "moderate": 750, "large": 750, "mate": 500})
        for count in (0, -200, 199, 201):
            with self.assertRaises(ValueError):
                quotas(count)
        for value, group in ((0.2, "near_equal"), (0.6, "moderate"),
                             (0.75, "large"), (0.975, "mate")):
            self.assertEqual(score_class(value), group)
            self.assertEqual(score_class(-value), group)

    def test_family_groups_turn_and_state_but_respects_pawn_direction(self):
        board = chess.Board()
        family = placement_family(board)
        self.assertEqual(placement_family(board.mirror()), family)
        board.turn = not board.turn
        board.castling_rights = 0
        self.assertEqual(placement_family(board), family)
        board = chess.Board()
        board.push_uci("a2a3")
        self.assertNotEqual(placement_family(board), family)

    def test_pair_and_game_provenance_after_promotion(self):
        fen = "4k3/P7/8/8/8/8/8/4K3 w - - 0 1"
        board = chess.Board(fen)
        board.push_uci("a7a8q")
        label = {"raw_engine_evaluation": {"kind": "cp", "value": 800},
                 "termination": None, "best_move": "e8d7", "engine_depth": 10, "engine_nodes": 10000}
        target, kind = translate_score(label)
        candidate = {"bitstring": encode(board), "fen": decode(encode(board)).fen(en_passant="fen"),
                     "ply": 1, "label": label, "target": target, "kind": kind,
                     "promotion_types": ["queen"]}
        left, right = make_pair(candidate, 7)
        self.assertEqual(left["target_score"], -right["target_score"])
        self.assertEqual(translate_score(right)[0], right["target_score"])
        self.assertEqual(left["promotion_types"], ["queen"])
        self.assertEqual(right["placement_family"], placement_family(decode(right["bitstring"])))

    def test_replay_rejects_illegal_history(self):
        with self.assertRaisesRegex(ValueError, "Invalid game"):
            replay_game({"game_id": 1, "moves": ["e2e5"]}, {1})
        game = {"game_id": 1, "moves": ["e2e4", "e7e5"], "split": "train", "ended": "ply_limit"}
        replay = replay_game(game, {1, 2})
        rows = []
        for ply, bits in replay["snapshots"].items():
            label = {"raw_engine_evaluation": {"kind": "cp", "value": 50}, "termination": None}
            target, kind = translate_score(label)
            rows.extend(make_pair({"bitstring": bits, "fen": decode(bits).fen(en_passant="fen"),
                                   "ply": ply, "label": label, "target": target, "kind": kind,
                                   "promotion_types": []}, 1))
        requested = {"train": {"near_equal": 2}}
        self.assertEqual(validate_dataset({"train": rows}, {1: replay}, requested)["records_checked"], 4)
        rows[2]["source_ply"] = 1
        with self.assertRaisesRegex(ValueError, "replayed legal game"):
            validate_dataset({"train": rows}, {1: replay}, requested)


if __name__ == "__main__":
    unittest.main()
