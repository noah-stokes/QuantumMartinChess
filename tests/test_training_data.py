"""Dataset contracts, symmetry isolation, cache recovery, and rook transitions."""

from contextlib import closing, contextmanager
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import chess
import chess.engine

from tools.board_bitstrings import board_to_bitstring, bitstring_to_board, moves_to_bitstrings
from tools.legal_moves import get_legal_moves
from tools.position_types import PositionType
from training.generate_rook_endgame import (
    AnalysisCache, BIN_NAMES, SYMMETRIES, analyze_board,
    evaluation_label, family_id, generate, make_record,
    normalize_score, prepare_engine, sample_pairs, score_bin, split_for_family,
    terminal_label, validate_records,
)


ROOKS = PositionType.ROOK_ENDGAME
FEN = "4k2r/8/8/8/8/8/8/R3K3 w - - 0 1"
MATE = "7k/5K2/8/8/8/8/8/7R b - - 0 1"
STALEMATE = "7k/6R1/5K2/8/8/8/8/8 b - - 0 1"


class RookEncodingTests(unittest.TestCase):
    def assert_round_trip(self, board):
        bits = board_to_bitstring(board, ROOKS)
        self.assertEqual(len(bits), 35)
        restored = bitstring_to_board(bits, ROOKS)
        # The format intentionally omits halfmove/fullmove clocks.
        self.assertEqual(restored.board_fen(), board.board_fen())
        self.assertEqual(restored.turn, board.turn)
        self.assertEqual(restored.castling_rights, board.castling_rights)
        self.assertEqual(restored.ep_square, board.ep_square)
        self.assertEqual(set(restored.legal_moves), set(board.legal_moves))
        self.assertEqual(board_to_bitstring(restored, ROOKS), bits)
        return bits

    def test_layout_and_both_rook_capture_directions(self):
        board = chess.Board("r3k3/8/8/8/8/8/8/R3K3 w - - 0 1")
        bits = self.assert_round_trip(board)
        self.assertEqual(bits, "000100" "0000000" "111100" "0111000" "000000000")
        restored = bitstring_to_board(bits, ROOKS)
        for current in (board, restored):
            current.push_uci("a1a8")
        after = self.assert_round_trip(board)
        self.assertEqual(after[19:26], "1000000")
        self.assertEqual(board_to_bitstring(restored, ROOKS), after)
        board = chess.Board("r3k3/8/8/8/8/8/8/R3K3 b - - 0 1")
        board.push_uci("a8a1")
        self.assertEqual(self.assert_round_trip(board)[6:13], "1000000")

    def test_black_only_and_bare_kings(self):
        board = chess.Board("4k2r/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertEqual(self.assert_round_trip(board)[6:13], "1000000")
        board.remove_piece_at(chess.H8)
        bits = self.assert_round_trip(board)
        self.assertEqual(bits[6:13], "1000000")
        self.assertEqual(bits[19:26], "1000000")

    def test_all_four_castling_variants_and_move_pipeline(self):
        for color in (chess.WHITE, chess.BLACK):
            for kingside in (True, False):
                board = chess.Board(None)
                board.set_piece_at(chess.E1, chess.Piece(chess.KING, chess.WHITE))
                board.set_piece_at(chess.E8, chess.Piece(chess.KING, chess.BLACK))
                rank = 0 if color else 7
                rook_square = chess.square(7 if kingside else 0, rank)
                board.set_piece_at(rook_square, chess.Piece(chess.ROOK, color))
                board.turn = color
                board.castling_rights = chess.BB_SQUARES[rook_square]
                restored = bitstring_to_board(self.assert_round_trip(board), ROOKS)
                move = chess.Move(chess.square(4, rank), chess.square(6 if kingside else 2, rank))
                encoded = moves_to_bitstrings(board.fen(), [move.uci()], ROOKS)
                self.assertIn(move.uci(), get_legal_moves(board.fen(), ROOKS))
                board.push(move)
                restored.push(move)
                self.assertEqual(self.assert_round_trip(board), encoded[move.uci()])
                self.assertEqual(board_to_bitstring(restored, ROOKS), encoded[move.uci()])

    def test_rejects_extra_material_and_corrupt_strings(self):
        for fen in (chess.STARTING_FEN, "4k3/8/8/8/8/8/8/R3K2R w - - 0 1"):
            with self.assertRaisesRegex(ValueError, "rook-endgame allows only"):
                board_to_bitstring(chess.Board(fen), ROOKS)
        bits = board_to_bitstring(chess.Board(FEN), ROOKS)
        for corrupt in (bits[:-1], bits + "0", "x" + bits[1:], bits[:6] + "1000001" + bits[13:]):
            with self.assertRaises(ValueError):
                bitstring_to_board(corrupt, ROOKS)


class DatasetTests(unittest.TestCase):
    def test_sampling_is_reproducible_unique_and_color_balanced(self):
        first = [(a.fen(), b.fen()) for a, b in sample_pairs(40, 8, 8, 42)]
        self.assertEqual(first, [(a.fen(), b.fen()) for a, b in sample_pairs(40, 8, 8, 42)])
        all_fens = [fen for pair in first for fen in pair]
        self.assertEqual(len(set(all_fens)), 56)
        for left, right in first:
            board = chess.Board(left)
            self.assertTrue(board.is_valid())
            self.assertEqual(board.mirror().fen(), right)
            self.assertFalse(board.castling_rights)
            self.assertIsNone(board.ep_square)
        with self.assertRaises(ValueError):
            list(sample_pairs(1, 0, 0, 42))

    def test_family_groups_every_geometry_color_and_turn_variant(self):
        board = chess.Board("3k4/8/1r6/8/5R2/8/2K5/8 w - - 0 1")
        expected = family_id(board)
        for transform in SYMMETRIES:
            variant = chess.Board(None)
            for square, piece in board.piece_map().items():
                variant.set_piece_at(transform[square], piece)
            for turn in (chess.WHITE, chess.BLACK):
                variant.turn = turn
                self.assertEqual(family_id(variant), expected)
                self.assertEqual(family_id(variant.mirror()), expected)
                self.assertEqual(split_for_family(family_id(variant), 42), split_for_family(expected, 42))
        changed = board.copy()
        changed.remove_piece_at(chess.F4)
        self.assertNotEqual(family_id(changed), expected)

    def test_continuous_targets_and_white_perspective(self):
        self.assertAlmostEqual(normalize_score(200), 0.4621171573)
        self.assertAlmostEqual(normalize_score(-200), -0.4621171573)
        self.assertEqual(normalize_score(0), 0)
        info = {"score": chess.engine.PovScore(chess.engine.Cp(200), chess.BLACK)}
        self.assertAlmostEqual(evaluation_label(info, 400)["target_score"], -0.4621171573)
        for relative, turn, expected in ((4, chess.WHITE, 1), (4, chess.BLACK, -1), (-4, chess.BLACK, 1)):
            info = {"score": chess.engine.PovScore(chess.engine.Mate(relative), turn)}
            self.assertEqual(evaluation_label(info, 400)["target_score"], expected)
        for invalid in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normalize_score(1, invalid)

    def test_bin_boundaries_and_nonfinite_rejection(self):
        for score, name in ((-1, 0), (-0.8, 1), (-0.2, 2), (0.2, 2), (0.8, 3), (1, 4)):
            self.assertEqual(score_bin(score), BIN_NAMES[name])
        for score in (-1.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                score_bin(score)

    def test_terminal_scores_and_mirror_provenance(self):
        for fen, score, reason in ((MATE, 1, "checkmate"), (STALEMATE, 0, "stalemate"),
                                    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", 0, "insufficient_material")):
            board = chess.Board(fen)
            self.assertTrue(board.is_valid())
            label = terminal_label(board)
            self.assertEqual(label["target_score"], score)
            self.assertEqual(label["termination"], reason)
            records = [make_record(board, label, fen), make_record(board.mirror(), label, fen, True)]
            self.assertEqual(validate_records(records, 400)["positions_checked"], 2)
            if score:
                self.assertEqual(records[1]["raw_engine_evaluation"]["winner"], "black")

    def test_bound_scores_are_ignored_and_terminal_boards_skip_engine(self):
        engine = FakeEngine()
        engine.infos = [
            {"score": chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE), "depth": 5},
            {"score": chess.engine.PovScore(chess.engine.Cp(900), chess.WHITE), "lowerbound": True},
        ]
        label = analyze_board(engine, chess.Board(FEN), 10000, 400)
        self.assertEqual(label["raw_engine_evaluation"]["value"], 200)
        self.assertEqual(engine.calls, 1)
        analyze_board(engine, chess.Board(MATE), 10000, 400)
        self.assertEqual(engine.calls, 1)

    def test_corrupt_score_and_duplicate_inputs_are_rejected(self):
        board = chess.Board(FEN)
        label = evaluation_label({"score": chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE)}, 400)
        records = [make_record(board, label, FEN), make_record(board.mirror(), label, FEN, True)]
        validate_records(records, 400)
        records[0]["target_score"] = 0.75
        with self.assertRaisesRegex(ValueError, "normalization mismatch"):
            validate_records(records, 400)
        with self.assertRaisesRegex(ValueError, "Duplicate FEN"):
            validate_records([records[0], records[0]], 400)

    def test_supplied_engine_checksum_must_match_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "engine"
            binary.write_bytes(b"not executed during verification")
            with self.assertRaisesRegex(ValueError, "requires"):
                prepare_engine(root / "cache", binary)
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare_engine(root / "cache", binary, "0" * 64)
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            _, provenance = prepare_engine(root / "cache", binary, digest)
            self.assertEqual(provenance["binary_sha256"], digest)

    def test_cache_persistence_and_configuration_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            with closing(AnalysisCache(path, {"nodes": 10000})) as cache:
                cache.put(FEN, {"target_score": 0.5}, 0.1)
            with closing(AnalysisCache(path, {"nodes": 10000})) as cache:
                self.assertEqual(cache.get(FEN), ({"target_score": 0.5}, 0.1))
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                AnalysisCache(path, {"nodes": 20000})

    def test_full_pipeline_resume_weights_and_deterministic_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(output_dir=root / "data", cache_dir=root / "cache",
                                   engine=None, engine_sha256=None, nodes=10000, score_scale=400,
                                   seed=42, two_rook_count=80, single_rook_count=16,
                                   bare_king_count=16, pilot_positions=20)
            engine = FakeEngine()
            with patch("training.generate_rook_endgame.prepare_engine", return_value=(root / "fake", {"binary_sha256": "verified"})), \
                    patch("training.generate_rook_endgame.chess.engine.SimpleEngine.popen_uci", return_value=engine):
                first = generate(args)
                hashes = {key: value["sha256"] for key, value in first["files"].items()}
                calls = engine.calls
                second = generate(args)
            self.assertEqual(engine.calls, calls)
            self.assertEqual(second["cache_hits"], 56)
            self.assertEqual(second["new_cached_pairs"], 0)
            self.assertEqual(hashes, {key: value["sha256"] for key, value in second["files"].items()})
            self.assertEqual(second["distribution"]["positions"], 112)
            self.assertEqual(second["distribution"]["material_counts"], {"KRvKR": 80, "KRvK": 8, "KvKR": 8, "KvK": 16})
            groups = {}
            for split in ("train", "validation", "test"):
                with gzip.open(root / "data" / f"{split}.jsonl.gz", "rt") as stream:
                    rows = [json.loads(line) for line in stream]
                groups[split] = {row["family_id"] for row in rows}
                for row in rows:
                    self.assertEqual("sampling_weight" in row, split == "train")
                if split == "train":
                    self.assertAlmostEqual(sum(row["sampling_weight"] for row in rows), len(rows))
            self.assertFalse(groups["train"] & groups["test"])
            self.assertFalse(groups["train"] & groups["validation"])
            self.assertFalse(groups["validation"] & groups["test"])


class FakeEngine:
    id = {"name": "Stockfish 19"}
    options = {}

    def __init__(self):
        self.calls = 0
        self.infos = [{"score": chess.engine.PovScore(chess.engine.Cp(200), chess.WHITE), "depth": 8, "nodes": 10000}]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def configure(self, options):
        pass

    @contextmanager
    def analysis(self, *args, **kwargs):
        self.calls += 1
        yield iter(self.infos)


if __name__ == "__main__":
    unittest.main()
