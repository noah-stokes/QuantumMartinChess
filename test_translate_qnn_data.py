"""Score semantics and safe, reproducible conversion of the QNN exports."""

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import chess

from board_bitstrings import board_to_bitstring
from position_types import PositionType
from translate_qnn_data import SPLITS, translate, translate_score


def score_record(kind, value, termination=None, **extra):
    return {"termination": termination,
            "raw_engine_evaluation": {"kind": kind, "value": value, **extra}}


class MateBandTests(unittest.TestCase):
    def test_ordinary_scores_and_old_target_is_ignored(self):
        for cp, expected in ((0, 0), (200, 0.3465878679450073), (-200, -0.3465878679450073)):
            record = score_record("cp", cp)
            record["target_score"] = -1  # A stale target must never determine the new label.
            actual, kind = translate_score(record)
            self.assertAlmostEqual(actual, expected)
            self.assertEqual(kind, "centipawn")
        for cp in (-100000, -800, 800, 100000):
            self.assertLessEqual(abs(translate_score(score_record("cp", cp))[0]), 0.75)

    def test_mate_examples_monotonicity_and_color_symmetry(self):
        for distance, expected in ((1, 0.975), (5, 0.8976225), (10, 0.837169610025)):
            positive, kind = translate_score(score_record("mate", distance, unit="moves"))
            negative, _ = translate_score(score_record("mate", -distance, unit="moves"))
            self.assertAlmostEqual(positive, expected)
            self.assertEqual(negative, -positive)
            self.assertEqual(kind, "nonterminal_mate")
        scores = [translate_score(score_record("mate", distance, unit="moves"))[0] for distance in range(1, 101)]
        self.assertTrue(all(0.75 < score < 1 for score in scores))
        self.assertTrue(all(earlier > later for earlier, later in zip(scores, scores[1:])))
        self.assertGreater(translate_score(score_record("mate", 10000, unit="moves"))[0], 0.75)

    def test_checkmate_zero_distance_uses_winner(self):
        for winner, expected in (("white", 1), ("black", -1)):
            record = score_record("mate", 0, "checkmate", winner=winner)
            record["target_score"] = -expected
            self.assertEqual(translate_score(record), (expected, "checkmate"))

    def test_terminal_draws_remain_zero(self):
        for termination in ("stalemate", "insufficient_material"):
            self.assertEqual(translate_score({"termination": termination, "raw_engine_evaluation": None}),
                             (0.0, "terminal_draw"))

    def test_ambiguous_or_unsupported_raw_evaluations_are_rejected(self):
        invalid = [
            score_record("mate", 0, unit="moves"),
            score_record("mate", 2, unit="plies"),
            score_record("mate", 0, "checkmate"),
            score_record("mate", 0, "checkmate", winner="unknown"),
            score_record("mate", 1, "checkmate", winner="white"),
            score_record("cp", True), score_record("cp", 1.5), score_record("cp", float("nan")),
            score_record("cp", "200"), score_record("unknown", 200),
            score_record("cp", 0, "stalemate"), score_record("cp", 0, "unknown"),
            {"termination": None, "raw_engine_evaluation": None},
        ]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                translate_score(record)


def rich_pair(fen, raw, termination=None):
    board = chess.Board(fen)
    if not board.is_valid():
        raise AssertionError(f"Invalid test board: {fen}")
    mirror = board.mirror()
    records = []
    for current, mirrored in ((board, False), (mirror, True)):
        evaluation = copy.deepcopy(raw)
        if mirrored and evaluation is not None:
            evaluation["value"] = -evaluation["value"]
            if "winner" in evaluation:
                evaluation["winner"] = "black" if evaluation["winner"] == "white" else "white"
        white = bool(current.pieces(chess.ROOK, chess.WHITE))
        black = bool(current.pieces(chess.ROOK, chess.BLACK))
        material = {(True, True): "KRvKR", (True, False): "KRvK", (False, True): "KvKR", (False, False): "KvK"}[white, black]
        records.append({"rook_bitstring": board_to_bitstring(current, PositionType.ROOK_ENDGAME),
                        "target_score": 0.0, "raw_engine_evaluation": evaluation,
                        "termination": termination, "material_class": material,
                        "family_id": "fixture:" + fen, "fen": current.fen(), "source_fen": fen,
                        "augmentation": "color_swap_vertical" if mirrored else "identity"})
    return records


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.source / "qnn"
        self.source.mkdir()
        self.output.mkdir()
        self.rows = {
            "train": [
                *rich_pair("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", {"kind": "cp", "value": 200}),
                *rich_pair("4k2r/8/8/8/8/8/8/R3K3 w - - 0 1", {"kind": "cp", "value": -400}),
                *rich_pair("4k3/8/8/8/8/8/8/4K3 w - - 0 1", None, "insufficient_material"),
                *rich_pair("7k/8/5K2/8/8/8/8/R7 w - - 0 1", {"kind": "mate", "value": 3, "unit": "moves"}),
                *rich_pair("7k/5K2/8/8/8/8/8/7R b - - 0 1", {"kind": "mate", "value": 0, "winner": "white"}, "checkmate"),
            ],
            "validation": [
                *rich_pair("8/4k3/8/8/8/8/8/1R3K2 w - - 0 1", {"kind": "cp", "value": 100}),
                *rich_pair("7k/5K2/8/8/8/8/8/R7 w - - 0 1", {"kind": "mate", "value": 1, "unit": "moves"}),
            ],
            "test": [
                *rich_pair("7k/6r1/8/8/8/8/8/R3K3 w - - 0 1", {"kind": "cp", "value": 0}),
                *rich_pair("6k1/8/4K3/8/8/8/8/1R6 w - - 0 1", {"kind": "mate", "value": 5, "unit": "moves"}),
                *rich_pair("7k/6R1/5K2/8/8/8/8/8 b - - 0 1", None, "stalemate"),
            ],
        }
        self.write_sources()
        for split in SPLITS:
            previous = [{"input": row["rook_bitstring"], "output": row["target_score"]} for row in self.rows[split]]
            (self.output / f"{split}.json").write_text(json.dumps(previous))

    def write_sources(self):
        manifest = {"files": {}}
        for split in SPLITS:
            path = self.source / f"{split}.jsonl.gz"
            with gzip.open(path, "wt") as stream:
                for row in self.rows[split]:
                    stream.write(json.dumps(row) + "\n")
            manifest["files"][split] = {"positions": len(self.rows[split]),
                                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (self.source / "manifest.json").write_text(json.dumps(manifest))

    def old_exports(self):
        return {path: path.read_bytes() for path in self.output.iterdir() if path.is_file()}

    def assert_unchanged(self, files):
        self.assertEqual(files, {path: path.read_bytes() for path in files})

    def test_full_conversion_provenance_baselines_and_rerun(self):
        protected = {path: path.read_bytes() for path in self.source.iterdir() if path.is_file()}
        manifest = translate(self.source, self.output)
        self.assert_unchanged(protected)
        self.assertEqual(manifest["distribution"]["positions"], 20)
        self.assertEqual(manifest["distribution"]["exact_endpoint_count"], 2)
        self.assertEqual(manifest["distribution"]["categories"],
                         {"centipawn": 8, "nonterminal_mate": 6, "checkmate": 2, "terminal_draw": 4})
        self.assertEqual(manifest["validation"]["mirror_pairs"], 10)
        for split in SPLITS:
            path = self.output / f"{split}.json"
            converted = json.loads(path.read_text())
            self.assertEqual(manifest["files"][split]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(len(converted), len(self.rows[split]))
            for original, row in zip(self.rows[split], converted):
                self.assertEqual(set(row), {"input", "output"})
                self.assertEqual(row["input"], original["rook_bitstring"])
            if split != "train":
                expected_zero = math.fsum(row["output"] ** 2 for row in converted) / len(converted)
                self.assertEqual(manifest["baselines"][split]["zero_prediction_mse"], expected_zero)
                means = manifest["baselines"]["training_material_means"]
                expected_material = math.fsum((row["output"] - means[source["material_class"]]) ** 2
                                              for source, row in zip(self.rows[split], converted)) / len(converted)
                self.assertEqual(manifest["baselines"][split]["material_mean_mse"], expected_material)
        first_files = self.old_exports()
        self.assertEqual(translate(self.source, self.output), manifest)
        self.assert_unchanged(first_files)

    def test_bad_final_split_aborts_before_any_export_is_replaced(self):
        before = self.old_exports()
        self.rows["test"][2]["raw_engine_evaluation"]["value"] = 0
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "nonzero signed distance"):
            translate(self.source, self.output)
        self.assert_unchanged(before)
        self.assertFalse((self.output / "manifest.json").exists())

    def test_source_checksum_mismatch_preserves_exports(self):
        before = self.old_exports()
        with (self.source / "test.jsonl.gz").open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            translate(self.source, self.output)
        self.assert_unchanged(before)

    def test_existing_input_order_must_match_source(self):
        path = self.output / "test.json"
        path.write_text(json.dumps(list(reversed(json.loads(path.read_text())))))
        before = self.old_exports()
        with self.assertRaisesRegex(ValueError, "input order does not match"):
            translate(self.source, self.output)
        self.assert_unchanged(before)

    def test_broken_mirror_and_cross_split_family_are_rejected(self):
        before = self.old_exports()
        self.rows["test"][3]["raw_engine_evaluation"]["value"] = 5
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "Color mirrors"):
            translate(self.source, self.output)
        self.assert_unchanged(before)
        self.rows["test"][3]["raw_engine_evaluation"]["value"] = -5
        self.rows["test"][0]["family_id"] = self.rows["train"][0]["family_id"]
        self.write_sources()
        with self.assertRaisesRegex(ValueError, "more than one split"):
            translate(self.source, self.output)
        self.assert_unchanged(before)

    def test_output_directory_cannot_replace_original_manifest(self):
        protected = {path: path.read_bytes() for path in self.source.iterdir() if path.is_file()}
        with self.assertRaisesRegex(ValueError, "must differ"):
            translate(self.source, self.source)
        self.assert_unchanged(protected)


if __name__ == "__main__":
    unittest.main()
