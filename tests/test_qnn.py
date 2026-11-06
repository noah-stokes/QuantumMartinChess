"""Circuit semantics, differentiability, data provenance, and training integration."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import chess
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
import torch

from tools.board_bitstrings import board_to_bitstring
from tools.position_types import PositionType
from qnns.rook_endgame.cli import evaluate, main, parser, predict_records, train
from qnns.rook_endgame.data import (evaluation_metrics, load_split, material_class, read_records,
                      select_indices, sha256_file, training_material_means)
from qnns.rook_endgame.model import (ModelConfig, PositionQNN, bitstrings_to_tensor, build_circuit,
                       load_checkpoint, save_checkpoint, validate_bitstring, z_observables)
from training.translate_qnn_data import MAPPING_VERSION, score_band


def pair(fen="4k3/8/8/8/8/8/8/5K1R w - - 0 1", score=.6):
    board = chess.Board(fen)
    return [{"input": board_to_bitstring(b, PositionType.ROOK_ENDGAME), "output": y}
            for b, y in ((board, score), (board.mirror(), -score))]


class ModelTests(unittest.TestCase):
    def test_layout_parameters_and_final_observables(self):
        circuit, inputs, weights = build_circuit()
        self.assertEqual(len(inputs), 35)
        self.assertEqual(len(weights), 147)
        occurrences = {p: [] for p in inputs}
        for instruction in circuit.data:
            if instruction.operation.name == "ry" and instruction.operation.params[0] in occurrences:
                occurrences[instruction.operation.params[0]].append(
                    circuit.find_bit(instruction.qubits[0]).index)
        for j, parameter in enumerate(inputs):
            self.assertEqual(occurrences[parameter], [j % 7, j % 7])
        self.assertEqual(occurrences[inputs[26]], [5, 5])
        self.assertNotIn("measure", circuit.count_ops())
        model = PositionQNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 190)
        self.assertTrue(model.quantum.neural_network.input_gradients)
        self.assertEqual(model.quantum.neural_network.default_precision, 0)
        self.assertTrue(torch.all(model.input_scales == torch.tensor(np.pi, dtype=torch.float32)))
        simple = QuantumCircuit(7)
        simple.x(0)
        result = StatevectorEstimator().run([(simple, z_observables())]).result()[0].data.evs
        np.testing.assert_allclose(result, [-1, 1, 1, 1, 1, 1, 1])

    def test_shapes_input_validation_and_state_isolation(self):
        model = PositionQNN()
        rows = pair()
        bits = bitstrings_to_tensor([r["input"] for r in rows])
        self.assertEqual(model(bits).shape, (2,))
        self.assertEqual(model(bits[:1]).shape, (1,))
        first = model.predict(rows[0]["input"])
        model.predict(rows[1]["input"])
        self.assertEqual(first, model.predict(rows[0]["input"]))
        np.testing.assert_allclose(model(bits).detach(),
                                   [model.predict(r["input"]) for r in rows], atol=1e-6)
        self.assertTrue(bool(torch.all(model(bits).abs() <= 1)))
        for bad in ("0" * 34, "0" * 36, "x" * 35, 123):
            with self.assertRaises(ValueError):
                validate_bitstring(bad)
        for bad in (torch.zeros(35), torch.zeros(2, 34), torch.zeros(0, 35),
                    torch.full((1, 35), .5), torch.full((1, 35), float("nan"))):
            with self.assertRaises(ValueError):
                model(bad)

    def test_turn_scale_is_independent_of_king_scale(self):
        board = chess.Board("4k3/8/8/8/8/8/8/5K1R w - - 0 1")
        white = board_to_bitstring(board, PositionType.ROOK_ENDGAME)
        board.turn = chess.BLACK
        black = board_to_bitstring(board, PositionType.ROOK_ENDGAME)
        self.assertEqual(white[26], "0")
        self.assertEqual(black[26], "1")
        model = PositionQNN()
        before = [model.predict(s) for s in (white, black)]
        king_scale = model.input_scales[5].item()
        with torch.no_grad():
            model.input_scales[26] += .3
        self.assertEqual(model.input_scales[5].item(), king_scale)
        self.assertEqual(model.predict(white), before[0])
        self.assertGreater(abs(model.predict(black) - before[1]), 1e-6)

    def test_seed_reproduces_predictions_and_spsa_gradients(self):
        bits = bitstrings_to_tensor([r["input"] for r in pair()])
        models = [PositionQNN(ModelConfig(seed=42, spsa_directions=1)) for _ in range(2)]
        torch.testing.assert_close(models[0](bits), models[1](bits), atol=0, rtol=0)
        for model in models:
            model(bits).square().mean().backward()
        for a, b in zip(models[0].parameters(), models[1].parameters()):
            torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_exact_gradients_reach_input_scales_and_match_finite_differences(self):
        model = PositionQNN(ModelConfig(gradient="parameter-shift"))
        board = chess.Board("4k3/8/8/8/8/8/8/5K1R b - - 0 1")
        bits = bitstrings_to_tensor([board_to_bitstring(board, PositionType.ROOK_ENDGAME)])
        target = torch.tensor([.6])
        loss = torch.nn.functional.mse_loss(model(bits), target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(model.input_scales.grad).all()))
        self.assertGreater(abs(model.input_scales.grad[5].item()), 1e-7)
        self.assertGreater(abs(model.input_scales.grad[26].item()), 1e-7)
        torch.testing.assert_close(model.input_scales.grad[bits[0] == 0],
                                   torch.zeros_like(model.input_scales.grad[bits[0] == 0]))
        for parameter, index in ((model.input_scales, 5), (model.input_scales, 26),
                                 (model.quantum.weight, 0)):
            exact = parameter.grad[index].item()
            original = parameter[index].item()
            epsilon = .001
            with torch.no_grad():
                parameter[index] = original + epsilon
                plus = torch.nn.functional.mse_loss(model(bits), target).item()
                parameter[index] = original - epsilon
                minus = torch.nn.functional.mse_loss(model(bits), target).item()
                parameter[index] = original
            self.assertAlmostEqual(exact, (plus - minus) / (2 * epsilon), delta=.0002)

    def test_spsa_learning_and_checkpoint_roundtrip(self):
        model = PositionQNN(ModelConfig(spsa_directions=1))
        rows = pair()
        bits = bitstrings_to_tensor([r["input"] for r in rows])
        target = torch.tensor([r["output"] for r in rows])
        initial_scales = model.input_scales.detach().clone()
        initial_quantum = model.quantum.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        initial = torch.nn.functional.mse_loss(model(bits), target).item()
        best = initial
        for _ in range(20):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(bits), target)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            best = min(best, torch.nn.functional.mse_loss(model(bits), target).item())
        self.assertLess(best, initial * .9)
        self.assertGreater(torch.linalg.vector_norm(model.input_scales - initial_scales).item(), 0)
        self.assertGreater(torch.linalg.vector_norm(model.quantum.weight - initial_quantum).item(), 0)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.pt"
            save_checkpoint(path, model, {"fixture": True})
            loaded, metadata = load_checkpoint(path)
            self.assertEqual(metadata, {"fixture": True})
            torch.testing.assert_close(loaded(bits), model(bits), atol=1e-6, rtol=0)


class DataTests(unittest.TestCase):
    def test_leading_zeros_and_bad_records(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "data.json"
            rows = pair()
            path.write_text(json.dumps(rows))
            self.assertEqual(read_records(path), rows)
            self.assertTrue(rows[0]["input"].startswith("0"))
            for value in (True, float("nan"), float("inf"), 1.1, "0.5"):
                path.write_text(json.dumps([{**rows[0], "output": value}]))
                with self.assertRaises(ValueError):
                    read_records(path)

    def test_stratified_samples_and_matched_metrics(self):
        rows = []
        for a, b in (("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")):
            for target in (-.9, -.4, 0, .4, .9):
                for _ in range(4):
                    bits = list("0" * 35)
                    bits[6], bits[19] = a, b
                    rows.append({"input": "".join(bits), "output": target})
        indices = select_indices(rows, 30, 42)
        self.assertEqual(indices, select_indices(rows, 30, 42))
        self.assertNotEqual(indices, select_indices(rows, 30, 43))
        self.assertEqual(len(set(indices)), 30)
        self.assertEqual(len({(material_class(rows[i]["input"]), score_band(rows[i]["output"]))
                              for i in indices}), 20)
        self.assertEqual(select_indices(rows, 0, 42), list(range(len(rows))))
        with self.assertRaises(ValueError):
            select_indices(rows, 19, 42)
        selected = [rows[i] for i in indices]
        means = training_material_means(selected)
        metrics = evaluation_metrics(selected, [r["output"] for r in selected], means)
        self.assertEqual(metrics["overall"]["mse"], 0)
        self.assertEqual(metrics["overall"]["count"], 30)
        self.assertAlmostEqual(metrics["overall"]["baselines"]["zero"]["mse"],
                               sum(r["output"] ** 2 for r in selected) / len(selected))

    def test_train_cli_artifacts_provenance_and_predict(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            data = root / "data"
            data.mkdir()
            fixtures = {"train": pair(),
                        "validation": pair("4k3/8/8/8/8/8/7R/5K2 w - - 0 1", .4)}
            manifest = {"mapping_version": MAPPING_VERSION, "files": {}}
            for split, rows in fixtures.items():
                path = data / f"{split}.json"
                path.write_text(json.dumps(rows))
                manifest["files"][split] = {"positions": len(rows), "sha256": sha256_file(path)}
            # Deliberately provide no test file: training must not try to open it.
            (data / "manifest.json").write_text(json.dumps(manifest))
            checksums = {p: sha256_file(p) for p in data.iterdir()}
            args = parser().parse_args(["train", "--data-dir", str(data), "--output-dir", str(root / "run"),
                                       "--train-limit", "0", "--validation-limit", "0", "--epochs", "1",
                                       "--spsa-directions", "1"])
            summary = train(args)
            self.assertFalse(summary["test_evaluated"])
            self.assertEqual(summary["epochs_completed"], 1)
            self.assertEqual(checksums, {p: sha256_file(p) for p in checksums})
            run = root / "run"
            for artifact in ("best.pt", "config.json", "selected_rows.json", "metrics.jsonl",
                             "summary.json", "loss.png", "overview.svg", "overview_circuit.svg"):
                self.assertTrue((run / artifact).is_file(), artifact)
            selected = json.loads((run / "selected_rows.json").read_text())
            self.assertEqual(selected, {"train": [0, 1], "validation": [0, 1]})
            output = io.StringIO()
            with redirect_stdout(output):
                main(["predict", "--checkpoint", str(run / "best.pt"), "--input", fixtures["train"][0]["input"]])
            self.assertEqual(json.loads(output.getvalue())["perspective"], "white")
            model, _ = load_checkpoint(run / "best.pt")
            self.assertEqual(json.loads(output.getvalue())["score"], model.predict(fixtures["train"][0]["input"]))
            result = evaluate(parser().parse_args(["evaluate", "--checkpoint", str(run / "best.pt"),
                                                   "--data-dir", str(data), "--split", "validation"]))
            self.assertEqual(result["metrics"]["overall"]["count"], 2)
            protected = parser().parse_args(["evaluate", "--checkpoint", str(run / "best.pt"),
                                             "--data-dir", str(data), "--output", str(data / "validation.json")])
            with self.assertRaisesRegex(ValueError, "overwrite"):
                evaluate(protected)
            self.assertEqual(checksums, {p: sha256_file(p) for p in checksums})
            (data / "validation.json").write_text("[]")
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_split(data, "validation")


if __name__ == "__main__":
    unittest.main()
