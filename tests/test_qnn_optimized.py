"""Optimized statevector equivalence, exact gradients, and checkpoint routing."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from qnns.rook_endgame.cli import main, parser, train
from qnns.rook_endgame.model import (ModelConfig, PositionQNN, bitstrings_to_tensor,
                       load_checkpoint, save_checkpoint)
from tests.test_qnn import pair


class OptimizedBackendTests(unittest.TestCase):
    def rows(self):
        return pair() + pair("1k5r/8/8/8/8/8/8/2K4R b - - 0 1", -.3)

    def make_model(self):
        return PositionQNN(ModelConfig(gradient="autograd", backend="torch"))

    def test_qiskit_equivalence_on_varied_parameters_and_batches(self):
        reference = PositionQNN()
        model = self.make_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 190)
        self.assertEqual(model.input_scales.dtype, torch.float64)
        self.assertEqual(set(model.state_dict()), set(reference.state_dict()))
        bits = bitstrings_to_tensor([r["input"] for r in self.rows()])
        rng = np.random.default_rng(123)
        for scale in (0, .2, 1.5):
            with torch.no_grad():
                if scale:
                    reference.input_scales.add_(torch.tensor(rng.normal(0, scale, 35), dtype=torch.float32))
                    reference.quantum.weight.add_(torch.tensor(rng.normal(0, scale, 147), dtype=torch.float32))
                model.load_state_dict(reference.state_dict())
                expected = reference(bits)
                actual = model(bits)
            torch.testing.assert_close(actual.float(), expected, atol=1e-6, rtol=1e-6)
            for i in range(len(bits)):
                torch.testing.assert_close(model(bits[i:i+1]), actual[i:i+1], atol=1e-12, rtol=0)

    def test_all_gradients_match_qiskit_parameter_shift(self):
        reference = PositionQNN(ModelConfig(gradient="parameter-shift"))
        model = self.make_model()
        board = "00010100001111111110000000010000000"
        bits = bitstrings_to_tensor([board])
        for m in (reference, model):
            target = torch.tensor([.6], dtype=m.input_scales.dtype)
            torch.nn.functional.mse_loss(m(bits), target).backward()
        for name, parameter in model.named_parameters():
            expected = dict(reference.named_parameters())[name].grad
            torch.testing.assert_close(parameter.grad.float(), expected, atol=1e-6, rtol=1e-4)
        zero_inputs = bits[0] == 0
        torch.testing.assert_close(model.input_scales.grad[zero_inputs],
                                   torch.zeros_like(model.input_scales.grad[zero_inputs]))

    def test_learning_and_checkpoint_roundtrip(self):
        model = self.make_model()
        rows = self.rows()
        bits = bitstrings_to_tensor([r["input"] for r in rows])
        targets = torch.tensor([r["output"] for r in rows], dtype=torch.float64)
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        initial = torch.nn.functional.mse_loss(model(bits), targets).item()
        best = initial
        for _ in range(25):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(bits), targets)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            best = min(best, torch.nn.functional.mse_loss(model(bits), targets).item())
        self.assertLess(best, initial * .9)
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "optimized.pt"
            save_checkpoint(checkpoint, model, {"optimized": True})
            loaded, metadata = load_checkpoint(checkpoint)
            self.assertEqual(loaded.config.backend, "torch")
            self.assertEqual(loaded.config.gradient, "autograd")
            self.assertEqual(metadata, {"optimized": True})
            torch.testing.assert_close(model(bits), loaded(bits), atol=0, rtol=0)

    def test_backend_gradient_validation(self):
        with self.assertRaisesRegex(ValueError, "requires autograd"):
            ModelConfig(backend="torch")
        with self.assertRaisesRegex(ValueError, "qiskit backend"):
            ModelConfig(gradient="autograd")
        args = parser().parse_args(["train", "--backend", "torch"])
        self.assertEqual(args.backend, "torch")
        self.assertIsNone(args.gradient)


if __name__ == "__main__":
    unittest.main()
