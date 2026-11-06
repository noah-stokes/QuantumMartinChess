"""Seven-qubit rook-endgame evaluator, with independently learned input scales."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient, SPSAEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
import torch
from torch import nn


INPUTS = 35
QUBITS = 7
GROUPS = 5
PASSES = 2
QUANTUM_WEIGHTS = PASSES * GROUPS * QUBITS * 2 + QUBITS
CHECKPOINT_VERSION = 1


def validate_bitstring(bits: str) -> str:
    if not isinstance(bits, str) or len(bits) != INPUTS or set(bits) - {"0", "1"}:
        raise ValueError("Input must be exactly 35 binary characters, including leading zeros")
    return bits


def bitstrings_to_tensor(bitstrings: list[str]) -> torch.Tensor:
    if not bitstrings:
        raise ValueError("At least one input is required")
    return torch.tensor([[int(b) for b in validate_bitstring(s)] for s in bitstrings],
                        dtype=torch.float32)


def build_circuit():
    """Return the circuit and explicitly ordered input/weight parameters.

    Input angles already include the learned scale. Reusing an input parameter
    on the second pass ties only that bit's input scale, not the mixing angles.
    Readout observables are supplied to the estimator, not measured mid-circuit.
    """
    inputs = ParameterVector("x", INPUTS)
    weights = ParameterVector("theta", QUANTUM_WEIGHTS)
    circuit = QuantumCircuit(QUBITS, name="rook_evaluator")
    k = 0
    for pass_index in range(PASSES):
        for group in range(GROUPS):
            start = group * QUBITS
            circuit.barrier(label=f"P{pass_index + 1} G{group + 1}: {start}-{start + 6}")
            for wire in range(QUBITS):
                circuit.ry(inputs[start + wire], wire)
            for wire in range(QUBITS):
                circuit.ry(weights[k], wire)
                circuit.rz(weights[k + 1], wire)
                k += 2
            for wire in range(QUBITS):
                circuit.cz(wire, (wire + 1) % QUBITS)
    circuit.barrier(label="Final readout")
    for wire in range(QUBITS):
        circuit.ry(weights[k], wire)
        k += 1
    return circuit, list(inputs), list(weights)


def z_observables():
    # Sparse wire indices avoid reversing Qiskit's little-endian Pauli labels.
    return [SparsePauliOp.from_sparse_list([("Z", [wire], 1.0)], num_qubits=QUBITS)
            for wire in range(QUBITS)]


@dataclass(frozen=True)
class ModelConfig:
    seed: int = 42
    gradient: str = "spsa"
    spsa_epsilon: float = 0.01
    spsa_directions: int = 4
    backend: str = "qiskit"

    def __post_init__(self):
        if self.backend not in ("qiskit", "torch"):
            raise ValueError("Backend must be qiskit or torch")
        if self.backend == "torch" and self.gradient != "autograd":
            raise ValueError("The torch backend requires autograd gradients")
        if self.backend == "qiskit" and self.gradient not in ("spsa", "parameter-shift"):
            raise ValueError("The qiskit backend requires spsa or parameter-shift gradients")
        if not math.isfinite(self.spsa_epsilon) or self.spsa_epsilon <= 0:
            raise ValueError("SPSA epsilon must be positive and finite")
        if self.spsa_directions < 1:
            raise ValueError("SPSA directions must be positive")


class TorchStatevector(nn.Module):
    """Execute the same circuit with batched, differentiable complex amplitudes.

    Gates on different wires commute, allowing each input RY and following
    learned RY to be combined. The seven diagonal RZs and seven diagonal CZs
    are likewise applied as phase vectors. Wire q uses bit q of the basis
    index, matching Qiskit's little-endian convention.
    """

    def __init__(self, initial_weights):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        # Match TorchConnector's checkpoint alias for shared circuit weights.
        self._weights = self.weight
        bits = (torch.arange(2 ** QUBITS)[:, None] >> torch.arange(QUBITS)) & 1
        self.register_buffer("z_signs", (1 - 2 * bits).double(), persistent=False)
        parity = (bits * torch.roll(bits, -1, dims=1)).sum(dim=1) % 2
        self.register_buffer("cz_phases", (1 - 2 * parity).double(), persistent=False)

    @staticmethod
    def ry(state, angle, wire):
        batch = state.shape[0]
        zero, one = state.reshape(batch, 2 ** (QUBITS - wire - 1), 2, 2 ** wire).unbind(2)
        cosine = torch.cos(angle / 2).reshape(-1, 1, 1)
        sine = torch.sin(angle / 2).reshape(-1, 1, 1)
        return torch.stack((cosine * zero - sine * one, sine * zero + cosine * one),
                           dim=2).reshape(batch, 2 ** QUBITS)

    def forward(self, angles):
        state = torch.zeros(angles.shape[0], 2 ** QUBITS, dtype=torch.complex128,
                            device=angles.device)
        state[:, 0] = 1
        mixing = self.weight[:-QUBITS].reshape(PASSES, GROUPS, QUBITS, 2)
        for pass_index in range(PASSES):
            for group in range(GROUPS):
                for wire in range(QUBITS):
                    angle = angles[:, group * QUBITS + wire] + mixing[pass_index, group, wire, 0]
                    state = self.ry(state, angle, wire)
                rz_phases = torch.exp(-0.5j * (self.z_signs @ mixing[pass_index, group, :, 1]))
                state = state * rz_phases * self.cz_phases
        for wire in range(QUBITS):
            state = self.ry(state, self.weight[-QUBITS + wire], wire)
        probabilities = state.real.square() + state.imag.square()
        return probabilities @ self.z_signs


class PositionQNN(nn.Module):
    """Map CPU tensors of shape [batch, 35] to White scores of shape [batch]."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        initial_weights = np.random.default_rng(self.config.seed).normal(
            0, 0.1, QUANTUM_WEIGHTS)
        self.input_scales = nn.Parameter(torch.full((INPUTS,), math.pi))
        # Keep construction reproducible without changing the caller's RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            if self.config.backend == "torch":
                self.quantum = TorchStatevector(initial_weights)
            else:
                circuit, inputs, weights = build_circuit()
                estimator = StatevectorEstimator(default_precision=0.0)
                if self.config.gradient == "spsa":
                    gradient = SPSAEstimatorGradient(
                        estimator, epsilon=self.config.spsa_epsilon,
                        batch_size=self.config.spsa_directions, seed=self.config.seed)
                else:
                    gradient = ParamShiftEstimatorGradient(estimator)
                network = EstimatorQNN(
                    circuit=circuit, estimator=estimator, observables=z_observables(),
                    input_params=inputs, weight_params=weights, input_gradients=True,
                    gradient=gradient, default_precision=0.0)
                self.quantum = TorchConnector(network, initial_weights=initial_weights)
            self.readout = nn.Linear(QUBITS, 1)
            nn.init.normal_(self.readout.weight, mean=0, std=0.1)
            nn.init.zeros_(self.readout.bias)
        if self.config.backend == "torch":
            # Double precision preserves state normalization through long gate
            # sequences; casting after initialization matches the Qiskit seed.
            self.double()

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        if not isinstance(bits, torch.Tensor) or bits.ndim != 2 or bits.shape[1] != INPUTS:
            raise ValueError("Expected a tensor of shape [batch, 35]")
        if bits.shape[0] < 1:
            raise ValueError("Batch must contain at least one position")
        if bits.device.type != "cpu":
            raise ValueError("This prototype uses CPU tensors and local Qiskit simulation")
        if not bool(torch.all((bits == 0) | (bits == 1))):
            raise ValueError("Input tensor must contain only binary values")
        angles = bits.to(dtype=self.input_scales.dtype) * self.input_scales
        features = self.quantum(angles).reshape(bits.shape[0], QUBITS)
        return torch.tanh(self.readout(features)).squeeze(-1)

    @torch.no_grad()
    def predict(self, bitstring: str) -> float:
        return float(self(bitstrings_to_tensor([bitstring]))[0])


def save_checkpoint(path: Path, model: PositionQNN, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"checkpoint_version": CHECKPOINT_VERSION,
                "model_config": asdict(model.config),
                "model_state": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_checkpoint(path: Path) -> tuple[PositionQNN, dict]:
    saved = torch.load(Path(path), map_location="cpu", weights_only=True)
    if saved.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported QNN checkpoint version")
    model = PositionQNN(ModelConfig(**saved["model_config"]))
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model, saved["metadata"]
