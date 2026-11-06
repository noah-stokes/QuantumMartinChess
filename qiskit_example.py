"""Create and simulate an entangled Bell pair locally with Qiskit Aer."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def main() -> None:
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()

    simulator = AerSimulator()
    compiled = transpile(circuit, simulator, seed_transpiler=42)
    result = simulator.run(compiled, shots=1024, seed_simulator=42).result()

    print(circuit.draw(output="text"))
    print("Measurement counts:", result.get_counts())


if __name__ == "__main__":
    main()
