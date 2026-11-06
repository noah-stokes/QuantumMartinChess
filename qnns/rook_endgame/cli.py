"""Train, evaluate, draw, and use the local 35-bit chess QNN."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from qnns.rook_endgame.data import (evaluation_metrics, load_split, select_indices,
                      training_material_means)
from qnns.rook_endgame.diagrams import save_diagrams, save_loss_plot
from qnns.rook_endgame.model import (ModelConfig, PositionQNN, bitstrings_to_tensor,
                       load_checkpoint, save_checkpoint)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/king_rook/qnn"


def write_json(path: Path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def progress(message):
    print(message, file=sys.stderr, flush=True)


@torch.no_grad()
def predict_records(model: PositionQNN, records: list[dict], batch_size: int = 8) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    model.eval()
    results = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        results.extend(model(bitstrings_to_tensor([r["input"] for r in batch])).tolist())
    return np.asarray(results)


def train(args) -> dict:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("Epochs, batch size, and patience must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite")
    gradient = args.gradient or ("autograd" if args.backend == "torch" else "spsa")
    config = ModelConfig(args.seed, gradient, args.spsa_epsilon, args.spsa_directions, args.backend)
    original_train, train_source = load_split(args.data_dir, "train")
    original_validation, validation_source = load_split(args.data_dir, "validation")
    train_indices = select_indices(original_train, args.train_limit, args.seed)
    validation_indices = select_indices(original_validation, args.validation_limit, args.seed + 1)
    training = [original_train[i] for i in train_indices]
    validation = [original_validation[i] for i in validation_indices]
    means = training_material_means(training)
    run_dir = args.output_dir or ROOT / "runs/qnn" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new run directory: {run_dir}")
    # Derived artifacts must never overwrite either source or converted data.
    if run_dir.is_relative_to(args.data_dir.resolve()) or args.data_dir.resolve().is_relative_to(run_dir):
        raise ValueError("Output directory must be separate from the dataset")
    run_dir.mkdir(parents=True)
    selected = {"train": train_indices, "validation": validation_indices}
    write_json(run_dir / "selected_rows.json", selected)
    training_config = {
        "model": asdict(config), "architecture": {"inputs": 35, "qubits": 7, "passes": 2,
                                                   "parameters": 190},
        "optimizer": "Adam", "learning_rate": args.learning_rate,
        "dtype": "float64" if config.backend == "torch" else "float32",
        "batch_size": args.batch_size, "epochs": args.epochs, "patience": args.patience,
        "min_delta": 0.00001, "gradient_clip_norm": 1.0,
        "train_positions": len(training), "validation_positions": len(validation),
        "train_limit": args.train_limit, "validation_limit": args.validation_limit,
        "data_dir": str(args.data_dir.resolve()),
        "dataset": {"train": train_source, "validation": validation_source},
        "versions": {p: version(p) for p in ("qiskit", "qiskit-machine-learning", "torch", "numpy")},
        "python": platform.python_version(),
    }
    write_json(run_dir / "config.json", training_config)
    progress(f"Run: {run_dir}\nTraining {len(training)} positions; validation {len(validation)}; "
             f"backend={config.backend}; gradient={config.gradient}. Test data is not loaded.")
    model = PositionQNN(config)
    initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
    tensors = bitstrings_to_tensor([r["input"] for r in training])
    targets = torch.tensor([r["output"] for r in training], dtype=model.input_scales.dtype)
    loader = DataLoader(TensorDataset(tensors, targets), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    started = time.perf_counter()
    history = []

    def measure(epoch):
        train_metrics = evaluation_metrics(training, predict_records(model, training, args.batch_size), means)
        validation_metrics = evaluation_metrics(validation, predict_records(model, validation, args.batch_size), means)
        record = {"epoch": epoch, "train_mse": train_metrics["overall"]["mse"],
                  "validation_mse": validation_metrics["overall"]["mse"],
                  "elapsed_seconds": time.perf_counter() - started}
        history.append(record)
        with (run_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        progress(f"Epoch {epoch}: train MSE={record['train_mse']:.6f}; "
                 f"validation MSE={record['validation_mse']:.6f}; "
                 f"elapsed={record['elapsed_seconds']:.1f}s")
        return train_metrics, validation_metrics

    def save_best(epoch, metrics):
        metadata = {"epoch": epoch, "validation_mse": metrics["overall"]["mse"],
                    "training_material_means": means, "training_config": training_config}
        save_checkpoint(run_dir / "best.pt", model, metadata)

    initial_train, initial_validation = measure(0)
    best_mse = initial_validation["overall"]["mse"]
    significant_best = best_mse
    best_epoch = 0
    stale = 0
    save_best(0, initial_validation)
    for epoch in range(1, args.epochs + 1):
        model.train()
        report_every = max(4, len(loader) // 20)
        for step, (bits, labels) in enumerate(loader, start=1):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(bits), labels)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Training produced a nonfinite loss")
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise ValueError("Training produced missing or nonfinite gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % report_every == 0 or step == len(loader):
                progress(f"  Epoch {epoch}, batch {step}/{len(loader)}, loss={loss.item():.6f}")
        _, current_validation = measure(epoch)
        current_mse = current_validation["overall"]["mse"]
        if current_mse < best_mse:
            best_mse, best_epoch = current_mse, epoch
            save_best(epoch, current_validation)
        if current_mse < significant_best - 0.00001:
            significant_best, stale = current_mse, 0
        else:
            stale += 1
        if stale >= args.patience:
            progress(f"Early stopping after {stale} epochs without significant improvement.")
            break

    best_model, _ = load_checkpoint(run_dir / "best.pt")
    final_training = evaluation_metrics(training, predict_records(best_model, training, args.batch_size), means)
    final_validation = evaluation_metrics(validation, predict_records(best_model, validation, args.batch_size), means)
    updates = {name: float(torch.linalg.vector_norm(p.detach() - initial_parameters[name]))
               for name, p in best_model.named_parameters()}
    save_loss_plot(run_dir / "loss.png", history, initial_validation["overall"]["baselines"])
    diagrams = save_diagrams(run_dir / "overview.svg")
    summary = {"run_dir": str(run_dir), "checkpoint": str(run_dir / "best.pt"),
               "best_epoch": best_epoch, "epochs_completed": history[-1]["epoch"],
               "initial_training": initial_train, "initial_validation": initial_validation,
               "training": final_training, "validation": final_validation,
               "parameter_update_norms": updates, "diagrams": diagrams,
               "elapsed_seconds": time.perf_counter() - started,
               "test_evaluated": False}
    write_json(run_dir / "summary.json", summary)
    progress(f"Saved best checkpoint from epoch {best_epoch}: {run_dir / 'best.pt'}")
    return summary


def evaluate(args) -> dict:
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    if args.output and (args.output.resolve().is_relative_to(args.data_dir.resolve())
                        or args.output.resolve() == args.checkpoint.resolve()):
        raise ValueError("Evaluation output must not overwrite the dataset or checkpoint")
    model, metadata = load_checkpoint(args.checkpoint)
    records, source = load_split(args.data_dir, args.split)
    trained = metadata["training_config"]["dataset"]
    if source["manifest_sha256"] != trained["train"]["manifest_sha256"]:
        raise ValueError("Evaluation dataset manifest differs from the training dataset")
    indices = select_indices(records, args.limit, model.config.seed + 1)
    selected = [records[i] for i in indices]
    result = {"split": args.split, "selected_indices": indices, "dataset": source,
              "metrics": evaluation_metrics(selected, predict_records(model, selected, args.batch_size),
                                            metadata["training_material_means"])}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
    return result


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a reproducible prototype; default 128/64 rows")
    training.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    training.add_argument("--output-dir", "--run-dir", type=Path)
    training.add_argument("--train-limit", type=int, default=128, help="0 uses every training row")
    training.add_argument("--validation-limit", type=int, default=64, help="0 uses every validation row")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=.01)
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--backend", choices=("qiskit", "torch"), default="qiskit",
                          help="Qiskit reference or optimized differentiable statevector simulator")
    training.add_argument("--gradient", choices=("spsa", "parameter-shift", "autograd"),
                          help="Default: spsa for qiskit; autograd for torch")
    training.add_argument("--spsa-epsilon", type=float, default=.01)
    training.add_argument("--spsa-directions", type=int, default=4)
    predicting = commands.add_parser("predict", help="Predict a White-perspective score")
    predicting.add_argument("--checkpoint", type=Path, required=True)
    predicting.add_argument("--input", required=True)
    evaluating = commands.add_parser("evaluate", help="Explicitly evaluate validation or test data")
    evaluating.add_argument("--checkpoint", type=Path, required=True)
    evaluating.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    evaluating.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluating.add_argument("--limit", type=int, default=0, help="0 uses the full selected split")
    evaluating.add_argument("--batch-size", type=int, default=8)
    evaluating.add_argument("--output", type=Path)
    drawing = commands.add_parser("diagram", help="Draw the overview and actual Qiskit circuit")
    drawing.add_argument("--output", type=Path, required=True)
    return cli


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "train":
            result = train(args)
        elif args.command == "predict":
            model, _ = load_checkpoint(args.checkpoint)
            result = {"score": model.predict(args.input), "perspective": "white"}
        elif args.command == "evaluate":
            result = evaluate(args)
        else:
            result = save_diagrams(args.output)
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        cli.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
