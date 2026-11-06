# Training the chess QNN

This prototype learns one White-perspective position score from the existing
35-bit `mate-bands-v1` examples. It runs locally on the CPU, without an IBM
account, a quantum device, or another Stockfish analysis.

The current 50,000 QNN positions use distribution-preserving splits of
40,000 training, 5,000 validation, and 5,000 test rows. Each split retains
approximately 19.48% Black advantage, 61.04% near-equal, and 19.48% White
advantage. Related boards stay together. See the
[dataset split details](../../data/king_rook/qnn/README.md#distribution-preserving-split).

## Run it

Run from the repository root with Python 3.12 and the project environment:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m qnns.rook_endgame train
```

The default run uses 128 training rows and 64 validation rows, batch size 8,
and up to 10 epochs. It covers every occupied material/score-band combination
before allocating the remaining sample places proportionally. The fixed
sample indices and seed are saved. Dataset files and split assignments stay
intact; training never loads the test file.

Progress goes to stderr, with a JSON summary on stdout. Each run gets its own
directory under `runs/qnn/`, which is ignored by Git. An explicit output
directory must be new:

```sh
.venv/bin/python -m qnns.rook_endgame train --output-dir runs/qnn/my-first-run
.venv/bin/python -m qnns.rook_endgame predict --checkpoint runs/qnn/my-first-run/best.pt \
  --input 00000000000010100100110010000000000
.venv/bin/python -m qnns.rook_endgame evaluate --checkpoint runs/qnn/my-first-run/best.pt \
  --split validation --output runs/qnn/my-first-run/full-validation.json
.venv/bin/python -m qnns.rook_endgame diagram --output runs/qnn/architecture.svg
```

Prediction returns `{"score": ..., "perspective": "white"}`. Keep leading
zeros in the 35-character input. Prediction validates binary layout, not chess
legality; existing `bitstring_to_board(..., PositionType.ROOK_ENDGAME)` can
validate legality when building an application around the evaluator.

Use the entire existing training and validation splits explicitly:

```sh
.venv/bin/python -m qnns.rook_endgame train --train-limit 0 --validation-limit 0
```

The default backend remains Qiskit. For the optimized simulator of the same
190-parameter circuit, use:

```sh
.venv/bin/python -m qnns.rook_endgame train --backend torch --train-limit 0 \
  --validation-limit 0 --epochs 10 --batch-size 8 --seed 42 \
  --output-dir runs/qnn/optimized-40k
```

The `torch` backend uses exact automatic differentiation instead of SPSA and
stores its backend choice in the checkpoint. The prediction/evaluation CLI
automatically restores the correct backend. Existing Qiskit checkpoints
continue to load, and diagrams still use the Qiskit reference circuit.
Choose a new output directory when repeating a run.

Full-data Qiskit training can take much longer than the prototype. Other controls include
`--epochs`, `--batch-size`, `--learning-rate`, `--patience`, `--seed`,
`--spsa-epsilon`, and `--spsa-directions`. A sample limit must cover every
occupied material/band combination (12 in the delivered train/validation
files). There is no automatic move search or test-set evaluation.

## What the model does

```text
35 bits → 35 learned input scales → 10 quantum stages
        → 7 Z averages → linear layer + tanh → one score
```

A fresh seven-qubit zero state is used for each board. Bits are read in five
groups: `0–6`, `7–13`, `14–20`, `21–27`, and `28–34`. In each group, bit `j`
controls `RY(input_scale[j] * bit[j])` on wire `j % 7`. A learned RY and RZ on
each wire, then a ring of CZ connections, process that state before the next
group. Two passes reuse the same input scales but have separate processing
angles. Seven final learned RY rotations precede the seven Z expectations.

There are **190 parameters**: 35 input scales, 140 processing angles, seven
final angles, and eight linear-layer parameters. A king-position bit at index
5 and the turn bit at index 26 share wire 5 at different stages, with separate
input scales. All boards use the same learned settings, and each board's
temporary quantum state starts fresh. There are no intermediate measurements.

The eight castling/en-passant bits are zero throughout this dataset. Their
input scales receive zero gradients, so the model cannot learn their effects.
The constant groups still contain trainable processing gates. Input scales
are rotation angles, and angle magnitude is not a feature-importance ranking.

`PositionQNN` in [model.py](model.py) accepts a CPU tensor `[batch, 35]` of binary
values and returns `[batch]`. Its default `qiskit` backend uses `EstimatorQNN`, individual Z
observables, `TorchConnector`, and enabled input gradients. Both estimator
and QNN use zero precision for exact expectation values. PyTorch learns the
input scales through the QNN's input gradients.

The optional `torch` backend evaluates the same circuit with batched complex
statevectors and exact autograd gradients. It combines consecutive RY
rotations and commuting diagonal RZ/CZ operations to avoid repeated circuit
execution. Double-precision parameters and complex128 amplitudes preserve
numerical accuracy. The input scales, gate order up to commuting operations,
190 trainable parameters, readout, and fresh state per board are preserved.
Use `ModelConfig(backend="torch", gradient="autograd")` from Python. Tests
compare predictions at varied parameter settings and all parameter gradients
against the Qiskit reference with exact parameter-shift gradients.

## Learning and results

Training uses unweighted MSE, Adam (learning rate 0.01), and gradient clipping
at norm 1. Default SPSA gradients average four random perturbation directions
of size 0.01. These gradients are approximate even though the expectation
values are exact. `--gradient parameter-shift` selects exact gradients and
requires many more circuit evaluations per update.
With `--backend torch`, the default gradient is `autograd`; SPSA options do
not affect that backend. Explicitly incompatible backend/gradient choices
are rejected.

Input scales start at π. Quantum angles and output weights start with normal
standard deviation 0.1; output bias starts at zero. Seed 42 controls initial
weights, sampling, training order, and SPSA perturbations. Early stopping
uses validation MSE, patience 3, and minimum improvement 0.00001. The initial
model is also a checkpoint candidate; `best.pt` always contains the lowest
observed validation-MSE model, even if later training makes it worse.

Each run saves:

- `best.pt`, model weights, architecture settings, and baseline metadata.
- `config.json` and `selected_rows.json`, including versions, dataset
  checksums, mapping version, and exact selected source-row indices.
- `metrics.jsonl`, epoch-zero and subsequent training/validation MSE.
- `summary.json`, best-checkpoint MSE/MAE overall, by material, and by target
  band, plus parameter-change norms and run timing.
- `loss.png`, `overview.svg`/`overview.png`, and `overview_circuit.svg`.

Baselines always use the same evaluation rows as the model. Material means
come only from the actual selected training rows. A prototype sample has a
different distribution from the full split, so compare its matched baselines
instead of copying full-dataset MSE numbers from the manifest. Evaluation
checks that the dataset manifest matches the training provenance.

Run held-out test evaluation explicitly after choosing a model:

```sh
.venv/bin/python -m qnns.rook_endgame evaluate --checkpoint runs/qnn/my-first-run/best.pt \
  --split test --output runs/qnn/my-first-run/test-results.json
```

Scores are approximate engine evaluations, not win probabilities or proofs of
mate. The mate-distance labels encourage progress but do not guarantee a
mating policy. This prototype does not establish an advantage over classical
models or transfer to other pieces.

Run verification with `.venv/bin/python -m unittest -v`. QNN tests include
exact gradients versus finite differences, independent input scales, fresh
state between boards, a tiny learning fixture, and checkpoint/CLI round trips.

## Recorded optimized 40k baseline

The optimized backend was trained from scratch on all 40,000 training
positions and selected on all 5,000 validation positions, using seed 42,
batch size 8, learning rate 0.01, and the same 190-parameter circuit. It
completed ten epochs in **1,049 seconds (17.5 minutes)**. The best checkpoint
was epoch eight. The 5,000 test positions were not evaluated.

| Evaluator | Validation MSE | Validation MAE |
| --- | ---: | ---: |
| Optimized QNN, best checkpoint | 0.073704 | 0.222869 |
| Training mean per material class | 0.099778 | 0.165024 |
| Always predict zero | 0.155671 | 0.248391 |

Lower is better for both metrics. The QNN reduced MSE by 26.1% versus the
material-only baseline, but its mean absolute error was higher. In
particular, near-equal positions had MAE 0.173313, compared with 0.015770
for the material baseline. Scores on advantage positions improved while
many near-equal positions acquired additional error. This is a mixed
regression result, not a percentage classification accuracy or a complete
chess-playing assessment.

Artifacts are in `runs/qnn/optimized-40k/`: `best.pt`, `summary.json`,
`metrics.jsonl`, `loss.png`, the selected rows, configuration, and diagrams.
The saved model automatically selects the optimized backend when loaded.
`qiskit-reference-check.json` records an independent comparison of the
trained checkpoint against the original Qiskit circuit on 16 validation
positions. Exact-gradient tests also compare all parameter gradients with
Qiskit's parameter-shift reference.

## Recorded original prototype result

These historical results use the original source-hash split, before the
QNN exports were regrouped into distribution-preserving 40,000/5,000/5,000
splits. The checkpoint and results remain available, but are not a valid
assessment of the new held-out sets. Train a fresh model for new experiments.
The evaluator's manifest check rejects mixing the old checkpoint with the
new dataset.

The default configuration was run with
`--output-dir runs/qnn/prototype`: 128 training positions, 64 validation
positions, and seed 42. Early stopping finished after eight epochs; the
lowest validation MSE was at epoch five. The run took approximately 369
seconds on this machine, excluding dependency installation.

| Evaluator | MSE on the same 64 validation positions |
| --- | ---: |
| Initial QNN | 0.215943 |
| Best QNN, epoch five | 0.194022 |
| Always predict zero | 0.194689 |
| Mean score per material class, fitted on the 128 training rows | 0.123001 |

Lower MSE is better. This run reduced prediction error and slightly beat
always predicting zero, but did not beat the material-only baseline. Its
validation MAE was 0.304712. These small-sample results demonstrate the
pipeline, not a strong chess evaluator. Test data was not evaluated.

The saved checkpoint, selected rows, full metrics, loss curve, and diagrams
are in the local ignored `runs/qnn/prototype/` directory. Loading that
checkpoint through the prediction CLI and evaluating the same validation
sample reproduced the saved report. Installed versions were Qiskit 2.6.2,
Qiskit Machine Learning 0.9.1, and PyTorch 2.15.0.

The exact old exports and manifest are archived locally under
`runs/qnn/dataset-3b6987176888/`. To reproduce the old validation result:

```sh
.venv/bin/python -m qnns.rook_endgame evaluate --checkpoint runs/qnn/prototype/best.pt \
  --data-dir runs/qnn/dataset-3b6987176888 --split validation --limit 64
```
