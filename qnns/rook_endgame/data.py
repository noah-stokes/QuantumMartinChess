"""Read-only dataset loading, reproducible subsets, and matched baselines."""

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from qnns.rook_endgame.model import validate_bitstring
from training.translate_qnn_data import BANDS, MAPPING_VERSION, score_band


MATERIALS = ("KRvKR", "KRvK", "KvKR", "KvK")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def material_class(bits: str) -> str:
    return {("0", "0"): "KRvKR", ("0", "1"): "KRvK",
            ("1", "0"): "KvKR", ("1", "1"): "KvK"}[(bits[6], bits[19])]


def read_records(path: Path) -> list[dict]:
    records = json.loads(Path(path).read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a nonempty JSON array")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"input", "output"}:
            raise ValueError(f"{path}, row {index}: expected input and output fields")
        validate_bitstring(record["input"])
        value = record["output"]
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not -1 <= value <= 1):
            raise ValueError(f"{path}, row {index}: target must be finite and in [-1, 1]")
    return records


def load_split(directory: Path, split: str) -> tuple[list[dict], dict]:
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown dataset split")
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("mapping_version") != MAPPING_VERSION:
        raise ValueError(f"Expected dataset mapping {MAPPING_VERSION}")
    path = directory / f"{split}.json"
    checksum = sha256_file(path)
    expected = manifest["files"][split]
    if checksum != expected["sha256"]:
        raise ValueError(f"{split}: dataset checksum does not match its manifest")
    records = read_records(path)
    if len(records) != expected["positions"]:
        raise ValueError(f"{split}: record count does not match its manifest")
    return records, {"mapping_version": manifest["mapping_version"],
                     "manifest_sha256": sha256_file(manifest_path),
                     "file_sha256": checksum, "positions": len(records)}


def select_indices(records: list[dict], limit: int, seed: int) -> list[int]:
    """Cover each occupied material/band stratum, then allocate proportionally."""
    if limit < 0:
        raise ValueError("Sample limits must be nonnegative; 0 means all rows")
    if limit == 0 or limit >= len(records):
        return list(range(len(records)))
    strata = defaultdict(list)
    for index, row in enumerate(records):
        strata[(material_class(row["input"]), score_band(row["output"]))].append(index)
    keys = sorted(strata)
    if limit < len(keys):
        raise ValueError(f"Sample limit must be at least {len(keys)} to cover every occupied stratum")
    capacities = np.array([len(strata[key]) - 1 for key in keys], dtype=int)
    remaining = limit - len(keys)
    quotas = remaining * capacities / capacities.sum() if capacities.sum() else capacities.astype(float)
    allocation = np.floor(quotas).astype(int)
    left = remaining - int(allocation.sum())
    for i in sorted(range(len(keys)), key=lambda i: (-(quotas[i] - allocation[i]), i))[:left]:
        allocation[i] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for key, count in zip(keys, allocation + 1):
        selected.extend(int(i) for i in rng.choice(strata[key], size=int(count), replace=False))
    return sorted(selected)


def training_material_means(records: list[dict]) -> dict[str, float]:
    groups = defaultdict(list)
    for row in records:
        groups[material_class(row["input"])].append(row["output"])
    return {key: math.fsum(values) / len(values) for key, values in groups.items()}


def evaluation_metrics(records: list[dict], predictions, means: dict[str, float]) -> dict:
    targets = np.array([r["output"] for r in records], dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != targets.shape or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be finite and match the evaluated rows")
    materials = np.array([material_class(r["input"]) for r in records])
    bands = np.array([score_band(r["output"]) for r in records])
    missing = set(materials) - means.keys()
    if missing:
        raise ValueError(f"Material baseline has no training examples for: {sorted(missing)}")
    baseline = np.array([means[key] for key in materials])

    def scores(mask):
        count = int(np.count_nonzero(mask))
        def errors(values):
            error = values[mask] - targets[mask]
            return {"mse": float(np.mean(error ** 2)) if count else None,
                    "mae": float(np.mean(np.abs(error))) if count else None}
        return {"count": count, **errors(predictions),
                "baselines": {"zero": errors(np.zeros_like(targets)),
                              "material_mean": errors(baseline)}}

    return {"overall": scores(np.ones(len(records), dtype=bool)),
            "by_material": {key: scores(materials == key) for key in MATERIALS},
            "by_target_band": {key: scores(bands == key) for key in BANDS}}
