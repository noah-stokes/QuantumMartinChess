"""Relabel existing QNN exports with reserved mate bands, without an engine.

Reads the original rich split files; writes only derived input/output JSON
arrays and their conversion manifest. Optional grouped stratification changes
the derived split membership without changing engine records or score labels.
No generation or analysis is performed.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import tempfile

from training.splits import METHOD as STRATIFIED_METHOD, assign_families


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
MAPPING_VERSION = "mate-bands-v1"
ORDINARY_LIMIT = 0.75
MATE_WIDTH = 0.25
MATE_DECAY = 0.9
CP_SCALE = 400.0
KINDS = ("centipawn", "nonterminal_mate", "checkmate", "terminal_draw")
BANDS = ("black_mate", "black_ordinary", "near_equal", "white_ordinary", "white_mate")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_value(raw: dict) -> int:
    value = raw.get("value")
    if type(value) is not int:
        raise ValueError("Raw evaluation must contain an integer value")
    return value


def translate_score(record: dict) -> tuple[float, str]:
    """Return the White-perspective target and its source category.

    Never derive a new label from the previous saturated target_score.
    """
    termination = record.get("termination")
    raw = record.get("raw_engine_evaluation")
    if termination == "checkmate":
        if not isinstance(raw, dict) or raw.get("kind") != "mate" or integer_value(raw) != 0:
            raise ValueError("Checkmate requires a zero-distance raw mate evaluation")
        if raw.get("winner") not in ("white", "black"):
            raise ValueError("Checkmate requires an explicit white/black winner")
        return (1.0 if raw["winner"] == "white" else -1.0), "checkmate"
    if termination in ("stalemate", "insufficient_material"):
        if raw is not None:
            raise ValueError("Terminal draw unexpectedly contains an engine evaluation")
        return 0.0, "terminal_draw"
    if termination is not None:
        raise ValueError(f"Unsupported termination: {termination}")
    if not isinstance(raw, dict):
        raise ValueError("A nonterminal position requires a raw engine evaluation")
    value = integer_value(raw)
    if raw.get("kind") == "cp":
        try:
            score = ORDINARY_LIMIT * math.tanh(value / CP_SCALE)
        except OverflowError as error:
            raise ValueError("Centipawn evaluation is too large") from error
        if not math.isfinite(score) or abs(score) > ORDINARY_LIMIT:
            raise ValueError("Centipawn target lies outside the ordinary band")
        return score, "centipawn"
    if raw.get("kind") != "mate":
        raise ValueError(f"Unsupported evaluation kind: {raw.get('kind')}")
    if raw.get("unit") != "moves" or value == 0:
        raise ValueError("Nonterminal mates require a nonzero signed distance in moves")
    magnitude = ORDINARY_LIMIT + MATE_WIDTH * MATE_DECAY ** abs(value)
    # Very long future mate distances can round back to 0.75 in binary64.
    # Preserve the strict band boundary without changing any current label.
    magnitude = max(magnitude, math.nextafter(ORDINARY_LIMIT, 1.0))
    if not ORDINARY_LIMIT < magnitude < 1:
        raise ValueError("Nonterminal mate target lies outside its reserved band")
    return math.copysign(magnitude, value), "nonterminal_mate"


def score_band(score: float) -> str:
    if score < -ORDINARY_LIMIT:
        return BANDS[0]
    if score < -0.2:
        return BANDS[1]
    if score <= 0.2:
        return BANDS[2]
    if score <= ORDINARY_LIMIT:
        return BANDS[3]
    return BANDS[4]


def distribution(rows: list[dict], kinds: list[str]) -> dict:
    scores = [row["output"] for row in rows]
    bands = Counter({name: 0 for name in BANDS})
    bands.update(score_band(score) for score in scores)
    categories = Counter({name: 0 for name in KINDS})
    categories.update(kinds)
    groups = {"black_advantage": bands["black_mate"] + bands["black_ordinary"],
              "near_equal": bands["near_equal"],
              "white_advantage": bands["white_mate"] + bands["white_ordinary"]}
    materials = Counter()
    for row in rows:
        bits = row["input"]
        material = {("0", "0"): "KRvKR", ("0", "1"): "KRvK",
                    ("1", "0"): "KvKR", ("1", "1"): "KvK"}[(bits[6], bits[19])]
        materials[material] += 1
    return {"positions": len(rows), "categories": dict(categories), "score_bands": dict(bands),
            "score_groups": groups,
            "score_group_percentages": {key: count * 100 / len(rows) if rows else 0
                                        for key, count in groups.items()},
            "material_counts": dict(materials),
            "minimum": min(scores) if scores else None, "maximum": max(scores) if scores else None,
            "exact_endpoint_count": sum(abs(score) == 1 for score in scores)}


def baseline_errors(source_rows: dict, converted: dict) -> dict:
    values = defaultdict(list)
    for source, row in zip(source_rows["train"], converted["train"]):
        values[source["material_class"]].append(row["output"])
    means = {material: math.fsum(scores) / len(scores) for material, scores in values.items()}
    results = {}
    for split in ("validation", "test"):
        targets = converted[split]
        missing = {row["material_class"] for row in source_rows[split]} - means.keys()
        if missing:
            raise ValueError(f"Held-out material classes missing from training: {sorted(missing)}")
        size = len(targets)
        results[split] = {
            "positions": size,
            "zero_prediction_mse": math.fsum(row["output"] ** 2 for row in targets) / size if size else None,
            "material_mean_mse": math.fsum(
                (row["output"] - means[source["material_class"]]) ** 2
                for source, row in zip(source_rows[split], targets)
            ) / size if size else None,
        }
    return {"scope": "unweighted converted targets; compare only within the same mapping version",
            "training_material_means": means, **results}


def validate_conversion(source_rows: dict, converted: dict, kinds: dict) -> dict:
    seen_inputs, families, mirror_pairs = set(), {}, defaultdict(list)
    checked = 0
    for split in SPLITS:
        original, rows = source_rows[split], converted[split]
        if len(original) != len(rows) or len(kinds[split]) != len(rows):
            raise ValueError(f"Record count changed in {split}")
        for source, row, kind in zip(original, rows, kinds[split]):
            bits = row.get("input")
            if set(row) != {"input", "output"} or not isinstance(bits, str) or len(bits) != 35 or set(bits) - {"0", "1"}:
                raise ValueError("Converted record must contain a 35-bit input and one output")
            if bits != source["rook_bitstring"]:
                raise ValueError(f"Input or ordering changed in {split}")
            if bits in seen_inputs:
                raise ValueError("Duplicate input in converted splits")
            seen_inputs.add(bits)
            score = row["output"]
            if type(score) is not float or not math.isfinite(score) or not -1 <= score <= 1:
                raise ValueError("Converted target must be a finite float in [-1, 1]")
            expected, expected_kind = translate_score(source)
            if score != expected or kind != expected_kind:
                raise ValueError("Converted target does not match its raw source evaluation")
            if (abs(score) == 1) != (kind == "checkmate"):
                raise ValueError("Only actual checkmates may have endpoint targets")
            family = source["family_id"]
            if families.setdefault(family, split) != split:
                raise ValueError("A position family appears in more than one split")
            if source["augmentation"] not in ("identity", "color_swap_vertical"):
                raise ValueError("Unsupported source augmentation")
            mirror_pairs[source["source_fen"]].append((score, split, family, source["augmentation"], kind))
            checked += 1
    for pair in mirror_pairs.values():
        if len(pair) != 2 or {entry[3] for entry in pair} != {"identity", "color_swap_vertical"}:
            raise ValueError("Missing color-mirror partner")
        if pair[0][0] != -pair[1][0] or pair[0][1:3] != pair[1][1:3] or pair[0][4] != pair[1][4]:
            raise ValueError("Color mirrors have inconsistent scores, categories, or splits")
    return {"records_checked": checked, "unique_inputs": len(seen_inputs), "families": len(families),
            "mirror_pairs": len(mirror_pairs), "inputs_order_and_splits": "preserved",
            "mirror_signs": "passed", "bands_and_endpoints": "passed", "family_overlap": 0}


def stratify(source_rows: dict, converted: dict, kinds: dict, seed: int):
    families = defaultdict(Counter)
    entries = []
    for split in SPLITS:
        for source, row, kind in zip(source_rows[split], converted[split], kinds[split]):
            families[source["family_id"]][(source["material_class"], score_band(row["output"]))] += 1
            entries.append((source, row, kind))
    assignment, metadata = assign_families(families, seed)
    new_source = {split: [] for split in SPLITS}
    new_rows = {split: [] for split in SPLITS}
    new_kinds = {split: [] for split in SPLITS}
    for source, row, kind in sorted(entries, key=lambda entry: entry[1]["input"]):
        split = assignment[source["family_id"]]
        new_source[split].append(source)
        new_rows[split].append(row)
        new_kinds[split].append(kind)
    return new_source, new_rows, new_kinds, metadata


def split_quality(converted: dict, kinds: dict) -> dict:
    all_rows = [row for split in SPLITS for row in converted[split]]
    all_kinds = [kind for split in SPLITS for kind in kinds[split]]
    overall = distribution(all_rows, all_kinds)
    quality = {}
    for split in SPLITS:
        part = distribution(converted[split], kinds[split])
        quality[split] = {}
        for field in ("score_groups", "score_bands", "material_counts"):
            deviations = [abs(part[field].get(key, 0) / part["positions"]
                              - count / overall["positions"]) * 100
                          for key, count in overall[field].items()] if part["positions"] else []
            quality[split][f"max_{field}_deviation_percentage_points"] = max(deviations, default=None)
    return quality


def translate(source_dir: Path, output_dir: Path, *, stratify_splits: bool | None = None,
              seed: int | None = None) -> dict:
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir:
        raise ValueError("Output directory must differ from the original data directory")
    source_manifest_path = source_dir / "manifest.json"
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    protected = {source_manifest_path: hashlib.sha256(source_manifest_bytes).hexdigest()}
    old_manifest_path = output_dir / "manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text()) if old_manifest_path.exists() else {}
    old_split = old_manifest.get("split", {})
    if stratify_splits is None:
        stratify_splits = old_split.get("method") == STRATIFIED_METHOD
    split_seed = seed if seed is not None else old_split.get("seed", 42)
    previous_rows = {}
    source_rows, converted, kinds, sources = {}, {}, {}, {}
    for split in SPLITS:
        path = source_dir / f"{split}.jsonl.gz"
        digest = sha256_file(path)
        expected = source_manifest["files"][split]
        if digest != expected["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
        protected[path] = digest
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            original = [json.loads(line) for line in stream]
        if len(original) != expected["positions"]:
            raise ValueError(f"Source record count mismatch: {path}")
        rows, categories = [], []
        for source in original:
            score, kind = translate_score(source)
            rows.append({"input": source["rook_bitstring"], "output": score})
            categories.append(kind)
        previous_path = output_dir / f"{split}.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            previous_rows[split] = previous
            if not stratify_splits and (len(previous) != len(rows) or any(
                    old.get("input") != new["input"] for old, new in zip(previous, rows))):
                raise ValueError(f"Existing QNN input order does not match the source: {previous_path}")
        source_rows[split], converted[split], kinds[split] = original, rows, categories
        sources[split] = {"path": path.name, "sha256": digest, "positions": len(original)}
    validation = validate_conversion(source_rows, converted, kinds)
    split_metadata = {"method": "source", "membership": "preserved from original engine records"}
    if stratify_splits:
        if previous_rows:
            if set(previous_rows) != set(SPLITS):
                raise ValueError("Cannot stratify partial QNN exports; all three files must be present")
            old_inputs = Counter(row["input"] for rows in previous_rows.values() for row in rows)
            new_inputs = Counter(row["input"] for rows in converted.values() for row in rows)
            if old_inputs != new_inputs:
                raise ValueError("Stratification must preserve the complete existing input set")
            if old_manifest.get("mapping_version") == MAPPING_VERSION:
                old_pairs = Counter((row["input"], row["output"]) for rows in previous_rows.values() for row in rows)
                new_pairs = Counter((row["input"], row["output"]) for rows in converted.values() for row in rows)
                if old_pairs != new_pairs:
                    raise ValueError("Stratification must preserve existing input/target pairs")
        source_rows, converted, kinds, split_metadata = stratify(source_rows, converted, kinds, split_seed)
        validation = validate_conversion(source_rows, converted, kinds)
        validation["inputs_order_and_splits"] = "regrouped deterministically; input/target pairs preserved"
    split_metadata["distribution_quality"] = split_quality(converted, kinds)
    baselines = baseline_errors(source_rows, converted)
    all_rows = [row for split in SPLITS for row in converted[split]]
    all_kinds = [kind for split in SPLITS for kind in kinds[split]]
    manifest = {
        "mapping_version": MAPPING_VERSION,
        "record_schema": {"input": "35-character binary string", "output": "White-perspective float in [-1,1]"},
        "mapping": {"ordinary_limit": ORDINARY_LIMIT, "mate_band_width": MATE_WIDTH,
                    "centipawn_scale": CP_SCALE, "mate_decay_per_move": MATE_DECAY,
                    "ordinary_formula": "0.75 * tanh(white_centipawns / 400)",
                    "nonterminal_mate_formula": "sign(m) * (0.75 + 0.25 * 0.9 ** abs(m))",
                    "mate_distance_unit": "engine-reported moves",
                    "checkmate": "signed endpoint from explicit winner", "terminal_draw": 0,
                    "rounding_guard": "round mate magnitudes up to nextafter(0.75, 1) if binary64 reaches the boundary"},
        "band_definitions": {"black_mate": "[-1,-0.75)", "black_ordinary": "[-0.75,-0.2)",
                             "near_equal": "[-0.2,0.2]", "white_ordinary": "(0.2,0.75]", "white_mate": "(0.75,1]"},
        "sources": sources, "source_manifest_sha256": protected[source_manifest_path],
        "converter_sha256": sha256_file(Path(__file__)),
        "split": split_metadata,
        "score_group_definitions": {"black_advantage": "[-1,-0.2)", "near_equal": "[-0.2,0.2]",
                                    "white_advantage": "(0.2,1]"},
        "distribution": distribution(all_rows, all_kinds), "validation": validation,
        "baselines": baselines,
        "limitations": ["Mate distances are engine-reported, not guaranteed globally shortest.",
                        "Distance-aware targets encourage progress but do not guarantee a mating policy.",
                        "Original rich-record targets, weights, and baseline scores use the old mapping."]}
    # Stage and read back every artifact before replacing any existing export.
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".translate-", dir=output_dir) as temporary:
        staging = Path(temporary)
        manifest["files"] = {}
        for split in SPLITS:
            path = staging / f"{split}.json"
            with path.open("w", encoding="utf-8") as stream:
                json.dump(converted[split], stream, indent=2, allow_nan=False)
                stream.write("\n")
            if json.loads(path.read_text()) != converted[split]:
                raise ValueError(f"Export read-back failed: {split}")
            manifest["files"][split] = {
                "path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                **distribution(converted[split], kinds[split]),
            }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if any(sha256_file(path) != digest for path, digest in protected.items()):
            raise ValueError("Source files changed during conversion; exports were not replaced")
        for split in SPLITS:
            (staging / f"{split}.json").replace(output_dir / f"{split}.json")
        # The conversion manifest is written last as the completion marker.
        manifest_path.replace(output_dir / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/king_rook")
    parser.add_argument("--output-dir", type=Path, help="Default: SOURCE_DIR/qnn")
    parser.add_argument("--stratify", action="store_true", default=None,
                        help="Re-split whole families 80/10/10, preserving material and score distributions")
    parser.add_argument("--seed", type=int, help="Stratification seed (default: existing seed or 42)")
    args = parser.parse_args()
    try:
        manifest = translate(args.source_dir, args.output_dir or args.source_dir / "qnn",
                             stratify_splits=args.stratify, seed=args.seed)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"Translation failed: {error}\n")
    print(json.dumps({"mapping_version": manifest["mapping_version"],
                      **manifest["distribution"], "split": manifest["split"],
                      "baselines": manifest["baselines"]}, indent=2))


if __name__ == "__main__":
    main()
