"""Generate resumable, engine-labeled rook endgames; no QNN dependency.

Run with the project's Python environment. See data/king_rook/README.md for
the schema, score convention, sampling limitations, and reproducibility.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import sqlite3
import tarfile
import time
from typing import Optional
import urllib.request

import chess
import chess.engine

from tools.board_bitstrings import board_to_bitstring, bitstring_to_board
from tools.position_types import PositionType


SCHEMA_VERSION = 1
STOCKFISH_TAG = "sf_19"
ARCHIVE_NAME = "stockfish-macos-universal.tar.gz"
ARCHIVE_SHA256 = "a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"
RELEASE_URL = "https://github.com/official-stockfish/Stockfish/releases/tag/sf_19"
DOWNLOAD_URL = f"https://github.com/official-stockfish/Stockfish/releases/download/{STOCKFISH_TAG}/{ARCHIVE_NAME}"
BIN_NAMES = ("black_large", "black_advantage", "near_equal", "white_advantage", "white_large")
ROOT = Path(__file__).resolve().parents[1]
ROOKS = PositionType.ROOK_ENDGAME


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_engine(directory: Path, supplied: Optional[Path] = None,
                   expected_sha256: Optional[str] = None) -> tuple[Path, dict]:
    """Verify before executing. Managed downloads are pinned, not 'latest'."""
    directory.mkdir(parents=True, exist_ok=True)
    if supplied is not None:
        if not expected_sha256 or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("--engine requires its independently verified --engine-sha256")
        binary = supplied.resolve()
        digest = sha256_file(binary)
        if digest != expected_sha256.lower():
            raise ValueError("Supplied engine SHA-256 does not match")
        return binary, {"release": RELEASE_URL, "binary_sha256": digest, "source": "supplied"}
    if platform.system() != "Darwin":
        raise ValueError("On other platforms provide Stockfish 19 with --engine and --engine-sha256")
    archive = directory / ARCHIVE_NAME
    if not archive.exists():
        partial = archive.with_suffix(".download")
        print(f"Downloading {DOWNLOAD_URL}", flush=True)
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        partial.replace(archive)
    if sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError(f"Stockfish release checksum mismatch: {archive}")
    # Extract only regular files and reject traversal, links and special files.
    destination = directory / "verified_release"
    destination.mkdir(exist_ok=True)
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if destination.resolve() not in target.parents and target != destination.resolve():
                raise ValueError("Unsafe path in release archive")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("wb") as output:
                    output.write(source.read())
                target.chmod(member.mode & 0o777)
            else:
                raise ValueError("Unsupported archive member")
    candidates = [p for p in destination.rglob("*")
                  if p.is_file() and p.name.startswith("stockfish") and p.stat().st_mode & 0o111]
    if len(candidates) != 1:
        raise ValueError(f"Expected one release executable, found {candidates}")
    binary = candidates[0]
    return binary, {"release": RELEASE_URL, "download_url": DOWNLOAD_URL,
                    "archive_sha256": ARCHIVE_SHA256, "binary_sha256": sha256_file(binary),
                    "source": "verified_release_archive"}


def score_bin(score: float) -> str:
    if not math.isfinite(score) or not -1 <= score <= 1:
        raise ValueError("Scores must be finite and within [-1, 1]")
    if score < -0.8:
        return BIN_NAMES[0]
    if score < -0.2:
        return BIN_NAMES[1]
    if score <= 0.2:
        return BIN_NAMES[2]
    if score <= 0.8:
        return BIN_NAMES[3]
    return BIN_NAMES[4]


def normalize_score(cp: int, scale: float = 400.0) -> float:
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Score scale must be positive and finite")
    return math.tanh(cp / scale)


def material_class(board: chess.Board) -> str:
    white = bool(board.rooks & board.occupied_co[chess.WHITE])
    black = bool(board.rooks & board.occupied_co[chess.BLACK])
    return { (True, True): "KRvKR", (True, False): "KRvK",
             (False, True): "KvKR", (False, False): "KvK" }[white, black]


def _symmetries() -> tuple[tuple[int, ...], ...]:
    maps = []
    for reflected in (False, True):
        for rotations in range(4):
            transform = []
            for square in chess.SQUARES:
                x, y = chess.square_file(square), chess.square_rank(square)
                if reflected:
                    x = 7 - x
                for _ in range(rotations):
                    x, y = 7 - y, x
                transform.append(chess.square(x, y))
            maps.append(tuple(transform) + (64,))
    return tuple(maps)


SYMMETRIES = _symmetries()


def family_id(board: chess.Board) -> str:
    """Ignore turn; group all D4 transforms and color swaps of a placement.

    This grouping is valid for this pawnless, castling-free dataset only.
    It is intentionally broader than an analysis cache key.
    """
    if board.castling_rights or board.ep_square is not None:
        raise ValueError("Dataset families require no castling or en passant")
    white_rook = next(iter(board.pieces(chess.ROOK, chess.WHITE)), 64)
    black_rook = next(iter(board.pieces(chess.ROOK, chess.BLACK)), 64)
    placement = (board.king(chess.WHITE), white_rook, board.king(chess.BLACK), black_rook)
    variants = []
    for transform in SYMMETRIES:
        wk, wr, bk, br = (transform[square] for square in placement)
        variants.extend(((wk, wr, bk, br), (bk, br, wk, wr)))
    return ",".join(map(str, min(variants)))


def split_for_family(family: str, seed: int) -> str:
    digest = hashlib.sha256(f"rook-split-v1:{seed}:{family}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def sample_board(rng: random.Random, material: str) -> chess.Board:
    pieces = [(chess.KING, chess.WHITE), (chess.KING, chess.BLACK)]
    if material in ("KRvKR", "KRvK"):
        pieces.append((chess.ROOK, chess.WHITE))
    if material == "KRvKR":
        pieces.append((chess.ROOK, chess.BLACK))
    while True:
        board = chess.Board(None)
        for square, (piece_type, color) in zip(rng.sample(range(64), len(pieces)), pieces):
            board.set_piece_at(square, chess.Piece(piece_type, color))
        board.turn = rng.choice((chess.WHITE, chess.BLACK))
        if board.is_valid():
            return board


def sample_pairs(two_rook_count: int, single_rook_count: int, bare_king_count: int,
                 seed: int):
    """Yield a deterministic uniform-valid-placement sample with color pairs."""
    counts = (two_rook_count, single_rook_count, bare_king_count)
    if any(count < 0 or count % 2 for count in counts) or not sum(counts):
        raise ValueError("Material counts must be nonnegative even integers with a nonzero total")
    # Bounds prevent endless rejection when a requested sample exceeds a domain.
    if single_rook_count > 798224 or bare_king_count > 7224 or two_rook_count > 30498048:
        raise ValueError("Requested count exceeds the material's position-space bound")
    schedule = (["KRvKR"] * (two_rook_count // 2)
                + ["KRvK"] * (single_rook_count // 2)
                + ["KvK"] * (bare_king_count // 2))
    rng = random.Random(seed)
    rng.shuffle(schedule)
    seen = set()
    for material in schedule:
        for _ in range(100000):
            board = sample_board(rng, material)
            mirror = board.mirror()
            first, second = board.fen(), mirror.fen()
            if first not in seen and second not in seen:
                seen.update((first, second))
                yield board, mirror
                break
        else:
            raise ValueError("Sampling exhausted its duplicate-rejection budget")


def terminal_label(board: chess.Board) -> Optional[dict]:
    if board.is_checkmate():
        winner = not board.turn
        return {"target_score": 1.0 if winner else -1.0,
                "raw_engine_evaluation": {"kind": "mate", "value": 0,
                                           "winner": "white" if winner else "black"},
                "termination": "checkmate", "engine_depth": None, "engine_nodes": 0}
    termination = ("stalemate" if board.is_stalemate() else
                   "insufficient_material" if board.is_insufficient_material() else None)
    if termination:
        return {"target_score": 0.0, "raw_engine_evaluation": None,
                "termination": termination, "engine_depth": None, "engine_nodes": 0}
    return None


def evaluation_label(info: dict, scale: float) -> dict:
    score = info["score"].white()
    if score.is_mate():
        mate = score.mate()
        if not mate:
            raise ValueError("Unexpected mate-zero score for a nonterminal board")
        target = 1.0 if mate > 0 else -1.0
        raw = {"kind": "mate", "value": mate, "unit": "moves"}
    else:
        cp = score.score()
        if cp is None:
            raise ValueError("Engine omitted its centipawn evaluation")
        target, raw = normalize_score(cp, scale), {"kind": "cp", "value": cp}
    return {"target_score": target, "raw_engine_evaluation": raw, "termination": None,
            "engine_depth": info.get("depth"), "engine_nodes": info.get("nodes")}


def analyze_board(engine, board: chess.Board, nodes: int, scale: float) -> dict:
    terminal = terminal_label(board)
    if terminal is not None:
        return terminal
    engine.configure({"Clear Hash": None})
    last_exact = None
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), game=object(),
                         info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC) as analysis:
        for info in analysis:
            if "score" in info and not info.get("lowerbound") and not info.get("upperbound"):
                last_exact = info
    if last_exact is None:
        raise ValueError(f"Engine produced no exact search score for {board.fen()}")
    return evaluation_label(last_exact, scale)


class AnalysisCache:
    def __init__(self, path: Path, configuration: dict):
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS analyses "
                                "(fen TEXT PRIMARY KEY, label TEXT NOT NULL, seconds REAL NOT NULL)")
        signature = json.dumps(configuration, sort_keys=True)
        previous = self.connection.execute("SELECT value FROM metadata WHERE key='configuration'").fetchone()
        if previous and previous[0] != signature:
            self.connection.close()
            raise ValueError("Cache configuration differs; use a different --cache-dir")
        self.connection.execute("INSERT OR IGNORE INTO metadata VALUES ('configuration', ?)", (signature,))
        self.connection.commit()

    def get(self, fen: str):
        row = self.connection.execute("SELECT label, seconds FROM analyses WHERE fen=?", (fen,)).fetchone()
        return (json.loads(row[0]), row[1]) if row else None

    def put(self, fen: str, label: dict, seconds: float):
        self.connection.execute("INSERT INTO analyses VALUES (?, ?, ?)",
                                (fen, json.dumps(label, sort_keys=True), seconds))
        self.connection.commit()

    def close(self):
        self.connection.close()


def make_record(board: chess.Board, label: dict, source_fen: str, mirrored=False) -> dict:
    copied = dict(label)
    if mirrored:
        copied["target_score"] = -label["target_score"] if label["target_score"] else 0.0
        raw = label["raw_engine_evaluation"]
        if raw is not None:
            raw = dict(raw)
            raw["value"] = -raw["value"]
            if "winner" in raw:
                raw["winner"] = "black" if raw["winner"] == "white" else "white"
            copied["raw_engine_evaluation"] = raw
    return {**copied, "bitstring": board_to_bitstring(board),
            "rook_bitstring": board_to_bitstring(board, ROOKS), "fen": board.fen(),
            "material_class": material_class(board), "family_id": family_id(board),
            "source_fen": source_fen, "augmentation": "color_swap_vertical" if mirrored else "identity"}


def distribution(records: list[dict]) -> dict:
    bins = Counter({name: 0 for name in BIN_NAMES})
    materials = Counter()
    turns = Counter()
    raw_types = Counter()
    exact_endpoints = near_endpoints = 0
    for row in records:
        score = row["target_score"]
        bins[score_bin(score)] += 1
        materials[row["material_class"]] += 1
        turns[row["fen"].split()[1]] += 1
        raw_types[(row["raw_engine_evaluation"] or {}).get("kind", "terminal_draw")] += 1
        exact_endpoints += abs(score) == 1
        near_endpoints += abs(score) >= 0.95
    count = len(records)
    return {"positions": count, "score_bins": dict(bins), "material_counts": dict(materials),
            "turn_counts": dict(turns), "raw_evaluation_kinds": dict(raw_types),
            "exact_endpoint_count": exact_endpoints,
            "exact_endpoint_fraction": exact_endpoints / count if count else 0,
            "abs_score_at_least_0_95_count": near_endpoints,
            "abs_score_at_least_0_95_fraction": near_endpoints / count if count else 0,
            "empty_bins": [name for name in BIN_NAMES if not bins[name]],
            "sparse_bins_below_one_percent": [name for name in BIN_NAMES if count and bins[name] / count < 0.01]}


def validate_records(records: list[dict], scale: float) -> dict:
    seen_standard, seen_rook = set(), set()
    by_fen = {row["fen"]: row for row in records}
    if len(by_fen) != len(records):
        raise ValueError("Duplicate FENs in dataset")
    for row in records:
        board = chess.Board(row["fen"])
        if not board.is_valid() or board.castling_rights or board.ep_square is not None:
            raise ValueError("Invalid dataset position/state")
        for field, kind, seen in (("bitstring", PositionType.STANDARD, seen_standard),
                                  ("rook_bitstring", ROOKS, seen_rook)):
            bits = row[field]
            if bits in seen:
                raise ValueError(f"Duplicate {field}")
            seen.add(bits)
            restored = bitstring_to_board(bits, kind)
            if restored.fen() != board.fen() or board_to_bitstring(restored, kind) != bits:
                raise ValueError(f"Encoding round-trip failed: {row['fen']}")
        if row["family_id"] != family_id(board) or row["material_class"] != material_class(board):
            raise ValueError("Incorrect record metadata")
        score_bin(row["target_score"])
        terminal = terminal_label(board)
        if terminal is not None:
            if row["target_score"] != terminal["target_score"] or row["termination"] != terminal["termination"]:
                raise ValueError("Terminal score mismatch")
        else:
            raw = row["raw_engine_evaluation"]
            expected = normalize_score(raw["value"], scale) if raw["kind"] == "cp" else math.copysign(1, raw["value"])
            if row["target_score"] != expected or row["termination"] is not None:
                raise ValueError("Score normalization mismatch")
        counterpart = by_fen.get(board.mirror().fen())
        if counterpart is None or counterpart["target_score"] != -row["target_score"]:
            raise ValueError("Missing or incorrectly signed color mirror")
        if counterpart["family_id"] != row["family_id"]:
            raise ValueError("Color mirrors assigned different families")
    return {"positions_checked": len(records), "unique_standard_inputs": len(seen_standard),
            "unique_rook_inputs": len(seen_rook), "encoding_round_trips": "passed",
            "color_mirrors": "passed", "terminal_labels_and_normalization": "passed"}


def baseline_errors(train: list[dict], heldout: list[dict]) -> dict:
    totals, counts = defaultdict(float), Counter()
    for row in train:
        totals[row["material_class"]] += row["target_score"]
        counts[row["material_class"]] += 1
    means = {key: totals[key] / counts[key] for key in counts}
    overall = sum(totals.values()) / len(train) if train else 0.0
    size = len(heldout)
    return {"positions": size, "training_material_means": means,
            "zero_prediction_mse": sum(row["target_score"] ** 2 for row in heldout) / size if size else None,
            "material_mean_mse": sum((row["target_score"] - means.get(row["material_class"], overall)) ** 2
                                     for row in heldout) / size if size else None}


def export_dataset(records: list[dict], output: Path, seed: int, metadata: dict) -> dict:
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in sorted(records, key=lambda row: row["rook_bitstring"]):
        splits[split_for_family(row["family_id"], seed)].append(dict(row))
    groups = {name: {row["family_id"] for row in rows} for name, rows in splits.items()}
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Family leakage between splits")
    train_bins = Counter(score_bin(row["target_score"]) for row in splits["train"])
    weights = {name: len(splits["train"]) / (len(train_bins) * count) for name, count in train_bins.items()}
    files = {}
    for name, rows in splits.items():
        target = output / f"{name}.jsonl.gz"
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                if name == "train":
                    row["sampling_weight"] = weights[score_bin(row["target_score"])]
                compressed.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
        temporary.replace(target)
        files[name] = {"path": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size,
                       "families": len(groups[name]), **distribution(rows)}
    manifest = {**metadata, "distribution": distribution(records), "files": files,
                "sampling_weights": {"applied_to": "train_only", "values_by_score_bin": weights,
                                     "formula": "N_train / (occupied_bins * bin_count)"},
                "baselines": {name: baseline_errors(splits["train"], splits[name]) for name in ("validation", "test")},
                "split_family_overlap": 0}
    atomic_json(output / "manifest.json", manifest)
    return manifest


def generate(args) -> dict:
    if args.nodes <= 0 or args.pilot_positions <= 0 or args.pilot_positions % 2:
        raise ValueError("Nodes must be positive; pilot size must be positive and even")
    normalize_score(0, args.score_scale)
    counts = (args.two_rook_count, args.single_rook_count, args.bare_king_count)
    # Validate before downloading/launching anything.
    if any(n < 0 or n % 2 for n in counts) or not sum(counts):
        raise ValueError("Material counts must be nonnegative even integers with a nonzero total")
    output, cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary, provenance = prepare_engine(cache_dir / "stockfish", args.engine, args.engine_sha256)
    configuration = {"schema_version": SCHEMA_VERSION, "binary_sha256": provenance["binary_sha256"],
                     "nodes": args.nodes, "threads": 1, "hash_mib": 64,
                     "clear_search_state_per_position": True, "syzygy_path": "",
                     "score_scale_cp": args.score_scale, "perspective": "white"}
    records, hits, misses, analysis_seconds = [], 0, 0, 0.0
    started = time.monotonic()
    total = sum(counts)
    pilot_size = min(args.pilot_positions, total)
    pilot = None
    with closing(AnalysisCache(cache_dir / "analyses.sqlite3", configuration)) as cache:
        with chess.engine.SimpleEngine.popen_uci(str(binary), timeout=60) as engine:
            if not re.search(r"\bStockfish 19\b", engine.id.get("name", "")):
                raise ValueError(f"Expected Stockfish 19, got {engine.id}")
            engine.configure({"Threads": 1, "Hash": 64, "SyzygyPath": ""})
            if "UCI_LimitStrength" in engine.options:
                engine.configure({"UCI_LimitStrength": False})
            engine_id = dict(engine.id)
            print(f"Engine: {engine_id}; generating {total:,} positions", flush=True)
            for board, mirror in sample_pairs(*counts, args.seed):
                fen = board.fen()
                cached = cache.get(fen)
                if cached is None:
                    before = time.monotonic()
                    label = analyze_board(engine, board, args.nodes, args.score_scale)
                    seconds = time.monotonic() - before
                    cache.put(fen, label, seconds)
                    misses += 1
                else:
                    label, seconds = cached
                    hits += 1
                analysis_seconds += seconds
                records.extend((make_record(board, label, fen), make_record(mirror, label, fen, mirrored=True)))
                if len(records) == pilot_size:
                    pilot = {**distribution(records), "elapsed_seconds": time.monotonic() - started,
                             "cached_pairs": hits, "new_pairs": misses,
                             "estimated_full_cold_analysis_seconds": analysis_seconds / (len(records) / 2) * (total / 2),
                             "estimated_remaining_cold_analysis_seconds": analysis_seconds / (len(records) / 2) * ((total - len(records)) / 2),
                             "included_in_final_dataset": True}
                    atomic_json(output / "pilot.json", pilot)
                    print("Pilot: " + json.dumps(pilot, sort_keys=True), flush=True)
                elif len(records) % 2000 == 0:
                    print(f"Progress: {len(records):,}/{total:,}; cache hits {hits:,}; elapsed {time.monotonic() - started:.1f}s", flush=True)
    print("Checking every record and exporting splits...", flush=True)
    checks = validate_records(records, args.score_scale)
    actual_counts = Counter(row["material_class"] for row in records)
    expected = Counter({"KRvKR": counts[0], "KRvK": counts[1] // 2, "KvKR": counts[1] // 2, "KvK": counts[2]})
    if actual_counts != +expected:
        raise ValueError(f"Material quota mismatch: {actual_counts} != {expected}")
    metadata = {"schema_version": SCHEMA_VERSION, "seed": args.seed, "engine": engine_id,
                "engine_provenance": provenance, "analysis_configuration": configuration,
                "requested_counts": dict(expected), "pilot": pilot, "validation": checks,
                "cache_hits": hits, "new_cached_pairs": misses,
                "encoding": {"bitstring": {"type": "standard", "bits": 231},
                             "rook_bitstring": {"type": "rook-endgame", "bits": 35}},
                "target": {"perspective": "white", "cp_formula": "tanh(white_centipawns / score_scale_cp)",
                           "score_scale_cp": args.score_scale, "engine_mate": "signed endpoint",
                           "terminal_draw": 0, "nonterminal_draws": "retain engine evaluation"},
                "bins": {"black_large": "[-1,-0.8)", "black_advantage": "[-0.8,-0.2)",
                         "near_equal": "[-0.2,0.2]", "white_advantage": "(0.2,0.8]", "white_large": "(0.8,1]"},
                "split": {"method": "seeded SHA-256 family hash modulo 100", "proportions": [0.8, 0.1, 0.1],
                          "family": "D4 geometry and color swap, ignoring turn", "heldout_weights": False},
                "sampling": {"method": "uniform distinct-square placements conditioned on validity, then color mirrors",
                             "castling": False, "en_passant": False, "history_dependent_draws": False,
                             "augmentation": "color swap plus vertical reflection; score negated without reanalysis"},
                "code_sha256": {name: sha256_file(ROOT / name) for name in
                                ("training/generate_rook_endgame.py", "tools/board_bitstrings.py",
                                 "tools/position_types.py")},
                "software": {"python": platform.python_version(), "python_chess": chess.__version__},
                "limitations": ["Engine estimates at a fixed node budget, not calibrated win probabilities.",
                                "Uniform legal placements are not a distribution of positions from human games.",
                                "Sampling weights balance occupied bins only; sparse/empty bins remain reported.",
                                "Score diversity and rook-endgame performance do not establish transfer to other pieces."]}
    manifest = export_dataset(records, output, args.seed, metadata)
    print("Complete: " + json.dumps({"output": str(output), **manifest["distribution"],
                                     "baselines": manifest["baselines"]}, sort_keys=True), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/king_rook")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/rook_endgame")
    parser.add_argument("--engine", type=Path, help="Verified Stockfish 19 executable (optional on macOS)")
    parser.add_argument("--engine-sha256", help="Independently verified digest for a supplied executable")
    parser.add_argument("--two-rook-count", type=int, default=40000)
    parser.add_argument("--single-rook-count", type=int, default=8000)
    parser.add_argument("--bare-king-count", type=int, default=2000)
    parser.add_argument("--pilot-positions", type=int, default=2000)
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--score-scale", type=float, default=400.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, OSError, sqlite3.Error, chess.engine.EngineError) as error:
        parser.exit(1, f"Generation failed: {error}\nCompleted analyses remain in the cache; rerun the same command to resume.\n")


if __name__ == "__main__":
    main()
