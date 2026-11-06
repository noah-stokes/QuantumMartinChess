"""Generate balanced, promotion-aware 265-bit examples from legal games.

The existing 35-bit dataset and model are independent of this generator.
See data/standard/README.md for the sampling and label conventions.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import gzip
import hashlib
import json
from pathlib import Path
import platform
import random
import sqlite3
import time

import chess
import chess.engine

from tools.board_bitstrings import board_to_bitstring, bitstring_to_board
from tools.position_types import PositionType
from training.generate_rook_endgame import (
    AnalysisCache, atomic_json, evaluation_label, prepare_engine, sha256_file,
    terminal_label,
)
from training.translate_qnn_data import MAPPING_VERSION, translate_score


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")
CLASSES = ("near_equal", "moderate", "large", "mate")
SCHEMA_VERSION = 1
FULL = PositionType.SQUARE_BOARD


def encode(board):
    return board_to_bitstring(board, FULL)


def decode(bits):
    return bitstring_to_board(bits, FULL)


# All continuations start from the normal initial board. Prefixes and the
# following random legal moves provide diversity before engine-guided play.
OPENINGS = (
    "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6",
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",
    "e2e4 e7e5 g1f3 g8f6 f3e5 d7d6",
    "e2e4 c7c5 g1f3 d7d6 d2d4 c5d4",
    "e2e4 c7c5 g1f3 b8c6 d2d4 c5d4",
    "e2e4 e7e6 d2d4 d7d5 b1c3 g8f6",
    "e2e4 c7c6 d2d4 d7d5 b1c3 d5e4",
    "e2e4 d7d5 e4d5 d8d5 b1c3 d5a5",
    "e2e4 d7d6 d2d4 g8f6 b1c3 g7g6",
    "d2d4 d7d5 c2c4 e7e6 b1c3 g8f6",
    "d2d4 d7d5 c2c4 c7c6 g1f3 g8f6",
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",
    "d2d4 g8f6 c2c4 c7c5 d4d5 e7e6",
    "d2d4 d7d5 g1f3 g8f6 c1f4 e7e6",
    "c2c4 e7e5 b1c3 g8f6 g1f3 b8c6",
    "g1f3 d7d5 g2g3 g8f6 f1g2 e7e6",
    "g2g3 d7d5 f1g2 e7e5 d2d3 g8f6",
    "e2e4 e7e5 b1c3 g8f6 f2f4 d7d5",
    "d2d4 f7f5 g2g3 g8f6 f1g2 e7e6",
)


def score_class(score: float) -> str:
    magnitude = abs(score)
    if magnitude <= 0.2:
        return "near_equal"
    if magnitude <= 0.6:
        return "moderate"
    if magnitude <= 0.75:
        return "large"
    return "mate"


def quotas(positions: int) -> dict:
    """Quotas count color-mirror pairs, rather than individual records."""
    if positions < 200 or positions % 200:
        raise ValueError("Positions must be a positive multiple of 200")
    return {split: dict(zip(CLASSES, (rows // 10, rows * 3 // 20,
                                     rows * 3 // 20, rows // 10)))
            for split, rows in zip(SPLITS, (positions * 8 // 10,
                                            positions // 10, positions // 10))}


def placement_family(board: chess.Board) -> str:
    """Group turn/state variants and vertical color swaps of a placement.

    The pawnless rook dataset's rotations are invalid symmetries for pawns.
    Ignoring turn, castling and en passant here is conservatively broader
    than equality of positions, and is never used as an analysis-cache key.
    """
    placement = min(board.board_fen(), board.mirror().board_fen())
    return hashlib.sha256(placement.encode()).hexdigest()


def mirror_bits(bits: str) -> str:
    """Color swap and vertical reflection preserve pawn move direction."""
    return encode(decode(bits).mirror())


def analyze_position(engine, board, nodes):
    terminal = terminal_label(board)
    if terminal is not None:
        return {**terminal, "best_move": None}
    engine.configure({"Clear Hash": None})
    last_exact = None
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), game=object(),
                         info=chess.engine.INFO_SCORE | chess.engine.INFO_BASIC
                         | chess.engine.INFO_PV) as analysis:
        for info in analysis:
            if "score" in info and not info.get("lowerbound") and not info.get("upperbound"):
                last_exact = info
    if last_exact is None or not last_exact.get("pv"):
        raise ValueError(f"No exact engine evaluation/principal variation: {board.fen()}")
    return {**evaluation_label(last_exact, 400),
            "best_move": last_exact["pv"][0].uci()}


def play_game(task, binary: str, cache_path: Path, configuration: dict) -> dict:
    game_id, split, focus = task
    rng = random.Random(f"standard-game-v1:{configuration['seed']}:{game_id}")
    with closing(AnalysisCache(cache_path, configuration)) as cache:
        previous = cache.connection.execute(
            "SELECT payload FROM games WHERE game_id=?", (game_id,)).fetchone()
        if previous:
            game = json.loads(gzip.decompress(previous[0]))
            if (game["split"], game["focus"]) != (split, focus):
                raise ValueError("Cached game schedule differs; use a different --cache-dir")
            return game
        board = chess.Board()
        moves = []
        for uci in rng.choice(OPENINGS).split():
            board.push_uci(uci)
            moves.append(uci)
        for _ in range(rng.randrange(7)):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            moves.append(move.uci())
        weak_color = rng.choice((chess.WHITE, chess.BLACK))
        strong_noise, weak_noise = {
            "near_equal": (0.015, 0.015), "moderate": (0.04, 0.12),
            "large": (0.02, 0.25), "mate": (0.01, 0.40),
        }[focus]
        # Clocks/repetitions are excluded from these position-only labels.
        board = decode(encode(board))
        candidates, promotions = [], []
        with chess.engine.SimpleEngine.popen_uci(binary, timeout=60) as engine:
            engine.configure({"Threads": 1, "Hash": 32, "SyzygyPath": "",
                              "UCI_LimitStrength": False})
            for _ in range(configuration["max_plies"] - len(moves) + 1):
                bits = encode(board)
                board = decode(bits)
                fen = board.fen(en_passant="fen")
                cached = cache.get(fen)
                if cached is None:
                    started = time.monotonic()
                    label = analyze_position(engine, board, configuration["nodes"])
                    # Two game workers can reach the same opening position.
                    try:
                        cache.put(fen, label, time.monotonic() - started)
                    except sqlite3.IntegrityError:
                        cache.connection.rollback()
                        label, _ = cache.get(fen)
                else:
                    label, _ = cached
                target, kind = translate_score(label)
                if len(moves) % 2 == game_id % 2 or label["termination"]:
                    candidates.append({"bitstring": bits, "fen": fen,
                                       "ply": len(moves), "label": label,
                                       "target": target, "kind": kind,
                                       "promotion_types": sorted(set(promotions))})
                if label["termination"]:
                    ended = label["termination"]
                    break
                if len(moves) >= configuration["max_plies"]:
                    ended = "ply_limit"
                    break
                move = chess.Move.from_uci(label["best_move"])
                noise = weak_noise if board.turn == weak_color else strong_noise
                # Follow a found mate line; random mistakes are introduced
                # before a mate is found, rather than corrupting mate labels.
                if kind == "centipawn" and rng.random() < noise:
                    move = rng.choice(list(board.legal_moves))
                if move.promotion:
                    # Include ordinary promotions and each underpromotion.
                    piece = rng.choices((chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT),
                                        weights=(7, 1, 1, 1))[0]
                    move = chess.Move(move.from_square, move.to_square, promotion=piece)
                    promotions.append(chess.piece_name(piece))
                board.push(move)
                moves.append(move.uci())
            else:
                ended = "ply_limit"
        game = {"game_id": game_id, "split": split, "focus": focus,
                "moves": moves, "ended": ended, "candidates": candidates}
        cache.connection.execute("INSERT INTO games VALUES (?, ?)", (
            game_id, gzip.compress(json.dumps(game, separators=(",", ":")).encode(), mtime=0)))
        cache.connection.commit()
        return game


def make_pair(candidate: dict, game_id: int) -> tuple[dict, dict]:
    bits, label = candidate["bitstring"], candidate["label"]
    board = decode(bits)
    base = {key: value for key, value in label.items()
            if key not in ("target_score", "best_move")}
    base.update({"bitstring": bits, "fen": candidate["fen"],
                 "target_score": candidate["target"], "score_category": candidate["kind"],
                 "source_game_id": game_id, "source_ply": candidate["ply"],
                 "promotion_types": candidate["promotion_types"],
                 "family_id": f"game:{game_id}", "placement_family": placement_family(board),
                 "source_fen": candidate["fen"], "augmentation": "identity"})
    mirrored = {**base, "bitstring": mirror_bits(bits),
                "target_score": -base["target_score"] if base["target_score"] else 0.0,
                "augmentation": "color_swap_vertical"}
    mirrored["fen"] = decode(mirrored["bitstring"]).fen(en_passant="fen")
    if base["raw_engine_evaluation"] is not None:
        raw = dict(base["raw_engine_evaluation"])
        raw["value"] = -raw["value"]
        if "winner" in raw:
            raw["winner"] = "black" if raw["winner"] == "white" else "white"
        mirrored["raw_engine_evaluation"] = raw
    return base, mirrored


def distribution(rows: list[dict]) -> dict:
    bins, kinds, phases, turns, promotions = Counter(), Counter(), Counter(), Counter(), Counter()
    for row in rows:
        value = row["target_score"]
        group = score_class(value)
        name = group if group == "near_equal" else ("white_" if value > 0 else "black_") + group
        bins[name] += 1
        kinds[row["score_category"]] += 1
        board = chess.Board(row["fen"])
        pieces = chess.popcount(board.occupied)
        phases["opening" if pieces >= 28 else "middlegame" if pieces > 12 else "endgame"] += 1
        turns["white" if board.turn else "black"] += 1
        promotions.update(row["promotion_types"])
    return {"positions": len(rows), "score_bins": dict(sorted(bins.items())),
            "categories": dict(sorted(kinds.items())), "phase_counts": dict(phases),
            "turn_counts": dict(turns), "castling_rights_count": sum("1" in r["bitstring"][257:261] for r in rows),
            "en_passant_count": sum(r["bitstring"][261:] != "0000" for r in rows),
            "post_promotion_count": sum(bool(r["promotion_types"]) for r in rows),
            "post_promotion_types": dict(promotions),
            "exact_endpoint_count": sum(abs(r["target_score"]) == 1 for r in rows)}


def validate_dataset(splits: dict, games: dict, requested: dict) -> dict:
    inputs, placements, game_splits = set(), {}, {}
    pair_count = 0
    for split, rows in splits.items():
        observed = Counter()
        for index in range(0, len(rows), 2):
            left, right = rows[index:index + 2]
            if mirror_bits(left["bitstring"]) != right["bitstring"]:
                raise ValueError("Missing color-mirror partner")
            if left["target_score"] != -right["target_score"]:
                raise ValueError("Color-mirror targets are not antisymmetric")
            observed[score_class(left["target_score"])] += 1
            pair_count += 1
        if observed != +Counter(requested[split]):
            raise ValueError(f"Score quotas differ in {split}: {observed}")
        for row in rows:
            bits = row["bitstring"]
            if bits in inputs:
                raise ValueError("Duplicate input")
            inputs.add(bits)
            board = decode(bits)
            if encode(board) != bits or board.fen(en_passant="fen") != row["fen"]:
                raise ValueError("Encoding round trip failed")
            if translate_score(row) != (row["target_score"], row["score_category"]):
                raise ValueError("Score mapping differs from the existing dataset")
            terminal = terminal_label(board)
            if (abs(row["target_score"]) == 1) != board.is_checkmate():
                raise ValueError("Only actual checkmates may have endpoint targets")
            if terminal and terminal["termination"] != row["termination"]:
                raise ValueError("Terminal label mismatch")
            placement = placement_family(board)
            if placement != row["placement_family"] or placements.setdefault(placement, split) != split:
                raise ValueError("Equivalent placement leaks between splits")
            game_id = row["source_game_id"]
            if game_splits.setdefault(game_id, split) != split:
                raise ValueError("A game leaks between splits")
            if row["augmentation"] == "identity":
                game = games[game_id]
                if game["split"] != split or row["source_ply"] not in game["snapshots"]:
                    raise ValueError("Missing legal-game provenance")
                if game["snapshots"][row["source_ply"]] != bits:
                    raise ValueError("Input disagrees with the replayed legal game")
    return {"records_checked": len(inputs), "mirror_pairs": pair_count,
            "encoding_round_trips": "passed", "raw_score_mapping": "passed",
            "legal_game_replays": "passed", "score_quotas": "passed",
            "unique_placements": len(placements), "games": len(game_splits),
            "game_overlap": 0, "placement_family_overlap": 0}


def replay_game(game: dict, selected_plies: set[int]) -> dict:
    board, snapshots = chess.Board(), {}
    for ply, uci in enumerate(game["moves"], 1):
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError(f"Invalid game provenance: game {game['game_id']}, ply {ply}")
        board.push(move)
        if ply in selected_plies:
            snapshots[ply] = encode(board)
    return {"split": game["split"], "moves": game["moves"], "ended": game["ended"],
            "snapshots": snapshots}


def write_gzip(path: Path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    temporary.replace(path)


def export_dataset(splits, games, output, metadata):
    qnn = output / "qnn"
    qnn.mkdir(parents=True, exist_ok=True)
    source_files, files = {}, {}
    for split, rows in splits.items():
        ordered = sorted(rows, key=lambda row: row["bitstring"])
        source = output / f"{split}.jsonl.gz"
        write_gzip(source, ordered)
        source_files[split] = {"path": source.name, "sha256": sha256_file(source),
                               "bytes": source.stat().st_size, **distribution(rows)}
        path = qnn / f"{split}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps([{"input": row["bitstring"], "output": row["target_score"]}
                                         for row in ordered], indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        files[split] = {"path": path.name, "sha256": sha256_file(path),
                        "bytes": path.stat().st_size, **distribution(rows)}
    game_path = output / "games.jsonl.gz"
    write_gzip(game_path, ({"game_id": game_id, "split": game["split"],
                            "moves": game["moves"], "ended": game["ended"]}
                           for game_id, game in sorted(games.items())))
    all_rows = [row for rows in splits.values() for row in rows]
    summary = distribution(all_rows)
    baselines = {split: {"zero_prediction_mse": sum(r["target_score"] ** 2 for r in rows) / len(rows)}
                 for split, rows in splits.items() if split != "train"}
    manifest = {**metadata, "distribution": summary, "files": source_files,
                "games_file": {"path": game_path.name, "sha256": sha256_file(game_path)},
                "baselines": baselines}
    atomic_json(output / "manifest.json", manifest)
    atomic_json(qnn / "manifest.json", {**metadata, "distribution": summary, "files": files,
                                        "baselines": baselines,
                                        "source_manifest_sha256": sha256_file(output / "manifest.json")})
    return manifest


def generate(args):
    requested = quotas(args.positions)
    if args.nodes < 1 or args.workers < 1 or args.max_plies < 20 or args.max_games < 1:
        raise ValueError("Nodes, workers and max-games must be positive; max-plies must be at least 20")
    output, cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary, provenance = prepare_engine(cache_dir / "stockfish", args.engine, args.engine_sha256)
    configuration = {"schema_version": SCHEMA_VERSION, "seed": args.seed,
                     "nodes": args.nodes, "workers": args.workers, "threads_per_engine": 1,
                     "hash_mib": 32, "max_plies": args.max_plies, "positions": args.positions,
                     "clear_search_state_per_position": True, "syzygy_path": "",
                     "binary_sha256": provenance["binary_sha256"],
                     "generator_sha256": sha256_file(Path(__file__)),
                     "encoding_sha256": sha256_file(ROOT / "tools/board_bitstrings.py"),
                     "mapping_sha256": sha256_file(ROOT / "training/translate_qnn_data.py")}
    cache_path = cache_dir / "analyses.sqlite3"
    with closing(AnalysisCache(cache_path, configuration)) as cache:
        cache.connection.execute("CREATE TABLE IF NOT EXISTS games (game_id INTEGER PRIMARY KEY, payload BLOB NOT NULL)")
        cache.connection.commit()
    with chess.engine.SimpleEngine.popen_uci(str(binary), timeout=60) as engine:
        engine_id = dict(engine.id)
        if not engine_id.get("name", "").startswith("Stockfish 19"):
            raise ValueError(f"Expected Stockfish 19, got {engine_id}")
    remaining = {split: dict(counts) for split, counts in requested.items()}
    splits, games, seen = {split: [] for split in SPLITS}, {}, set()
    game_id, considered, previous_count = 0, 0, 0
    started = time.monotonic()
    print(f"Engine: {engine_id}; target {args.positions:,} rows; {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while any(sum(counts.values()) for counts in remaining.values()):
            if game_id >= args.max_games:
                raise ValueError(f"Reached --max-games with quotas unfilled: {remaining}")
            tasks = []
            for offset in range(min(args.workers, args.max_games - game_id)):
                index = game_id + offset
                rng = random.Random(f"standard-schedule-v1:{args.seed}:{index}")
                split = rng.choices(SPLITS, weights=[sum(remaining[s].values()) for s in SPLITS])[0]
                focus = rng.choices(CLASSES, weights=[remaining[split][c] for c in CLASSES])[0]
                tasks.append((index, split, focus))
            results = list(executor.map(lambda task: play_game(task, str(binary), cache_path, configuration), tasks))
            for game in results:
                split = game["split"]
                selected = set()
                for candidate in game["candidates"]:
                    considered += 1
                    category = score_class(candidate["target"])
                    if not remaining[split][category]:
                        continue
                    family = placement_family(chess.Board(candidate["fen"]))
                    if family in seen:
                        continue
                    seen.add(family)
                    splits[split].extend(make_pair(candidate, game["game_id"]))
                    remaining[split][category] -= 1
                    selected.add(candidate["ply"])
                if selected:
                    games[game["game_id"]] = replay_game(game, selected)
            game_id += len(tasks)
            count = sum(len(rows) for rows in splits.values())
            if count - previous_count >= 1000 or count == args.positions or game_id % 40 == 0:
                progress = {"accepted_rows": count, "target_rows": args.positions,
                            "completed_games": game_id, "candidate_pairs": considered,
                            "elapsed_seconds": round(time.monotonic() - started, 1),
                            "remaining_pairs": remaining}
                atomic_json(cache_dir / "progress.json", progress)
                print(json.dumps(progress, sort_keys=True), flush=True)
                previous_count = count
    print("Checking every encoding, label, game replay, quota and split...", flush=True)
    checks = validate_dataset(splits, games, requested)
    metadata = {"schema_version": SCHEMA_VERSION, "mapping_version": MAPPING_VERSION,
                "seed": args.seed, "engine": engine_id, "engine_provenance": provenance,
                "analysis_configuration": configuration, "encoding": {"type": "square-board", "bits": 265,
                    "square_order": "a1, b1, ... h1, a2, ... h8", "piece_codes": {
                        "empty": 0, "white_pawn": 1, "white_knight": 2, "white_bishop": 3,
                        "white_rook": 4, "white_queen": 5, "white_king": 6,
                        "black_pawn": 7, "black_knight": 8, "black_bishop": 9,
                        "black_rook": 10, "black_queen": 11, "black_king": 12}},
                "target": {"perspective": "white", "cp_formula": "0.75 * tanh(cp / 400)",
                           "mate_formula": "sign(m) * (0.75 + 0.25 * 0.9 ** abs(m))",
                           "checkmate": "signed endpoint", "terminal_draw": 0},
                "score_bins": {"near_equal": "[-0.2, 0.2]",
                               "moderate": "0.2 < abs(score) <= 0.6",
                               "large": "0.6 < abs(score) <= 0.75",
                               "mate": "0.75 < abs(score) <= 1"},
                "split": {"method": "quota-stratified-games-v1", "proportions": [0.8, 0.1, 0.1],
                          "whole_games": True, "placement_family_overlap": 0,
                          "color_mirrors_together": True, "requested_pair_quotas": requested},
                "sampling": {"method": "legal opening prefixes followed by engine/random legal play",
                             "candidate_interval_plies": 2, "random_move_probabilities": {
                                 "near_equal": [0.015, 0.015], "moderate": [0.04, 0.12],
                                 "large": [0.02, 0.25], "mate": [0.01, 0.40]},
                             "completed_games": game_id, "candidate_pairs": considered,
                             "promotions": "continue play; queen/rook/bishop/knight weights 7/1/1/1",
                             "history_dependent_draws": False,
                             "augmentation": "color swap plus vertical reflection; score negated"},
                "validation": checks, "software": {"python": platform.python_version(), "chess": chess.__version__},
                "limitations": ["Fixed-node engine estimates, not tablebase proofs or win probabilities.",
                                "Mate distances are engine-reported, not guaranteed shortest.",
                                "Synthetic mixed-strength games with quotas, not a human-game distribution.",
                                "Repetition history and move clocks are outside this input format.",
                                "The existing QNN prototype accepts 35-bit inputs; a 265-bit model is separate work."]}
    manifest = export_dataset(splits, games, output, metadata)
    print("Complete: " + json.dumps({"output": str(output), **manifest["distribution"]}, sort_keys=True), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=50000)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/standard")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/standard")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--engine-sha256")
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-plies", type=int, default=180)
    parser.add_argument("--max-games", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        generate(args)
    except (ValueError, OSError, sqlite3.Error, chess.engine.EngineError) as error:
        parser.exit(1, f"Generation failed: {error}\nCompleted games/analyses remain cached; rerun the same command to resume.\n")


if __name__ == "__main__":
    main()
