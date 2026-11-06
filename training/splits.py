"""Deterministic 80/10/10 stratification of whole position families.

Families with the same material/score-band histogram form a stratum. Allocate
families proportionally inside each stratum, then move whole families to
balance row counts while minimizing changes to the feature distributions.
No chess engine, quantum package, or third-party numerical library is needed.
"""

from collections import Counter, defaultdict
from functools import reduce
import hashlib
import math
from pathlib import Path


SPLITS = ("train", "validation", "test")
WEIGHTS = (8, 1, 1)
METHOD = "group-stratified-v1"


def apportion(total: int) -> list[int]:
    """Largest-remainder allocation; deterministic ties use split order."""
    counts = [total * weight // 10 for weight in WEIGHTS]
    remaining = total - sum(counts)
    order = sorted(range(3), key=lambda i: (-(total * WEIGHTS[i] % 10), i))
    for i in order[:remaining]:
        counts[i] += 1
    return counts


def assign_families(families: dict[str, Counter], seed: int = 42) -> tuple[dict[str, str], dict]:
    if not families or any(not histogram or any(n < 1 for n in histogram.values())
                           for histogram in families.values()):
        raise ValueError("Stratification requires nonempty families with positive histogram counts")
    strata = defaultdict(list)
    totals = Counter()
    for family, histogram in families.items():
        signature = tuple(sorted(histogram.items()))
        strata[signature].append(family)
        totals.update(histogram)
    assigned, pools = {}, {}
    sizes = [0, 0, 0]
    features = [Counter(), Counter(), Counter()]
    for signature in sorted(strata):
        ordered = sorted(strata[signature], key=lambda family: (
            hashlib.sha256(f"{METHOD}:{seed}:{family}".encode()).digest(), family))
        pools[signature] = [[], [], []]
        offset = 0
        for i, count in enumerate(apportion(len(ordered))):
            pools[signature][i] = ordered[offset:offset + count]
            offset += count
            for family in pools[signature][i]:
                assigned[family] = i
                sizes[i] += sum(families[family].values())
                features[i].update(dict(signature))
    size_gcd = reduce(math.gcd, (sum(hist.values()) for hist in families.values()))
    targets = [count * size_gcd for count in apportion(sum(sizes) // size_gcd)]
    feature_targets = [{key: count * WEIGHTS[i] / 10 for key, count in totals.items()}
                       for i in range(3)]

    def change(i, signature, direction):
        return sum(((features[i][key] + direction * n - feature_targets[i][key]) ** 2
                    - (features[i][key] - feature_targets[i][key]) ** 2)
                   / max(1, feature_targets[i][key]) for key, n in signature)

    moves = 0
    while sizes != targets:
        candidates = []
        for origin in range(3):
            for destination in range(3):
                if sizes[origin] <= targets[origin] or sizes[destination] >= targets[destination]:
                    continue
                room = min(sizes[origin] - targets[origin], targets[destination] - sizes[destination])
                for signature, pool in pools.items():
                    size = sum(n for _, n in signature)
                    if pool[origin] and size <= room:
                        cost = change(origin, signature, -1) + change(destination, signature, 1)
                        candidates.append((cost, origin, destination, signature, size))
        if not candidates:
            # Some tiny datasets cannot meet row targets without splitting a
            # family. Keep the family intact and report the actual sizes.
            break
        _, origin, destination, signature, size = min(candidates)
        family = pools[signature][origin].pop()
        pools[signature][destination].append(family)
        assigned[family] = destination
        sizes[origin] -= size
        sizes[destination] += size
        features[origin].subtract(dict(signature))
        features[destination].update(dict(signature))
        moves += 1
    return {family: SPLITS[i] for family, i in assigned.items()}, {
        "method": METHOD, "seed": seed,
        "proportions": dict(zip(SPLITS, (0.8, 0.1, 0.1))),
        "target_rows": dict(zip(SPLITS, targets)), "actual_rows": dict(zip(SPLITS, sizes)),
        "exact_target_rows": sizes == targets, "families": len(families),
        "family_size_gcd": size_gcd, "histogram_strata": len(strata),
        "size_balancing_moves": moves,
        "stratification": "material class and five mate-band score bins, grouped by family_id",
        "family_policy": "D4 symmetries, color swaps, and turn variants stay together",
        "splitter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
