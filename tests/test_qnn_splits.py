"""Whole-family stratification preserves score mixtures without split leakage."""

from collections import Counter
import unittest

from training.splits import SPLITS, assign_families


class StratifiedFamilyTests(unittest.TestCase):
    def families(self):
        families = {}
        for i in range(60):
            families[f"equal-{i}"] = Counter({("KRvKR", "near_equal"): 2})
        for i in range(20):
            families[f"ordinary-{i}"] = Counter({("KRvKR", "black_ordinary"): 1,
                                                  ("KRvKR", "white_ordinary"): 1})
            families[f"mate-{i}"] = Counter({("KvKR", "black_mate"): 1,
                                              ("KRvK", "white_mate"): 1})
        return families

    def test_exact_sizes_and_matching_feature_distributions(self):
        families = self.families()
        assignment, metadata = assign_families(families)
        self.assertEqual(set(assignment), set(families))
        self.assertEqual(metadata["actual_rows"], {"train": 160, "validation": 20, "test": 20})
        self.assertTrue(metadata["exact_target_rows"])
        overall = Counter()
        actual = {split: Counter() for split in SPLITS}
        for family, counts in families.items():
            overall.update(counts)
            actual[assignment[family]].update(counts)
        for split in SPLITS:
            for key, count in overall.items():
                self.assertEqual(actual[split][key] / metadata["actual_rows"][split], count / 200)

    def test_determinism_independent_of_input_order_and_seed(self):
        families = self.families()
        first, metadata = assign_families(families, 42)
        reverse, reverse_metadata = assign_families(dict(reversed(list(families.items()))), 42)
        self.assertEqual(first, reverse)
        self.assertEqual(metadata, reverse_metadata)
        changed, _ = assign_families(families, 43)
        self.assertNotEqual(first, changed)

    def test_large_family_stays_whole_and_row_counts_are_balanced(self):
        families = {f"small-{i}": Counter({("KRvKR", "near_equal"): 2}) for i in range(100)}
        families["large"] = Counter({("KvK", "near_equal"): 18})
        assignment, metadata = assign_families(families)
        self.assertEqual(metadata["actual_rows"], {"train": 174, "validation": 22, "test": 22})
        self.assertIn(assignment["large"], SPLITS)
        self.assertGreater(metadata["size_balancing_moves"], 0)
        singleton, report = assign_families({"only-family": Counter({("KvK", "near_equal"): 18})})
        self.assertEqual(len(singleton), 1)
        self.assertEqual(sum(report["actual_rows"].values()), 18)
        self.assertEqual(sorted(report["actual_rows"].values()), [0, 0, 18])

    def test_empty_or_invalid_histograms_are_rejected(self):
        for families in ({}, {"bad": Counter()}, {"bad": Counter({("x", "y"): -1})}):
            with self.assertRaises(ValueError):
                assign_families(families)


if __name__ == "__main__":
    unittest.main()
