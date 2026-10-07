"""Tests for the pre-registered ablation ladder analysis.

The analysis is pre-registered, so it has to be right before any result is fed
through it. These tests pin down the parts that would silently corrupt a
conclusion: the Holm correction, the TOST equivalence logic, and above all the
rule that a non-significant difference is never reported as parity.
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.analyze_ablation_ladder import (
    ACCURACY_EQUIVALENCE_MARGIN_PP,
    ALPHA,
    ContrastResult,
    _holm,
    _rung_key,
    _tost_paired,
    analyze_configuration,
    compare,
    load_runs,
)

SEEDS = tuple(range(42, 62))


def _run(algorithm, cache, seed, accuracy, seconds=100.0, energy=10.0, memory=500.0):
    return {
        "experiment_name": f"{algorithm}_{cache}",
        "algorithm": algorithm,
        "dataset": "MNIST",
        "architecture": [1000, 1000],
        "activation_cache": cache,
        "seed": seed,
        "test_accuracy": accuracy,
        "training_duration_sec": seconds,
        "total_gpu_energy_wh": energy,
        "peak_torch_alloc_mib": memory,
    }


class TestHolmCorrection(unittest.TestCase):
    def test_matches_hand_computed_values(self) -> None:
        # m=4. Sorted: .01 -> x4, .02 -> x3, .03 -> x2, .04 -> x1, then step-down.
        adjusted = _holm([0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(adjusted[0], 0.04)
        self.assertAlmostEqual(adjusted[1], 0.06)
        self.assertAlmostEqual(adjusted[2], 0.06)
        self.assertAlmostEqual(adjusted[3], 0.06)

    def test_is_monotone_non_decreasing(self) -> None:
        raw = [0.001, 0.2, 0.04, 0.5, 0.009]
        adjusted = _holm(raw)
        by_raw = [adjusted[i] for i in np.argsort(raw)]
        for earlier, later in zip(by_raw, by_raw[1:]):
            self.assertLessEqual(earlier, later)

    def test_never_exceeds_one(self) -> None:
        for value in _holm([0.4, 0.5, 0.6, 0.99]):
            self.assertLessEqual(value, 1.0)

    def test_preserves_input_order(self) -> None:
        adjusted = _holm([0.5, 0.001])
        self.assertGreater(adjusted[0], adjusted[1])

    def test_ignores_missing_p_values(self) -> None:
        adjusted = _holm([0.01, math.nan, 0.02])
        self.assertTrue(math.isnan(adjusted[1]))
        # Only two real tests, so the smallest is multiplied by two, not three.
        self.assertAlmostEqual(adjusted[0], 0.02)


class TestTOST(unittest.TestCase):
    def test_tight_data_inside_the_margin_is_equivalent(self) -> None:
        differences = np.full(20, 0.01) + np.linspace(-0.001, 0.001, 20)
        tost_p, equivalent = _tost_paired(differences, ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertTrue(equivalent)
        self.assertLess(tost_p, ALPHA)

    def test_a_real_difference_is_not_equivalent(self) -> None:
        rng = np.random.default_rng(0)
        differences = rng.normal(loc=2.0, scale=0.1, size=20)
        _, equivalent = _tost_paired(differences, ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertFalse(equivalent)

    def test_noisy_data_centred_on_zero_is_not_equivalent(self) -> None:
        # THE CENTRAL GUARD. The mean difference is ~0 so a difference test will
        # not reject, but the noise is far wider than the margin. Calling this
        # 'equivalent' is exactly the error the pre-registration forbids.
        rng = np.random.default_rng(1)
        differences = rng.normal(loc=0.0, scale=3.0, size=20)
        _, equivalent = _tost_paired(differences, ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertFalse(equivalent)

    def test_identical_measurements_are_equivalent(self) -> None:
        tost_p, equivalent = _tost_paired(np.zeros(20), ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertTrue(equivalent)
        self.assertEqual(tost_p, 0.0)

    def test_constant_offset_larger_than_margin_is_not_equivalent(self) -> None:
        offset = np.full(20, ACCURACY_EQUIVALENCE_MARGIN_PP * 2)
        _, equivalent = _tost_paired(offset, ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertFalse(equivalent)

    def test_too_few_pairs_cannot_establish_equivalence(self) -> None:
        _, equivalent = _tost_paired(np.array([0.0]), ACCURACY_EQUIVALENCE_MARGIN_PP)
        self.assertFalse(equivalent)


class TestVerdictVocabulary(unittest.TestCase):
    """A non-significant p-value must never be reported as parity."""

    @staticmethod
    def _contrast(paired_holm, welch_holm, equivalent):
        return ContrastResult(
            metric="test_accuracy",
            family="accuracy",
            rung_a="bp_ds",
            rung_b="mf_recompute",
            n_pairs=20,
            mean_a=98.0,
            mean_b=98.0,
            sd_a=0.1,
            sd_b=0.1,
            paired_p=paired_holm,
            welch_p=welch_holm,
            wilcoxon_p=paired_holm,
            paired_p_holm=paired_holm,
            welch_p_holm=welch_holm,
            equivalent=equivalent,
        )

    def test_non_significant_without_tost_is_inconclusive(self) -> None:
        self.assertEqual(self._contrast(0.6, 0.7, False).verdict, "INCONCLUSIVE")

    def test_non_significant_with_tost_is_equivalent(self) -> None:
        self.assertEqual(self._contrast(0.6, 0.7, True).verdict, "EQUIVALENT")

    def test_significant_is_different(self) -> None:
        self.assertEqual(self._contrast(0.001, 0.002, False).verdict, "DIFFERENT")

    def test_significant_but_inside_margin_is_flagged_as_negligible(self) -> None:
        self.assertEqual(
            self._contrast(0.001, 0.002, True).verdict, "DIFFERENT-BUT-NEGLIGIBLE"
        )

    def test_the_conservative_test_decides(self) -> None:
        # Paired says significant, Welch does not. We must not claim a difference.
        self.assertEqual(self._contrast(0.001, 0.40, False).verdict, "INCONCLUSIVE")

    def test_parity_is_never_claimed_without_tost(self) -> None:
        for paired in (0.06, 0.2, 0.5, 0.99):
            self.assertNotEqual(
                self._contrast(paired, paired, False).verdict, "EQUIVALENT"
            )


class TestRungIdentification(unittest.TestCase):
    def test_each_rung_maps_to_a_distinct_key(self) -> None:
        cases = {
            ("bp", "recompute"): "bp",
            ("BP_DS", "recompute"): "bp_ds",
            ("MF_JOINT", "recompute"): "mf_joint",
            ("MF", "recompute"): "mf_recompute",
            ("MF", "cache_device"): "mf_cache_device",
            ("MF", "cache_host"): "mf_cache_host",
        }
        keys = set()
        for (algorithm, cache), expected in cases.items():
            record = {"algorithm": algorithm, "activation_cache": cache}
            self.assertEqual(_rung_key(record), expected)
            keys.add(expected)
        self.assertEqual(len(keys), 6, "The six rungs must not collide.")

    def test_unrelated_algorithms_are_not_rungs(self) -> None:
        for algorithm in ("ff", "cafo"):
            self.assertIsNone(
                _rung_key({"algorithm": algorithm, "activation_cache": "recompute"})
            )


class TestContrastStatistics(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(7)

    def _runs_by_rung(self, accuracies_a, accuracies_b):
        return {
            "bp_ds": {
                seed: _run("bp_ds", "recompute", seed, value)
                for seed, value in zip(SEEDS, accuracies_a)
            },
            "mf_recompute": {
                seed: _run("mf", "recompute", seed, value)
                for seed, value in zip(SEEDS, accuracies_b)
            },
        }

    def test_detects_a_genuine_accuracy_gap(self) -> None:
        base = 98.0 + self.rng.normal(0, 0.05, len(SEEDS))
        result = compare(
            self._runs_by_rung(base, base - 1.5),
            "bp_ds",
            "mf_recompute",
            "test_accuracy",
            "accuracy",
            self.rng,
        )
        self.assertLess(result.paired_p, ALPHA)
        self.assertFalse(result.equivalent)
        self.assertLess(result.difference_ci.high, 0.0)

    def test_declares_equivalence_when_rungs_really_do_match(self) -> None:
        base = 98.0 + self.rng.normal(0, 0.03, len(SEEDS))
        result = compare(
            self._runs_by_rung(base, base + 0.01),
            "bp_ds",
            "mf_recompute",
            "test_accuracy",
            "accuracy",
            self.rng,
        )
        self.assertTrue(result.equivalent)
        self.assertEqual(result.tost_margin, ACCURACY_EQUIVALENCE_MARGIN_PP)

    def test_wide_noise_yields_inconclusive_not_equivalent(self) -> None:
        a = 98.0 + self.rng.normal(0, 2.0, len(SEEDS))
        b = 98.0 + self.rng.normal(0, 2.0, len(SEEDS))
        result = compare(
            self._runs_by_rung(a, b),
            "bp_ds",
            "mf_recompute",
            "test_accuracy",
            "accuracy",
            self.rng,
        )
        result.paired_p_holm = result.paired_p
        result.welch_p_holm = result.welch_p
        self.assertEqual(result.verdict, "INCONCLUSIVE")

    def test_confidence_interval_brackets_the_observed_difference(self) -> None:
        base = 98.0 + self.rng.normal(0, 0.1, len(SEEDS))
        result = compare(
            self._runs_by_rung(base, base - 0.8),
            "bp_ds",
            "mf_recompute",
            "test_accuracy",
            "accuracy",
            self.rng,
        )
        self.assertLessEqual(result.difference_ci.low, result.difference_ci.point)
        self.assertLessEqual(result.difference_ci.point, result.difference_ci.high)
        self.assertAlmostEqual(result.difference_ci.point, -0.8, places=6)

    def test_ratio_interval_is_reported_for_timing(self) -> None:
        runs = {
            "bp": {
                seed: _run("bp", "recompute", seed, 98.0, seconds=100.0)
                for seed in SEEDS
            },
            "mf_recompute": {
                seed: _run("mf", "recompute", seed, 98.0, seconds=50.0)
                for seed in SEEDS
            },
        }
        result = compare(runs, "bp", "mf_recompute", "training_duration_sec", "time", self.rng)
        self.assertIsNotNone(result.ratio_ci)
        self.assertAlmostEqual(result.ratio_ci.point, 0.5, places=6)
        # A 5 percent band around a 100 second reference.
        self.assertAlmostEqual(result.tost_margin, 5.0, places=6)

    def test_unpaired_seeds_are_dropped(self) -> None:
        runs = self._runs_by_rung([98.0] * len(SEEDS), [98.0] * len(SEEDS))
        del runs["mf_recompute"][SEEDS[0]]
        result = compare(
            runs, "bp_ds", "mf_recompute", "test_accuracy", "accuracy", self.rng
        )
        self.assertEqual(result.n_pairs, len(SEEDS) - 1)

    def test_missing_rung_produces_no_contrast(self) -> None:
        runs = self._runs_by_rung([98.0] * len(SEEDS), [98.0] * len(SEEDS))
        self.assertIsNone(
            compare(
                runs, "bp_ds", "mf_cache_host", "test_accuracy", "accuracy", self.rng
            )
        )

    def test_identical_measurements_read_as_p_one_not_missing(self) -> None:
        # scipy hands back nan when both samples have zero variance. If that
        # leaked through, Holm would drop the test and the contrast would vanish
        # from the report instead of being recorded as a perfect match.
        result = compare(
            self._runs_by_rung([98.0] * len(SEEDS), [98.0] * len(SEEDS)),
            "bp_ds",
            "mf_recompute",
            "test_accuracy",
            "accuracy",
            self.rng,
        )
        self.assertEqual(result.paired_p, 1.0)
        self.assertEqual(result.welch_p, 1.0)
        self.assertEqual(result.wilcoxon_p, 1.0)
        result.paired_p_holm = result.paired_p
        result.welch_p_holm = result.welch_p
        self.assertEqual(result.verdict, "EQUIVALENT")


class TestEndToEnd(unittest.TestCase):
    def test_reads_engine_summaries_and_reports_every_family(self) -> None:
        rng = np.random.default_rng(11)
        records = []
        for algorithm, cache in (
            ("bp", "recompute"),
            ("bp_ds", "recompute"),
            ("mf_joint", "recompute"),
            ("mf", "recompute"),
        ):
            for seed in SEEDS:
                records.append(
                    _run(
                        algorithm,
                        cache,
                        seed,
                        98.0 + rng.normal(0, 0.1),
                        seconds=100.0 + rng.normal(0, 5.0),
                        energy=10.0 + rng.normal(0, 0.5),
                        memory=500.0 + rng.normal(0, 2.0),
                    )
                )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, record in enumerate(records):
                path = root / record["experiment_name"] / f"run{index}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(record), encoding="utf-8")

            loaded = load_runs(root)
            self.assertEqual(len(loaded), len(records))

            report = analyze_configuration("mnist_[1000, 1000]", loaded, rng)

        self.assertEqual(len(report.seeds), len(SEEDS))
        self.assertEqual(
            report.rungs_present, ["bp", "bp_ds", "mf_joint", "mf_recompute"]
        )
        self.assertIn("test_accuracy", report.observed_sd)
        families = {c.family for c in report.contrasts}
        self.assertEqual(families, {"accuracy", "time", "energy", "memory"})
        for contrast in report.contrasts:
            self.assertFalse(math.isnan(contrast.paired_p_holm))
            self.assertGreaterEqual(contrast.paired_p_holm, contrast.paired_p)

    def test_failed_runs_are_excluded(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = _run("bp", "recompute", 42, 98.0)
            bad = _run("bp", "recompute", 43, float("nan"))
            bad["error"] = "CUDA out of memory"
            (root / "a.json").write_text(json.dumps(good), encoding="utf-8")
            (root / "b.json").write_text(json.dumps(bad), encoding="utf-8")
            self.assertEqual(len(load_runs(root)), 1)


if __name__ == "__main__":
    unittest.main()
