import unittest

from src.registry import PromotionConflict, check_metrics


class PromotionGateTest(unittest.TestCase):
    def test_quality_rules(self):
        baseline = {"images": 50, "support": {"rice": 50}, "macro_f1": .8,
                    "accuracy": .85, "mean_cross_entropy": .3}
        improved = {**baseline, "macro_f1": .81}
        check_metrics(baseline, improved)
        for change in ({"macro_f1": .8}, {"accuracy": .84}, {"mean_cross_entropy": .31},
                       {"mean_cross_entropy": float("nan")}, {"images": 40}, {"support": {"rice": 49}}):
            with self.subTest(change=change), self.assertRaises(PromotionConflict):
                check_metrics(baseline, {**improved, **change})
