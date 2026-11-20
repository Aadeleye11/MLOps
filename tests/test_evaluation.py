import unittest

from src.evaluation import score, test_exclusions as select_exclusions


class EvaluationTest(unittest.TestCase):
    def test_selection_prioritizes_prior_splits_and_keeps_first_test_copy(self):
        groups = [
            {"members": [{"split": "test", "path": "z"}, {"split": "test", "path": "a"}]},
            {"members": [{"split": "test", "path": "b"}, {"split": "val", "path": "c"},
                         {"split": "test", "path": "d"}]},
            {"members": [{"split": "train", "path": "e"}, {"split": "test", "path": "f"}]},
        ]
        self.assertEqual(select_exclusions(groups), {
            "a": "repeat_in_test", "b": "shared_with_train_or_val",
            "d": "shared_with_train_or_val", "f": "shared_with_train_or_val",
        })

    def test_metrics_use_true_class_support_and_unweighted_macro_average(self):
        rows = [{"crop": crop, "prediction": predicted, "cross_entropy": loss}
                for crop, predicted, loss in [
                    ("rice", "rice", 0.1), ("rice", "maize", 1.0), ("maize", "maize", 0.4)
                ]]
        result = score(rows)
        self.assertEqual(result["support"]["rice"], 2)
        self.assertEqual(result["confusion_matrix"][3][2], 1)
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertAlmostEqual(result["macro_precision"], 1.5 / 5)
        self.assertAlmostEqual(result["macro_recall"], 1.5 / 5)
        self.assertAlmostEqual(result["macro_f1"], (2 / 3 + 2 / 3) / 5)
        self.assertAlmostEqual(result["mean_cross_entropy"], 0.5)
