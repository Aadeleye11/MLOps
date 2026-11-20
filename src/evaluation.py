"""Selection and scoring for the frozen crop checkpoint."""

import numpy as np

from src import CROP_CLASSES


def test_exclusions(duplicate_groups: list) -> dict[str, str]:
    excluded = {}
    for group in duplicate_groups:
        members = group["members"]
        test_paths = [m["path"] for m in members if m["split"] == "test"]
        # The archive audit lists members in their original manifest order.
        if len(test_paths) == len(members):
            excluded.update(dict.fromkeys(test_paths[1:], "repeat_in_test"))
        else:
            excluded.update(dict.fromkeys(test_paths, "shared_with_train_or_val"))
    return excluded


def score(rows: list) -> dict:
    if not rows:
        raise ValueError("Cannot score an empty evaluation set")
    matrix = np.zeros((len(CROP_CLASSES), len(CROP_CLASSES)), dtype=int)
    for row in rows:
        matrix[CROP_CLASSES.index(row["crop"]), CROP_CLASSES.index(row["prediction"])] += 1
    correct = matrix.diagonal()
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    # A class with zero denominator contributes zero to the five-class macro mean.
    precision = correct / np.maximum(predicted, 1)
    recall = correct / np.maximum(support, 1)
    f1 = 2 * correct / np.maximum(predicted + support, 1)
    return {
        "images": len(rows),
        "correct": int(correct.sum()),
        "accuracy": float(correct.sum() / len(rows)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "mean_cross_entropy": float(np.mean([row["cross_entropy"] for row in rows])),
        "support": dict(zip(CROP_CLASSES, support.tolist())),
        "confusion_matrix": matrix.tolist(),
    }
