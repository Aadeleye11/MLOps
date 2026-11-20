"""Prepare a targeted visual review of retained test images."""

import json
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_similar_images import ARCHIVES, draw_sheet


def main():
    root = Path("artifacts")
    evaluation = json.loads((root / "crop-evaluation.json").read_text())
    retained = {row["path"]: row for row in evaluation["predictions"] if row["exclusion"] is None}
    report = json.loads((root / "similarity/pairs.json").read_text())
    candidates = [pair for pair in report["comparisons"]["train-test"]["pairs"]
                  if pair["right"] in retained]
    annotations = json.loads(Path("docs/train-test-review.json").read_text())
    candidate_paths = {(pair["left"], pair["right"]) for pair in candidates}
    reviewed_paths = {(pair["left"], pair["right"]) for pair in annotations}
    if not reviewed_paths <= candidate_paths or len(reviewed_paths) != len(annotations):
        raise ValueError("Review annotations must identify distinct retained train–test pairs")
    excluded = {pair["right"] for pair in annotations if pair["verdict"] == "likely_same_photo"}
    sensitivity_rows = [row for path, row in retained.items() if path not in excluded]
    correct = sum(row["prediction"] == row["crop"] for row in sensitivity_rows)
    groups = defaultdict(list)
    for pair in candidates:
        groups[pair["right_crop"] == "rice", pair["distance"]].append(pair)

    # Review every zero-distance non-rice pair and one example from each other group.
    selected = []
    for (rice, distance), pairs in sorted(groups.items()):
        examples = pairs if not rice and distance == 0 else pairs[len(pairs) // 2:len(pairs) // 2 + 1]
        selected.extend({"splits": "train-test", **pair} for pair in examples)

    destination = root / "similarity/train-test"
    destination.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        files = {key: stack.enter_context(zipfile.ZipFile(root / "archives" / name))
                 for key, name in ARCHIVES.items()}
        draw_sheet(files, selected, destination / "review.jpg")
    output = {"candidate_pairs": len(candidates),
              "flagged_test_images": len({pair["right"] for pair in candidates}),
              "pairs_by_distance": dict(sorted(Counter(p["distance"] for p in candidates).items())),
              "review": selected,
              "verdict_counts": dict(Counter(pair["verdict"] for pair in annotations)),
              "sensitivity": {"excluded_test_paths": sorted(excluded),
                              "images": len(sensitivity_rows), "correct": correct,
                              "accuracy": correct / len(sensitivity_rows)}}
    (destination / "review.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
