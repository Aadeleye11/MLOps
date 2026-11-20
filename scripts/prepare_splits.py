"""Prepare portable crop manifests from the original train, validation and test CSVs."""

import argparse
import csv
import filecmp
import sys
import tempfile
from collections import Counter
from pathlib import Path, PurePosixPath

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import CROP_CLASSES


SPLITS = ("train", "val", "test")


def portable_path(value: str) -> str:
    path = PurePosixPath(value.removeprefix("/content/work/"))
    if (
        len(path.parts) < 2
        or path.parts[0] not in {"ccmt", "rice"}
        or ".." in path.parts
        or "\\" in value
        or "\x00" in value
        or value.endswith("/")
    ):
        raise ValueError(f"expected an image path under ccmt/ or rice/, got {value!r}")
    return path.as_posix()


def prepare_splits(input_dir: Path, output_dir: Path) -> dict[str, Counter]:
    seen = {}
    counts = {}
    duplicates = []
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temp:
        staged = Path(temp) / "splits"
        staged.mkdir()
        for split in SPLITS:
            name = f"{split}.csv"
            counts[split] = Counter()
            with (
                (input_dir / name).open(encoding="utf-8-sig", newline="") as source,
                (staged / name).open("w", encoding="utf-8", newline="") as dest,
            ):
                reader = csv.DictReader(source, strict=True)
                fields = reader.fieldnames or []
                if not {"path", "crop"}.issubset(fields) or len(set(fields)) != len(fields):
                    raise ValueError(f"{name}: expected unique columns including path and crop")
                writer = csv.DictWriter(dest, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                for row in reader:
                    location = f"{name} line {reader.line_num}"
                    if None in row or None in row.values():
                        raise ValueError(f"{location}: wrong number of columns")
                    if row["crop"] not in CROP_CLASSES:
                        raise ValueError(f"{location}: unknown crop {row['crop']!r}")
                    try:
                        path = portable_path(row["path"])
                    except ValueError as error:
                        raise ValueError(f"{location}: {error}") from error
                    if path in seen:
                        duplicates.append(f"{path}: {seen[path]} and {location}")
                    else:
                        seen[path] = location
                    row["path"] = path
                    writer.writerow(row)
                    counts[split][row["crop"]] += 1
                if not counts[split]:
                    raise ValueError(f"{name}: empty split")
        if duplicates:
            raise ValueError("Repeated image paths:\n" + "\n".join(duplicates))

        # Publish all three CSVs together. An identical previous run is reusable.
        try:
            staged.rename(output_dir)
        except OSError:
            if not output_dir.is_dir():
                raise
            for split in SPLITS:
                name = f"{split}.csv"
                target = output_dir / name
                if not target.is_file() or not filecmp.cmp(staged / name, target, shallow=False):
                    raise ValueError(f"{target} differs or is missing. Choose a new output directory.")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("artifacts/original-splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/splits"))
    args = parser.parse_args()
    try:
        counts = prepare_splits(args.input_dir, args.output_dir)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"{error}\n")
    for split, crops in counts.items():
        breakdown = " ".join(f"{crop}={crops[crop]}" for crop in CROP_CLASSES)
        print(f"{split} total={crops.total()} {breakdown}")
    print(f"Repeated image paths: 0. Manifests: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
