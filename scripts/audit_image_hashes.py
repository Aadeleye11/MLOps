"""Find byte-identical images in the original LeafVision splits."""

import argparse
import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_splits import SPLITS, portable_path


ARCHIVES = {"ccmt": "ccmt_raw.zip", "rice": "rice_original.zip"}


def hashed_images(split_dir: Path, archive_dir: Path):
    """Yield each manifest row with a digest computed from its archive member."""
    seen_paths = set()
    with ExitStack() as files:
        archives = {
            key: files.enter_context(zipfile.ZipFile(archive_dir / name))
            for key, name in ARCHIVES.items()
        }
        member_counts = {key: Counter(z.namelist()) for key, z in archives.items()}
        for split in SPLITS:
            count = 0
            with (split_dir / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
                rows = csv.DictReader(stream, strict=True)
                if not {"path", "crop"}.issubset(rows.fieldnames or []):
                    raise ValueError(f"{split}.csv requires path and crop columns")
                for row in rows:
                    path = portable_path(row["path"] or "")
                    if path in seen_paths:
                        raise ValueError(f"Repeated manifest path in {split}: {path}")
                    seen_paths.add(path)
                    source, member = path.split("/", 1)
                    if member_counts[source][member] != 1:
                        raise ValueError(f"Expected exactly one archive member for {split}: {path}")
                    with archives[source].open(member) as image:
                        digest = hashlib.file_digest(image, "sha256").hexdigest()
                    yield digest, {"split": split, "path": path, "crop": row["crop"]}
                    count += 1
            if not count:
                raise ValueError(f"{split}.csv is empty")


def audit(split_dir: Path, archive_dir: Path) -> dict:
    by_digest = defaultdict(list)
    counts = Counter()
    for digest, row in hashed_images(split_dir, archive_dir):
        by_digest[digest].append(row)
        counts[row["split"]] += 1

    duplicates = []
    within = dict.fromkeys(SPLITS, 0)
    cross_split = 0
    for digest, members in sorted(by_digest.items()):
        if len(members) < 2:
            continue
        splits = Counter(item["split"] for item in members)
        if len(splits) > 1:
            cross_split += 1
        for split, count in splits.items():
            if count > 1:
                within[split] += 1
        duplicates.append({"sha256": digest, "members": members})
    return {
        "images_hashed": counts,
        "unique_sha256": len(by_digest),
        "within_split_duplicate_groups": within,
        "cross_split_duplicate_groups": cross_split,
        "duplicate_groups": duplicates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/archives"))
    parser.add_argument("--report", type=Path, default=Path("artifacts/image-hash-audit.json"))
    args = parser.parse_args()
    try:
        report = audit(args.split_dir, args.archive_dir)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.report.parent) as temp:
            staged = Path(temp) / "report.json"
            staged.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            staged.replace(args.report)
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as error:
        parser.exit(2, f"Audit incomplete: {error}\n")
    print(f"Hashed {sum(report['images_hashed'].values())} images. "
          f"Unique SHA-256 digests: {report['unique_sha256']}.")
    print(f"Duplicate groups within splits: {report['within_split_duplicate_groups']}")
    print(f"Duplicate groups crossing splits: {report['cross_split_duplicate_groups']}")
    print(f"Report: {args.report}")
    return int(report["cross_split_duplicate_groups"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
