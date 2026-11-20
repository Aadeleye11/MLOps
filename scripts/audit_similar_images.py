"""Screen cross-split image pairs by perceptual difference hash."""

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_image_hashes import ARCHIVES
from scripts.prepare_splits import SPLITS, portable_path


MAX_DISTANCE = 4


@dataclass(frozen=True)
class Photo:
    path: str
    crop: str
    sha256: bytes
    dhash: int


def difference_hash(image: Image.Image) -> int:
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = list(gray.resize((9, 8), Image.Resampling.LANCZOS).getdata())
    result = 0
    for y in range(8):
        for x in range(8):
            i = y * 9 + x
            result = (result << 1) | (pixels[i] > pixels[i + 1])
    return result


def read_bytes(files, path):
    source, member = path.split("/", 1)
    return files[source].read(member)


def read_photos(manifests, files):
    photos = {}
    for split in SPLITS:
        photos[split] = []
        with (manifests / f"{split}.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                path = portable_path(row["path"])
                data = read_bytes(files, path)
                with Image.open(io.BytesIO(data)) as image:
                    fingerprint = difference_hash(image)
                photos[split].append(Photo(path, row["crop"], hashlib.sha256(data).digest(), fingerprint))
        if not photos[split]:
            raise ValueError(f"{split}.csv is empty")
        print(f"Hashed {len(photos[split])} {split} images", flush=True)
    return photos


def bands(value):
    # At most four bit differences leave at least one of five disjoint bands equal.
    for shift in (0, 13, 26, 39, 52):
        yield shift, (value >> shift) & 0x1fff


def find_pairs(left, right):
    index = defaultdict(list)
    for position, photo in enumerate(right):
        for key in bands(photo.dhash):
            index[key].append(position)
    pairs = []
    exact = 0
    for a in left:
        candidates = {i for key in bands(a.dhash) for i in index.get(key, ())}
        for i in candidates:
            b = right[i]
            distance = (a.dhash ^ b.dhash).bit_count()
            if a.sha256 == b.sha256:
                exact += 1
            elif distance <= MAX_DISTANCE:
                pairs.append({"left": a.path, "right": b.path, "distance": distance,
                              "left_crop": a.crop, "right_crop": b.crop})
    pairs.sort(key=lambda pair: (pair["distance"], pair["left"], pair["right"]))
    return pairs, exact


def draw_sheet(files, rows, destination):
    sheet = Image.new("RGB", (680, max(1, len(rows)) * 200), "white")
    draw = ImageDraw.Draw(sheet)
    if not rows:
        draw.text((10, 10), "No candidates at this threshold", fill="black")
    for number, pair in enumerate(rows, 1):
        y = (number - 1) * 200
        draw.text((8, y + 2), f"{number}. {pair['splits']}    distance {pair['distance']}", fill="black")
        for x, side in ((0, "left"), (340, "right")):
            with Image.open(io.BytesIO(read_bytes(files, pair[side]))) as image:
                thumb = ImageOps.exif_transpose(image).convert("RGB")
                thumb.thumbnail((330, 155))
                sheet.paste(thumb, (x + (340 - thumb.width) // 2, y + 21))
            draw.text((x + 5, y + 180), pair[side + "_crop"], fill="black")
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, default=Path("artifacts/splits"))
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/similarity"))
    args = parser.parse_args()
    with ExitStack() as stack:
        files = {key: stack.enter_context(zipfile.ZipFile(args.archive_dir / name))
                 for key, name in ARCHIVES.items()}
        photos = read_photos(args.split_dir, files)
        comparisons = {}
        review = []
        for a, b in combinations(SPLITS, 2):
            key = f"{a}-{b}"
            pairs, exact = find_pairs(photos[a], photos[b])
            comparisons[key] = {"exact_pairs": exact, "candidate_count": len(pairs), "pairs": pairs}
            if pairs:
                for i in sorted({0, len(pairs) // 2, len(pairs) - 1}):
                    review.append({"splits": key, **pairs[i]})
            print(f"{key}: {len(pairs)} candidates, {exact} byte-identical pairs", flush=True)
        report = {"method": "64-bit horizontal dHash", "max_hamming_distance": MAX_DISTANCE,
                  "images": {k: len(v) for k, v in photos.items()},
                  "comparisons": comparisons, "review_sample": review}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
            staged = Path(temp)
            draw_sheet(files, review, staged / "review.jpg")
            (staged / "pairs.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            for name in ("review.jpg", "pairs.json"):
                (staged / name).replace(args.output_dir / name)


if __name__ == "__main__":
    main()
