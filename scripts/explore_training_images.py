"""Measure and plot visual features of the unique LeafVision training images."""

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import zipfile
from contextlib import ExitStack
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageOps

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.audit_image_hashes import ARCHIVES
from scripts.prepare_splits import portable_path
from src import CROP_CLASSES

matplotlib.use("Agg")
from matplotlib import pyplot as plt

FEATURES = {
    "brightness": ("Image brightness", "Mean grayscale intensity (0–255)"),
    "green_dominance": ("Green dominance", "Mean green minus max(red, blue)"),
    "edge_density": ("Edge density", "Pixels with a neighbor difference ≥24 (%)"),
}


def measure(image):
    rgb = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR),
                     dtype=np.float32)
    red, green, blue = rgb.transpose(2, 0, 1)
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    base = gray[:-1, :-1]
    edge = (np.abs(gray[:-1, 1:] - base) >= 24) | (np.abs(gray[1:, :-1] - base) >= 24)
    return {
        "brightness": float(gray.mean()),
        "green_dominance": float((green - np.maximum(red, blue)).mean()),
        "edge_density": 100 * float(edge.mean()),
        "near_white": 100 * float((rgb.min(axis=2) >= 245).mean()),
    }


def read_image(files, path):
    archive, member = path.split("/", 1)
    return files[archive].read(member)


def collect(manifest, files):
    rows = []
    seen = {}
    with manifest.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, strict=True):
            crop, path = row["crop"], portable_path(row["path"])
            if crop not in CROP_CLASSES:
                raise ValueError(f"Unknown crop {crop!r} for {path}")
            data = read_image(files, path)
            digest = hashlib.sha256(data).hexdigest()
            if digest in seen:
                if seen[digest] != crop:
                    raise ValueError(f"Conflicting crop labels for SHA-256 {digest}")
                continue
            seen[digest] = crop
            with Image.open(io.BytesIO(data)) as image:
                oriented = ImageOps.exif_transpose(image)
                rows.append({"path": path, "crop": crop, "sha256": digest,
                             "width": oriented.width, "height": oriented.height,
                             **measure(oriented)})
    return rows


def summarize(rows):
    grouped = {crop: [row for row in rows if row["crop"] == crop] for crop in CROP_CLASSES}
    summary = {}
    for crop, images in grouped.items():
        if not images:
            raise ValueError(f"No training images for {crop}")
        stats = {}
        for key in (*FEATURES, "near_white", "width", "height"):
            quantiles = np.percentile([r[key] for r in images], [10, 50, 90])
            stats[key] = dict(zip(("p10", "median", "p90"), quantiles.tolist()))
        summary[crop] = {"images": len(images), **stats}
    return summary, grouped


def write_plots(grouped, destination):
    for feature, (title, units) in FEATURES.items():
        fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
        ax.boxplot(
            [[r[feature] for r in images] for images in grouped.values()],
            vert=False, whis=(10, 90), showfliers=False, patch_artist=True,
            tick_labels=[f"{crop.title()}  ·  {len(images):,}" for crop, images in grouped.items()],
            boxprops={"facecolor": "#c8e6df", "edgecolor": "#1c6555"},
            medianprops={"color": "#124638", "linewidth": 2},
        )
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold", pad=14)
        ax.set_xlabel(units)
        ax.grid(axis="x", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.text(0.99, 0.99, "Training split · one image per SHA-256",
                 ha="right", va="top", fontsize=8, color="#485855")
        fig.savefig(destination / f"{feature}.png", dpi=160)
        plt.close(fig)


def write_examples(files, grouped, summary, destination):
    fig, axes = plt.subplots(1, len(grouped), figsize=(10, 2.5), layout="constrained")
    for ax, (crop, images) in zip(axes, grouped.items(), strict=True):
        center = summary[crop]["brightness"]["median"]
        row = min(images, key=lambda item: abs(item["brightness"] - center))
        with Image.open(io.BytesIO(read_image(files, row["path"]))) as image:
            preview = ImageOps.exif_transpose(image).convert("RGB")
            preview.thumbnail((256, 256))
        ax.imshow(preview)
        ax.set_title(crop.title())
        ax.set_axis_off()
        summary[crop]["example_path"] = row["path"]
    fig.suptitle("Training examples near each crop’s median brightness")
    fig.savefig(destination / "examples.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/splits/train.csv"))
    parser.add_argument("--archive-dir", type=Path, default=Path("artifacts/archives"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training-features"))
    args = parser.parse_args()

    with ExitStack() as stack:
        files = {key: stack.enter_context(zipfile.ZipFile(args.archive_dir / name))
                 for key, name in ARCHIVES.items()}
        rows = collect(args.manifest, files)
        summary, grouped = summarize(rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=args.output_dir) as temp:
            staged = Path(temp)
            write_plots(grouped, staged)
            write_examples(files, grouped, summary, staged)
            with (staged / "features.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (staged / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                 encoding="utf-8")
            for output in staged.iterdir():
                output.replace(args.output_dir / output.name)
    print(f"Measured {len(rows):,} unique training images. Results in {args.output_dir}.")


if __name__ == "__main__":
    main()
