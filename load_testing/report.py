"""Collect the completed local trials and plot throughput and tail latency."""

import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def collect(directory):
    rows = []
    for replicas in (1, 2, 3):
        with (directory / f"trials-r{replicas}.csv").open(newline="") as stream:
            trials = list(csv.DictReader(stream))
        expected = {(str(replicas), str(users), str(trial)) for users in (1, 4, 8) for trial in (1, 2)}
        if len(trials) != 6 or {(r["replicas"], r["users"], r["trial"]) for r in trials} != expected:
            raise ValueError(f"Incomplete trial matrix for {replicas} API containers")
        rows.extend(sorted(trials, key=lambda r: (int(r["users"]), int(r["trial"]))))
    return rows


def plot(rows, destination):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    levels = [1, 4, 8]
    for ax, metric, label in zip(axes, ("rps", "p95_ms"), ("Requests per second", "p95 response time (ms)")):
        for replicas, color in enumerate(("#246144", "#1d64a3", "#a54d16"), start=1):
            pairs = [[float(r[metric]) for r in rows
                      if int(r["replicas"]) == replicas and int(r["users"]) == users] for users in levels]
            centers = [mean(values) for values in pairs]
            ax.errorbar(levels, centers, yerr=[(max(values) - min(values)) / 2 for values in pairs],
                        marker="o", capsize=4, color=color, label=f"{replicas} API")
        ax.set(xlabel="Concurrent users", ylabel=label, xticks=levels)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = collect(args.trials)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "trials.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, args.output / "scaling.png")


if __name__ == "__main__":
    main()
