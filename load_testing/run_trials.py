"""Run one local replica configuration; pipe the API container IDs from Compose to stdin."""

import argparse
import csv
from pathlib import Path
import re
import subprocess
import sys
import time


LOCUSTFILE = Path(__file__).with_name("locustfile.py")
GROUPS = Path("/sys/fs/cgroup/system.slice")


def snapshot(groups):
    usage_usec = working_bytes = 0
    for group in groups:
        cpu = dict(line.split() for line in (group / "cpu.stat").read_text().splitlines())
        memory = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
        usage_usec += int(cpu["usage_usec"])
        working_bytes += int((group / "memory.current").read_text()) - int(memory["inactive_file"])
    return time.monotonic(), usage_usec, working_bytes


def command(users, seconds):
    return [sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
            "--host", "http://127.0.0.1:18090", "-u", str(users), "-r", str(users),
            "-t", f"{seconds}s", "--stop-timeout", "30", "--only-summary"]


def run_trial(directory, groups, users, trial):
    prefix = directory / f"r{len(groups)}-u{users}-t{trial}"
    with prefix.with_suffix(".log").open("x") as log:
        start, initial_cpu, peak_bytes = snapshot(groups)
        with subprocess.Popen(command(users, 30) + ["--csv", str(prefix)],
                              stdout=log, stderr=subprocess.STDOUT) as run:
            while run.poll() is None:
                time.sleep(1)
                end, cpu, memory = snapshot(groups)
                peak_bytes = max(peak_bytes, memory)
    if run.returncode not in (0, 1):
        raise RuntimeError(f"Locust exited with {run.returncode}; see {log.name}")
    with Path(f"{prefix}_completed.csv").open(newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if row["Name"] == "/predict")
    with Path(f"{prefix}_backends.csv").open(newline="") as stream:
        hits = list(csv.DictReader(stream))
    requests, failures = int(row["Request Count"]), int(row["Failure Count"])
    if ((run.returncode == 1 and not failures) or len(hits) != len(groups)
            or sum(int(hit["validated_responses"]) for hit in hits) != requests - failures):
        raise RuntimeError(f"Routing or Locust totals inconsistent; inspect {prefix}")
    return [len(groups), users, trial, requests, failures, row["Requests/s"],
            row["Average Response Time"], row["Median Response Time"], row["95%"], row["99%"],
            round(100 * (cpu - initial_cpu) / 1e6 / (end - start), 2),
            round(peak_bytes / 2**20, 1)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ids = sys.stdin.read().split()
    if len(set(ids)) != args.replicas or len(ids) != args.replicas or not all(
            re.fullmatch(r"[a-f0-9]{64}", cid) for cid in ids):
        parser.error("Expected one full Docker container ID per API replica on stdin")
    groups = [GROUPS / f"docker-{cid}.scope" for cid in ids]
    snapshot(groups)
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "raw"
    raw.mkdir(exist_ok=True)
    with (args.out / f"trials-r{args.replicas}.csv").open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["replicas", "users", "trial", "requests", "failures", "rps", "mean_ms",
                         "median_ms", "p95_ms", "p99_ms", "api_cpu_pct", "api_peak_working_mib"])
        for trial, levels in enumerate(((1, 4, 8), (8, 4, 1)), start=1):
            for users in levels:
                with (raw / f"r{args.replicas}-u{users}-t{trial}-warmup.log").open("x") as warmup:
                    subprocess.run(command(users, 5), check=True, stdout=warmup, stderr=subprocess.STDOUT)
                result = run_trial(raw, groups, users, trial)
                writer.writerow(result)
                stream.flush()
                print(f"replicas={args.replicas} users={users} trial={trial} "
                      f"requests={result[3]} failures={result[4]} RPS={float(result[5]):.2f} "
                      f"p95={result[8]} ms", flush=True)


if __name__ == "__main__":
    main()
