# Reproduce the local prediction benchmark

Run from the repository root on a Linux Docker host with systemd cgroup v2. The trial runner reads per-container CPU and memory counters in `/sys/fs/cgroup/system.slice/docker-<id>.scope`. Docker Compose and the benchmark run on the same host. Prepare the [original labeled rice JPEG](../docs/technical-guide.md#run-the-first-prediction) at `artifacts/HL_423.jpg`. The workload verifies its SHA-256 before use.

Create a separate Python 3.12 environment for Locust. The plotting command uses the project's regular Python environment with Matplotlib installed.

```bash
python3.12 -m venv .venv/locust
.venv/locust/bin/python -m pip install -r load_testing/requirements.txt
```

The example below uses one isolated Compose project, the same baseline image at every scale, and an API proxy bound to localhost on port 18090. Set `LEAFVISION_DB_PASSWORD` as described in the [Compose setup](../docs/technical-guide.md#run-the-dashboard-api-and-postgresql-with-docker-compose). Only the proxy, API and PostgreSQL services are needed. Keep the test machine available for the full matrix.

```bash
export COMPOSE_PROJECT_NAME=leafvision-locust
export LEAFVISION_API_PORT=18090
mkdir -p artifacts
run_dir=$(mktemp -d artifacts/locust-XXXXXX)
(
  set -e -o pipefail
  docker compose build api
  for replicas in 1 2 3; do
    docker compose up --detach --wait --scale "api=$replicas" proxy
    docker compose ps -q api | .venv/locust/bin/python load_testing/run_trials.py \
      --replicas "$replicas" --out "$run_dir"
  done
)
# After all three configurations have finished:
.venv/bin/python load_testing/report.py "$run_dir" "$run_dir/report"
docker compose down
```

Each virtual user continuously submits one multipart JPEG to `POST /predict` and waits for its response before the next upload. A request succeeds when the API returns HTTP 200, the rice label, the expected baseline version and score within tolerance, and a proxy backend identifier. Every replica must serve a validated response in every measured run.

At each replica count the runner tests 1, 4 and 8 users, then repeats in reverse order. Each trial has a separate five-second warm-up at its target concurrency, followed by a fresh 30-second measured run. Users spawn at the configured user count per second. Locust allows in-flight requests up to 30 seconds to finish. The average CPU percentage sums the API cgroups (100% means one logical CPU). Peak memory sums their working sets (current usage less inactive file cache) with one-second sampling over the Locust process lifetime. Both exclude the generator, proxy and database.

`trials.csv` has one row per measured run, including p99 latency and sampled API resources. The chart shows the average of the two trial values with whiskers spanning their observed range. For per-request statistics and backend counts, see the raw Locust exports in the Git-ignored `$run_dir/raw/` directory.
