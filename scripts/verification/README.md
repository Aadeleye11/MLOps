# Development verification scripts

These are the original checks and demo launchers used during development. Run them from the repository root using the project environment. The repeatable test suite is documented in the [project README](../../docs/technical-guide.md#run-the-tests).

- `python -m scripts.verification.check_uploads` runs the tests against a disposable PostgreSQL database and checks an upload across an API restart. It checks public prediction and operator job submission using `artifacts/HL_423.jpg`, both labeled tomato JPEGs in `artifacts/retraining-inputs/`, and `embedded-postgres==18.6.3`.
- Checks that operate a live dashboard or change a live model read `LEAFVISION_OPERATOR_TOKEN` from the verification process environment. For checks inside the UI container, export the token in your shell and pass it to that one process with `docker compose exec -e LEAFVISION_OPERATOR_TOKEN ui …`. The disposable database check generates its own token.
- The dashboard and version-switch checks exercise live services. Some refer to specific batch or candidate IDs from earlier runs. Inspect their inputs and select matching records from your own database before using them.
- The Compose UI and worker checks run inside the UI container and read sample photos from `/tmp`. Several other checks use the local sample photos and model files described in the project README.
- `serve_with_uploads.py`, `serve_worker.py` and `run_retraining_demo.py` use the local development PostgreSQL directory under `artifacts/retraining-inputs/`. Use the documented Compose setup for a fresh installation.
