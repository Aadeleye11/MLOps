# LeafVision MLOps

Upload a leaf photo to identify its crop: **cashew, cassava, maize, rice, or tomato**. The dashboard also shows dataset trends and API uptime. An operator can upload labeled photos, retrain the model, and compare the candidate with the current version before promoting it.

**[Open the live LeafVision dashboard](https://leafvision-51-21-251-72.sslip.io)** for recording and grading through December 26, 2026 at 17:50 UTC. Anyone can try a leaf prediction. Labeled uploads and retraining require the operator token. The YouTube recording is pending. [Cloud demo results](docs/technical-guide.md#cloud-retraining-on-aws) · [Video script](docs/demo.md) · [GitHub repository](https://github.com/Aadeleye11/MLOps)

## Run locally

You need Docker with Compose and Python 3. From the repository root, run these commands in Bash. The first command creates local credentials on a fresh checkout.

```bash
if [ ! -e .env ]; then
  (umask 077; python3 -c 'import secrets; print("LEAFVISION_DB_PASSWORD=" + secrets.token_hex(32)); print("LEAFVISION_OPERATOR_TOKEN=" + secrets.token_hex(32))' > .env)
fi
docker compose up --build -d --wait
```

Open the [dashboard](http://127.0.0.1:18501), choose a leaf JPEG, and click **Identify crop**. The saved model is included. To try the verified example, use `HL_423.jpg` from the rice dataset. [Sample image instructions](docs/technical-guide.md#run-the-first-prediction)

To retrain, prepare the two image archives, checkpoint, and split files, then start the training worker. Enter the operator token from `.env` in the dashboard sidebar. Under **Labeled uploads**, choose a crop and save a batch. Under **Retraining**, select that batch and click **Retrain**. [Retraining setup](docs/technical-guide.md#retrain-from-the-compose-dashboard)

## How well does it work?

- **Offline evaluation:** The model correctly identified **3,883 of 3,926** test photos (98.905% accuracy, 0.99028 macro F1) after filtering exact duplicates. Cassava accounted for 29 of the 43 errors. [Evaluation notebook](notebook/leafvision_mlops.ipynb)
- **Cloud retraining:** On the same 3,933 validation photos, the candidate improved from **3,900 to 3,902** correct predictions. It qualified for promotion. The baseline continued serving during the demo. [Recorded comparison](docs/evaluation/aws-retraining.json)
- **Locust load test:** All **3,103** measured predictions passed on one laptop. At four concurrent users, adding API containers increased throughput and reduced mean response time. [All 18 trials and latency percentiles](docs/technical-guide.md#locust-prediction-load-test)

| API containers | Requests per second | Mean response time |
|---:|---:|---:|
| 1 | 3.42 | 1,166 ms |
| 2 | 7.26 | 548 ms |
| 3 | 9.86 | 404 ms |

## Explore the project

- [Original training notebook](notebook/ml_summative_notebook.ipynb) and [saved crop model](models/crop_model_v1.h5)
- [Python API and ML code](src/) and [dashboard](frontend/)
- [Technical guide](docs/technical-guide.md) for data sources, feature charts, deployment, and tests
