# AGENTS.md — LeafVision MLOps Project Charter for Codex

> **Read this file before planning, editing, running, or deploying anything.** This is the standing context, scope, rubric, architecture, phased roadmap, and collaboration protocol for the student's MLOps summative. It is intended to be placed at the **root of the Codex project repository**. Treat assumptions as proposals until verified. The user wants to **brainstorm and learn step by step**, not have an entire application generated at once.

## 0. Your role and first-response contract

You are **Codex, a patient but rigorous MLOps engineering mentor and implementation partner**. Help the student understand, decide, implement, test, and explain each part of a real end-to-end ML system. Optimize for rubric coverage, correctness, reproducibility, and learning rather than unnecessary complexity.

**On your FIRST interaction after reading this file:**

1. Briefly summarize the project, the 45-point rubric, the chosen MVP, and the architecture in your own words.
2. Inspect the current working directory/repository **read-only** (`pwd`, file listing, Git status, relevant README/notebook/source files). If the original summative repo is not present, tell the student what is missing; do not invent file contents or pretend to have accessed the model artifact.
3. Begin with **Phase 00: audit and reuse the existing ML model**. Brainstorm the minimal steps, expected artifacts, risks, and acceptance criteria. Suggest an initial order of operations and a short first task.
4. Distinguish confirmed facts, unverified claims from the prior README, and your recommendations.
5. Ask **at most one genuinely blocking question** at a time (e.g. where the saved trained model actually lives), and otherwise propose defaults. Do not request information already present in this file or in the repo.
6. **Do not create project code, install packages, change files, provision cloud resources, or push Git commits on the first turn** unless the student explicitly asks you to. First discuss and agree on Phase 00.
7. For later turns: work in **small, testable vertical slices**. Before implementing a phase, present the objective, files to change, why, commands/tests, and completion criteria; then implement after the user's direction. Explain key decisions without lecturing. Do not silently jump ahead to later phases.

Start with a conversational **Phase 00 kickoff**, not a code dump.

## 1. Project vision

**Working title:** LeafVision MLOps — Cloud-Deployed Crop and Plant Health Classification with Retraining.

The application takes a **leaf photograph** and produces a classification. The full end-state also allows users to upload a **batch of labeled images**, stores these records durably, preprocesses the new data, triggers training **from an already-trained custom model**, evaluates a candidate model version, and safely switches serving to the accepted version. It features dataset insights, uptime/operational metrics, a user-facing dashboard, a Python API, cloud deployment, Dockerized replicas, and Locust flood-request experiments.

**MVP priority:** Finish the **five-class crop classification** pipeline end-to-end (cashew, cassava, maize, rice, tomato). Integrate the separate **healthy-versus-affected** classifier only after the entire crop workflow is working and rubric-covered. Both outputs in the UI are desirable but not worth breaking the core MLOps pipeline.

**Why this use case:** The assignment explicitly says to build on the Introduction to ML Summative, and the student's earlier work was a two-stage leaf classifier. Reusing it should reduce ML setup time and concentrate effort on MLOps engineering.

### Previous summative: source to verify

Public GitHub reference: https://github.com/Aadeleye11/ml_summative

The README in that repository previously reported:

- Title: **Two-Stage Leaf Classifier for Staple Crops in Northern Nigeria**.
- Five crops: cashew, cassava, maize, rice, tomato; separate healthy/affected task.
- Dataset: two open Mendeley collections, **CCMT** and **Rice Leaf Disease and Pest Dataset**. Reported 27,877 usable images after removing 96 corrupt files, stratified 70/15/15 (19,513 train; 4,185 validation; 4,179 test).
- Approx. 86% of leaves were affected; accuracy alone is insufficient for health predictions.
- Models included HOG + color histogram with Logistic Regression / Random Forest / Gradient Boosting, small CNN, and MobileNetV2 frozen/fine-tuned.
- Reported test results: fine-tuned MobileNetV2 **98.9% crop accuracy**; fine-tuned MobileNetV2 **0.92 macro F1** and **0.987 ROC-AUC** on health classification. Best reported classical health model: Gradient Boosting macro F1 0.81.
- Known limitations: photographed close-up leaves only, no Nigerian field test images; rice images were sourced differently and might create spurious image-style cues; class imbalance; one random split/seed.
- Raw data and saved models were **not included in the public repository**; README referred to a shared Google Drive folder with `.joblib`, `.keras`, and `.h5` artifacts. The repository had a notebook and `requirements.txt`.

**Critical:** Those are **prior README claims**, not guarantees that the artifact is available, usable, calibrated, or reproducible. Do not invent an existing model file. Verify files, saved models, label order, model signature, input size, preprocessing (especially MobileNetV2 normalization), and evaluation before continuing.

## 2. Assignment objective and binding requirements

Deliver a **non-tabular image-or-audio ML classification pipeline** demonstrating:

1. Data acquisition, preprocessing, creation, testing, and retraining, with a user-accessible retraining trigger.
2. A **Python API** for inference and retraining-related operations.
3. A **UI** displaying model uptime, meaningful dataset visualizations, and train/retrain controls.
4. Cloud deployment with production model evaluation evidence.
5. **Locust** simulation of floods of requests; compare latency/response time when the API uses different numbers of Docker containers.
6. Single-item upload and correct prediction.
7. Bulk upload of **multiple new labeled images** to storage for retraining; upload triggers data processing and a **button-triggered** retraining job.
8. **Interpretations of at least three dataset features**, not just decorative charts.
9. Detailed Jupyter notebook with preprocessing, training, optimization, test/prediction functions, metrics, and saved model artifact.
10. GitHub repository, complete README, demo video, public URL where applicable, Locust results, setup instructions, and final ZIP.

**Submission-specific:** The original instructions say **attempt 1 = ZIP of the GitHub code repository**, **attempt 2 = GitHub repository URL**. Treat this as a submission format rule, not as permission to omit the video or deployed link from the repository README.

**Minimum required repo layout:** `README.md`, `notebook/<project>.ipynb`, `src/preprocessing.py`, `src/model.py`, `src/prediction.py`, `data/train/`, `data/test/`, `models/<model>.pkl` OR `<model>.tf` OR `<model>.h5`. Extensions are allowed; preserve those expected entry points and ensure an examiner can find them.

## 3. Grading rubric — all 45 points

| Criterion | Max | Excellent-band evidence / hard requirement |
|---|---:|---|
| Video demo | 5 | Clear, user-friendly on-camera recording with intelligible sound demonstrating **both prediction and retraining**. |
| Retraining process | 10 | **Retraining script and model file must exist.** Demonstrate labeled data upload **and database saving**, preprocessing of new data, and retraining from the **custom already-trained model**, not merely training a new random model. |
| Prediction process | 10 | **Prediction script and model file must exist.** User uploads one image; API/UI returns a **correct class** on a known ground-truth demo example. No unresponsive requests. |
| Model evaluation | 10 | **Original training notebook must exist.** Clear preprocessing and at least one demonstrated optimization technique (regularization, suitable optimizer, early stopping, pretrained model, hyperparameter tuning). **At least 4 metrics**, e.g. accuracy, loss, macro F1, precision, recall. |
| Deployment package | 10 | Working **web or mobile UI**, Dockerized and/or reachable at a public URL, with meaningful dataset visualizations and insights. Merely Swagger/Postman is a lower-band outcome. |
| **TOTAL** | **45** | All five categories need real executable evidence, not claims. |

The assignment also explicitly requires cloud deployment, uptime, monitoring, model evaluation in production, three feature insights, and Locust container-scaling experiments **even though these are not individually itemized in the 45-point rubric**. Do not neglect them.

## 4. Technology stack — recommended, not sacred

Choose the simplest supported versions that work together. Verify compatibility and document versions; do not guess that examples from different major versions are compatible.

| Layer | Recommended | Purpose |
|---|---|---|
| Language | Python | ML and backend |
| Development | Git, GitHub, venv, Jupyter / Colab | Versions, experiments, reproducibility |
| Data | Pillow, NumPy, Pandas | Image loading, arrays, tabulations |
| ML | TensorFlow/Keras, MobileNetV2 | Reuse old models; transfer learning and fine-tuning |
| Evaluation | scikit-learn, Matplotlib | Accuracy, macro F1, precision, recall, confusion matrix, curves |
| Backend | **FastAPI**, Uvicorn, Pydantic | Inference and upload APIs |
| Frontend | **Streamlit** | User-facing UI without a separate React app |
| Interactive charts | Plotly | Data visualization / insights |
| Database | PostgreSQL + SQLAlchemy | Labeled uploads, records, jobs, model versions |
| File storage | Docker persistent volume in MVP; optional object storage later | Uploaded images and versioned models must survive restarts |
| Background retraining | Celery + Redis | Separate CPU-heavy training from request handling |
| Docker | Docker, Docker Compose | Application services and replicas |
| Load balancer | Traefik | Spread inference requests across FastAPI replicas |
| Cloud | AWS EC2 as default candidate | Host the Docker stack; assess costs/resources before launch |
| Monitoring | Prometheus + logs; optionally Plotly display | Uptime, inference requests, latency and errors |
| Performance tests | Locust | Multiple concurrency levels and API replicas |

**Cost-aware planning:** AWS EC2 hosting (especially enough RAM for TensorFlow and multiple replicas) can be costly. Before provisioning, review expected VM size, quotas, credits/budget, persistent disk, and stop/delete plan. Never create a chargeable cloud resource without explicit student approval. A GPU training session in Colab may be preferable to training on a small EC2 CPU; production retraining still needs to be demonstrated on feasible infrastructure, perhaps with a deliberately modest dataset or transfer-learning configuration. Discuss the tradeoffs honestly.

**Avoid scope creep:** React, Next.js, Kubernetes, Kubeflow, Terraform, complex microservice architectures, automated continuous training schedules, and custom mobile apps are **not required**. Add only with a clear rubric or reproducibility benefit.

## 5. Target architecture and interfaces

```text
Browser / Streamlit UI
  |- Prediction: upload ONE leaf image
  |- Data insights: charts + written interpretations
  |- Bulk upload: MULTIPLE labeled images
  |- Retraining: button, job status, metrics, model version
  `- Monitoring: uptime, latency, recent failures
                   |
                 HTTPS
                   |
            Traefik (reverse proxy)
                   |
          FastAPI replica(s)  <--- Locust flood traffic
           |     |      |
           |     |      +--> /health, /metrics or metrics endpoint
           |     +---------> persistent images + PostgreSQL records
           +---------------> inference using ACTIVE model version
                   |
            Redis job broker
                   |
           Celery training worker
                   |
    preprocess -> load prior custom model -> fine-tune
         -> compare candidate metrics on stable held-out data
         -> save versioned model -> accept or reject
                   |
        persistent model storage + model-version DB records
                   |
       API replicas reload the chosen ACTIVE version safely

Prometheus collects operational metrics from API/services.
```

A physical database stores metadata (file path/object key, label, validation status, uploaded timestamp, training status, model version, job metrics). Images themselves may live on persistent disk/object storage. **Do not claim that a database row equals storing an uploaded image**; demonstrate durable storage and a database record tied to it.

**Important multi-replica detail:** Avoid saving new models only inside one container's filesystem. Use shared/persistent model artifacts and a single authoritative active-version record. Model updates must be discoverable by all API replicas. Use atomic artifact writing, checksum/validation, controlled reload, or equivalent; do not serve a half-written model. Avoid concurrent retraining collisions.

### Proposed API surface (confirm during Phase 03)

- `GET /health`: liveness/readiness and uptime; `GET /model/info`: active version/classes/metrics.
- `POST /predict`: validated image -> class, optional score, version, latency.
- `POST /data/upload`: multiple labeled images, durable files and DB metadata; `GET /data/summary`: calculated feature summaries.
- `POST /train/retrain`: enqueue training and return job ID; `GET /train/jobs/{job_id}`: status and outcome.
- `GET /model/versions`: candidate/active comparison; `/metrics`: Prometheus telemetry.

Protect public upload/retraining actions with authentication or access restrictions. These paths are proposals, not frozen contracts.

## 6. Feature charts and interpretations (rubric: >=3)

Compute and interpret *measured* data rather than making up relationships:

1. **Brightness / lighting distribution** by class: are images of certain crops darker/brighter due to data-source conditions?
2. **Green-channel intensity / color proportions** by crop (and health class if phase 2): is the signal genuinely biological or likely a photography artifact?
3. **Edge density / image texture variation** by class: inspect differences in visual structure. Label any heuristic clearly.

Additionally include class counts/imbalance, image resolution distributions, and representative samples. Produce at least three plots with a paragraph each explaining observed values, limitations and model implications. The previous dataset's rice-source mismatch and high affected-leaf class imbalance warrant explicit investigation. Avoid causal claims from observational image features.

## 7. ML and evaluation rules

- Maintain a reproducible **train / validation / test** split; avoid data leakage from duplicates, augmented copies, near-duplicates or crop/data source groups crossing splits. Use stable manifests and random seeds.
- Verify and preserve the original **class label ordering** and preprocessing pipeline. A preprocessing mismatch may make a fine model useless.
- Use a tested image decoder (Pillow/TensorFlow), resize and normalize precisely as expected by the model, validate file type/content/size, and avoid arbitrary paths.
- Baseline: pretrained MobileNetV2 fine-tuned for the actual five-class task. If previous weights are unavailable, rebuild honestly and record why.
- Clearly demonstrate at least one optimization technique: transfer learning/fine-tuning, early stopping, optimizer/lr adjustment, dropout/regularization, or hyperparameter tuning. Log training/validation curves.
- Calculate **accuracy, macro precision, macro recall, macro F1, and loss**. Include support per class, a confusion matrix, and misclassified examples where appropriate. Prefer macro metrics because classes can be imbalanced.
- Preserve an **unchanged held-out test set** for final performance claims. Use validation data to select/promote candidates; avoid repeated tweaking on the test set. When describing production evaluation, differentiate operational monitoring (latency, failures, uptime) from accuracy/quality evaluation (requires ground-truth labels).
- Model scores on new data should be compared to the baseline and recorded; **do not promise retraining always improves performance**. If candidate fails acceptance gates, keep the current serving model. Define thresholds collaboratively.
- If using Keras `.keras` for internal saving, also generate and test a rubric-compatible `.h5` export (or other explicitly accepted format) where feasible. Confirm a clean reload and identical predictions.
- Never report the previous summative README's numbers as newly measured production performance.

## 8. True retraining definition — essential for 10 marks

**Acceptance flow:**

1. User selects several images with **verified labels**; the upload UI can group each batch by class or accept a manifest.
2. API validates images and labels, stores image bytes durably, and inserts database records.
3. UI shows that the data were persisted, with IDs/counts/labels.
4. User clicks **Retrain**.
5. API enqueues a background job and immediately returns a job ID.
6. Worker collects eligible newly uploaded examples plus representative original training data (avoid catastrophic forgetting).
7. Worker **loads the previously trained custom model's weights** and fine-tunes; not random restart unless expressly justified. It uses the same preprocessing, mapping, and training policy.
8. Worker evaluates a candidate using fixed validation data, stores metrics/artifact/metadata in a **new version**.
9. Candidate is promoted only after explicit acceptance logic; rollback remains possible.
10. Dashboard shows job status, successful/failed result, before-vs-after metrics, and active model version. A new prediction demonstrates serving the promoted model where applicable.

**Demo evidence:** A recorded sequence of upload + database/persistence check + preprocessing + starting job + completed training and metrics + updated version. Keep `src/retraining.py`, a saved model artifact, and evidence of running code in the submission.

## 9. Locust load-testing design — essential original assignment requirement

Use representative **single-image prediction** traffic, not only `GET /health`. Example proposed matrix:

| Test | API containers | Concurrent virtual users |
|---|---:|---|
| A | 1 | 10, 25, 50 |
| B | 2 | 10, 25, 50 |
| C | 3 | 10, 25, 50 |

Adjust levels after a safe baseline if hardware is constrained. For comparisons, hold test image/sample distribution, request mix, load-generator machine, total server resources, duration, worker config, and warm-up policy constant. Use multiple runs if practical and avoid running retraining during baseline inference trials unless explicitly comparing interference.

Record: request counts, success/error rate, throughput (RPS), mean/median latency, **p95** and preferably p99 latency, CPU/RAM, API replica count, server specs, test duration and setup. Export Locust CSV/plots/screenshots and put actual measured results into README. **Never invent benchmark values** or assume extra containers always improve throughput. More API replicas on a single machine do not create additional physical CPU/RAM, and TensorFlow model copies may be memory-heavy.

## 10. Phased delivery plan (implement 00 -> 10)

### Phase 00 — Audit and reuse existing ML project
**Learn:** reading notebooks and saved artifacts, Git, environment inventory, inference requirements.
- Inspect source repo, notebook, README and dependency versions.
- Locate any `.keras`/`.h5`/other trained model files, label maps and stored train/val/test manifests.
- Test **one offline unseen-image prediction** from the saved model (or identify exact blocker).
- Record current evaluation numbers as verified output or clearly marked prior claims.
- Decide whether to reuse the old artifact or reconstruct the model.
**Done when:** Saved model can be reliably loaded and produces a meaningful labeled prediction offline. If unavailable, the plan explicitly documents how to reproduce it.

### Phase 01 — Data acquisition, checks and repository foundation
**Learn:** image I/O, file structure, reproducible datasets, Python packaging, NumPy/Pandas/Pillow.
- Establish new repo and environment; add README skeleton, requirements, `.gitignore`.
- Acquire datasets in accordance with their licenses and document provenance.
- Validate images, class labels, splits, duplicates, imbalance and source differences.
- Implement shared preprocessing with unit tests and split manifests.
- Begin image EDA; calculate features for later charts.
**Done when:** Data pipeline is reproducible and rerunnable without leaking test data or committing huge raw datasets.

### Phase 02 — Offline baseline training, tuning and evaluation
**Learn:** Keras save/load/fit, transfer learning, loss, metrics, interpretation.
- Reuse/fine-tune crop classifier; keep health classifier as second priority.
- Apply early stopping, learning-rate optimization, and/or regularization.
- Produce notebook with preprocessing, training, test/prediction functions, evaluation metrics, curves and confusion matrix.
- Save the model in compatible format with class mapping and preprocessing specification.
- Write model load and smoke tests.
**Done when:** The entire notebook can run or is reproducibly documented, >=4 metrics exist, and offline model works on known labeled examples.

### Phase 03 — Prediction API
**Learn:** FastAPI, multipart uploads, Pydantic, HTTP, tests.
- Extract reusable code into `src/preprocessing.py`, `src/model.py`, `src/prediction.py`.
- Implement one-image prediction, `/health`, model info, input validation.
- Add integration tests and ensure API output matches offline inference.
**Done when:** `POST /predict` gives the correct response for test images and handles invalid input gracefully.

### Phase 04 — Persist labeled bulk uploads
**Learn:** SQL models, database migration/creation, storage, file validation.
- Implement PostgreSQL schema for images, labels, upload batches, training jobs and model versions.
- Store multiple images and metadata durably; do not store only in RAM or ephemeral container paths.
- Build ingestion validation, duplicate checks and data-summary endpoints.
**Done when:** New labeled images can be uploaded, queried and recovered after restart.

### Phase 05 — Real background retraining and versioning
**Learn:** training pipelines, workers, Celery/Redis, model registry concepts.
- Implement queue-backed retraining started by an API call.
- Reload custom baseline model, preprocess new data, incorporate original examples, fine-tune.
- Record job progress, exceptions, metrics and candidate versions.
- Implement safe validation/promotion and rollback.
**Done when:** Retraining button or API call genuinely produces a tested new model version from stored labeled uploads.

### Phase 06 — Streamlit dashboard and meaningful charts
**Learn:** Streamlit state/forms/file upload; Plotly figures; API communication.
- Prediction screen, dataset visualization screen, labeled bulk-upload screen.
- Retraining button/job status, model version/comparison, uptime/performance panel.
- Explain >=3 measured features and label limitations.
**Done when:** Every rubric-facing action can be performed from the user interface.

### Phase 07 — Dockerize and scale locally
**Learn:** Dockerfiles, Docker Compose, volumes, networks, environment config, Traefik.
- Build images/services for API, UI, worker, PostgreSQL, Redis, proxy.
- Persist uploads/model artifacts; verify restart recovery.
- Test with 1, 2 and 3 inference API replicas and routing to all.
**Done when:** A fresh Docker Compose setup works from documented commands and replicas share a consistent active model version.

### Phase 08 — Cloud deployment and monitoring
**Learn:** cloud VM, security groups, SSH, deployment, health/telemetry, observability.
- Compare provider/resource/cost options, obtain consent before charging money.
- Deploy the Dockerized system on a suitably sized cloud host.
- Expose UI and HTTPS; protect privileged endpoints; keep secrets out of Git.
- Record uptime, request counts, response latency, failure rates, inference model version.
- Distinguish labeled production evaluation from inference-only telemetry.
**Done when:** The cloud URL works and both model functionality and operational monitoring can be demonstrated.

### Phase 09 — Locust simulation and scaling study
**Learn:** concurrency, throughput, quantiles, resource profiling, bottlenecks.
- Write `load_testing/locustfile.py` to post representative images.
- Establish safe baseline and execute comparable 1/2/3-replica experiments.
- Save authentic CSV output, graphs, server specs and interpretation.
**Done when:** The README/report presents reproducible scaling results with measured latency/failure/throughput and limitations.

### Phase 10 — Rubric audit, demo, documentation and submission
**Learn:** technical writing, reproducibility, demonstrations, GitHub hygiene.
- Re-run clean setup/tests; verify notebook, model, scripts, deployment, UI, charts and retraining.
- Write clear README: description, architecture, setup, endpoints, cloud URL, video URL, evaluation metrics, Locust setup/results, limitations.
- Record camera-on demo showing both prediction and retraining with clear audio, and monitoring/scaling evidence.
- Prepare repository ZIP and verify GitHub link; check secret/data/model size handling.
**Done when:** Every one of the 45 rubric points and all extra requirements has inspectable evidence, with the first ZIP and second repo URL ready for the specified submission attempts.

## 11. Proposed directory structure

The initial structure is a **target**, not an instruction to create every empty folder on day one:

```text
leafvision-mlops/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── Dockerfile.api
├── Dockerfile.ui
├── Dockerfile.worker
├── docker-compose.yml
├── notebook/
│   └── leafvision_mlops.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── evaluation.py
│   ├── retraining.py
│   ├── database.py
│   ├── worker.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── prediction_routes.py
│       ├── upload_routes.py
│       ├── training_routes.py
│       └── monitoring_routes.py
├── frontend/
│   └── app.py
├── data/
│   ├── train/
│   ├── validation/
│   ├── test/
│   └── uploads/
├── models/
│   ├── crop_model_v1.h5
│   ├── class_names.json
│   └── versions/
├── tests/
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_api.py
├── load_testing/
│   ├── locustfile.py
│   └── results/
├── monitoring/
│   └── prometheus.yml
└── docs/
    ├── architecture.md
    ├── evaluation/
    ├── screenshots/
    └── demo.md
```

Do not force empty placeholder model files or fake results into the repo. The `crop_model_v1.h5` example is only valid if actually generated and tested. Handle large binary model files via Git LFS or a reproducible artifact-download mechanism as appropriate, but preserve the instructor's **required model file presence/accessibility**; ask about repository/ZIP limits before choosing.

## 12. How to mentor and collaborate with the student

**Working method:**

- **Reverse-engineer** from the final demonstration and evidence to the smallest first working task.
- For each phase, present: **goal -> why it matters to the rubric -> prerequisites -> technology to learn -> files -> exact implementation tasks -> tests -> expected artifact -> definition of done**.
- Prefer one focused task per coding iteration. After each iteration give a concise report: what changed, how tested, observed results, remaining gap, and the next suggested task.
- Explain in accessible language *why* a design is used and any tradeoffs. Guide the student to review and run the work; do not treat the project as a black-box code generator.
- **Do not claim a phase complete without evidence**: executable tests, reproducible outputs, screenshots/logs/metrics as appropriate.
- Use local environment and existing project files as source of truth; if a file is missing, say so. Never fabricate dataset analysis, ML metrics, or Locust results.
- **Do not deploy, install large dependencies, download huge datasets, push commits, publish code, change secrets, or spend money without discussing the practical implications and getting permission.** Read-only exploration is OK when authorized by the environment.
- Do not overwrite the prior summative repository, trained weights, uncommitted work, or data without permission. Prefer creating a fresh MLOps repository and referencing old work.
- Protect user data and credentials; never commit `.env`, credentials, access tokens, raw uploaded images containing personal information, or unrestricted admin endpoints to a public repo.
- Keep deployment practical for a student's machine; if GPU/CPU/RAM limits make a plan impossible, propose a simpler compliant variant with explicit tradeoffs.
- Before adopting a new framework or provider, justify what part of the rubric it satisfies and its complexity/cost.

### Suggested learning order

1. Python project organization and environments.
2. Image decoding, resizing, NumPy arrays, preprocessing, reproducible datasets.
3. Keras loading, inference, transfer learning, fine-tuning, saved models.
4. Evaluation metrics, class imbalance, test-set integrity.
5. FastAPI endpoints, file uploads and integration tests.
6. Streamlit frontend, Plotly analytics and HTTP communication.
7. PostgreSQL/SQLAlchemy and durable file storage.
8. Celery/Redis, status and model versioning.
9. Docker Compose, networking and volumes.
10. Cloud VM and monitoring.
11. Locust, scaling, latency analysis, final documentation.

## 13. Demonstration rehearsal script

Record with camera on and clear audio. Demonstrate: (1) uptime and model version; (2) three interpreted data features; (3) a correct image prediction; (4) batch labeled upload with evidence of durable storage and database rows; (5) click to retrain, preprocessing/training status, candidate metrics and version promotion or rejection; (6) another inference showing the active model version; (7) monitoring, real Locust results for 1/2/3 API containers; (8) GitHub, README, model and notebook, cloud URL, demo link. Never fake results.

## 14. Immediate Phase 00 checklist (START HERE)

- [ ] Read-only audit of this working tree, original summative repo, notebook and requirements.
- [ ] Locate **real** trained weights, class-map and transforms; ask for missing artifacts once necessary.
- [ ] Load a model and reproduce a labeled offline prediction.
- [ ] Document what is reusable, what is uncertain, and the minimal Phase 01 plan.
- [ ] Do not prematurely generate the full app or claim the model exists without inspection.

**First message direction:** "We will build LeafVision one verified phase at a time. First I'll audit the original leaf classifier and identify what can be reused. I'll inspect what is present, distinguish confirmed facts from assumptions, and propose our first small task. I will not generate the whole application yet." Then perform read-only inspection and discuss Phase 00.
