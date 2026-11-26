# LeafVision AWS demo video script

**Suggested finished length: 10–12 minutes.** The timestamps are pacing cues. Keep your camera on, speak in your own voice, and give the viewer a few seconds to read each important result. Text in quotation marks is narration you can read aloud. Instructions in bold are things to do on screen.

Record prediction, labeled upload, and retraining on the [public AWS dashboard](https://leafvision-51-21-251-72.sslip.io). The recording and grading window ends December 26, 2026 at 19:50 Kigali time. During the video, show the job ID and metrics from your own take. The November 26 screenshots document earlier AWS runs. The Locust chart documents a separate laptop experiment.

## Before you press Record

1. **Prepare a clean recording window.** Put the browser and a terminal side by side or in separate tabs. A small camera overlay in the upper-right usually leaves the buttons and results visible. Make a ten-second test recording and play it back to check that your voice is clear and the metric labels are readable. Silence notifications. Keep your password manager, secrets file, AWS settings, and personal tabs outside the capture.
2. **Open the public dashboard.** Use the AWS link above and keep that tab open. On **Monitoring**, note the serving version and current prediction count. The deployment has already handled a verification request, and other visitors can use it.
3. **Sign in before recording.** Enter the cloud operator token in the sidebar, check for **Operator access enabled**, and close the private token file before sharing your screen.
4. **Stage the cloud evidence terminal.** On this laptop, open a terminal and run the commands below. The helper connects to the grading instance. Leave that SSH session in the project directory so the database commands later in the script query the same deployment as the dashboard. In the browser, note the starting counts under **Labeled uploads** and return to **Crop prediction**.

   ```bash
   cd /home/ayo/Documents/MLOps
   "$HOME/Downloads/mlops leftovers/artifacts/aws-grading-2026-11-26/remote"
   # After the SSH prompt appears:
   cd ~/leafvision
   sudo docker compose --profile training ps
   ```

5. **Put your three demo photos within easy reach.** Use [HL_423.jpg](../artifacts/HL_423.jpg) for prediction. Use [tomato-259.jpg](../artifacts/retraining-inputs/tomato-259.jpg) and [tomato-308.jpg](../artifacts/retraining-inputs/tomato-308.jpg) together for retraining. These are local demo inputs. Pre-open the file picker folders so the recording shows the files you intend to use.
6. **Pre-open the supporting evidence.** Bookmark the rice row in the [test manifest](../data/test/test.csv), sections 10.6–10.7 of the [original training notebook](../notebook/ml_summative_notebook.ipynb), and **Evaluate both test subsets** in the [executed baseline notebook](../notebook/leafvision_mlops.ipynb). Open the [AWS prediction](evaluation/aws-rice-prediction.png), [AWS upload](evaluation/aws-labeled-upload.png), [completed AWS job](evaluation/aws-retraining-validation.png), [AWS job metrics](evaluation/aws-retraining.json), and [Locust graph](../load_testing/results/2026-11-23-local/scaling.png). Local files work even if you haven't pushed the latest evidence to GitHub.
7. **Decide how to film the wait.** The recorded AWS job took about 1 minute 54 seconds from worker start to finish. The current job can take a different amount of time. Start the live job early, then cover the notebook and charts while it runs. If you cut a longer wait, say how much time elapsed and show the same job ID before and after the edit.

## 0:00–0:35 | Introduce the project

**Show:** Your face for the opening line. Then share **Crop prediction**, with the title and file uploader visible. Keep the camera overlay on.

> “Hi, I’m [your name]. This is LeafVision, my MLOps project for identifying a crop from a close-up leaf photograph. It recognizes five crops: cashew, cassava, maize, rice, and tomato.
>
> “I’ll start with a leaf whose label we already know. Then I’ll upload two new labeled leaves, start a training job, and check whether the new model earned a place in the application. I’m running today’s live demonstration on AWS in Stockholm. Later I’ll compare what we see today with the results from my earlier cloud tests.”

## 0:35–1:30 | Make one prediction the viewer can verify

**Show:** In the test manifest, search for `HL_423.jpg`. Point at **crop = rice**. Switch to **Crop prediction**, choose the file under **Leaf photo**, and click **Identify crop**. Pause with the image preview, **Predicted crop**, **Model score**, and **Serving model** visible.

> “Before giving this image to the model, I’m checking its known label. This row identifies the crop as rice. Now I’ll upload the JPEG and click Identify crop.”
>
> **After the result appears:** “The model returned [read the crop]. That agrees with the recorded label. The score on this image is [read the displayed score], and the version that served it is [read the version]. That score belongs to this one photo. I’ll show accuracy across thousands of labeled photos in the notebook.”

With the verified baseline, the expected screen reads **Rice**, **0.99999988**, and **crop_model_v1**. If your live result differs, read what is on screen and check the version before using the baseline figures.

**Click Monitoring.** Point at **API uptime**, **Prediction requests**, **Failed requests**, and the server-side latency measurement. Let the page refresh if needed.

> “This API process has been running for [read uptime]. It has handled [count] prediction requests and recorded [count] failures. Uptime starts when the process starts. The timer here measures how long the API took to handle a prediction. I use labeled examples separately to measure whether the answers are right.”

## 1:30–2:40 | Upload two labeled photos and prove they were saved

**Show:** Click **Labeled uploads**. Point at the starting counts. Choose **Tomato** under **Crop label**, select both tomato JPEGs under **Leaf photos**, and click **Save labeled batch** once. Show **Last saved batch**, its ID, the image count, and the Tomato row under **Saved uploads**.

> “Prediction needed only a photograph. Training also needs a verified answer. These two photographs have the label tomato, so I select Tomato for the batch before I save it.”
>
> “Here’s the receipt. One batch contains two tomato images, and the dashboard gave it a batch ID. That ID is what I’ll select for retraining. The actual JPEG files are on a persistent volume. PostgreSQL stores the batch label and links each file to its storage path, byte count, and SHA-256 fingerprint. If I upload the same bytes twice, the system can recognize them.”

**Show a database row and its files.** Use the SSH terminal you prepared in step 4. Check that the batch ID in the output matches the dashboard receipt. In the first output, point at the two rows sharing a batch ID. In the second, point at a filename whose hash begins with the same characters. A few seconds on each output is enough.

```bash
sudo docker compose exec -T db psql -U leafvision -d leafvision -c \
  "SELECT b.id AS batch, b.crop, left(i.sha256, 12) AS hash_prefix, i.size_bytes
   FROM leafvision_upload_batch b
   JOIN leafvision_uploaded_image i ON i.batch_id = b.id
   ORDER BY b.created_at DESC, i.sha256 LIMIT 2;"
sudo docker compose exec -T api find /app/artifacts/uploads -type f -name '*.jpg'
```

> “Here are the two database records. These filenames on disk start with the same fingerprints. That’s the link between the training labels and the image bytes the worker will actually read.”

## 2:40–3:30 | Click Retrain and explain what changes

**Show:** Click **Retraining**. Under **Saved batch**, select the tomato batch you just created. Show its two-image count, click **Retrain** once, and hold on **Job ID** and the first status you see. Write down the last few characters of the job ID for the return shot.

> “I’m choosing the same batch and clicking Retrain. The API gives me a job ID right away. A separate Python worker picks up the job, which lets the dashboard keep answering requests while training runs.”
>
> “The worker loads the weights from my already-trained crop classifier. It checks the upload fingerprints against the original splits, mixes the two uploads with 20 original training images per crop, and fine-tunes the final classification layer. That’s 102 training images in this run. The existing backbone stays frozen. Using examples from all five crops helps retain what the model already learned.”
>
> “I’ll leave this job running while I show how the original model was trained and evaluated.”

**Keep the dashboard session open.** Switch to another tab for the notebook. The worker continues independently.

## 3:30–5:15 | Show training, preprocessing, optimization, and test results

**Show:** In the original notebook, show the crop MobileNetV2 section and the early-stopping callback. In the LeafVision notebook, scroll from **Original training and optimization** to **Run a labeled prediction**, then **Evaluate both test subsets** and the confusion matrix. Present the saved outputs at a readable zoom.

> “The original data came from two image collections. CCMT supplied cashew, cassava, maize, and tomato. The rice photographs came from the Rice Leaf Disease and Pest Dataset. The split manifests list 19,513 training images, 4,185 validation images, and 4,179 test images.”
>
> “For prediction, the preprocessing function decodes the JPEG into three RGB channels and resizes it to 128 by 128 pixels. The saved model applies normalization internally. The API and the offline prediction function use that same preprocessing.”
>
> “For the original model, I started with MobileNetV2 pretrained on ImageNet and fine-tuned the last 30 backbone layers. The classifier uses dropout and a five-class softmax output. Early stopping watches validation loss and restores the best weights after three epochs without improvement. That’s how I chose the weights that became the baseline.”
>
> “I also checked for data leakage. Out of the original test set, 191 images had a byte-identical copy in training. For the result I’m highlighting, I excluded test images whose exact bytes appeared in training or validation, and kept one copy of each remaining test image.”
>
> “On those 3,926 test images, the model got 3,883 correct. That’s **98.905 percent accuracy**. Macro precision was **0.98942**, macro recall **0.99125**, macro F1 **0.99028**, and cross-entropy loss **0.04485**. Macro averaging gives each crop equal weight. The confusion matrix shows where the remaining mistakes went: 29 of the 43 errors were on cassava photos. That gives me a specific crop to investigate.”
>
> “Exact matching can miss a resized or recompressed copy. I also reviewed visually similar examples. These scores describe these image collections. I would want an independently collected field set before making claims about performance on a farm.”

**Point while speaking:** Trace one row of the confusion matrix horizontally. Rows are actual crops, columns are predictions. Give the viewer a moment to read the table instead of saying each number twice.

## 5:15–6:35 | Tell the story behind three image features

**Show:** Return to the dashboard and click **Dataset insights**. Hover over a dot in each chart. A dot is the median for one crop. Its line spans the 10th to the 90th percentile. These charts use whole training images and count each unique JPEG byte sequence once.

**Brightness.** Hover over rice, then maize or cassava.

> “Look how far the rice dot sits from the others. On a zero-to-255 brightness scale, rice has a median of 228.1. The other four crops sit between 126.2 and 132.4. In the rice collection, the median share of near-white pixels is 82.7 percent. Rice came from a separate collection, so a pale background is a plausible shortcut. An interesting follow-up would be to photograph rice against darker backgrounds and see whether performance holds.”

**Green dominance.** Hover over maize, cassava, and tomato.

> “This chart measures how much green exceeds the larger of red and blue across the frame. Maize and cassava have medians of plus 15.6 and plus 13.5, and their ranges overlap. Tomato’s median is minus 6.3. Color carries some information here, but two different crops can look similarly green. Lighting and background also change the number.”

**Edge density.** Point to the length of the maize interval and compare it with rice.

> “Here an edge is a sharp brightness change between neighboring pixels. The middle 80 percent of maize photos spans about 1.54 to 14.72 percent edge pixels. For rice the range is just 2.91 to 4.63 percent. Large uniform backgrounds can lower that value. Together, these charts explain why different framing and lighting belong in the next round of testing.”

## 6:35–7:50 | Return to the same job and interpret its result

**Show:** Click **Retraining** and select the same batch if needed. Match the job ID to the one you showed earlier. Scroll to **Training and validation completed**, **Candidate version**, and **Validation results**. Keep both columns in view.

> “This is the job ending in [read the same suffix]. It has finished, and the candidate now has its own version. Both models were evaluated on the same [read count] validation images. That gives me a fair comparison.”
>
> “Accuracy went from [baseline] to [candidate]. Macro F1 went from [baseline] to [candidate]. Cross-entropy loss went from [baseline] to [candidate]. Our promotion rule requires higher macro F1, equal or better accuracy, and equal or lower loss. Based on the results on screen, this candidate [meets / does not meet] that rule.”
>
> “The serving version shown above the table is still [read version]. The worker saves the candidate and its report. An operator decides whether to switch traffic to an eligible candidate.”

If the screen is still validating, show the AWS evidence or Locust segment next and return to the live job afterward. If you choose to promote on the grading server, first show the eligible metrics, click **Promote candidate** once, return to **Crop prediction**, run the same rice photo, and read the new **Serving model**. **Restore baseline** is available on **Retraining** after a successful switch. A promotion changes what visitors receive. For a recording that leaves the grading baseline in place, finish with the comparison and show the current serving version.

## 7:50–9:00 | Compare with the earlier AWS run

**Show:** Switch from the live grading dashboard to the saved AWS prediction screenshot. Open the cloud monitoring response, the AWS upload screenshot, then the cloud retraining JSON. Search within the JSON for `correct`, `max_reload_score_difference`, and `active_model`. Use the technical guide’s table for a more readable side-by-side view. [Cloud prediction evidence](technical-guide.md#cloud-prediction-on-aws) · [Cloud retraining evidence](technical-guide.md#cloud-retraining-on-aws)

> “I’m switching to the saved evidence from AWS. On November 26, I tested LeafVision on an EC2 instance in Stockholm with two virtual CPUs and eight GiB of RAM. Caddy provided public HTTPS. Streamlit called the FastAPI prediction service through HAProxy. PostgreSQL stored the upload and job records, and a separate container ran the training worker.”
>
> “In the public prediction check, this rice photo returned **Rice**, with a model score of **0.99999988**. The API recorded one prediction, zero failures, and **241.3 milliseconds** of server-side request time for that check.”
>
> “In the cloud retraining run, the dashboard saved both tomato photos and started the worker. The baseline got **3,900** of **3,933** validation photos right, and the candidate got **3,902**. Macro F1 rose from **0.99268** to **0.99306**, and loss fell from **0.02906** to **0.02785**. Reloading the saved candidate reproduced its validation scores exactly.”
>
> “That’s a small improvement on this validation set. The candidate qualified for promotion. I kept version one serving, exported the candidate and report, and closed that earlier test instance. The grading deployment you saw today has its own public link.”

Give the metrics below a few seconds of screen time. They are from the **completed AWS job**. Use the numbers from your live recording in the preceding scene.

| Measure | AWS baseline | AWS candidate |
|---|---:|---:|
| Correct out of 3,933 | 3,900 | 3,902 |
| Accuracy | 0.99161 | 0.99212 |
| Macro precision | 0.99208 | 0.99248 |
| Macro recall | 0.99332 | 0.99367 |
| Macro F1 | 0.99268 | 0.99306 |
| Cross-entropy loss | 0.02906 | 0.02785 |

## 9:00–10:15 | Explain the Locust flood test

**Show:** Open the [Locust request code](../load_testing/locustfile.py), point at the image upload to `POST /predict`, then show the [scaling graph](../load_testing/results/2026-11-23-local/scaling.png). In the [benchmark table](technical-guide.md#locust-prediction-load-test), follow the four-user rows across one, two, and three API containers.

> “I used Locust to send real image uploads through HAProxy to the prediction endpoint. The load generator checked the returned crop, score, model version, and backend identifier. I varied the number of API containers from one to three and tested one, four, and eight concurrent users. Each combination had two measured runs after warm-up. These experiments ran locally on the same laptop.”
>
> “At four users, one container handled **3.42 requests per second**, with a mean response time of **1,166 milliseconds**. Two containers reached **7.26 requests per second** and **548 milliseconds**. Three reached **9.86 requests per second** and **404 milliseconds**. Across all 18 trials, all **3,103 measured predictions** passed the response checks.”
>
> “I also looked at slower requests. At eight users, the two-container 95th-percentile latency was **2.6 seconds** in both trials. With three containers it was **3.1** and **2.6 seconds**. So the third container increased throughput, while some users still waited longer. Those containers shared one laptop, and each request used the same photo. I’d repeat the experiment with varied images and a separate load generator before predicting cloud capacity.”

## 10:15–10:55 | Repository and closing

**Show:** Open the repository tree. Point at `notebook/`, `src/preprocessing.py`, `src/prediction.py`, `src/retraining.py`, `data/train/`, `data/test/`, and `models/crop_model_v1.h5`. Show the public dashboard link at the top of the README and the model and load-test results below it. End on camera.

> “The repository contains the original training notebook, the evaluation notebook with saved outputs, the Python prediction and retraining code, the split manifests, and the saved model. The README explains how to run the dashboard and reproduce the measurements.”
>
> “LeafVision identified a known crop, saved new labeled examples, trained a candidate from the existing weights, and measured whether that candidate improved. The question I’d investigate next is how well the classifier handles field photos with more varied backgrounds, particularly for rice, and where it still confuses cassava. Thank you for watching.”

## If something changes during the take

| What you see | What to do |
|---|---|
| The upload or Retrain control is disabled | Off camera, verify the operator token in this browser session. For upload, also select a crop and both JPEGs. For retraining, select a saved batch and check whether a job is already running. |
| The upload reports that the images already exist | Select the existing batch for retraining. For a fresh upload on camera, choose new, verified photos with the correct label. Renaming identical JPEGs leaves their fingerprints unchanged. |
| The job stays queued | In the grading SSH terminal, run `sudo docker compose --profile training ps`. Check the worker's health and use `sudo docker compose logs --tail 30 worker` off camera if needed. A queued job is evidence of submission only. Return to the results once the dashboard reports completion. |
| Your candidate misses a promotion threshold | Show the actual comparison and explain why the baseline keeps serving. A correctly rejected candidate demonstrates the safeguard. |
| A viewer asks whether 98.905% is the cloud retraining accuracy | Use the denominators. **3,926** is the filtered offline *test* set. **3,933** is the fixed *validation* set for the cloud candidate comparison. The single cloud prediction and the Locust requests answer separate operational questions. |
| The video needs to be shorter | Keep the on-camera intro, correct prediction, two-file upload, database receipt, Retrain click, live comparison, three feature insights, cloud evidence, and the four-user scaling result. Cut installation footage, typing, long waits, and the optional promotion sequence first. |

After filming on AWS, leave the grading dashboard running. Clear the operator token from the sidebar and close the SSH terminal. Play back the upload, retraining-result, and audio portions before publishing. Add the actual YouTube link to the README once the video is available.
