# PoppyPure v2

Reproducible image classification of poppy seeds mixed with semolina. The seven labels are **semolina adulteration percentages by volume**: `0`, `10`, `20`, `30`, `40`, `50`, and `100`. Includes the machine-learning workflow, a command-line predictor and a minimal local browser app.

Model weights are distributed separately from the source. Follow the download and setup instructions below to install the bundle in `models/poppypure-v1/` before prediction.

## Status

**Phases 0–6 are complete.** Final model **poppypure-v1** uses frozen ResNet18 + a 128-unit MLP on five pooled detail regions. It was freshly fitted on **466 training+validation images for 28 fixed epochs**, sealed, and evaluated on all **82 reserved test images**: **91.46% accuracy (75/82), macro F1 0.9138**. The fixed SVM reference, fitted on the same 466 images, scored **90.24% (74/82), macro F1 0.8976**. Seven errors remain, including one 10% mixture classified as pure.

The original 548-image allocation was **384 training / 82 validation / 82 test**. Development selected the recipe using **92.68% validation accuracy / 0.9249 macro F1**; those development results concern the earlier 384-image checkpoint. The final 82 test images never enter fitting. See [final report](reports/phase6/FINAL_REPORT.md), [graphs](reports/phase6/plots/README.md), [model card](reports/phase6/MODEL_CARD.md), [dataset card](reports/phase6/DATASET_CARD.md) and [Phase 5 selection](reports/phase5/MODEL_SELECTION.json).

## Download the model and run the app

**Model download:** get the verified bundle from [Release v1.0.0](https://github.com/shadesin/PoppyPurev2/releases/tag/v1.0.0): **[poppypure-v1.zip](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip)** (40.9 MiB) and **[checksum](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip.sha256)**. On the current development Mac, the installed environment and model are already present.

### 1. Get the source

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then clone the repository:

```sh
git clone https://github.com/shadesin/PoppyPurev2.git
cd PoppyPurev2
```

You can also use GitHub's **Code → Download ZIP**, extract it and open a terminal inside the extracted folder containing this README. Git is only needed for the clone option. The application uses Python; Node/npm and a GPU are not required.

### 2. Get the trained model

Open [Release v1.0.0](https://github.com/shadesin/PoppyPurev2/releases/tag/v1.0.0) and download the attached **`poppypure-v1.zip`** and **`poppypure-v1.zip.sha256`** assets using the links above. Save them beside this README.

A [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) is a versioned download page. Our attached ZIP supplies the trained model bundle; GitHub's automatic **Source code (zip)** download contains the repository source and does not include the ignored model weights. The model ZIP contains weights, saved preprocessing/scaler settings, class mapping, checksums and model documentation. It does not contain the raw training dataset.

### 3. Install, verify and extract

These commands work in a macOS terminal or Windows PowerShell from the repository root:

```sh
uv sync --locked --python 3.12
uv run --locked python scripts/package_model.py verify poppypure-v1.zip
uv run --locked python -m zipfile -e poppypure-v1.zip .
```

Only extract if verification reports **`"status": "passed"`**. The verifier compares the ZIP against the committed [asset manifest](releases/poppypure-v1/ASSET_MANIFEST.json) and verifies every packaged file. The `.sha256` sidecar also provides the archive checksum for other download tools.

The ZIP already contains the `models/poppypure-v1/` folders. Extract it into the **repository root**, not into `models/` again. Check the resulting layout:

```text
PoppyPurev2/
  README.md
  app/server.py
  models/poppypure-v1/model.pt
  models/poppypure-v1/bundle.json
  models/poppypure-v1/RELEASE_MANIFEST.json
  ...
```

### 4. Start

```sh
uv run --locked python app/server.py --device cpu --open
```

CPU mode works on both intended platforms. On macOS, `--device auto` can use Apple MPS acceleration. Actual browser/app verification was performed on macOS; Windows dependency wheels were checked, but the app has not been tested on a Windows machine.

On the current Mac you can also double-click **`run_app.command`** in Finder, or run `.venv/bin/python app/server.py --open` from this directory. Keep the terminal window open while using the app. The browser opens at **http://127.0.0.1:8765**; you can also open that address manually.

- **Drag and drop** an image onto the page, **paste** an image from the clipboard with **⌘V** (Ctrl+V on other keyboards), or click **Choose image**. Pasting requires an image on the clipboard, not a copied filename or URL.
- Analysis starts automatically. The result is a predicted adulteration class **by volume**, not a continuous measurement: 0%, 10%, 20%, 30%, 40%, 50% or 100%. Zero means pure poppy; 100 means semolina.
- Click the preview to replace the image, or **Clear** to reset. Invalid or undersized images show an error instead of a percentage. The app does not display softmax scores as confidence.
- Supported files: **JPEG, PNG and still WebP**, up to **20 MB**. Export HEIC/HEIF photos as JPEG first. Use an original camera photo, not a screenshot.
- Stop the app with **Ctrl+C** in its terminal. If port 8765 is occupied, use `uv run --locked python app/server.py --device cpu --port 8766 --open`. If Apple acceleration is unavailable, use `--device cpu`.

The app binds only to your computer's loopback interface. The model loads once at startup; uploads are processed locally and temporary upload files are deleted after each request. No image history is kept. The app needs the model in `models/poppypure-v1/` (or pass `--model-dir /path/to/poppypure-v1`); **Backup Plus and an internet connection are not needed for prediction** once dependencies and model files are present. No Node, npm or extra Python dependencies are needed to run it.

### Photo capture specifications

A phone is sufficient: all 548 included source images record a **TECNO CE7** phone camera. These practical capture rules aim to match the training images; they do not establish accuracy for every phone or setup.

| Aspect | Capture specification |
| --- | --- |
| Background | Plain, flat, **matte white or matte black**; both occur in training. Avoid patterns, coloured surfaces, reflections and transparent trays. Black provides visible contrast for pale grains, but better prediction accuracy on black has not been established. |
| Sample preparation | Gently mix the sample, then spread a **thin, fairly even layer**. Grains may touch; avoid thick heaps, heavy overlap and separate ingredient pockets. Photograph a representative portion rather than selecting a visibly pure or adulterant-rich patch. |
| Framing | Fill most of the image with the spread material, including near the corners and centre. Avoid large empty borders, tray edges, hands and unrelated objects. Do not photograph an entire distant tray. |
| Angle and distance | Shoot directly down with the camera approximately parallel to the surface. Get close enough to resolve grain shapes and texture while staying within the camera's focusing range; no universal distance in centimetres has been validated. |
| Focus | Clean the lens, tap the grains to focus and hold the phone steady. Zoom into the saved photo to check grain edges. Move slightly farther away if the camera cannot focus. |
| Lighting | Bright, even, diffuse light. Avoid direct flash, harsh shadows, glare, strong sunlight and mixed coloured lighting. Keep the phone's shadow off the sample. |
| Camera settings | Normal photo mode at the camera's normal full resolution. Avoid digital zoom, portrait blur, filters and beauty effects. Use macro mode only when it preserves clear grain detail. |
| Resolution and file | Use the **original full-resolution file**, preferably at least **1600 × 1200** pixels (1200 × 1600 portrait is also suitable as a size guideline). That guideline does not guarantee sharpness or suitability. The runtime specifically requires five distinct native detail regions after its fixed crop; undersized files are rejected. Never enlarge a small image to bypass this check. Avoid screenshots and reduced-resolution messaging copies. |

The model applies EXIF orientation, crops to **5–95% of width and 5–88% of height**, then samples four corner regions and one centre region at native 512-pixel scale before CNN resizing. It does **not** automatically locate the sample, remove the background, assess focus or correct lighting. Keep representative material throughout that area. A file passing the size check is not necessarily a good photograph.

These instructions reduce avoidable capture problems. They cannot guarantee a correct prediction; the model can confuse pure poppy with an adulteration class. Its 91.46% test accuracy is the result on the reserved dataset test images, not a measured accuracy for arbitrary daily-use phone photos.

## Predict an image from the command line

From this project directory:

```sh
uv run --locked poppypure predict /path/to/sample.jpg --device cpu
```

The output includes the model version, predicted semolina percentage **by volume**, derived adulteration flag and seven class probabilities. Use `--device auto` for Apple MPS when available, `--model-dir /path/to/poppypure-v1` for a copied bundle, and `--output prediction.json` to save a new JSON file. Inference requires the installed package, the model bundle and the input image. It works without Backup Plus, training caches, project configuration or pretrained downloads; this was verified in a fresh offline-installed environment.

## Set up and check

Use Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). From this directory:

```sh
uv sync --locked
uv run --locked poppypure doctor
uv run --locked poppypure smoke --output experiments/runs/phase0-smoke.json
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m compileall -q src
```

`smoke` checks that configuration loads and writes an experiment record. It does **not** train or assess a model. It refuses to overwrite an existing record; use a new output name for another run.

For formatting and linting, use the Ruff version pinned in the lockfile. CI or a clean checkout can run these commands without creating generated files in Git:

```sh
uv run --locked ruff format --check src tests
uv run --locked ruff check src tests
```

## Data storage

The original Mendeley archive is in `/Volumes/Backup Plus/PoppyPurev2/source_archives/` and extracted raw images are in `/Volumes/Backup Plus/PoppyPurev2/raw/`. Local paths `data/source_archives` and `data/raw` are symbolic links to those drive folders. Keep Backup Plus connected when using the images. The later Mendeley dataset provides overlap and label provenance, not extra training data.

The expected local paths and the primary metric are in [`configs/base.toml`](configs/base.toml). Relative paths there resolve from this directory. Raw images, run outputs, and model files are excluded by `.gitignore`; the later manifest and split definitions can be versioned.

## Experiment records

Each run writes a JSON file using [`experiments/SCHEMA.md`](experiments/SCHEMA.md). The common fields include the code version, Git revision when available, config checksum, seed, package versions, dataset manifest checksum, split checksum, preprocessing and model settings, and metrics. Training commands fill dataset-specific fields and save per-image predictions alongside the record.

## Phase 1 commands

Connect Backup Plus before acquisition or audit. Acquisition downloads only the original dataset, verifies Mendeley's published SHA-256, extracts it without changing image bytes, and creates the two local data links. Interrupted byte-range downloads can be resumed with the same command.

```sh
uv run --locked poppypure source-metadata
uv run --locked poppypure acquire --workers 16
uv run --locked poppypure audit
```

Inspect `reports/phase1/` and record manual decisions in `data/review_decisions.json`, then rerun the audit. The audit checks every extracted image against its ZIP member and writes the manifest, exclusion reasons, duplicate reports, grouping information, and contact sheets. Sharpness and exposure flags prompt review; they do not remove images automatically. Hash similarity alone is not proof that two photos show the same scene.

Once all image and near-duplicate reviews are resolved:

```sh
uv run --locked poppypure freeze-data
```

The freeze records the final manifest checksum and refuses a different replacement. See [`reports/phase1/DATASET_CARD.md`](reports/phase1/DATASET_CARD.md) for source attribution and updated preparation provenance. No training or evaluation split is created in Phase 1.

## Phase 2 commands

```sh
uv run --locked poppypure freeze-evaluation
```

Creates or verifies active artifacts in `data/splits/v4/`. It rejects changed frozen inputs or altered outputs. Add `--record experiments/runs/NEW-NAME.json` to save an immutable run record. Use `--version v1`, `v2` or `v3` to verify historical protocols. No model is fitted. See [the train/validation/test specification](reports/phase2/TRAIN_VAL_TEST.md). After selection, the locked recipe may refit on 466 train+validation images before the 82-image final test.

## Phase 3 commands

```sh
uv run --locked poppypure preprocess-qc
uv run --locked poppypure mask-qc
uv run --locked poppypure extract-features --output data/features/NEW-VERSION
```

Connect Backup Plus. These commands verify source checksums and process training images only. Saved outputs include crop metadata, mask review sheets, and three feature matrices under `data/features/v1/` (384×196 whole/crop, 384×392 pooled detail). Extraction refuses an existing output directory. See [the initial processing report](reports/phase3/PROCESSING.md) and [feature definitions and mask decisions](reports/phase3/FEATURES.md). Phase 4 compared the representations on validation. Phase 3 did not process test images.

## Phase 4 commands

```sh
uv run --locked poppypure extract-features --partition validation --output data/features/validation-v1
uv run --locked poppypure prepare-controls --output data/features/controls-v1
uv run --locked poppypure train-baselines --output experiments/runs/phase4-NEW
```

The validation and control inputs already exist in this workspace; skip their creation commands when rerunning training. The settings are in `configs/phase4-v1.json`. The training command refuses an existing experiment directory, saves every model/prediction/run record and regenerates development reports in `reports/phase4/`. Saved first-run artifacts are in `experiments/runs/phase4-v1/`. No test feature-extraction option is exposed by the development command. See [findings](reports/phase4/FINDINGS.md) and [all-run comparison](reports/phase4/BASELINES.md).

See [the figure index](reports/phase4/plots/README.md) for rendered loss/F1/accuracy curves and important result charts. Reproduce into a new directory with `uv run --locked poppypure plot-phase4 --output reports/phase4/plots-NEW`. The command verifies two identical MLP diagnostic replays; it does not evaluate test images or replace original model runs.

## Phase 5 commands

```sh
uv run --locked poppypure train-deep --output experiments/runs/phase5-v1
uv run --locked python scripts/review_phase5.py
```

The first run is complete and saved under `experiments/runs/phase5-v1/`. For a clean rebuild, the commands require verified pretrained weight files under `models/pretrained/` and Backup Plus connected. Training runs four frozen ResNet18 heads, two frozen ResNet50 heads and two limited fine-tuning configurations; then verifies checkpoint predictions and renders learning/result graphs. It refuses existing experiment/embedding directories. The review script preserves source/provenance snapshots, locks the selected recipe and writes the report, class/condition figures and error gallery. MPS accelerates encoders when macOS exposes it; small frozen heads run on CPU. See [methods and saved-artifact locations](reports/phase5/METHODS.md). No final-test option is exposed.

## Phase 6 commands and artifacts

The completed final run is preserved. Reproduce in a fresh artifact workspace with the verified earlier-phase inputs:

```sh
uv run --locked poppypure refit-final
uv run --locked poppypure evaluate-final --device auto
uv run --locked poppypure report-final
```

`refit-final` refuses existing model/run outputs and fits exactly 28 epochs without refit validation. `evaluate-final` verifies the pretest seal and refuses repeated test access. `report-final` regenerates reports from saved predictions without model rescoring. No test-based tuning or all-548-image deployment refit was performed.

- Model/scaler/preprocessing/version/dependencies: `models/poppypure-v1/`.
- Refit membership, training curves, source snapshot and pretest checks: `experiments/runs/phase6-v1/`.
- All 82 primary/reference predictions, metrics and bootstrap replicates: `experiments/runs/phase6-v1/test/`.
- Technical report, five PNG/PDF figure sets and seven error images: `reports/phase6/`.
- The first pretest attempt is preserved separately; a float32-versus-float64 verification calculation was corrected before test access. The accepted retry produced identical checkpoint bytes.
