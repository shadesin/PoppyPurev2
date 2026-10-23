# Experiment record, schema version 1

One JSON record is written per run. `experiments/runs/` contains generated records and is ignored by Git. Run IDs are filenames chosen by the caller; the writer refuses to overwrite one.

| Field | Meaning |
| --- | --- |
| `schema_version` | Integer schema revision, initially `1`. |
| `kind` | Run type, such as `phase0_smoke`, `training`, or `evaluation`. |
| `created_at_utc` | ISO 8601 UTC creation time. |
| `status` | Run outcome. |
| `project_version`, `git_revision` | Package version and commit when available. |
| `source_tree_sha256` | Fingerprint of source paths and bytes, including uncommitted code. |
| `python_version`, `python_executable`, `package_versions` | Runtime identity. |
| `config_sha256`, `seed` | Exact config bytes and random seed. |
| `dataset_manifest_sha256`, `split_sha256` | Exact dataset and split definitions, or `null` before Phase 1–2. |
| `preprocessing`, `model`, `metrics` | Run settings/results, or `null` for a Phase 0 smoke run. |

Training and evaluation records must include source archive checksum, manifest checksum, split checksum, full preprocessing/model configuration, aggregate metrics, and paths to per-image predictions. These fields will be defined as implementation begins; a Phase 0 smoke record does not claim any model performance.

A `phase1_audit` record also includes `source_archive_sha256` and `audit_summary`. Counts there describe the dataset, not model accuracy. Use `poppypure audit --record experiments/runs/phase1-audit.json` to save one; record paths cannot be overwritten.

## Phase 4 training records

Each subdirectory of `experiments/runs/phase4-v1/` contains `record.json`, `model.joblib`, `validation_predictions.csv` and `metrics.json`. MLP runs also contain `learning_curve.csv`. Settings/provenance and screening selections are saved at the experiment root.

`phase4_training` adds `source_archive_sha256`, explicit fit/selection partitions, input-artifact/config hashes, complete estimator parameters, training time, fitted parameter counts, separate preprocessing statistic counts, warnings, training metrics, condition summaries, selected MLP epoch and artifact SHA-256 values. `status` distinguishes `passed`, `not_converged` and `failed`; failed or non-converged runs cannot win selection.

Prediction rows identify the source/preparation, true and predicted by-volume class, capture condition and seven output scores in fixed class order. `score_kind` distinguishes probabilities from SVM decision margins. Probabilities are not presented as calibrated confidence. The evaluation aggregate is validation-only; test data is not passed to the training command. `ACCEPTANCE.json` records saved-model/prediction reproduction checks.

## Phase 5 training records

Each `phase5_training` run saves `checkpoint.pt`, `record.json`, `result.json`, `metrics.json`, `validation_predictions.csv` and `learning_curve.csv`. The record includes Torch/Torchvision versions, full-source fingerprint, config/manifest/split hashes, actual device, pretrained weight and embedding provenance, architecture/representation/head, selected epoch, total versus fitted parameter counts, timing and artifact checksums. Fine-tuning records identify their initialization and backbone learning rate. The shared settings specify optimizer, augmentation and stopping rules.

The checkpoint stores the complete encoder/head state, embedding scaler buffers, seven by-volume class mapping, selected epoch, settings and input provenance. Source crop settings are also snapshotted in `source_preprocessing.json`. The companion `dataset_provenance.json` preserves the source-archive and frozen-manifest identity; these link the raw inputs to individual model records without changing the prior dataset freeze.

Prediction CSVs contain image ID, true/predicted by-volume class and seven softmax probabilities. Preparation and capture-condition attributes join through the immutable split membership. Softmax values are prediction probabilities, without a calibration claim. Loss traces use full-partition, deterministic cross-entropy for both training and validation, plus macro F1 and accuracy. Selection restores the earliest best-validation-macro-F1 checkpoint.

Embedding summaries identify separate training/validation NPZ files, membership, dimensions, source/config/weight hashes and encoder time. Frozen-head fit timing excludes this separately reported extraction; fine-tuning timing includes image loading and epoch evaluation. `ACCEPTANCE.json` checks persisted predictions, scaler statistics, membership, tensor integrity, checkpoint selection and fixed-state boundaries. Test scoring is excluded throughout Phase 5.

## Phase 6 records and final bundle

`phase6_final_fit` records the approved selection/config/split/archive identity, 466-image fitting membership, fresh scaling, 28 fixed MLP epochs, actual timing and training metrics. The training trace has no validation columns. `REFIT_ACCEPTANCE.json`, `CLI_PORTABILITY.json` and `CLEAN_ENV_ACCEPTANCE.json` verify persistence, membership/scaling, unchanged pretrained weights/BatchNorm and actual copied-bundle prediction without project/data/pretrained dependencies. Exact source/config/environment bytes are preserved under `source_snapshot/`.

`PRETEST_LOCK.json` seals the primary checkpoint, fixed SVM reference, runtime metadata, settings, dependencies and code before test access. `TEST_ACCESS_STARTED.json` and exclusive `test/` creation prevent a second model-scoring run. `phase6_final_test` and `TEST_ACCESS_COMPLETED.json` identify the exact checkpoint and all 82 test outputs. Both models were fixed before scoring; the reference cannot replace the primary using test results.

Prediction rows include original source/preparation/condition/capture IDs, targets, outputs and fixed-order probability or decision-margin columns. `inference_arrays.npz` stores logits/features for arithmetic verification without rescoring models. `bootstrap_samples.csv`/`bootstrap.json` store 2,000 paired preparation-unit, true-class-stratified resamples and percentile intervals. Reports regenerate only from saved predictions.

`models/poppypure-v1/model.pt` contains the complete frozen CNN/MLP/scaler state, preprocessing settings, class mapping and version. `bundle.json` identifies its checksum and label meaning. Inference never reads source datasets, pretrained downloads or project configuration. Model-card/report additions do not modify the sealed checkpoint. The failed first pretest-verification attempt is preserved separately; its classifier bytes match the accepted retry exactly.
