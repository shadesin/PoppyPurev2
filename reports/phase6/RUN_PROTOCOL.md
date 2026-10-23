# Phase 6 execution protocol

**Status at declaration:** implementation verified; refit and final test not yet run.

**Execution outcome:** completed and preserved. Primary test accuracy **75/82 = 91.46%**, macro F1 **0.9138**; fixed SVM reference **74/82 = 90.24% / 0.8976**. See [final report](FINAL_REPORT.md) and the workflow for the pretest verification correction and completed acceptance checks.

The user approved retaining the 466-image refit and starting Phase 6. The seven labels remain semolina adulteration **by volume**, `[0,10,20,30,40,50,100]`.

## Fixed primary recipe

- Phase 5 selection: `resnet18_crop_detail_mlp_frozen`.
- ImageNet V1 ResNet18 with fixed weights and BatchNorm; five native 512-pixel detail regions, 224-pixel aspect-preserving letterboxing, ImageNet normalization, image-level mean pooling.
- Fresh StandardScaler fitted on all **466 training+validation embeddings**, excluding test. Original v4 assignments define the row order. The known same-scene pair stays in the fitting partition.
- Fresh MLP: 512 → 128 → ReLU → dropout 0.3 → 7, AdamW lr 0.001, weight decay 0.0001, batch 32, seed 42.
- **28 fixed epochs**, retaining the final epoch. No refit validation, early stopping, augmentation or fine-tuning.

## Fixed classical reference

The selected Phase 4 SVM (`svm_input1_C10`) is refitted on the same 466 images: whole-image colour/texture descriptors, StandardScaler, RBF SVM C=10, gamma=scale, no probability calibration. This is a predeclared comparison, with no test-driven replacement of the selected primary.

## Gates before test access

Verify selected recipe/config/checkpoint/split hashes; cached feature IDs, labels and preparation units; all 466 raw image hashes; exact refit scaler membership and epoch count; portable checkpoint inference against cached predictions; unchanged encoder/BatchNorm weights; saved SVM predictions. Preserve exact source/config/environment snapshots. Seal primary/reference checkpoints and settings before opening test images.

## Final scoring

Score the complete **82-image test set** in one scoring event, running both fixed predictors. Persist image-level outputs, logits/features, source/preparation/condition IDs, metrics and artifact hashes. Exclusive access records prevent a second model-scoring run. Report regeneration reads saved predictions and never reruns the predictors.

Primary metrics: macro F1, accuracy, balanced accuracy, per-class precision/recall/F1, count/normalized confusion matrix and volume-class MAE. Derive pure/adulterated sensitivity/specificity. Include background/magnification/condition results and descriptive capture-date summaries.

Uncertainty: 2,000 percentile-bootstrap replicates, seed 1042, preparation units sampled within each true class. Use the same sampled units for paired primary/reference comparisons. This calculates intervals from saved predictions without training or changing either model.

## Deliverables

`models/poppypure-v1/` contains full CNN/head/scaler weights, embedded preprocessing/labels, bundle metadata, recipe and dependency lockfiles. CLI inference must work with a copied bundle and input image without access to Backup Plus, training caches, project configuration or pretrained downloads.

Save training-only refit curves, final result plots, all test errors, technical report, dataset/model cards and reproduction commands. Preserve the evaluated 466-image version. No all-548-image deployment refit is part of this run.

```sh
uv run --locked poppypure refit-final
uv run --locked poppypure evaluate-final --device auto
uv run --locked poppypure report-final
uv run --locked poppypure predict sample.jpg --device cpu
```

The first two commands refuse existing final-model/run or test-access outputs. Reproduction uses a fresh artifact workspace with the verified Phase 1–5 inputs.
