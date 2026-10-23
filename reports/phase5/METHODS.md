# Phase 5 — ResNet comparison methods

**Status:** all eight configurations completed and passed acceptance. See [findings](FINDINGS.md) and [the selected recipe](MODEL_SELECTION.json). Class labels are semolina adulteration **by volume**, `[0, 10, 20, 30, 40, 50, 100]`.

## Data and inputs

The unchanged v4 allocation contains **384 training / 82 validation / 82 reserved test images**. Training and validation embeddings are separate artifacts. Test images are not decoded or scored by this workflow.

1. Check source-image SHA-256 against the manifest. Apply EXIF orientation, convert to RGB, discard ancillary metadata.
2. Apply the existing fractional crop `[0.05, 0.05, 0.95, 0.88]`.
3. Compare the complete cropped overview with five native 512-pixel regions: four corners and centre. Regions stay together under their original image and target.
4. Resize each input with bilinear interpolation while preserving aspect ratio; centre it on 224×224. Fill padding with exact floating ImageNet RGB means, producing zero normalized padding. Normalize by mean `[0.485,0.456,0.406]` and standard deviation `[0.229,0.224,0.225]`.
5. Remove the ImageNet classifier and mean-pool region embeddings. Produce one 512-dimensional ResNet18 or 2048-dimensional ResNet50 vector per source image.

No segmentation mask, background removal, illumination correction, EXIF feature or folder metadata enters the classifier. The 28-image input review and GPU benchmark are saved beside this report. The letterbox transform adapts the pretrained model's usual resize/centre-crop input; its exact implementation is part of this experiment.

## Bounded sequence

| Stage | Configurations | Selection before next stage |
| --- | ---: | --- |
| Frozen ResNet18, ImageNet V1 | Crop/detail × linear/MLP = 4 | Select representation using validation macro F1 and existing tie rule |
| Frozen ResNet50, ImageNet V2 | Selected representation × linear/MLP = 2 | Select best of all six frozen configurations |
| Limited fine-tuning | Two layer4 learning rates = 2 | Compare all eight with classical reference |

Frozen backbone parameters and BatchNorm statistics are fixed. Fit StandardScaler on **training embeddings only**. A given representation's linear/MLP heads receive the same embedding definition and scaling statistics.

- Linear: embedding → 7 logits.
- MLP: embedding → 128 → ReLU → dropout 0.3 → 7 logits.
- Frozen-head optimizer: AdamW, lr 0.001, weight decay 0.0001, batch 32, maximum 50 epochs, patience 8.
- Fine-tuning: initialize each trial independently from the best frozen checkpoint; unfreeze only layer4 and the head. Keep earlier blocks, all BatchNorm running statistics and the training-fitted embedding scaler fixed.
- Fine-tuning backbone lr: 0.00001 or 0.00003; head lr 0.0001, AdamW weight decay 0.0001, batch 8 source images, maximum 20 epochs, patience 5.
- Fine-tuning augmentation: source-crop flips, quarter-turn rotations, brightness/contrast ±10%, applied consistently before region extraction. Validation is deterministic.
- Seed 42. Cross-entropy on image-level logits; softmax is used for saved prediction probabilities. Loss labels use indices 0–6 internally and map back to the seven volume classes.

Every run retains the **earliest epoch with highest validation macro F1**. Near ties between models within 0.005 macro F1 prefer fewer fitted parameters, then measured fitting time. Frozen-head fitting time excludes separately recorded encoder extraction time; fine-tuning time includes image loading, training and epoch evaluation. Total encoder parameter count is reported separately from fitted parameter count.

## Outputs and acceptance

Command, from the project directory:

```sh
.venv/bin/poppypure train-deep --output experiments/runs/phase5-v1
.venv/bin/python scripts/review_phase5.py
```

MPS training needs execution where macOS exposes the GPU; the sandbox reports MPS unavailable. Actual devices are recorded. Pretrained weights live in `models/pretrained/`; raw images remain on Backup Plus. Existing experiment/cache directories are not overwritten.

For a clean environment, first download the explicit official weights:

```sh
mkdir -p models/pretrained
curl --fail --location --output models/pretrained/resnet18-f37072fd.pth https://download.pytorch.org/models/resnet18-f37072fd.pth
curl --fail --location --output models/pretrained/resnet50-11ad3fa6.pth https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

Accepted full SHA-256 values:

- ResNet18: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- ResNet50: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

The loader checks the official filename hash prefix before loading and records the full digest. See the [pinned Torchvision model definitions](https://github.com/pytorch/vision/blob/v0.24.1/torchvision/models/resnet.py) for the weight identifiers and URLs.

- `data/embeddings/phase5-v1/`: train/validation matrices, original IDs, targets and preparation IDs, weight/config/split/source hashes and timings.
- `experiments/runs/phase5-v1/<run>/`: complete checkpoint, training/validation cross-entropy/F1/accuracy traces, 82 validation predictions with class probabilities, per-class/condition metrics and provenance record.
- Selection records precede ResNet50 and fine-tuning. Exceptions create `FAILURE.json` and preserve completed runs.
- `ACCEPTANCE.json`: artifact integrity, membership, train-only scaler, persisted prediction reproduction, RGB-versus-embedding agreement, earliest-best epoch, and frozen-state checks.
- `source_snapshot/`: exact training source/configuration/environment bytes and checksums, preserved by the review script. The script also preserves source/preparation provenance and its own executed bytes.
- `reports/phase5/plots/`: all eight learning curves, selected confusion matrix and model comparison, each in PNG/PDF.

Phase 6 will lock a selected recipe and handle final fitting, the reserved test evaluation and inference packaging.
