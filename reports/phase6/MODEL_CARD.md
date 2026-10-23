# Model card — poppypure-v1

## Task

Seven-class classification of semolina adulteration in poppy seed close-up images. Labels are percentages **by volume**: 0, 10, 20, 30, 40, 50 and 100. Zero is pure poppy; 100 is pure semolina.

## Model and fitting

ImageNet V1 ResNet18 with frozen weights/BatchNorm; mean of five native detail-region embeddings; a 466-image fitted StandardScaler; MLP 512 → 128 → ReLU → dropout 0.3 → 7 logits. Final fitting uses seed 42, AdamW lr 0.001, weight decay 0.0001, batch 32 and 28 fixed epochs. No refit validation, augmentation or fine-tuning.

## Evaluation

The exact packaged checkpoint scored **91.46% accuracy (75/82), 0.9138 macro F1** on the reserved test set after selection and refitting. The checkpoint was sealed before test access and remains unchanged. See FINAL_REPORT.md for class errors, binary detection, condition summaries and bootstrap intervals.

## Input/output

Input: an original close-up RGB-compatible image with enough resolution for five distinct 512-pixel detail regions after cropping. EXIF is used for orientation only. Output: model version, one documented by-volume class, derived adulteration flag and seven softmax probabilities. Probabilities are uncalibrated; the output is a discrete class.

## Provenance

Dataset: Sarkar, Mukherjee and Chatterjee (2023), Mendeley Data [10.17632/p2gvvnh8y4.1](https://doi.org/10.17632/p2gvvnh8y4.1), CC BY 4.0. Related label provenance: Ganguly and Sarkar (2026), [10.17632/nx8m9kxy26.1](https://doi.org/10.17632/nx8m9kxy26.1). Source images retain their attribution; crop/feature/model outputs are PoppyPure v2 additions. Preparation units follow user confirmation, with the verified same-scene pair kept together.

Encoder: Torchvision ResNet18, explicit `IMAGENET1K_V1`; weight file `resnet18-f37072fd.pth`. Runtime package versions are pinned in the included pyproject.toml/uv.lock; exact fitting source is snapshotted.

Checkpoint SHA-256: `520d47d80da1c3d8f5c2d689974a6ecbceee792dba41eae4c162fd072b4d69d1`.
