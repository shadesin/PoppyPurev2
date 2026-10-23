# Phase 3 — initial image processing

**Started and completed 14 October 2026.** This document preserves the initial processing pass; subsequent [feature extraction and mask review](FEATURES.md) complete Phase 3. The active [v4 split](../phase2/TRAIN_VAL_TEST.md) is frozen before processing. No classifier has been trained.

## Completed first pass

- Inspected one training image from each of the 28 class/capture strata in four initial contact sheets. TECNO watermarks are visible near the lower edge; border/grid lines, colour balance, focus and grain scale also vary.
- Implemented consistent RGB decoding, EXIF orientation and removal of ancillary metadata. EXIF is not a model feature. Source files remain unchanged.
- Added whole-image and crop candidates, aspect-preserving Lanczos resizing without upscaling, optional square padding with content bounds, and five deterministic detail regions inside a crop.
- The crop candidate removes 5% from left/right/top and 12% from bottom, retaining approximately **74.7% of the original area**. It is a development candidate, not an established optimal crop. Cropping removes material at the edges too; the source volume label stays unchanged.
- The initial long-side size is 768 pixels. Native 512-pixel detail regions are available for pooled image-level features if overview resizing loses useful texture. Regions are not independent concentration-labelled examples.
- Ran training-only integrity/processing checks and saved output dimensions, crop coordinates and pixel hashes, with selected original/crop/detail review sheets. The command excludes validation and test image decoding.
- All **384 training images** passed checksum, decode, RGB/metadata and crop/resize checks. Visually inspected all four original/crop/detail sheets, covering 28 selected training images. The observed lower-edge markings are absent from their cropped views; both materials remain visible in mixed-class examples. Saved the findings in `review_decisions.json`. This review covers those 28 examples, not every crop individually.

## Current choices

No background mask, illumination correction, smoothing or sharpening is enabled. The output remains RGB. The whole-image reference allows later measurement of the crop's effect. Some internal grid lines cannot be removed by a simple margin crop.

The crop is designed to exclude the observed lower-edge camera markings. Review the saved overlays before accepting it, and compare whole versus crop on validation once baseline models are available. Do not assume that all possible markings are removed solely because metadata is absent.

## Subsequent work — completed in this phase

1. Extend mask prototypes and visual checks, especially white semolina against white background; keep masks optional.
2. Implement normalized colour histograms and consistent LBP/GLCM texture descriptors, with tile/region pooling and padding/mask exclusion.
3. Check feature dimensions and numerical outputs; save feature definitions.
4. Compare the processing candidates with baseline classifiers on the fixed train/validation allocation in Phase 4. Reserve test scoring for the final locked pipeline.

Items 1–3 are now implemented and documented in [FEATURES.md](FEATURES.md); item 4 is the next phase. Mask prototypes failed visual retention checks and remain disabled. Three training feature matrices and schemas are saved in `data/features/v1/`.

## Reproduce and inspect

```sh
uv run --locked poppypure preprocess-qc
```

Configuration: [preprocessing-v1.json](../../configs/preprocessing-v1.json). Implementation: [preprocessing.py](../../src/poppypurev2/preprocessing.py). Outputs: `training_preprocessing.csv`, `summary.json`, and `crop_review_{black,white}_{4x,micro}.jpg` here. Add `--record` with a new JSON path for another run record. QC may regenerate its development reports; experiment records refuse overwrites.
