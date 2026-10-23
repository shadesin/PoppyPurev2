# Phase 3 — colour/texture features and mask review

## Background-mask result

Prototyped two pixel-derived methods on **56 training images**, two per class/background/folder-magnification stratum: the earlier representative image and a different low-sharpness image. Inspected all eight review sheets with removed pixels shown in red. No validation/test photographs were used.

1. **Lab lightness Otsu:** choose bright/dark foreground from an image-derived background-polarity rule, then apply a one-pixel-radius morphological closing.
2. **Colour/lightness:** use lightness for inferred dark backgrounds and HSV saturation Otsu for inferred light backgrounds, with the same cleanup.

An initial border-median polarity rule incorrectly classified grain-filled borders. Replaced it with a global dark-tail rule: fifth-percentile normalized Lab lightness below 0.20 indicates a dark background. This improved many black-background examples, but dense/blurred cases still fail. Both prototypes remove visible material on white backgrounds; pure semolina examples make the failure clear. Saturation also favours warmer particles over pale material.

**Decision: enable no foreground mask.** Neither prototype passes the visual retention requirement across the four conditions. Use the checked crop and unmasked region descriptors. This is a result of the inspected prototypes, not a claim that every segmentation method would fail. No masked classifier is trained simply to obtain a score from visibly invalid masks.

Evidence: [measurements](masks/measurements.csv), [review decisions](masks/review_decisions.json), and eight `masks/{background}_{magnification}_{selection}.jpg` sheets. Pixel coverage is not segmentation accuracy or a volume measurement. Mask measurement run records retain the initial and corrected versions; the final corrected trial is `experiments/runs/phase3-mask-dark-tail.json`.

## Feature definitions

Configuration: [features-v1.json](../../configs/features-v1.json). Implementation: [features.py](../../src/poppypurev2/features.py).

| Family | Definition | Dimensions |
| --- | --- | ---: |
| Colour | HSV and Lab, 16 bins for each of six channels | 96 |
| LBP | Rotation-invariant uniform patterns, P=8/R=1 and P=16/R=2; P+2 bins per scale | 28 |
| GLCM | 32 gray levels, distances 1/2/4, angles 0/45/90/135°, six properties | 72 |
| **Single-region total** | Concatenation in saved schema order | **196** |

Each colour-channel histogram sums to one. HSV ranges are [0,1]; Lab ranges are L=[0,100], a/b=[−128,128]. LBP uses rounded integer grayscale and excludes a border equal to its radius, preventing artificial image-edge samples. GLCM is symmetric and normalized; properties are contrast, dissimilarity, homogeneity, energy, correlation and ASM, with explicit distance/angle order.

Only valid RGB image regions are described. No square-padding pixels or blacked-out mask edges enter histograms or texture calculations. Features are deterministic and contain no EXIF, filenames, condition tags or volume labels. Histograms normalize within each image; dataset scaling/PCA/selection remain unfitted until training in Phase 4.

## Three candidate image representations

| Variant | Processing | Dimensions per image |
| --- | --- | ---: |
| Whole | Oriented RGB overview, long side ≤768 | 196 |
| Crop | Checked crop, long side ≤768 | 196 |
| Crop detail | Native 512-pixel regions at four crop corners and centre; pool descriptor mean and population standard deviation | 392 |

Whole-image features are a reference, including any visible markings; crop features are the candidate that excludes observed lower-edge camera marks. Region pooling makes one row per source image, without inventing independently labelled local concentrations. Regions use valid pixels without padding or stretching. Their pixel scale is defined by the saved preprocessing settings.

## Saved training features

`data/features/v1/train.npz` contains `whole`, `crop`, `crop_detail`, `image_ids`, `volume_percent` and `preparation_ids`. Labels and identifiers are saved separately from the numeric feature matrices. `schema.json` defines every feature name/order, and `summary.json` records checksums, dimensions, input configuration hashes and split identity. Load with `numpy.load(..., allow_pickle=False)`.

The extraction command is training-only and refuses an existing output directory. Every raw training image is checked against its frozen SHA-256. Validation features will be computed using these same definitions for model comparison in Phase 4; test images stay reserved.

Completed extraction for all **384 training images**: whole **384×196**, crop **384×196**, pooled detail **384×392**. All values are finite; no dataset scaler or classifier was fitted. Training NPZ SHA-256: `1ad5fc7a7d95229c624cd467ed38cdb7c024f799ee8cd36225d9915883423985`; feature schema SHA-256: `1329a0f699d73dc626567a2b84ba6b96220f8908a343d81a7278d4cb23281fc0`. Run record: `experiments/runs/phase3-training-features.json`.

```sh
uv run --locked poppypure mask-qc
uv run --locked poppypure extract-features --output data/features/NEW-VERSION
```

Optional `--record` writes a new immutable run record. No classifier is fitted by either command.

## Verification

Fourteen tests pass. New checks verify normalized colour/LBP histograms, expected LBP bins for constant regions, zero GLCM contrast and unit energy/correlation for constant regions, analytically correct checkerboard contrast at different offsets, deterministic descriptors, image-level region pooling, and synthetic mask/overlay behaviour including grain-filled borders. Compilation and Ruff checks also pass.

Scikit-image **0.26.0** is pinned in `pyproject.toml` and `uv.lock`; experiment records now include its version. Initial use of the deprecated `binary_closing` API produced warnings; replaced it with the supported `closing` operation and regenerated the mask trial. The sandboxed dependency download failed DNS checks; an approved network retry installed the pinned environment.

Method definitions follow the official [LBP/GLCM API](https://scikit-image.org/docs/stable/api/skimage.feature.html), [Otsu threshold API](https://scikit-image.org/docs/stable/api/skimage.filters.html#skimage.filters.threshold_otsu), and [colour conversion API](https://scikit-image.org/docs/stable/api/skimage.color.html).

**Next: Phase 4 baselines on training/validation.** Compare representation and feature families, include trivial/metadata/background controls, and save predictions. No classification accuracy has been produced in this phase.
