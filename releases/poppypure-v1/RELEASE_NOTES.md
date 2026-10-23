# PoppyPure v2 — model poppypure-v1

First release of the local prediction app and the verified poppypure-v1 model bundle.

## Included

- Local Python/browser app with paste, drag-and-drop, image preview and automatic prediction.
- Seven semolina adulteration classes **by volume**: 0%, 10%, 20%, 30%, 40%, 50% and 100%.
- Frozen ResNet18 + MLP, fitted on 466 images for 28 fixed epochs.
- Reserved-test accuracy **91.46% (75/82)**; macro F1 **0.9138**.
- Capture specifications, evaluation reports and pinned dependencies in the source repository.

## Downloads

1. [poppypure-v1.zip](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip) — complete verified model bundle, 40.9 MiB; extract into the source checkout root.
2. [poppypure-v1.zip.sha256](https://github.com/shadesin/PoppyPurev2/releases/download/v1.0.0/poppypure-v1.zip.sha256) — download checksum.

The asset ZIP includes `models/poppypure-v1/`, not a copy of the app or raw dataset. Users also need the source checkout. GitHub's automatically generated source archives do not include model weights.

The repository's `releases/poppypure-v1/ASSET_MANIFEST.json` records the expected ZIP checksum and exact model identity. Use `uv run --locked python scripts/package_model.py build` to package an existing verified model bundle. End-user download, verification, extraction and launch instructions are in `README.md`.

## Run

After downloading the source and both assets into the repository root:

```sh
uv sync --locked --python 3.12
uv run --locked python scripts/package_model.py verify poppypure-v1.zip
uv run --locked python -m zipfile -e poppypure-v1.zip .
uv run --locked python app/server.py --device cpu --open
```

Install uv first. The application opens at http://127.0.0.1:8765. No GPU or external dataset drive is needed. Model predictions depend on image quality and are discrete by-volume classes, not laboratory measurements. macOS was tested; Windows dependencies were checked but no Windows-machine test was performed.

## Verification

The ZIP contains 52 files, each verified against its release checksum. A clean source checkout with a fresh environment and this extracted model started the app and reproduced CLI predictions. The raw dataset and Backup Plus drive are not required for prediction.

Archive SHA-256: `adc394eac6a7fcdede9ee146242f389a22100b2548f8068ed567221090069d9c`.

For full photo capture and installation instructions, see the [project README](https://github.com/shadesin/PoppyPurev2#readme).
