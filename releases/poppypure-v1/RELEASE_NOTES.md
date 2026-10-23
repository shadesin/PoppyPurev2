# PoppyPure v2 — model poppypure-v1

Prepared release notes. This release has not been uploaded or published yet.

## Included

- Local Python/browser app with paste, drag-and-drop, image preview and automatic prediction.
- Seven semolina adulteration classes **by volume**: 0%, 10%, 20%, 30%, 40%, 50% and 100%.
- Frozen ResNet18 + MLP, fitted on 466 images for 28 fixed epochs.
- Reserved-test accuracy **91.46% (75/82)**; macro F1 **0.9138**.
- Capture specifications, evaluation reports and pinned dependencies in the source repository.

## Attach these assets

1. `poppypure-v1.zip` — complete verified model bundle; extract into the source checkout root.
2. `poppypure-v1.zip.sha256` — download checksum.

The asset ZIP includes `models/poppypure-v1/`, not a copy of the app or raw dataset. Users also need the source checkout. GitHub's automatically generated source archives do not include model weights.

The repository's `releases/poppypure-v1/ASSET_MANIFEST.json` records the expected ZIP checksum and exact model identity. Build/verification commands are documented in `GIT_GUIDE.md`; end-user download, verification, extraction and launch instructions are in `README.md`.

## Run

After downloading the source and both assets into the repository root:

```sh
uv sync --locked --python 3.12
uv run --locked python scripts/package_model.py verify poppypure-v1.zip
uv run --locked python -m zipfile -e poppypure-v1.zip .
uv run --locked python app/server.py --device cpu --open
```

Install uv first. The application opens at http://127.0.0.1:8765. No GPU or external dataset drive is needed. Model predictions depend on image quality and are discrete by-volume classes, not laboratory measurements. macOS was tested; Windows dependencies were checked but no Windows-machine test was performed.

## Publication checklist for later

- Push the approved source commit to `shadesin/PoppyPurev2` when authorized.
- Create a release/tag for that exact commit, attach both assets and use these notes as the release description.
- Update the README's preparation-status paragraph when the downloads are actually published.

No publication action is authorized by preparing these files.
