"""Compare local app inference with the sealed CLI path without test-set access."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from poppypurev2.inference import predict_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--device", choices=("cpu", "auto", "mps"), default="cpu")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = root / "models/poppypure-v1"
    output = root / "reports/generated/local-app"
    output.mkdir(parents=True, exist_ok=True)

    with urlopen(args.url + "/api/health", timeout=10) as response:
        assert json.load(response)["status"] == "ready"
    request = Request(
        args.url + "/api/predict",
        data=args.image.read_bytes(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        app_result = json.load(response)
    cli_result = predict_image(bundle, args.image, args.device)
    for key in (
        "model_version",
        "percentage_basis",
        "predicted_volume_percent",
        "adulterated",
        "image_sha256",
    ):
        assert app_result[key] == cli_result[key], key
    labels = ["0", "10", "20", "30", "40", "50", "100"]
    np.testing.assert_allclose(
        [app_result["class_probabilities"][key] for key in labels],
        [cli_result["class_probabilities"][key] for key in labels],
        atol=1e-6,
        rtol=1e-5,
    )
    try:
        urlopen(Request(args.url + "/api/predict", data=b"not an image"), timeout=10)
        raise AssertionError("Malformed upload was accepted")
    except HTTPError as error:
        assert error.code == 400
        assert "could not be read" in json.load(error)["error"]

    release = json.loads((bundle / "RELEASE_MANIFEST.json").read_text())
    for relative, expected in release["files"].items():
        assert hashlib.sha256((bundle / relative).read_bytes()).hexdigest() == expected
    current = {
        str(p.relative_to(root / "src")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root / "src").rglob("*.py")
    }
    archived = {
        str(p.relative_to(bundle / "runtime_source/src")): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in (bundle / "runtime_source/src").rglob("*.py")
    }
    assert current == archived, "Sealed prediction/training source changed"
    result = {
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "checks": [
            "ready health endpoint",
            "real image upload",
            "app/CLI class, volume basis, flag, image hash and probabilities agree",
            "malformed image rejected",
            "release file hashes unchanged",
            "current src matches sealed runtime source",
        ],
        "release_files_checked": len(release["files"]),
        "checkpoint_sha256": release["checkpoint_sha256"],
        "test_set_accessed": False,
        "note": "Functional prediction check only; no new accuracy result.",
    }
    (output / "API_ACCEPTANCE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
