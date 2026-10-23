"""Local browser UI for the sealed PoppyPure model; no training dependencies added."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import tempfile
import threading
import warnings
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError

from poppypurev2.deep_inputs import image_regions
from poppypurev2.inference import load_bundle, probability_rows
from poppypurev2.preprocessing import decode_rgb

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


class ImagePredictor:
    """Load once and use the release's unchanged image processing and classifier."""

    def __init__(self, model_dir: Path, device: str):
        self.model, self.checkpoint, self.metadata, self.device = load_bundle(
            model_dir, device
        )

    def predict(self, content: bytes) -> dict:
        # Temporary upload bytes are deleted on success and on every failure.
        with tempfile.TemporaryDirectory(prefix="poppypure-upload-") as directory:
            path = Path(directory) / "image"
            path.write_bytes(content)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as source:
                        if source.format not in {"JPEG", "PNG", "WEBP"}:
                            raise ValueError("Please use a JPEG, PNG or WebP image.")
                        if getattr(source, "is_animated", False):
                            raise ValueError(
                                "Please use a still image, not an animation."
                            )
                        source.verify()
                    rgb = decode_rgb(path)
            except (
                UnidentifiedImageError,
                OSError,
                Image.DecompressionBombError,
                Image.DecompressionBombWarning,
            ) as error:
                raise ValueError(
                    "This image could not be read. Export it as JPEG or PNG and retry."
                ) from error
            try:
                regions = image_regions(
                    rgb,
                    "crop_detail",
                    self.checkpoint["config"],
                    self.checkpoint["source_preprocessing"],
                )
            except ValueError as error:
                raise ValueError(
                    "This image is too small. Use the original close-up photo; "
                    "the model needs five distinct 512-pixel detail regions "
                    "after cropping. Do not enlarge a small image."
                ) from error
            with torch.inference_mode():
                probabilities = (
                    self.model(regions[None].to(self.device)).softmax(1).cpu().numpy()
                )
            labels, rows = probability_rows(probabilities)
            return {
                "model_version": self.metadata["model_version"],
                "percentage_basis": "volume",
                "predicted_volume_percent": int(labels[0]),
                "adulterated": bool(labels[0] > 0),
                "class_probabilities": rows[0],
                "image_sha256": hashlib.sha256(content).hexdigest(),
                "device": str(self.device),
            }


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], predictor: ImagePredictor):
        self.predictor = predictor
        self.inference_lock = threading.Lock()
        super().__init__(address, AppHandler)


class AppHandler(BaseHTTPRequestHandler):
    server: AppServer

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def reply(self, status: int, content: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            pass  # A closed browser tab must not terminate the server.

    def json_reply(self, status: int, payload: dict):
        self.reply(status, json.dumps(payload).encode(), "application/json")

    def local_request(self) -> bool:
        port = self.server.server_address[1]
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in hosts:
            self.json_reply(403, {"error": "Open the app using its localhost URL."})
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in hosts}:
            self.json_reply(403, {"error": "Upload from the local app page."})
            return False
        return True

    def do_GET(self):
        if not self.local_request():
            return
        if self.path == "/api/health":
            self.json_reply(
                200,
                {
                    "status": "ready",
                    "model_version": self.server.predictor.metadata["model_version"],
                },
            )
        elif self.path in ASSETS:
            name, content_type = ASSETS[self.path]
            self.reply(200, (ROOT / name).read_bytes(), content_type)
        else:
            self.json_reply(404, {"error": "Not found."})

    def do_POST(self):
        # Every POST uses a fresh connection, including rejected partial uploads.
        self.close_connection = True
        if not self.local_request():
            return
        if self.path != "/api/predict":
            self.json_reply(404, {"error": "Not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.json_reply(400, {"error": "Invalid image upload length."})
            return
        if length <= 0:
            self.json_reply(400, {"error": "Please choose an image first."})
            return
        if length > MAX_UPLOAD_BYTES:
            self.json_reply(413, {"error": "Please choose an image under 20 MB."})
            return
        if self.headers.get("Transfer-Encoding"):
            self.json_reply(400, {"error": "Upload the image with a known file size."})
            return
        if not self.server.inference_lock.acquire(blocking=False):
            self.json_reply(429, {"error": "An image is being analysed. Please retry."})
            return
        try:
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("The upload was incomplete. Please retry.")
            self.json_reply(200, self.server.predictor.predict(content))
        except (ValueError, TimeoutError) as error:
            self.json_reply(400, {"error": str(error)})
        except Exception:
            logging.exception("Local image prediction failed")
            self.json_reply(
                500,
                {"error": "Prediction failed. Retry or check the app terminal."},
            )
        finally:
            self.server.inference_lock.release()

    def log_message(self, format, *args):
        # No image names, content or prediction history are logged.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local PoppyPure app")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--model-dir", type=Path, default=ROOT.parent / "models/poppypure-v1"
    )
    parser.add_argument(
        "--open", action="store_true", help="Open the local app in a browser"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    print("Loading PoppyPure…", flush=True)
    try:
        predictor = ImagePredictor(args.model_dir, args.device)
        server = AppServer(("127.0.0.1", args.port), predictor)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Could not start: {error}", file=sys.stderr)
        print(
            "Try --device cpu, or --port 8766 if the port is in use.", file=sys.stderr
        )
        return 1
    url = f"http://127.0.0.1:{args.port}"
    print(f"PoppyPure is ready: {url}\nPress Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nPoppyPure stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
