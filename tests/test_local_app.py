"""Local HTTP contract: upload validation, recovery and loopback request boundaries."""

import http.client
import json
import threading
import unittest
from unittest.mock import Mock

from app.server import MAX_UPLOAD_BYTES, AppServer


class LocalAppTests(unittest.TestCase):
    def setUp(self):
        self.predictor = Mock()
        self.predictor.metadata = {"model_version": "test-local"}
        self.predictor.predict.return_value = {
            "percentage_basis": "volume",
            "predicted_volume_percent": 20,
        }
        self.server = AppServer(("127.0.0.1", 0), self.predictor)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_address[1]

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_page_assets_health_and_path_boundaries(self):
        for path in ("/", "/style.css", "/app.js", "/favicon.svg", "/api/health"):
            status, body, headers = self.request("GET", path)
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertEqual(headers["Cache-Control"], "no-store")
        status, body, _ = self.request("GET", "/api/health")
        self.assertEqual(json.loads(body)["model_version"], "test-local")
        self.assertEqual(
            self.request("GET", "/../models/poppypure-v1/model.pt")[0], 404
        )
        self.assertEqual(self.request("POST", "/not-an-api", b"x")[0], 404)

    def test_upload_returns_by_volume_prediction(self):
        status, body, _ = self.request("POST", "/api/predict", b"image bytes")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["predicted_volume_percent"], 20)
        self.assertEqual(json.loads(body)["percentage_basis"], "volume")
        self.predictor.predict.assert_called_once_with(b"image bytes")

    def test_rejects_invalid_sizes_before_reading_or_predicting(self):
        for length, expected in (
            ("0", 400),
            ("-1", 400),
            ("bad", 400),
            (str(MAX_UPLOAD_BYTES + 1), 413),
        ):
            status, _, _ = self.request(
                "POST", "/api/predict", headers={"Content-Length": length}
            )
            self.assertEqual(status, expected)
        self.predictor.predict.assert_not_called()

    def test_rejects_foreign_browser_origins_and_hosts(self):
        for headers in ({"Origin": "https://other.example"}, {"Host": "other.example"}):
            self.assertEqual(
                self.request("POST", "/api/predict", b"image", headers)[0], 403
            )
        self.predictor.predict.assert_not_called()

    def test_bad_image_error_releases_lock_for_next_upload(self):
        self.predictor.predict.side_effect = [
            ValueError("This image is too small."),
            {"percentage_basis": "volume", "predicted_volume_percent": 0},
        ]
        status, body, _ = self.request("POST", "/api/predict", b"small")
        self.assertEqual(status, 400)
        self.assertIn("too small", json.loads(body)["error"])
        self.assertEqual(self.request("POST", "/api/predict", b"valid")[0], 200)

    def test_parallel_upload_is_rejected_without_second_inference(self):
        self.server.inference_lock.acquire()
        try:
            self.assertEqual(self.request("POST", "/api/predict", b"image")[0], 429)
            self.predictor.predict.assert_not_called()
        finally:
            self.server.inference_lock.release()


if __name__ == "__main__":
    unittest.main()
