"""
Unit tests for FastAPI Endpoints: Health, Samples List, Sample Prediction, and Guide.
"""

import unittest
from fastapi.testclient import TestClient

from backend.main import app, startup_event


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        startup_event()
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        """Verify API health status, architecture metadata, and channel count."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["eeg_channels"], 32)
        self.assertEqual(data["sampling_rate_hz"], 128)
        self.assertEqual(len(data["target_dimensions"]), 4)

    def test_samples_list_endpoint(self):
        """Verify sample DEAP files are discovered and listed."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)

    def test_predict_sample_endpoint(self):
        """Verify prediction for a preloaded sample trial."""
        payload = {
            "sample_filename": "s01.dat",
            "trial_index": 0
        }
        response = self.client.post("/api/predict/sample", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        
        preds = data["predictions"]
        for dim in ["Valence", "Arousal", "Dominance", "Liking"]:
            self.assertIn(dim, preds)
            self.assertGreaterEqual(preds[dim], 1.0)
            self.assertLessEqual(preds[dim], 9.0)

        self.assertIn("circumplex", data)
        self.assertIn("quadrant", data["circumplex"])
        self.assertIn("disclaimer", data)

    def test_affect_guide_endpoint(self):
        """Verify educational affective guide and non-diagnostic disclaimer."""
        response = self.client.get("/api/affect-guide")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("dimensions", data)
        self.assertIn("circumplex_quadrants", data)
        self.assertIn("ethics_and_disclaimer", data)


if __name__ == "__main__":
    unittest.main()

