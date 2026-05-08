"""
tests/test_score.py

Unit tests for src/score.py Flask API.
Run with: pytest tests/
"""

import json
import os
import pickle
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import score


@pytest.fixture
def client(tmp_path):
    """Create a test Flask client with a mock model loaded."""
    # Build a minimal mock SVD model
    mock_model = MagicMock()
    mock_model.predict = lambda user_id, movie: MagicMock(est=7.5)

    # Patch the globals in the score module
    score._model = mock_model
    score._all_movies = ["Inception", "The Dark Knight", "Interstellar", "Parasite"]

    score.app.config["TESTING"] = True
    with score.app.test_client() as client:
        yield client

    # Cleanup
    score._model = None
    score._all_movies = None


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = json.loads(response.data) if hasattr(client.get("/health"), 'data') else {}
        resp = client.get("/health")
        data = json.loads(resp.data)
        assert data["status"] == "ok"


class TestScoreEndpoint:
    def test_score_requires_user_id(self, client):
        response = client.post("/score",
                               data=json.dumps({}),
                               content_type="application/json")
        assert response.status_code == 400

    def test_score_returns_recommendations(self, client):
        response = client.post("/score",
                               data=json.dumps({"user_id": 123, "top_n": 2}),
                               content_type="application/json")
        assert response.status_code == 200
        data = json.loads(response.data)
        assert "recommendations" in data
        assert len(data["recommendations"]) == 2

    def test_score_returns_user_id_in_response(self, client):
        response = client.post("/score",
                               data=json.dumps({"user_id": 42}),
                               content_type="application/json")
        data = json.loads(response.data)
        assert data["user_id"] == 42

    def test_score_default_top_n_is_10(self, client):
        # We only have 4 mock movies, so it returns all 4
        response = client.post("/score",
                               data=json.dumps({"user_id": 1}),
                               content_type="application/json")
        data = json.loads(response.data)
        assert data["top_n"] == 10
        assert len(data["recommendations"]) <= 10

    def test_score_without_model_returns_503(self, client):
        score._model = None
        response = client.post("/score",
                               data=json.dumps({"user_id": 1}),
                               content_type="application/json")
        assert response.status_code == 503
        # Restore
        score._model = MagicMock()
        score._model.predict = lambda u, m: MagicMock(est=7.5)
