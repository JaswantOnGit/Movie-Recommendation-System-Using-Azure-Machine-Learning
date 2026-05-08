"""
score.py

Flask inference API for the SVD movie recommendation model.
Loads a trained model pickle and serves top-N recommendations
for a given user via a REST endpoint.

Endpoints:
    GET  /health            -- liveness probe
    POST /score             -- returns top-N recommendations
        body: {"user_id": 123, "top_n": 10}

Usage (local):
    MODEL_PATH=models/svd_model.pkl \
    DATA_PATH=data/processed/cleaned_ratings.csv \
    python src/score.py

Deployment (ACI / AKS):
    gunicorn --bind 0.0.0.0:5000 --workers 2 src.score:app
"""

import logging
import os
import pickle

import pandas as pd
from flask import Flask, jsonify, request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --------------------------------------------------------------------------- #
# Model + data loaded once at startup                                          #
# --------------------------------------------------------------------------- #
MODEL_PATH = os.getenv("MODEL_PATH", "models/svd_model.pkl")
DATA_PATH  = os.getenv("DATA_PATH",  "data/processed/cleaned_ratings.csv")

_model = None
_all_movies = None


def _load_resources():
    global _model, _all_movies
    logger.info("Loading model from %s", MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)
    logger.info("Loading movie list from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    _all_movies = df["Movie Name"].unique().tolist()
    logger.info("Ready. %d unique movies loaded.", len(_all_movies))


# --------------------------------------------------------------------------- #
# Routes                                                                       #
# --------------------------------------------------------------------------- #
@app.route("/health", methods=["GET"])
def health():
    """Liveness probe used by ACI / Kubernetes."""
    return jsonify({"status": "ok"}), 200


@app.route("/score", methods=["POST"])
def score():
    """
    Return top-N predicted movies for a user.

    Request body (JSON):
        user_id  (int)  -- user to generate recommendations for
        top_n    (int)  -- number of recommendations (default 10)

    Returns (JSON):
        recommendations: list of {movie, predicted_rating}
        user_id, top_n, model_info
    """
    if _model is None:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(silent=True) or {}
    user_id = body.get("user_id")
    top_n   = int(body.get("top_n", 10))

    if user_id is None:
        return jsonify({"error": "user_id is required"}), 400

    # Predict ratings for all movies the user hasn't rated yet
    predictions = [
        (_model.predict(str(user_id), movie).est, movie)
        for movie in _all_movies
    ]
    predictions.sort(reverse=True)
    top_predictions = [
        {"movie": movie, "predicted_rating": round(rating, 3)}
        for rating, movie in predictions[:top_n]
    ]

    return jsonify({
        "user_id":         user_id,
        "top_n":           top_n,
        "recommendations": top_predictions,
        "model_info": {
            "type":    "SVD (scikit-surprise)",
            "note":    "Ratings are on a 1-10 scale. Predictions are interpolated estimates.",
        },
    }), 200


# --------------------------------------------------------------------------- #
# Entry point                                                                  #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    _load_resources()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=False)
