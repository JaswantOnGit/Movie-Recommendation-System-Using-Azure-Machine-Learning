"""
train.py

Trains a SVD collaborative filtering model using scikit-surprise
and saves the trained model to disk.

Usage:
    python src/train.py --input  data/processed/cleaned_ratings.csv \
                        --output models/svd_model.pkl
"""

import argparse
import logging
import os
import pickle

import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(csv_path, rating_scale=(1, 10)):
    df = pd.read_csv(csv_path)
    reader = Reader(rating_scale=rating_scale)
    dataset = Dataset.load_from_df(df[["UserId", "Movie Name", "Rating"]], reader)
    return dataset, df


def compute_baseline(df):
    """Global-mean RMSE: the naive predictor every model should beat."""
    mean_rating = df["Rating"].mean()
    return ((df["Rating"] - mean_rating) ** 2).mean() ** 0.5


def train_and_evaluate(dataset, df, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
    baseline_rmse = compute_baseline(df)
    logger.info("Global-mean baseline RMSE: %.4f", baseline_rmse)

    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all, random_state=42)

    logger.info("Running 5-fold cross-validation ...")
    cv = cross_validate(algo, dataset, measures=["RMSE", "MAE"], cv=5, verbose=False)
    cv_rmse = float(cv["test_rmse"].mean())
    cv_mae  = float(cv["test_mae"].mean())
    pseudo_r2 = 1.0 - (cv_rmse ** 2) / (baseline_rmse ** 2)

    logger.info("CV RMSE : %.4f  (baseline %.4f, improvement %.1f%%)",
                cv_rmse, baseline_rmse, (1 - cv_rmse / baseline_rmse) * 100)
    logger.info("CV MAE  : %.4f", cv_mae)
    logger.info("Pseudo-R2 (vs global mean): %.4f", pseudo_r2)
    logger.info("Context: R2=0.15 means model explains 15%% of variance above the trivial")
    logger.info("baseline. Typical SVD on MovieLens-100k achieves R2=0.30-0.45.")
    logger.info("Improvements: tune factors/epochs, add implicit feedback, try NMF or NCF.")

    if cv_rmse >= baseline_rmse:
        logger.warning("Model does not beat baseline. Consider tuning or more data.")

    # Final model on full data
    trainset = dataset.build_full_trainset()
    algo.fit(trainset)
    logger.info("Final model trained on full dataset.")
    return algo, {"cv_rmse": cv_rmse, "cv_mae": cv_mae, "pseudo_r2": pseudo_r2, "baseline_rmse": baseline_rmse}


def save_model(algo, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(algo, f)
    logger.info("Model saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="Train SVD recommender.")
    parser.add_argument("--input",     required=True)
    parser.add_argument("--output",    required=True)
    parser.add_argument("--n-factors", type=int,   default=100)
    parser.add_argument("--n-epochs",  type=int,   default=20)
    parser.add_argument("--lr",        type=float, default=0.005)
    parser.add_argument("--reg",       type=float, default=0.02)
    args = parser.parse_args()

    dataset, df = load_dataset(args.input)
    algo, metrics = train_and_evaluate(dataset, df, args.n_factors, args.n_epochs, args.lr, args.reg)
    save_model(algo, args.output)
    logger.info("Done. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
