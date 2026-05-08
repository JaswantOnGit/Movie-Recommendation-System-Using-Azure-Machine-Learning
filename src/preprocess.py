"""
preprocess.py

Loads and cleans the MovieLens-style ratings dataset, merges with movie
metadata, and writes a cleaned CSV ready for model training.

Usage:
    python src/preprocess.py --ratings data/raw/movie_ratings.csv \
                             --movies  data/raw/imdb_movie_titles.csv \
                             --output  data/processed/cleaned_ratings.csv
                             """

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

REQUIRED_COLS = ["UserId", "Movie Name", "Rating"]


def load_data(ratings_path: str, movies_path: str) -> pd.DataFrame:
      """Load raw ratings and movie-title files and merge on movie ID."""
      logger.info("Loading ratings from %s", ratings_path)
      ratings = pd.read_csv(ratings_path)

    logger.info("Loading movie titles from %s", movies_path)
    movies = pd.read_csv(movies_path)

    # Flexible join: works whether the common key is 'movieId' or 'Movie ID'
    join_key = _find_common_key(ratings.columns, movies.columns)
    logger.info("Joining on column: %s", join_key)
    merged = ratings.merge(movies, on=join_key, how="inner")
    return merged


def _find_common_key(cols_a, cols_b):
      """Return the first column name that appears in both column lists."""
      candidates = ["movieId", "Movie ID", "movie_id"]
      for c in candidates:
                if c in cols_a and c in cols_b:
                              return c
                      common = set(cols_a) & set(cols_b)
            if not common:
                      raise ValueError("No common column found between ratings and movies files.")
                  return sorted(common)[0]


def clean(df: pd.DataFrame) -> pd.DataFrame:
      """Select required columns, drop duplicates and NaNs, validate ratings range."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
              raise KeyError(f"Missing expected columns after merge: {missing}")

    df = df[REQUIRED_COLS].copy()
    before = len(df)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    logger.info("Removed %d rows (duplicates / NaN). Remaining: %d", before - len(df), len(df))

    # Ratings outside [1, 10] are noise — log a warning but keep them so the
    # caller can decide how to handle edge cases.
    out_of_range = df[(df["Rating"] < 1) | (df["Rating"] > 10)]
    if not out_of_range.empty:
              logger.warning("%d ratings outside expected [1, 10] range.", len(out_of_range))

    return df


def save(df: pd.DataFrame, output_path: str) -> None:
      os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset (%d rows) to %s", len(df), output_path)


def main():
      parser = argparse.ArgumentParser(description="Preprocess movie ratings data.")
    parser.add_argument("--ratings", required=True, help="Path to raw ratings CSV")
    parser.add_argument("--movies", required=True, help="Path to movie titles CSV")
    parser.add_argument("--output", required=True, help="Path for cleaned output CSV")
    args = parser.parse_args()

    df = load_data(args.ratings, args.movies)
    df = clean(df)
    save(df, args.output)


if __name__ == "__main__":
      main()
