"""
tests/test_preprocess.py

Unit tests for src/preprocess.py
Run with: pytest tests/
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

# Make sure src/ is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import clean, _find_common_key, REQUIRED_COLS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def clean_df():
    """Minimal valid DataFrame that should survive clean() unchanged."""
    return pd.DataFrame({
        "UserId":     [1, 2, 3],
        "Movie Name": ["Inception", "The Dark Knight", "Interstellar"],
        "Rating":     [8.0, 9.0, 8.5],
    })


@pytest.fixture
def dirty_df(clean_df):
    """DataFrame with duplicates and NaN rows."""
    extra = pd.DataFrame({
        "UserId":     [1, None],
        "Movie Name": ["Inception", "Unknown"],
        "Rating":     [8.0, 7.0],
    })
    return pd.concat([clean_df, extra], ignore_index=True)


# ---------------------------------------------------------------------------
# clean()
# ---------------------------------------------------------------------------
class TestClean:
    def test_returns_only_required_columns(self, clean_df):
        result = clean(clean_df)
        assert list(result.columns) == REQUIRED_COLS

    def test_removes_duplicates(self, dirty_df):
        result = clean(dirty_df)
        assert result.duplicated().sum() == 0

    def test_removes_nan_rows(self, dirty_df):
        result = clean(dirty_df)
        assert result.isnull().sum().sum() == 0

    def test_raises_on_missing_columns(self):
        bad_df = pd.DataFrame({"A": [1], "B": [2]})
        with pytest.raises(KeyError):
            clean(bad_df)

    def test_preserves_valid_rows(self, clean_df):
        result = clean(clean_df)
        assert len(result) == len(clean_df)


# ---------------------------------------------------------------------------
# _find_common_key()
# ---------------------------------------------------------------------------
class TestFindCommonKey:
    def test_finds_movieId(self):
        cols_a = ["movieId", "Rating", "UserId"]
        cols_b = ["movieId", "title"]
        assert _find_common_key(cols_a, cols_b) == "movieId"

    def test_finds_movie_id(self):
        cols_a = ["Movie ID", "Rating"]
        cols_b = ["Movie ID", "title"]
        assert _find_common_key(cols_a, cols_b) == "Movie ID"

    def test_raises_when_no_common_key(self):
        with pytest.raises(ValueError):
            _find_common_key(["A", "B"], ["C", "D"])
