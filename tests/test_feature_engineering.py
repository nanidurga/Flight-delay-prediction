"""
Unit tests for FeatureEngineer.
Run from MTP root: pytest tests/test_feature_engineering.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import joblib

from api.services.feature_engineering import FeatureEngineer


def test_fit_returns_self(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer()
    result = fe.fit(sample_X, sample_y_total, sample_y_status)
    assert result is fe, "fit() must return self for chaining"


def test_transform_adds_15_columns(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    expected_cols = len(sample_X.columns) + 15
    assert out.shape == (len(sample_X), expected_cols), (
        f"Expected shape ({len(sample_X)}, {expected_cols}), got {out.shape}"
    )


def test_transform_requires_fit_first(sample_X):
    fe = FeatureEngineer()
    with pytest.raises(RuntimeError, match="fit"):
        fe.transform(sample_X)


def test_weather_severity_sunny_is_zero(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    sev_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("origin_weather_severity")
    assert out[0, sev_idx] == 0.0, f"Sunny should give severity 0, got {out[0, sev_idx]}"


def test_weather_severity_heavy_rain_is_8(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    sev_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("origin_weather_severity")
    assert out[2, sev_idx] == 8.0, f"Heavy rain should give severity 8, got {out[2, sev_idx]}"


def test_unseen_carrier_uses_global_mean(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    X_unknown = sample_X.copy()
    for c in fe.carrier_cols_:
        X_unknown[c] = 0.0
    out = fe.transform(X_unknown)
    mean_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("carrier_hist_mean_delay")
    assert out[0, mean_idx] == pytest.approx(fe.global_mean_, abs=0.01)


def test_serialize_deserialize_gives_same_output(sample_X, sample_y_total, sample_y_status, tmp_path):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    path = tmp_path / "fe.pkl"
    joblib.dump(fe, path)
    fe2 = joblib.load(path)
    out1 = fe.transform(sample_X)
    out2 = fe2.transform(sample_X)
    np.testing.assert_array_almost_equal(out1, out2, decimal=5)


def test_single_row_transform(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    single = sample_X.iloc[[0]]
    out = fe.transform(single)
    assert out.shape == (1, len(sample_X.columns) + 15)


def test_route_hist_n_flights_is_positive(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    n_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("route_hist_n_flights")
    assert out[:, n_idx].max() > 0


def test_engineered_col_names_length():
    assert len(FeatureEngineer.ENGINEERED_COL_NAMES) == 15
