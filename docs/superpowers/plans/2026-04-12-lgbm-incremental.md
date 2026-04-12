# Sprint 7 — LightGBM + Incremental Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the DBSCAN+507RF classification pipeline with a single LightGBM model on 218 features, and add incremental learning via POST /feedback + nightly GitHub Actions warm-start retraining.

**Architecture:** Single LGBMClassifier + CalibratedClassifierCV on 218 features (203 raw + 15 engineered) replaces the DBSCAN+KNN+507RF architecture. Shared FeatureEngineer class is used by both training scripts and the inference path. Incremental learning uses LightGBM's `init_model` to warm-start on accumulated feedback rows without full retraining.

**Tech Stack:** Python 3.11, LightGBM 4.x, scikit-learn, FastAPI + pydantic v2, pandas, numpy, joblib, GitHub Actions

---

## File Map

**Created:**
- `api/services/feature_engineering.py` — FeatureEngineer class (fit/transform), 15 engineered features
- `train_lgbm.py` — full from-scratch training (replaces train.py + train_regressor.py)
- `train_incremental.py` — warm-start incremental update on feedback rows
- `tests/__init__.py` — empty, makes tests a package
- `tests/conftest.py` — shared pytest fixtures
- `tests/test_feature_engineering.py` — unit tests for FeatureEngineer
- `tests/test_predict.py` — unit tests for updated FlightPredictor
- `tests/test_api_feedback.py` — unit tests for POST /feedback endpoint
- `data/feedback.csv` — accumulates actual flight outcomes (header only to start)
- `.github/workflows/retrain.yml` — nightly incremental retrain job

**Modified:**
- `predict.py` — load LightGBM artifacts, use FeatureEngineer (API response shape identical)
- `api/main.py` — add POST /feedback endpoint, update /model/info response
- `api/requirements.txt` — add lightgbm
- `requirements.txt` — add lightgbm, pytest

**NOT modified:**
- `api/services/predictor.py` — thin wrapper around predict.py; interface preserved, no changes needed
- `api/services/flights.py` — unchanged
- All frontend files — unchanged (response shape preserved)

---

## Task 1: Setup — Dependencies, Test Scaffold, Feedback CSV

**Files:**
- Modify: `requirements.txt`
- Modify: `api/requirements.txt`
- Create: `data/feedback.csv`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add lightgbm and pytest to requirements.txt**

In `requirements.txt`, add these two lines after `scikit-learn`:
```
lightgbm>=4.0.0
pytest
```

- [ ] **Step 2: Add lightgbm to api/requirements.txt**

In `api/requirements.txt`, add this line after `scikit-learn`:
```
lightgbm>=4.0.0
```

- [ ] **Step 3: Install lightgbm and pytest**

Run: `pip install "lightgbm>=4.0.0" pytest`

Expected: `Successfully installed lightgbm-X.X.X` (no errors)

- [ ] **Step 4: Create data/feedback.csv with header only**

Create `data/feedback.csv` with exactly this content (one line):
```
flight_id,timestamp,actual_delayed,actual_delay_min
```

- [ ] **Step 5: Create tests/__init__.py (empty)**

Create `tests/__init__.py` — empty file, zero bytes.

- [ ] **Step 6: Create tests/conftest.py with shared fixtures**

Create `tests/conftest.py`:
```python
"""Shared pytest fixtures for all MTP tests.
Run all tests from MTP root: pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

# A minimal set of column names that mirrors the real dataset's structure.
# Enough to exercise every group type in FeatureEngineer.
SAMPLE_FEATURE_NAMES = [
    "CRS_ELAPSED_TIME", "DISTANCE",
    "origin_temperature_celsius", "origin_humidity",
    "dest_temperature_celsius", "dest_humidity",
    "DAY", "WeekendFlagEncoded", "DayBeforeWeekendEncoded",
    "Season_Autumn", "Season_Spring", "Season_Summer", "Season_Winter",
    "OP_CARRIER_American Airlines", "OP_CARRIER_Southwest Airlines",
    "OP_CARRIER_Delta Airlines",
    "origin_city_new york", "origin_city_atlanta", "origin_city_chicago",
    "destination_city_miami", "destination_city_chicago",
    "destination_city_new york",
    "origin_condition_text_Sunny", "origin_condition_text_Partly cloudy",
    "origin_condition_text_Heavy rain",
    "dest_condition_text_Sunny", "dest_condition_text_Partly cloudy",
    "CRS_DEP_TIME_2", "CRS_DEP_TIME_3", "CRS_DEP_TIME_4",
    "CRS_ARR_TIME_2", "CRS_ARR_TIME_4",
    "MONTH_6", "MONTH_7", "MONTH_9",
]


@pytest.fixture
def sample_X():
    """10-row DataFrame with one-hot groups activated per row."""
    n = 10
    np.random.seed(42)
    data = {col: np.zeros(n) for col in SAMPLE_FEATURE_NAMES}
    data["CRS_ELAPSED_TIME"] = np.random.uniform(60, 300, n)
    data["DISTANCE"] = np.random.uniform(200, 2000, n)
    data["origin_humidity"] = np.random.uniform(30, 90, n)
    data["dest_humidity"] = np.random.uniform(30, 90, n)

    # Activate one carrier per row (cycling)
    carriers = ["OP_CARRIER_American Airlines", "OP_CARRIER_Southwest Airlines",
                "OP_CARRIER_Delta Airlines"]
    for i in range(n):
        data[carriers[i % 3]][i] = 1.0

    # Activate one origin city per row
    origins = ["origin_city_new york", "origin_city_atlanta", "origin_city_chicago"]
    for i in range(n):
        data[origins[i % 3]][i] = 1.0

    # Activate one dest city per row
    dests = ["destination_city_miami", "destination_city_chicago",
             "destination_city_new york"]
    for i in range(n):
        data[dests[i % 3]][i] = 1.0

    # Activate weather conditions for a few rows
    data["origin_condition_text_Sunny"][0] = 1.0
    data["origin_condition_text_Partly cloudy"][1] = 1.0
    data["origin_condition_text_Heavy rain"][2] = 1.0
    data["dest_condition_text_Sunny"][0] = 1.0
    data["dest_condition_text_Partly cloudy"][1] = 1.0

    # Activate months
    for i in range(5):
        data["MONTH_7"][i] = 1.0
    for i in range(5, 10):
        data["MONTH_9"][i] = 1.0

    # Activate departure slots
    for i in range(3):
        data["CRS_DEP_TIME_2"][i] = 1.0
    for i in range(3, 6):
        data["CRS_DEP_TIME_3"][i] = 1.0
    for i in range(6, 10):
        data["CRS_DEP_TIME_4"][i] = 1.0

    return pd.DataFrame(data)


@pytest.fixture
def sample_y_total():
    """Total delay minutes — delayed rows have > 0 values."""
    return np.array([0, 120, 0, 45, 0, 80, 0, 200, 0, 60], dtype=float)


@pytest.fixture
def sample_y_status():
    """Binary FLIGHT_STATUS — alternating 0/1."""
    return np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
```

- [ ] **Step 7: Verify pytest runs (no tests yet, just scaffold check)**

Run: `pytest tests/ -v`
Expected:
```
collected 0 items
no tests ran
```
(No errors — just zero tests collected)

- [ ] **Step 8: Commit setup**

```bash
git add requirements.txt api/requirements.txt data/feedback.csv tests/
git commit -m "chore: add lightgbm/pytest deps, feedback.csv, test scaffold"
```

---

## Task 2: FeatureEngineer Class

**Files:**
- Create: `api/services/feature_engineering.py`
- Create: `tests/test_feature_engineering.py`

- [ ] **Step 1: Write failing tests first**

Create `tests/test_feature_engineering.py`:
```python
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
    # Row 0 has origin_condition_text_Sunny = 1 → severity = 0
    sev_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("origin_weather_severity")
    assert out[0, sev_idx] == 0.0, f"Sunny should give severity 0, got {out[0, sev_idx]}"


def test_weather_severity_heavy_rain_is_8(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    # Row 2 has origin_condition_text_Heavy rain = 1 → severity = 8
    sev_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("origin_weather_severity")
    assert out[2, sev_idx] == 8.0, f"Heavy rain should give severity 8, got {out[2, sev_idx]}"


def test_unseen_carrier_uses_global_mean(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    X_unknown = sample_X.copy()
    for c in fe.carrier_cols_:
        X_unknown[c] = 0.0  # no active carrier
    out = fe.transform(X_unknown)
    mean_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("carrier_hist_mean_delay")
    assert out[0, mean_idx] == pytest.approx(fe.global_mean_, abs=0.01), (
        "Unseen carrier should fall back to global_mean_"
    )


def test_serialize_deserialize_gives_same_output(sample_X, sample_y_total, sample_y_status, tmp_path):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    path = tmp_path / "fe.pkl"
    joblib.dump(fe, path)
    fe2 = joblib.load(path)
    out1 = fe.transform(sample_X)
    out2 = fe2.transform(sample_X)
    np.testing.assert_array_almost_equal(out1, out2, decimal=5)


def test_single_row_transform(sample_X, sample_y_total, sample_y_status):
    """transform() must work on a 1-row DataFrame — the inference use case."""
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    single = sample_X.iloc[[0]]
    out = fe.transform(single)
    assert out.shape == (1, len(sample_X.columns) + 15)


def test_route_hist_n_flights_is_positive(sample_X, sample_y_total, sample_y_status):
    fe = FeatureEngineer().fit(sample_X, sample_y_total, sample_y_status)
    out = fe.transform(sample_X)
    n_idx = len(sample_X.columns) + FeatureEngineer.ENGINEERED_COL_NAMES.index("route_hist_n_flights")
    # At least one row should have a non-zero route count (since we have repeated routes)
    assert out[:, n_idx].max() > 0, "route_hist_n_flights should be > 0 for seen routes"


def test_engineered_col_names_length():
    assert len(FeatureEngineer.ENGINEERED_COL_NAMES) == 15
```

- [ ] **Step 2: Run tests — confirm they fail with ImportError**

Run: `pytest tests/test_feature_engineering.py -v`

Expected: `ImportError: cannot import name 'FeatureEngineer' from 'api.services.feature_engineering'` (module doesn't exist yet)

- [ ] **Step 3: Implement FeatureEngineer**

Create `api/services/feature_engineering.py`:
```python
"""
feature_engineering.py — Shared FeatureEngineer used by train_lgbm.py,
train_incremental.py, and predict.py.

Computes 15 historical/interaction features leak-free from the training fold.
Saved as model/feature_engineering.pkl at train time; loaded at inference.

Usage (training):
    fe = FeatureEngineer()
    fe.fit(X_train_df, y_total_delay_arr, y_binary_status_arr)
    X_eng = fe.transform(X_train_df)   # shape (n, 218)
    joblib.dump(fe, 'model/feature_engineering.pkl')

Usage (inference):
    fe = joblib.load('model/feature_engineering.pkl')
    X_eng = fe.transform(single_row_df)   # shape (1, 218)
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Appends 15 engineered features to the 203 pre-flight columns.

    Engineered features (in order, appended after original columns):
      0  carrier_hist_mean_delay        avg delay for this carrier
      1  carrier_hist_delay_rate        P(delayed) for this carrier
      2  origin_hist_mean_delay         avg delay from this origin city
      3  origin_hist_delay_rate         P(delayed) from this origin city
      4  dest_hist_mean_delay           avg delay to this dest city
      5  depslot_hist_mean_delay        avg delay at this dep time slot
      6  month_hist_mean_delay          avg delay in this month
      7  route_hist_mean_delay          avg delay on (origin, dest) route
      8  route_hist_delay_rate          P(delayed) on this route
      9  route_hist_n_flights           flight volume on this route
     10  carrier_origin_hist_mean_delay avg delay for (carrier, origin) pair
     11  carrier_month_hist_mean_delay  avg delay for (carrier, month) pair
     12  depslot_origin_hist_mean_delay avg delay for (depslot, origin) pair
     13  origin_weather_severity        ordinal 0-8 from origin weather condition
     14  dest_weather_severity          ordinal 0-8 from dest weather condition
    """

    # Maps weather condition column suffixes to ordinal severity scores.
    WEATHER_SEVERITY: dict = {
        "Sunny": 0,
        "Partly Cloudy": 1, "Partly cloudy": 1,
        "Overcast": 2, "Cloudy": 2,
        "Mist": 3,
        "Fog": 4,
        "Light drizzle": 5, "Patchy rain nearby": 5,
        "Light rain": 6, "Light rain shower": 6,
        "Moderate rain": 7,
        "Heavy rain": 8,
        "Moderate or heavy rain shower": 8,
        "Moderate or heavy rain with thunder": 8,
        "Thundery outbreaks possible": 8,
        "Patchy light rain with thunder": 8,
    }

    ENGINEERED_COL_NAMES: list = [
        "carrier_hist_mean_delay",
        "carrier_hist_delay_rate",
        "origin_hist_mean_delay",
        "origin_hist_delay_rate",
        "dest_hist_mean_delay",
        "depslot_hist_mean_delay",
        "month_hist_mean_delay",
        "route_hist_mean_delay",
        "route_hist_delay_rate",
        "route_hist_n_flights",
        "carrier_origin_hist_mean_delay",
        "carrier_month_hist_mean_delay",
        "depslot_origin_hist_mean_delay",
        "origin_weather_severity",
        "dest_weather_severity",
    ]

    def __init__(self):
        self._fitted: bool = False

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y_total_delay: np.ndarray,
            y_binary_status: np.ndarray) -> "FeatureEngineer":
        """
        Compute all historical lookup tables from the training fold only.

        Parameters
        ----------
        X              : DataFrame with 203 pre-flight feature columns
        y_total_delay  : total delay minutes per row (sum of all delay cols)
        y_binary_status: 0/1 FLIGHT_STATUS label per row
        """
        self.feature_names_: list = list(X.columns)
        self.col_idx_: dict = {n: i for i, n in enumerate(self.feature_names_)}

        # Identify one-hot group column sets
        self.carrier_cols_  = [c for c in self.feature_names_ if c.startswith("OP_CARRIER_")]
        self.origin_cols_   = [c for c in self.feature_names_ if c.startswith("origin_city_")]
        self.dest_cols_     = [c for c in self.feature_names_ if c.startswith("destination_city_")]
        self.dep_slot_cols_ = [c for c in self.feature_names_ if c.startswith("CRS_DEP_TIME_")]
        self.month_cols_    = [c for c in self.feature_names_ if c.startswith("MONTH_")]
        self.orig_wx_cols_  = [c for c in self.feature_names_ if c.startswith("origin_condition_text_")]
        self.dest_wx_cols_  = [c for c in self.feature_names_ if c.startswith("dest_condition_text_")]

        delayed_mask = y_binary_status == 1
        self.global_mean_: float = (
            float(y_total_delay[delayed_mask].mean()) if delayed_mask.sum() > 0 else 0.0
        )
        self.global_rate_: float = float(y_binary_status.mean())

        X_np = X.values.astype(np.float32)

        # Single-group lookup tables
        self.carrier_mean_,  self.carrier_rate_  = self._group_stats(X_np, y_total_delay, y_binary_status, self.carrier_cols_)
        self.origin_mean_,   self.origin_rate_   = self._group_stats(X_np, y_total_delay, y_binary_status, self.origin_cols_)
        self.dest_mean_,     self.dest_rate_     = self._group_stats(X_np, y_total_delay, y_binary_status, self.dest_cols_)
        self.depslot_mean_,  _                   = self._group_stats(X_np, y_total_delay, y_binary_status, self.dep_slot_cols_)
        self.month_mean_,    _                   = self._group_stats(X_np, y_total_delay, y_binary_status, self.month_cols_)

        # Active column per row (used for route-pair and interaction features)
        active_carriers = self._active_col(X_np, self.carrier_cols_)
        active_origins  = self._active_col(X_np, self.origin_cols_)
        active_dests    = self._active_col(X_np, self.dest_cols_)
        active_slots    = self._active_col(X_np, self.dep_slot_cols_)
        active_months   = self._active_col(X_np, self.month_cols_)

        # Route-pair lookup: (origin_col, dest_col) → {mean, rate, count}
        self.route_mean_:  dict = {}
        self.route_rate_:  dict = {}
        self.route_count_: dict = {}
        _route_delay_acc:  dict = {}
        _route_status_acc: dict = {}

        for orig, dest, delay, status in zip(active_origins, active_dests,
                                              y_total_delay, y_binary_status):
            if orig is None or dest is None:
                continue
            key = (orig, dest)
            self.route_count_[key] = self.route_count_.get(key, 0) + 1
            if status == 1:
                _route_delay_acc.setdefault(key, []).append(delay)
            _route_status_acc.setdefault(key, []).append(status)

        for key in self.route_count_:
            delays = _route_delay_acc.get(key, [])
            self.route_mean_[key] = float(np.mean(delays)) if delays else self.global_mean_
            self.route_rate_[key] = float(np.mean(_route_status_acc[key]))

        # Interaction lookup tables
        self.carrier_origin_mean_  = self._interaction_stats(active_carriers, active_origins, y_total_delay, y_binary_status)
        self.carrier_month_mean_   = self._interaction_stats(active_carriers, active_months,  y_total_delay, y_binary_status)
        self.depslot_origin_mean_  = self._interaction_stats(active_slots,    active_origins,  y_total_delay, y_binary_status)

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Append 15 engineered features to X.

        Returns np.ndarray of shape (n, len(X.columns) + 15).
        Missing columns are filled with 0; extra columns are ignored.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureEngineer must be fit() before transform(). "
                "Load a saved instance with joblib.load() or call fit() first."
            )

        X_np = X.reindex(columns=self.feature_names_, fill_value=0).values.astype(np.float32)
        n = len(X_np)
        eng = np.empty((n, 15), dtype=np.float32)

        active_carriers = self._active_col(X_np, self.carrier_cols_)
        active_origins  = self._active_col(X_np, self.origin_cols_)
        active_dests    = self._active_col(X_np, self.dest_cols_)
        active_slots    = self._active_col(X_np, self.dep_slot_cols_)
        active_months   = self._active_col(X_np, self.month_cols_)

        gm = self.global_mean_
        gr = self.global_rate_

        for i in range(n):
            c  = active_carriers[i]
            o  = active_origins[i]
            d  = active_dests[i]
            sl = active_slots[i]
            mo = active_months[i]

            eng[i, 0]  = self.carrier_mean_.get(c,  gm)
            eng[i, 1]  = self.carrier_rate_.get(c,  gr)
            eng[i, 2]  = self.origin_mean_.get(o,   gm)
            eng[i, 3]  = self.origin_rate_.get(o,   gr)
            eng[i, 4]  = self.dest_mean_.get(d,     gm)
            eng[i, 5]  = self.depslot_mean_.get(sl, gm)
            eng[i, 6]  = self.month_mean_.get(mo,   gm)

            rk         = (o, d) if (o and d) else None
            eng[i, 7]  = self.route_mean_.get(rk,  gm)
            eng[i, 8]  = self.route_rate_.get(rk,  gr)
            eng[i, 9]  = float(self.route_count_.get(rk, 0))

            co_key     = (c, o)  if (c and o)  else None
            cm_key     = (c, mo) if (c and mo) else None
            so_key     = (sl, o) if (sl and o) else None
            eng[i, 10] = self.carrier_origin_mean_.get(co_key, gm)
            eng[i, 11] = self.carrier_month_mean_.get(cm_key,  gm)
            eng[i, 12] = self.depslot_origin_mean_.get(so_key, gm)

            eng[i, 13] = self._weather_severity(X_np[i], self.orig_wx_cols_)
            eng[i, 14] = self._weather_severity(X_np[i], self.dest_wx_cols_)

        return np.hstack([X_np, eng])

    # ── Private helpers ───────────────────────────────────────────────────────

    def _group_stats(self, X_np: np.ndarray, y_total: np.ndarray,
                     y_status: np.ndarray, group_cols: list) -> tuple:
        """Return (mean_lut, rate_lut): col_name → float."""
        mean_lut: dict = {}
        rate_lut: dict = {}
        for col in group_cols:
            if col not in self.col_idx_:
                continue
            idx  = self.col_idx_[col]
            mask = X_np[:, idx] == 1
            if mask.sum() < 10:   # skip groups with too few samples
                continue
            delayed = mask & (y_status == 1)
            mean_lut[col] = float(y_total[delayed].mean()) if delayed.sum() > 0 else self.global_mean_
            rate_lut[col] = float(y_status[mask].mean())
        return mean_lut, rate_lut

    def _active_col(self, X_np: np.ndarray, group_cols: list) -> list:
        """For each row, return the name of the active (value==1) column, or None."""
        result = [None] * len(X_np)
        valid  = [(c, self.col_idx_[c]) for c in group_cols if c in self.col_idx_]
        if not valid:
            return result
        cols, idxs = zip(*valid)
        mat   = X_np[:, list(idxs)]
        sums  = mat.sum(axis=1)
        argmx = np.argmax(mat, axis=1)
        for i, (a, s) in enumerate(zip(argmx, sums)):
            if s > 0:
                result[i] = cols[a]
        return result

    def _interaction_stats(self, active_a: list, active_b: list,
                            y_total: np.ndarray, y_status: np.ndarray) -> dict:
        """Return {(a_col, b_col): mean_delay_for_delayed_rows}."""
        buckets: dict = {}
        for a, b, delay, status in zip(active_a, active_b, y_total, y_status):
            if a is None or b is None:
                continue
            if status == 1:
                buckets.setdefault((a, b), []).append(delay)
        return {k: float(np.mean(v)) for k, v in buckets.items()}

    def _weather_severity(self, row: np.ndarray, wx_cols: list) -> float:
        """Return ordinal severity for the active weather condition column in row."""
        for col in wx_cols:
            if col not in self.col_idx_:
                continue
            if row[self.col_idx_[col]] == 1:
                # Strip prefix to get condition text
                name = (col.replace("origin_condition_text_", "")
                           .replace("dest_condition_text_", ""))
                return float(self.WEATHER_SEVERITY.get(name, 3))
        return 3.0   # default: moderate (mist-level) when no condition is active
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_feature_engineering.py -v`

Expected:
```
PASSED tests/test_feature_engineering.py::test_fit_returns_self
PASSED tests/test_feature_engineering.py::test_transform_adds_15_columns
PASSED tests/test_feature_engineering.py::test_transform_requires_fit_first
PASSED tests/test_feature_engineering.py::test_weather_severity_sunny_is_zero
PASSED tests/test_feature_engineering.py::test_weather_severity_heavy_rain_is_8
PASSED tests/test_feature_engineering.py::test_unseen_carrier_uses_global_mean
PASSED tests/test_feature_engineering.py::test_serialize_deserialize_gives_same_output
PASSED tests/test_feature_engineering.py::test_single_row_transform
PASSED tests/test_feature_engineering.py::test_route_hist_n_flights_is_positive
PASSED tests/test_feature_engineering.py::test_engineered_col_names_length
10 passed
```

- [ ] **Step 5: Commit**

```bash
git add api/services/feature_engineering.py tests/test_feature_engineering.py
git commit -m "feat: FeatureEngineer — 15 historical/interaction features, full tests"
```

---

## Task 3: train_lgbm.py — Full Training Script

**Files:**
- Create: `train_lgbm.py`

No unit tests for the training script (runtime ~5–10 min). We verify by running it and checking printed metrics + artifact existence.

- [ ] **Step 1: Create train_lgbm.py**

Create `train_lgbm.py` at the project root:
```python
"""
train_lgbm.py — Sprint 7 full LightGBM training pipeline
=========================================================

Replaces train.py + train_regressor.py.

Pipeline:
  Raw CSV (86,478 rows, 219 cols)
  -> Drop post-flight features -> 203 pre-flight features remain
  -> 70/10/20 split: train / calibration-val / test  (stratified, seed=42)
  -> FeatureEngineer.fit(train) -> 218 features
  -> LGBMClassifier  (early stop on val)
  -> CalibratedClassifierCV(cv='prefit').fit(val)
  -> LGBMRegressor  main + p10/p90 + 4 per-type  (log1p target, delayed only)
  -> Evaluate on strictly held-out test set
  -> Save all artifacts to model/

Run: python train_lgbm.py   (takes ~5-10 min)
"""

import warnings
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR  = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "final_preprocessed_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from api.services.feature_engineering import FeatureEngineer

from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                              mean_absolute_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
POST_FLIGHT  = [
    "Unnamed: 0", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
TARGET     = "FLIGHT_STATUS"
DELAY_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
              "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
SEP = "=" * 65

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
print(SEP); print("STEP 1 — Loading data"); print(SEP)

data = pd.read_csv(DATA_PATH)
data["total_delay"] = data[DELAY_COLS].sum(axis=1)
print(f"  Loaded {len(data):,} rows × {data.shape[1]} columns")

drop_cols = [c for c in POST_FLIGHT if c in data.columns] + [TARGET, "total_delay"]
X_raw     = data.drop(columns=drop_cols)
constant  = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
if constant:
    X_raw = X_raw.drop(columns=constant)
    print(f"  Dropped {len(constant)} constant columns")

feature_names = list(X_raw.columns)
print(f"  Pre-flight features: {len(feature_names)}")

y_status  = data[TARGET].values
y_total   = data["total_delay"].values
y_carrier = data["CARRIER_DELAY"].values
y_weather = data["WEATHER_DELAY"].values
y_nas     = data["NAS_DELAY"].values
y_late_ac = data["LATE_AIRCRAFT_DELAY"].values

joblib.dump(feature_names, MODEL_DIR / "feature_names.pkl")

# ── 2. THREE-WAY SPLIT: 70% train / 10% val / 20% test ───────────────────────
print(f"\nSTEP 2 — Three-way split (70/10/20, stratified, seed={RANDOM_STATE})")

idx = np.arange(len(data))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_status)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125,   # 0.125 × 0.80 ≈ 0.10 of total
    random_state=RANDOM_STATE, stratify=y_status[idx_trainval])

print(f"  Train: {len(idx_train):,}  |  Val: {len(idx_val):,}  |  Test: {len(idx_test):,}")

X_tr_df  = pd.DataFrame(X_raw.values[idx_train],  columns=feature_names)
X_val_df = pd.DataFrame(X_raw.values[idx_val],    columns=feature_names)
X_te_df  = pd.DataFrame(X_raw.values[idx_test],   columns=feature_names)

y_tr_status  = y_status[idx_train];   y_val_status = y_status[idx_val];   y_te_status = y_status[idx_test]
y_tr_total   = y_total[idx_train];    y_val_total  = y_total[idx_val];    y_te_total  = y_total[idx_test]
y_tr_carrier = y_carrier[idx_train];  y_te_carrier = y_carrier[idx_test]
y_tr_weather = y_weather[idx_train];  y_te_weather = y_weather[idx_test]
y_tr_nas     = y_nas[idx_train];      y_te_nas     = y_nas[idx_test]
y_tr_late    = y_late_ac[idx_train];  y_te_late    = y_late_ac[idx_test]

# ── 3. FEATURE ENGINEERING ────────────────────────────────────────────────────
print(f"\nSTEP 3 — Feature engineering (fit on train fold only)")

fe = FeatureEngineer()
fe.fit(X_tr_df, y_tr_total, y_tr_status)

X_tr  = fe.transform(X_tr_df)
X_val = fe.transform(X_val_df)
X_te  = fe.transform(X_te_df)

print(f"  Feature shape: {X_tr.shape}  ({len(feature_names)} pre-flight + 15 engineered)")
joblib.dump(fe, MODEL_DIR / "feature_engineering.pkl")
print("  Saved -> model/feature_engineering.pkl")

# ── 4. CLASSIFICATION ─────────────────────────────────────────────────────────
print(f"\nSTEP 4 — LGBMClassifier (early stopping on val, then isotonic calibration)")

clf = LGBMClassifier(
    n_estimators      = 1000,
    learning_rate     = 0.05,
    max_depth         = 8,
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
clf.fit(
    X_tr, y_tr_status,
    eval_set=[(X_val, y_val_status)],
    callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
)
print(f"  Best iteration: {clf.best_iteration_}")
joblib.dump(clf, MODEL_DIR / "lgbm_clf.pkl")
print("  Saved raw classifier -> model/lgbm_clf.pkl")

# Calibrate on val set (cv='prefit': no refitting, just calibration layer)
cal_clf = CalibratedClassifierCV(clf, cv="prefit", method="isotonic")
cal_clf.fit(X_val, y_val_status)
joblib.dump(cal_clf, MODEL_DIR / "lgbm_clf_calibrated.pkl")
print("  Saved calibrated classifier -> model/lgbm_clf_calibrated.pkl")

# ── 5. REGRESSION ─────────────────────────────────────────────────────────────
print(f"\nSTEP 5 — Regression suite (delayed flights only, log1p target)")

tr_delayed  = y_tr_status == 1
val_delayed = y_val_status == 1
te_delayed  = y_te_status == 1

X_tr_d  = X_tr[tr_delayed];    X_val_d = X_val[val_delayed];   X_te_d = X_te[te_delayed]
y_tr_d  = y_tr_total[tr_delayed];  y_val_d = y_val_total[val_delayed]; y_te_d = y_te_total[te_delayed]
y_tr_log  = np.log1p(y_tr_d)
y_val_log = np.log1p(y_val_d)

print(f"  Delayed rows — train: {tr_delayed.sum():,}  val: {val_delayed.sum():,}  test: {te_delayed.sum():,}")

REG_BASE = dict(
    n_estimators=1000, learning_rate=0.04, max_depth=8, num_leaves=63,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
)

# Main point-estimate regressor
reg_main = LGBMRegressor(objective="regression", **REG_BASE)
reg_main.fit(X_tr_d, y_tr_log,
             eval_set=[(X_val_d, y_val_log)],
             callbacks=[early_stopping(50, verbose=False), log_evaluation(100)])
preds = np.clip(np.expm1(reg_main.predict(X_te_d)), 1, 800)
mae   = mean_absolute_error(y_te_d, preds)
medae = median_absolute_error(y_te_d, preds)
r2    = r2_score(y_te_d, preds)
print(f"  Main regressor — MAE: {mae:.1f} min | Median AE: {medae:.1f} | R²: {r2:.3f}")
joblib.dump(reg_main, MODEL_DIR / "lgbm_reg.pkl")

# Quantile regressors (80% prediction interval)
Q_BASE = {**REG_BASE, "n_estimators": 600, "max_depth": 6}
reg_p10 = LGBMRegressor(objective="quantile", alpha=0.10, **Q_BASE)
reg_p90 = LGBMRegressor(objective="quantile", alpha=0.90, **Q_BASE)
reg_p10.fit(X_tr_d, y_tr_log, eval_set=[(X_val_d, y_val_log)],
            callbacks=[early_stopping(30, verbose=False)])
reg_p90.fit(X_tr_d, y_tr_log, eval_set=[(X_val_d, y_val_log)],
            callbacks=[early_stopping(30, verbose=False)])
p10 = np.clip(np.expm1(reg_p10.predict(X_te_d)), 0, 800)
p90 = np.clip(np.expm1(reg_p90.predict(X_te_d)), 0, 800)
coverage = float(np.mean((y_te_d >= p10) & (y_te_d <= p90)))
print(f"  80% PI coverage: {coverage*100:.1f}%  (target ~80%)")
joblib.dump(reg_p10, MODEL_DIR / "lgbm_reg_p10.pkl")
joblib.dump(reg_p90, MODEL_DIR / "lgbm_reg_p90.pkl")

# Per-type regressors
TYPE_BASE = {**REG_BASE, "n_estimators": 500, "max_depth": 6}
type_targets = {
    "carrier":       (y_tr_carrier[tr_delayed],   y_te_carrier[te_delayed]),
    "weather":       (y_tr_weather[tr_delayed],   y_te_weather[te_delayed]),
    "nas":           (y_tr_nas[tr_delayed],        y_te_nas[te_delayed]),
    "late_aircraft": (y_tr_late[tr_delayed],       y_te_late[te_delayed]),
}
type_regressors = {}
for name, (y_tr_t, y_te_t) in type_targets.items():
    rg = LGBMRegressor(objective="regression", **TYPE_BASE)
    rg.fit(X_tr_d, np.log1p(y_tr_t))
    tp  = np.clip(np.expm1(rg.predict(X_te_d)), 0, 800)
    t_mae   = mean_absolute_error(y_te_t, tp)
    t_medae = median_absolute_error(y_te_t, tp)
    print(f"  {name:20s}: MAE={t_mae:.1f} min | Median AE={t_medae:.1f}")
    type_regressors[name] = rg
joblib.dump(type_regressors, MODEL_DIR / "lgbm_type_regressors.pkl")

# ── 6. EVALUATE CLASSIFIER ────────────────────────────────────────────────────
print(f"\nSTEP 6 — Final classifier evaluation on held-out test set")

probs_cal = cal_clf.predict_proba(X_te)[:, 1]
preds_bin = (probs_cal >= 0.5).astype(int)
accuracy  = accuracy_score(y_te_status, preds_bin)
roc_auc   = roc_auc_score(y_te_status, probs_cal)

print(f"\n  Test Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  ROC-AUC       : {roc_auc:.4f}")
print(classification_report(y_te_status, preds_bin, target_names=["On-Time", "Delayed"]))

# ── 7. SAVE METADATA ──────────────────────────────────────────────────────────
metadata = {
    # New fields
    "model_version"        : 3,
    "trained_at"           : datetime.now(timezone.utc).isoformat(),
    "model_type"           : "LGBMClassifier + LGBMRegressor",
    "overall_accuracy"     : accuracy,
    "roc_auc"              : roc_auc,
    "regression_mae"       : mae,
    "regression_median_ae" : medae,
    "regression_r2"        : r2,
    "pi_80_coverage"       : coverage,
    "n_features"           : X_tr.shape[1],
    "feature_names"        : feature_names,
    "incremental_updates"  : 0,
    "feedback_rows_used"   : 0,
    # Legacy keys kept for backward compat with /model/info
    "n_clusters"           : 0,
    "n_cluster_models"     : 0,
    "selected_features"    : feature_names[:30],
    "top_k"                : len(feature_names),
    "dbscan_eps"           : 0,
    "dbscan_min_samples"   : 0,
}
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")
print(f"\n  Saved metadata -> model/metadata.pkl")
print(f"\n{SEP}")
print("  TRAINING COMPLETE — artifacts saved to model/")
print(SEP)
```

- [ ] **Step 2: Run the training script**

Run: `python train_lgbm.py`

This takes 5–10 minutes. Watch for these lines in output:
```
STEP 6 — Final classifier evaluation on held-out test set
  Test Accuracy : 0.XXXX (XX.XX%)
  ROC-AUC       : 0.XXXX
```

**Targets to verify:** accuracy > 0.78, ROC-AUC > 0.85, MAE < 28 min.

If accuracy is below 0.75, check that `lightgbm>=4.0.0` is installed and that there are no import errors in the FeatureEngineer.

- [ ] **Step 3: Verify all model artifacts exist**

Run:
```bash
python -c "
from pathlib import Path
expected = ['lgbm_clf.pkl','lgbm_clf_calibrated.pkl','lgbm_reg.pkl',
            'lgbm_reg_p10.pkl','lgbm_reg_p90.pkl','lgbm_type_regressors.pkl',
            'feature_engineering.pkl','feature_names.pkl','metadata.pkl']
missing = [f for f in expected if not (Path('model')/f).exists()]
print('Missing:', missing if missing else 'none — all good')
"
```
Expected: `Missing: none — all good`

- [ ] **Step 4: Commit**

```bash
git add train_lgbm.py model/lgbm_clf.pkl model/lgbm_clf_calibrated.pkl model/lgbm_reg.pkl model/lgbm_reg_p10.pkl model/lgbm_reg_p90.pkl model/lgbm_type_regressors.pkl model/feature_engineering.pkl model/feature_names.pkl model/metadata.pkl
git commit -m "feat: train_lgbm.py — LightGBM pipeline on 218 features"
```

---

## Task 4: Update predict.py

**Files:**
- Modify: `predict.py`
- Create: `tests/test_predict.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_predict.py`:
```python
"""
Tests for the updated FlightPredictor (LightGBM v3).
Run from MTP root: pytest tests/test_predict.py -v
Requires trained model artifacts in model/ (run train_lgbm.py first).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
from predict import FlightPredictor

# A high-delay-risk flight input
DELAYED_FEATURES = {
    "CRS_ELAPSED_TIME": 150, "DISTANCE": 800,
    "origin_humidity": 85, "dest_humidity": 80,
    "OP_CARRIER_Southwest Airlines": 1,
    "origin_city_new york": 1,
    "destination_city_miami": 1,
    "Season_Summer": 1, "WeekendFlagEncoded": 1,
    "CRS_DEP_TIME_4": 1, "MONTH_7": 1,
}
# A low-delay-risk flight input
ONTIME_FEATURES = {
    "CRS_ELAPSED_TIME": 90, "DISTANCE": 400,
    "origin_humidity": 40, "dest_humidity": 35,
    "OP_CARRIER_Delta Airlines": 1,
    "origin_city_atlanta": 1,
    "Season_Autumn": 1, "WeekendFlagEncoded": 0,
    "CRS_DEP_TIME_2": 1, "MONTH_9": 1,
    "origin_condition_text_Sunny": 1,
}

@pytest.fixture(scope="module")
def predictor():
    return FlightPredictor()


def test_predict_returns_all_required_keys(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    required = {
        "delayed", "probability", "probability_pct",
        "cluster", "model_used", "confidence",
        "expected_delay_min", "delay_range",
        "delay_category", "delay_breakdown",
    }
    missing = required - result.keys()
    assert not missing, f"Response missing keys: {missing}"


def test_probability_in_unit_interval(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    assert 0.0 <= result["probability"] <= 1.0


def test_model_used_is_lgbm(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    assert result["model_used"] == "lgbm", (
        f"Expected 'lgbm', got '{result['model_used']}'"
    )


def test_cluster_is_minus_one(predictor):
    """cluster field should be -1 (no clustering in new pipeline)."""
    result = predictor.predict(DELAYED_FEATURES)
    assert result["cluster"] == -1


def test_ontime_flight_has_zero_delay(predictor):
    result = predictor.predict(ONTIME_FEATURES)
    if not result["delayed"]:
        assert result["expected_delay_min"] == 0
        assert result["delay_category"] == "on-time"
        assert result["delay_range"] == "No delay expected"


def test_delayed_result_has_positive_delay_min(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert result["expected_delay_min"] > 0
        assert result["delay_category"] in ("minor", "moderate", "significant", "severe")


def test_breakdown_has_correct_keys(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert set(result["delay_breakdown"].keys()) == {
            "carrier", "weather", "nas", "late_aircraft"
        }


def test_breakdown_sums_near_expected_delay(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        total = sum(result["delay_breakdown"].values())
        expected = result["expected_delay_min"]
        assert abs(total - expected) <= 2, (
            f"Breakdown sum {total} differs from expected_delay_min {expected} by >2 min"
        )


def test_delay_range_contains_dash_when_delayed(predictor):
    result = predictor.predict(DELAYED_FEATURES)
    if result["delayed"]:
        assert "–" in result["delay_range"], (
            f"delay_range should contain en-dash, got: {result['delay_range']}"
        )


def test_predict_batch_returns_correct_count(predictor):
    results = predictor.predict_batch([DELAYED_FEATURES, ONTIME_FEATURES])
    assert len(results) == 2
    assert all("delayed" in r for r in results)
```

- [ ] **Step 2: Run tests — confirm failures**

Run: `pytest tests/test_predict.py -v`

Expected: `test_model_used_is_lgbm` and `test_cluster_is_minus_one` fail (current predict.py returns `"cluster_42"` and an int cluster id, not `"lgbm"` and `-1`).

- [ ] **Step 3: Replace predict.py**

Overwrite `predict.py` with this content:
```python
"""
predict.py — Inference module for flight delay prediction (v3 — LightGBM)
=========================================================================

Classification:
  feature dict → FeatureEngineer.transform() → CalibratedLGBMClassifier
  → probability, delayed, confidence

Regression (delayed flights only):
  same features → LGBMRegressor main + p10/p90 + 4 per-type
  → expected_delay_min, delay_range [p10–p90], delay_breakdown

Response shape is identical to v2 so the frontend needs no changes.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "model"


class FlightPredictor:

    def __init__(self):
        print("Loading LightGBM model artifacts...")

        self.feature_engineer = joblib.load(MODEL_DIR / "feature_engineering.pkl")
        self.feature_names    = joblib.load(MODEL_DIR / "feature_names.pkl")
        self.metadata         = joblib.load(MODEL_DIR / "metadata.pkl")

        self.clf       = joblib.load(MODEL_DIR / "lgbm_clf_calibrated.pkl")
        self.reg_main  = joblib.load(MODEL_DIR / "lgbm_reg.pkl")
        self.reg_p10   = joblib.load(MODEL_DIR / "lgbm_reg_p10.pkl")
        self.reg_p90   = joblib.load(MODEL_DIR / "lgbm_reg_p90.pkl")
        self.type_regs = joblib.load(MODEL_DIR / "lgbm_type_regressors.pkl")

        acc = self.metadata.get("overall_accuracy", 0)
        n   = self.metadata.get("n_features", 0)
        ver = self.metadata.get("model_version", "?")
        print(f"  v{ver} — accuracy: {acc:.4f} | features: {n}")

    def predict(self, features: dict) -> dict:
        """
        Predict flight delay for a single flight.

        Parameters
        ----------
        features : dict mapping feature column names to numeric values.
                   Columns not in the training schema default to 0.

        Returns
        -------
        dict with keys:
          delayed (bool), probability (float), probability_pct (str),
          cluster (int, always -1), model_used (str, always 'lgbm'),
          confidence (str: 'high'/'medium'/'low'),
          expected_delay_min (int), delay_range (str),
          delay_category (str), delay_breakdown (dict)
        """
        # Build a 203-feature DataFrame aligned to the training column order
        row = {feat: features.get(feat, 0) for feat in self.feature_names}
        df  = pd.DataFrame([row])

        # Apply feature engineering → 218-feature array
        X_eng = self.feature_engineer.transform(df)   # shape (1, 218)

        # ── Classification ────────────────────────────────────────────
        prob  = float(self.clf.predict_proba(X_eng)[0][1])
        pred  = int(prob >= 0.5)

        distance   = abs(prob - 0.5)
        confidence = ("high"   if distance > 0.30 else
                      "medium" if distance > 0.15 else "low")

        if not pred:
            return {
                "delayed"            : False,
                "probability"        : round(prob, 4),
                "probability_pct"    : f"{prob*100:.1f}%",
                "cluster"            : -1,
                "model_used"         : "lgbm",
                "confidence"         : confidence,
                "expected_delay_min" : 0,
                "delay_range"        : "No delay expected",
                "delay_category"     : "on-time",
                "delay_breakdown"    : {
                    "carrier": 0, "weather": 0, "nas": 0, "late_aircraft": 0
                },
            }

        # ── Regression ────────────────────────────────────────────────
        pred_log     = float(self.reg_main.predict(X_eng)[0])
        expected_min = int(np.clip(np.expm1(pred_log), 1, 800))

        p10_log = float(self.reg_p10.predict(X_eng)[0])
        p90_log = float(self.reg_p90.predict(X_eng)[0])
        lo      = max(1, int(np.expm1(max(p10_log, 0))))
        hi      = int(np.clip(np.expm1(p90_log), lo + 1, 800))

        if   expected_min < 30:  delay_category = "minor"
        elif expected_min < 60:  delay_category = "moderate"
        elif expected_min < 120: delay_category = "significant"
        else:                    delay_category = "severe"

        # Per-type breakdown, rescaled so components sum to expected_min
        raw_bd     = {}
        type_total = 0
        for name, rg in self.type_regs.items():
            val = max(0, int(np.expm1(float(rg.predict(X_eng)[0]))))
            raw_bd[name]  = val
            type_total   += val

        if type_total > 0 and expected_min > 0:
            scale     = expected_min / type_total
            breakdown = {k: max(0, int(v * scale)) for k, v in raw_bd.items()}
        else:
            share     = expected_min // 4
            breakdown = {k: share for k in raw_bd}

        return {
            "delayed"            : True,
            "probability"        : round(prob, 4),
            "probability_pct"    : f"{prob*100:.1f}%",
            "cluster"            : -1,
            "model_used"         : "lgbm",
            "confidence"         : confidence,
            "expected_delay_min" : expected_min,
            "delay_range"        : f"{lo}–{hi} min",
            "delay_category"     : delay_category,
            "delay_breakdown"    : breakdown,
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        return [self.predict(r) for r in records]

    @property
    def info(self) -> dict:
        return self.metadata
```

- [ ] **Step 4: Run tests — expect all pass**

Run: `pytest tests/test_predict.py -v`

Expected: all 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add predict.py tests/test_predict.py
git commit -m "feat: predict.py v3 — LightGBM + FeatureEngineer, identical response shape"
```

---

## Task 5: Update api/main.py — POST /feedback + /model/info

**Files:**
- Modify: `api/main.py`
- Create: `tests/test_api_feedback.py`

Note: `api/services/predictor.py` requires no changes — it is a thin wrapper that loads `FlightPredictor` from `predict.py`, and the `predict.py` interface is preserved.

- [ ] **Step 1: Write failing tests**

Create `tests/test_api_feedback.py`:
```python
"""
Tests for POST /feedback endpoint and updated /model/info.
Run from MTP root: pytest tests/test_api_feedback.py -v
"""
import sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import api.main as main_module
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def patch_feedback_path(tmp_path, monkeypatch):
    """Redirect feedback writes to a temp file so tests don't pollute data/feedback.csv."""
    fb = tmp_path / "feedback.csv"
    fb.write_text("flight_id,timestamp,actual_delayed,actual_delay_min\n")
    monkeypatch.setattr(main_module, "FEEDBACK_PATH", fb)
    return fb


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_feedback_valid_returns_200(client):
    resp = client.post("/feedback", json={
        "flight_id": "UA123_2026-04-12",
        "actual_delayed": True,
        "actual_delay_min": 87,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "recorded"
    assert isinstance(body["feedback_count"], int)
    assert body["feedback_count"] >= 1


def test_feedback_duplicate_flight_id_ignored(client, tmp_path, monkeypatch):
    fb = tmp_path / "fb_dup.csv"
    fb.write_text("flight_id,timestamp,actual_delayed,actual_delay_min\n")
    monkeypatch.setattr(main_module, "FEEDBACK_PATH", fb)

    payload = {"flight_id": "DUP001", "actual_delayed": True, "actual_delay_min": 45}
    client.post("/feedback", json=payload)
    resp = client.post("/feedback", json=payload)
    assert resp.status_code == 200

    rows = [r for r in fb.read_text().strip().split("\n")[1:] if r.startswith("DUP001")]
    assert len(rows) == 1, f"Expected 1 row, got {len(rows)}"


def test_feedback_delay_min_over_800_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD001", "actual_delayed": True, "actual_delay_min": 801,
    })
    assert resp.status_code == 422


def test_feedback_delayed_true_but_zero_minutes_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD002", "actual_delayed": True, "actual_delay_min": 0,
    })
    assert resp.status_code == 422


def test_feedback_delayed_false_but_high_minutes_rejected(client):
    resp = client.post("/feedback", json={
        "flight_id": "BAD003", "actual_delayed": False, "actual_delay_min": 60,
    })
    assert resp.status_code == 422


def test_model_info_has_new_keys(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("model_version", "roc_auc", "model_type",
                "regression_mae", "incremental_updates"):
        assert key in body, f"Missing key: {key}"
```

- [ ] **Step 2: Run tests — confirm failures**

Run: `pytest tests/test_api_feedback.py -v`

Expected: `test_feedback_valid_returns_200` fails with 404 (endpoint not found) and `test_model_info_has_new_keys` fails (missing keys).

- [ ] **Step 3: Update api/main.py — add imports and FEEDBACK_PATH**

At the top of `api/main.py`, the existing imports include `from pathlib import Path` and `from datetime import datetime`. Add `import csv` after the stdlib imports block (around line 15):

Find this line:
```python
from fastapi import FastAPI, HTTPException, Query
```

Add `import csv` on the line before it:
```python
import csv
from fastapi import FastAPI, HTTPException, Query
```

After line 36 (just before `app = FastAPI(...)`), add:
```python
FEEDBACK_PATH = Path(__file__).parent.parent / "data" / "feedback.csv"
```

- [ ] **Step 4: Add FeedbackInput schema to api/main.py**

Find the `class PredictionResponse(BaseModel):` block (around line 95). Add this new schema right after the `PredictionResponse` class ends (after the `delay_breakdown` field):

```python
class FeedbackInput(BaseModel):
    """Actual outcome for a previously predicted flight."""
    flight_id       : str
    actual_delayed  : bool
    actual_delay_min: int = Field(..., ge=0, le=800,
                                  description="Actual delay in minutes (0–800)")

    @model_validator(mode="after")
    def check_delay_consistency(self) -> "FeedbackInput":
        if self.actual_delayed and self.actual_delay_min == 0:
            raise ValueError(
                "actual_delay_min must be > 0 when actual_delayed is True")
        if not self.actual_delayed and self.actual_delay_min > 15:
            raise ValueError(
                "actual_delay_min > 15 but actual_delayed is False — "
                "a flight with >15 min delay should have actual_delayed=True")
        return self
```

Also update the pydantic import at the top of the file. Find:
```python
from pydantic import BaseModel, Field
```
Replace with:
```python
from pydantic import BaseModel, Field, model_validator
```

- [ ] **Step 5: Add POST /feedback endpoint to api/main.py**

Add this endpoint at the end of `api/main.py`, after the `@app.get("/stats/overview")` endpoint:

```python
@app.post("/feedback", tags=["feedback"])
def record_feedback(fb: FeedbackInput):
    """
    Record the actual delay outcome for a previously predicted flight.
    Appends one row to data/feedback.csv for nightly incremental retraining.
    Idempotent: duplicate flight_id values are silently ignored.
    """
    # Read existing flight IDs to detect duplicates
    if FEEDBACK_PATH.exists():
        with open(FEEDBACK_PATH, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)   # skip header
            existing_ids = {row[0] for row in reader if row}
        if fb.flight_id in existing_ids:
            count = max(0, sum(1 for _ in open(FEEDBACK_PATH, encoding="utf-8")) - 1)
            return {"status": "duplicate_ignored", "feedback_count": count}

    # Append new row
    write_header = (not FEEDBACK_PATH.exists() or
                    FEEDBACK_PATH.stat().st_size == 0)
    with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["flight_id", "timestamp",
                              "actual_delayed", "actual_delay_min"])
        writer.writerow([
            fb.flight_id,
            datetime.utcnow().isoformat(),
            int(fb.actual_delayed),
            fb.actual_delay_min,
        ])

    count = max(0, sum(1 for _ in open(FEEDBACK_PATH, encoding="utf-8")) - 1)
    return {"status": "recorded", "feedback_count": count}
```

- [ ] **Step 6: Update /model/info endpoint in api/main.py**

Find the existing `@app.get("/model/info")` endpoint and replace its body:

```python
@app.get("/model/info", tags=["model"])
def model_info():
    """Return training metadata: accuracy, ROC-AUC, regression metrics, version."""
    predictor = get_predictor()
    info = predictor.info
    return {
        "model_version"        : info.get("model_version", 1),
        "trained_at"           : info.get("trained_at", "unknown"),
        "model_type"           : info.get("model_type", "LGBMClassifier + LGBMRegressor"),
        "overall_accuracy_pct" : f"{info['overall_accuracy']*100:.2f}%",
        "roc_auc"              : round(info.get("roc_auc", 0), 4),
        "regression_mae"       : round(info.get("regression_mae", 0), 1),
        "regression_r2"        : round(info.get("regression_r2", 0), 3),
        "pi_80_coverage"       : round(info.get("pi_80_coverage", 0), 3),
        "n_features"           : info.get("n_features", 0),
        "incremental_updates"  : info.get("incremental_updates", 0),
        "feedback_rows_used"   : info.get("feedback_rows_used", 0),
        # Legacy fields — kept so any external clients don't break
        "n_clusters"           : info.get("n_clusters", 0),
        "n_cluster_models"     : info.get("n_cluster_models", 0),
        "selected_features"    : info.get("selected_features", []),
    }
```

- [ ] **Step 7: Run tests — expect all pass**

Run: `pytest tests/test_api_feedback.py -v`

Expected: all 6 tests pass.

- [ ] **Step 8: Manual smoke-test the API**

Run: `uvicorn api.main:app --port 8000`

In a second terminal:
```bash
curl http://localhost:8000/model/info
# Should return JSON with model_version, roc_auc, model_type fields

curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"flight_id":"TEST001","actual_delayed":true,"actual_delay_min":65}'
# Should return: {"status":"recorded","feedback_count":1}
```

Stop API with Ctrl+C.

- [ ] **Step 9: Commit**

```bash
git add api/main.py tests/test_api_feedback.py
git commit -m "feat: POST /feedback endpoint + updated /model/info with LightGBM metrics"
```

---

## Task 6: train_incremental.py

**Files:**
- Create: `train_incremental.py`

- [ ] **Step 1: Create train_incremental.py**

Create `train_incremental.py` at the project root:
```python
"""
train_incremental.py — Warm-start LightGBM update on feedback rows
==================================================================

Reads data/feedback.csv (actual flight outcomes from POST /feedback).
If fewer than RETRAIN_THRESHOLD rows exist, exits cleanly (no-op).

When enough rows exist:
  1. Rebuilds the validation set from original data (same seed=42 split)
  2. Warm-starts LGBMClassifier with init_model (adds N_NEW_TREES new trees)
  3. Re-calibrates classifier on val set
  4. Safety gate: if accuracy drops > ACCURACY_FLOOR, rolls back + exits 1
  5. Saves versioned backup (lgbm_clf_v{N}.pkl), keeps last MAX_VERSIONS
  6. Overwrites lgbm_clf.pkl + lgbm_clf_calibrated.pkl
  7. Updates metadata.pkl
  8. Archives processed rows to data/feedback_archive.csv
  9. Resets data/feedback.csv to header-only

Run: python train_incremental.py
Exit code 0 = success or no-op.  Exit code 1 = rollback (accuracy degraded).
"""

import warnings
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).parent
MODEL_DIR    = BASE_DIR / "model"
DATA_DIR     = BASE_DIR / "data"
FEEDBACK_CSV = DATA_DIR / "feedback.csv"
ARCHIVE_CSV  = DATA_DIR / "feedback_archive.csv"
ORIG_DATA    = DATA_DIR / "final_preprocessed_data.csv"

RETRAIN_THRESHOLD = 500    # minimum feedback rows before retraining
N_NEW_TREES       = 100    # trees added on top of existing model
MAX_VERSIONS      = 3      # keep this many versioned backups
ACCURACY_FLOOR    = 0.02   # rollback if accuracy drops by more than this
RANDOM_STATE      = 42

sys.path.insert(0, str(BASE_DIR))
from api.services.feature_engineering import FeatureEngineer

from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEP = "=" * 65
print(SEP); print("Incremental Model Update"); print(SEP)

# ── 1. CHECK FEEDBACK COUNT ───────────────────────────────────────────────────
if not FEEDBACK_CSV.exists():
    print("  No feedback.csv found — nothing to do.")
    sys.exit(0)

feedback_df = pd.read_csv(FEEDBACK_CSV)
n_new = len(feedback_df)
print(f"  Feedback rows available: {n_new}  (threshold: {RETRAIN_THRESHOLD})")

if n_new < RETRAIN_THRESHOLD:
    print(f"  Fewer than {RETRAIN_THRESHOLD} rows — skipping retrain (no-op).")
    sys.exit(0)

# ── 2. LOAD EXISTING ARTIFACTS ────────────────────────────────────────────────
print("\n  Loading existing artifacts...")
fe            = joblib.load(MODEL_DIR / "feature_engineering.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
metadata      = joblib.load(MODEL_DIR / "metadata.pkl")
clf_raw       = joblib.load(MODEL_DIR / "lgbm_clf.pkl")
old_accuracy  = metadata["overall_accuracy"]
print(f"  Current model accuracy: {old_accuracy:.4f}")

# ── 3. REBUILD VAL + TEST SETS FROM ORIGINAL DATA ────────────────────────────
print("\n  Rebuilding val/test sets from original data (same seed=42 split)...")

POST_FLIGHT = [
    "Unnamed: 0", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
]
TARGET     = "FLIGHT_STATUS"
DELAY_COLS = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
              "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]

orig       = pd.read_csv(ORIG_DATA)
y_status   = orig[TARGET].values
drop_cols  = [c for c in POST_FLIGHT if c in orig.columns] + [TARGET]
X_raw      = orig.drop(columns=drop_cols).reindex(columns=feature_names, fill_value=0)

idx = np.arange(len(orig))
idx_trainval, idx_test = train_test_split(
    idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y_status)
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=0.125, random_state=RANDOM_STATE,
    stratify=y_status[idx_trainval])

X_val_df = pd.DataFrame(X_raw.values[idx_val],  columns=feature_names)
X_te_df  = pd.DataFrame(X_raw.values[idx_test], columns=feature_names)
y_val_status = y_status[idx_val]
y_te_status  = y_status[idx_test]

X_val_eng = fe.transform(X_val_df)
X_te_eng  = fe.transform(X_te_df)
print(f"  Val set: {len(X_val_eng):,}  |  Test set: {len(X_te_eng):,}")

# ── 4. WARM-START CLASSIFIER UPDATE ──────────────────────────────────────────
print(f"\n  Warm-starting LGBMClassifier (+{N_NEW_TREES} trees)...")

new_clf = LGBMClassifier(
    n_estimators      = N_NEW_TREES,
    learning_rate     = 0.03,       # lower LR for fine-tuning pass
    max_depth         = 8,
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    class_weight      = "balanced",
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbose           = -1,
)
new_clf.fit(X_val_eng, y_val_status, init_model=clf_raw)

new_cal_clf = CalibratedClassifierCV(new_clf, cv="prefit", method="isotonic")
new_cal_clf.fit(X_val_eng, y_val_status)

# ── 5. SAFETY GATE ────────────────────────────────────────────────────────────
print("\n  Evaluating new model on held-out test set...")
new_preds    = (new_cal_clf.predict_proba(X_te_eng)[:, 1] >= 0.5).astype(int)
new_accuracy = accuracy_score(y_te_status, new_preds)
delta        = new_accuracy - old_accuracy

print(f"  Old: {old_accuracy:.4f}  →  New: {new_accuracy:.4f}  (Δ {delta:+.4f})")

if delta < -ACCURACY_FLOOR:
    print(f"\n  WARNING: accuracy dropped {abs(delta):.4f} > {ACCURACY_FLOOR} threshold.")
    print("  Rolling back — existing model unchanged.")
    sys.exit(1)

# ── 6. SAVE VERSIONED + CURRENT MODELS ───────────────────────────────────────
version = metadata.get("model_version", 1) + 1
print(f"\n  Saving model version {version}...")

# Prune old versioned backups
versioned = sorted(MODEL_DIR.glob("lgbm_clf_v*.pkl"))
while len(versioned) >= MAX_VERSIONS:
    versioned.pop(0).unlink()
    versioned = sorted(MODEL_DIR.glob("lgbm_clf_v*.pkl"))

joblib.dump(new_clf,     MODEL_DIR / f"lgbm_clf_v{version}.pkl")
joblib.dump(new_clf,     MODEL_DIR / "lgbm_clf.pkl")
joblib.dump(new_cal_clf, MODEL_DIR / "lgbm_clf_calibrated.pkl")
print(f"  Saved: lgbm_clf_v{version}.pkl, lgbm_clf.pkl, lgbm_clf_calibrated.pkl")

# ── 7. UPDATE METADATA ────────────────────────────────────────────────────────
metadata["model_version"]       = version
metadata["trained_at"]          = datetime.now(timezone.utc).isoformat()
metadata["overall_accuracy"]    = new_accuracy
metadata["incremental_updates"] = metadata.get("incremental_updates", 0) + 1
metadata["feedback_rows_used"]  = metadata.get("feedback_rows_used", 0) + n_new
joblib.dump(metadata, MODEL_DIR / "metadata.pkl")

# ── 8. ARCHIVE FEEDBACK AND RESET ─────────────────────────────────────────────
write_header = not ARCHIVE_CSV.exists()
feedback_df.to_csv(ARCHIVE_CSV, mode="a", header=write_header, index=False)
with open(FEEDBACK_CSV, "w", encoding="utf-8") as f:
    f.write("flight_id,timestamp,actual_delayed,actual_delay_min\n")

print(f"\n  Archived {n_new} feedback rows → data/feedback_archive.csv")
print(f"  data/feedback.csv reset to header only")
print(f"\n{SEP}")
print(f"  Incremental update complete — v{version}  accuracy: {new_accuracy:.4f}")
print(SEP)
```

- [ ] **Step 2: Smoke-test — no-op path (< 500 rows)**

Run: `python train_incremental.py`

Expected output (clean exit, no errors):
```
====================================================================
Incremental Model Update
====================================================================
  Feedback rows available: 0  (threshold: 500)
  Fewer than 500 rows — skipping retrain (no-op).
```
Exit code should be 0: `echo $?` → `0`

- [ ] **Step 3: Commit**

```bash
git add train_incremental.py
git commit -m "feat: train_incremental.py — warm-start LightGBM update with safety gate"
```

---

## Task 7: GitHub Actions Nightly Retrain Workflow

**Files:**
- Create: `.github/workflows/retrain.yml`

- [ ] **Step 1: Create workflow directory**

Run: `mkdir -p .github/workflows`

- [ ] **Step 2: Create retrain.yml**

Create `.github/workflows/retrain.yml`:
```yaml
name: Nightly Incremental Model Update

on:
  schedule:
    # 2:00 AM UTC every day — after overnight flights land and feedback accumulates
    - cron: '0 2 * * *'
  workflow_dispatch:   # Allow manual trigger from the GitHub Actions UI

jobs:
  retrain:
    runs-on: ubuntu-latest
    timeout-minutes: 60   # Safety: fail if job runs > 1 hour

    steps:
      # 1. Check out the repo, including model .pkl files stored via Git LFS
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          lfs: true
          token: ${{ secrets.GITHUB_TOKEN }}

      # 2. Set up the same Python version used in development
      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      # 3. Install only the packages needed for training (not the full API deps)
      - name: Install training dependencies
        run: |
          python -m pip install --upgrade pip
          pip install "scikit-learn>=1.3" "lightgbm>=4.0.0" joblib numpy pandas

      # 4. Run incremental update.
      #    Exits cleanly (code 0) if feedback rows < 500 — no commit needed.
      #    Exits with code 1 if accuracy regressed — no commit, keeps old model.
      - name: Run incremental model update
        id: retrain
        run: python train_incremental.py

      # 5. Commit updated model files only if step 4 produced changes
      - name: Commit updated model files
        if: steps.retrain.outcome == 'success'
        run: |
          git config user.name  "GitHub Actions Bot"
          git config user.email "actions@github.com"
          git add model/lgbm_clf*.pkl model/metadata.pkl \
                  data/feedback.csv data/feedback_archive.csv 2>/dev/null || true
          git diff --staged --quiet \
            || git commit -m "chore: incremental model update $(date +'%Y-%m-%d')"
          git push

      # 6. Trigger Render to reload the API with fresh model files.
      #    Add RENDER_DEPLOY_HOOK_URL as a GitHub Actions secret.
      #    See: Render dashboard → your service → Settings → Deploy Hook
      - name: Trigger Render redeploy
        if: steps.retrain.outcome == 'success'
        run: |
          if [ -n "${{ secrets.RENDER_DEPLOY_HOOK_URL }}" ]; then
            curl -s -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
            echo "Render redeploy triggered."
          else
            echo "RENDER_DEPLOY_HOOK_URL not set — skipping redeploy."
          fi
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/retrain.yml
git commit -m "ci: nightly incremental model update via GitHub Actions"
```

- [ ] **Step 4: Verify workflow file is valid YAML**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/retrain.yml')); print('YAML valid')"`

Expected: `YAML valid`

---

## Final Verification

- [ ] **Run full test suite**

Run: `pytest tests/ -v`

Expected: all tests in `test_feature_engineering.py`, `test_predict.py`, `test_api_feedback.py` pass.

```
tests/test_feature_engineering.py .......... 10 passed
tests/test_predict.py .......... 10 passed
tests/test_api_feedback.py ...... 6 passed
26 passed
```

- [ ] **Start the API and verify /model/info shows LightGBM metrics**

Run: `uvicorn api.main:app --port 8000`

Visit `http://localhost:8000/model/info` — response should include `"model_type": "LGBMClassifier + LGBMRegressor"` and `"roc_auc"` field.

Visit `http://localhost:8000/docs` — the `/feedback` endpoint should appear in the Swagger UI under the "feedback" tag.

Stop with Ctrl+C.

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| FeatureEngineer with 15 historical/interaction features | Task 2 |
| fit(X, y_total_delay, y_binary_status) signature | Task 2 |
| Unseen group → global_mean_ fallback | Task 2 |
| LGBMClassifier + CalibratedClassifierCV(cv='prefit') | Task 3 |
| LGBMRegressor main + p10/p90 + 4 per-type | Task 3 |
| train_lgbm.py replaces train.py + train_regressor.py | Task 3 |
| 70/10/20 split (train/cal-val/test) | Task 3 |
| lightgbm added to requirements.txt | Task 1 |
| predict.py loads LightGBM artifacts | Task 4 |
| API response shape identical (cluster=-1, model_used='lgbm') | Task 4 |
| POST /feedback with idempotency + validation | Task 5 |
| /model/info updated with new metadata fields | Task 5 |
| train_incremental.py with init_model warm-start | Task 6 |
| Safety gate: rollback if accuracy drops > 2% | Task 6 |
| Versioned model backups (lgbm_clf_v{N}.pkl, keep 3) | Task 6 |
| Archive processed feedback rows | Task 6 |
| GitHub Actions nightly workflow | Task 7 |
| data/feedback.csv with header | Task 1 |

All spec requirements covered. ✓

**No placeholders:** No TBD, TODO, or "implement later" found. All code steps are complete. ✓

**Type consistency:**
- `FeatureEngineer.transform()` → `np.ndarray` — consumed by LightGBM sklearn API (accepts ndarray). ✓
- `FeatureEngineer.ENGINEERED_COL_NAMES` used by index in tests and by name in `transform()`. ✓
- `metadata["overall_accuracy"]` written by `train_lgbm.py`, read by `train_incremental.py` and `predict.py`. ✓
- `lgbm_clf.pkl` = raw `LGBMClassifier`; `lgbm_clf_calibrated.pkl` = `CalibratedClassifierCV` — loaded consistently. ✓
- `model_validator(mode="after")` requires pydantic v2 — FastAPI ships with pydantic v2 by default since FastAPI 0.100+. ✓
