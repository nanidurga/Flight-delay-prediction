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
      0  carrier_hist_mean_delay
      1  carrier_hist_delay_rate
      2  origin_hist_mean_delay
      3  origin_hist_delay_rate
      4  dest_hist_mean_delay
      5  depslot_hist_mean_delay
      6  month_hist_mean_delay
      7  route_hist_mean_delay
      8  route_hist_delay_rate
      9  route_hist_n_flights
     10  carrier_origin_hist_mean_delay
     11  carrier_month_hist_mean_delay
     12  depslot_origin_hist_mean_delay
     13  origin_weather_severity
     14  dest_weather_severity
    """

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

        self.carrier_mean_,  self.carrier_rate_  = self._group_stats(X_np, y_total_delay, y_binary_status, self.carrier_cols_)
        self.origin_mean_,   self.origin_rate_   = self._group_stats(X_np, y_total_delay, y_binary_status, self.origin_cols_)
        self.dest_mean_,     self.dest_rate_     = self._group_stats(X_np, y_total_delay, y_binary_status, self.dest_cols_)
        self.depslot_mean_,  _                   = self._group_stats(X_np, y_total_delay, y_binary_status, self.dep_slot_cols_)
        self.month_mean_,    _                   = self._group_stats(X_np, y_total_delay, y_binary_status, self.month_cols_)

        active_carriers = self._active_col(X_np, self.carrier_cols_)
        active_origins  = self._active_col(X_np, self.origin_cols_)
        active_dests    = self._active_col(X_np, self.dest_cols_)
        active_slots    = self._active_col(X_np, self.dep_slot_cols_)
        active_months   = self._active_col(X_np, self.month_cols_)

        # Route-pair lookup
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

    def _group_stats(self, X_np, y_total, y_status, group_cols):
        mean_lut: dict = {}
        rate_lut: dict = {}
        for col in group_cols:
            if col not in self.col_idx_:
                continue
            idx  = self.col_idx_[col]
            mask = X_np[:, idx] == 1
            if mask.sum() < 10:
                continue
            delayed = mask & (y_status == 1)
            mean_lut[col] = float(y_total[delayed].mean()) if delayed.sum() > 0 else self.global_mean_
            rate_lut[col] = float(y_status[mask].mean())
        return mean_lut, rate_lut

    def _active_col(self, X_np, group_cols):
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

    def _interaction_stats(self, active_a, active_b, y_total, y_status):
        buckets: dict = {}
        for a, b, delay, status in zip(active_a, active_b, y_total, y_status):
            if a is None or b is None:
                continue
            if status == 1:
                buckets.setdefault((a, b), []).append(delay)
        return {k: float(np.mean(v)) for k, v in buckets.items()}

    def _weather_severity(self, row, wx_cols):
        for col in wx_cols:
            if col not in self.col_idx_:
                continue
            if row[self.col_idx_[col]] == 1:
                name = (col.replace("origin_condition_text_", "")
                           .replace("dest_condition_text_", ""))
                return float(self.WEATHER_SEVERITY.get(name, 3))
        return 3.0
