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


# ─────────────────────────────────────────────────────────────────────
# Quick smoke-test
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = FlightPredictor()
    tests = [
        {
            "name": "Evening Southwest flight, high humidity",
            "features": {
                "CRS_ELAPSED_TIME": 150, "DISTANCE": 800,
                "origin_humidity": 85, "dest_humidity": 80,
                "OP_CARRIER_Southwest Airlines": 1,
                "origin_city_new york": 1,
                "destination_city_miami": 1,
                "Season_Summer": 1, "WeekendFlagEncoded": 1,
                "CRS_DEP_TIME_4": 1, "MONTH_7": 1,
            },
        },
        {
            "name": "Morning Delta flight, sunny",
            "features": {
                "CRS_ELAPSED_TIME": 90, "DISTANCE": 400,
                "origin_humidity": 40, "dest_humidity": 35,
                "OP_CARRIER_Delta Airlines": 1,
                "origin_city_atlanta": 1,
                "Season_Autumn": 1, "WeekendFlagEncoded": 0,
                "CRS_DEP_TIME_2": 1, "MONTH_9": 1,
                "origin_condition_text_Sunny": 1,
            },
        },
    ]
    for t in tests:
        r = p.predict(t["features"])
        print(f"\n--- {t['name']} ---")
        print(f"  Delayed       : {r['delayed']}  ({r['probability_pct']})")
        print(f"  Delay min     : {r['expected_delay_min']} min  [{r['delay_range']}]")
        print(f"  Category      : {r['delay_category']}")
        print(f"  Breakdown     : {r['delay_breakdown']}")
        print(f"  Model         : {r['model_used']}")
