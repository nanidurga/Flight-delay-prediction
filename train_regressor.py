"""
train_regressor.py -- Enhanced delay-minutes regression  (v2)
=============================================================

Key improvements over v1:
  1. Historical group features (carrier / origin / destination / time-slot /
     month delay stats) computed leak-free from the training fold only.
     Research shows these carry the strongest signal for delay magnitude.
  2. Log1p-transform target  (skewness 4.7 -> near-normal), predictions
     back-transformed with expm1.
  3. HistGradientBoostingRegressor on ALL 203 pre-flight features + 7
     historical features (= 210 total).  Tree-based models need no scaling
     or ANOVA selection and handle 200+ features natively via feature
     importance splitting.
  4. Quantile regressors (alpha=0.10 & 0.90) give honest 80% prediction
     intervals instead of a hand-coded ±35% band.
  5. Four per-type HistGBR regressors (carrier / weather / NAS / late-
     aircraft) -- each one actually varies with the input features rather
     than applying a fixed global proportion.

Outputs written to model/:
  delay_regressor.pkl          -- main point-estimate regressor
  delay_regressor_p10.pkl      -- 10th-percentile regressor
  delay_regressor_p90.pkl      -- 90th-percentile regressor
  delay_type_regressors.pkl    -- dict of 4 per-type regressors
  hist_feature_lookup.pkl      -- lookup tables for historical features
                                 (needed at inference time in predict.py)
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

warnings.filterwarnings("ignore")

BASE_DIR     = Path(__file__).parent
DATA_PATH    = BASE_DIR / "data" / "final_preprocessed_data.csv"
MODEL_DIR    = BASE_DIR / "model"
RANDOM_STATE = 42

DELAY_COLS  = ["CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
               "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"]
POST_FLIGHT = [
    "Unnamed: 0", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "ACTUAL_ELAPSED_TIME", "AIR_TIME",
] + DELAY_COLS
TARGET = "FLIGHT_STATUS"

print("=" * 65)
print("Enhanced Delay-Minutes Regression  (v2)")
print("=" * 65)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. LOAD & PREPARE DATA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
data = pd.read_csv(DATA_PATH)
data["total_delay"] = data[DELAY_COLS].sum(axis=1)
print(f"Loaded {len(data):,} rows")
print(f"  Delayed flights : {(data[TARGET]==1).sum():,}")
print(f"  Delay mean/med  : {data.loc[data[TARGET]==1,'total_delay'].mean():.1f} / "
      f"{data.loc[data[TARGET]==1,'total_delay'].median():.1f} min")
print(f"  Delay skewness  : {data.loc[data[TARGET]==1,'total_delay'].skew():.2f}  "
      f"(log1p -> {np.log1p(data.loc[data[TARGET]==1,'total_delay']).skew():.2f})")

# Load the feature list saved by train.py (203 pre-flight features)
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

drop_cols = [c for c in POST_FLIGHT if c in data.columns] + [TARGET, "total_delay"]
X_all = (data
         .drop(columns=drop_cols)
         .reindex(columns=feature_names, fill_value=0))

y_total    = data["total_delay"].values
y_status   = data[TARGET].values
y_carrier  = data["CARRIER_DELAY"].values
y_weather  = data["WEATHER_DELAY"].values
y_nas      = data["NAS_DELAY"].values
y_late_ac  = data["LATE_AIRCRAFT_DELAY"].values

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. TRAIN / TEST SPLIT (stratified -- same seed as classifier so the
#    held-out rows are equivalent for fair comparison)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nSTEP 1 -- Train / test split (80/20, stratified)")

idx = np.arange(len(data))
idx_tr, idx_te = train_test_split(
    idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y_status
)

X_tr_raw   = X_all.values[idx_tr]
X_te_raw   = X_all.values[idx_te]
y_tr_total = y_total[idx_tr]
y_te_total = y_total[idx_te]
y_tr_status = y_status[idx_tr]
y_te_status = y_status[idx_te]
y_tr_carrier  = y_carrier[idx_tr];   y_te_carrier  = y_carrier[idx_te]
y_tr_weather  = y_weather[idx_tr];   y_te_weather  = y_weather[idx_te]
y_tr_nas      = y_nas[idx_tr];       y_te_nas      = y_nas[idx_te]
y_tr_late_ac  = y_late_ac[idx_tr];   y_te_late_ac  = y_late_ac[idx_te]

print(f"  Train : {len(idx_tr):,}   |   Test : {len(idx_te):,}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 3. HISTORICAL FEATURE ENGINEERING  (leak-free)
#    Computes group-level statistics from TRAINING rows only,
#    then applies the same lookup to test rows.
#
#    7 features added:
#      0  carrier_hist_mean_delay  -- avg total delay for this carrier
#      1  carrier_hist_delay_rate  -- P(delayed) for this carrier
#      2  origin_hist_mean_delay   -- avg total delay from origin city
#      3  origin_hist_delay_rate   -- P(delayed) from origin city
#      4  dest_hist_mean_delay     -- avg total delay to dest city
#      5  depslot_hist_mean_delay  -- avg delay at this dep time slot
#      6  month_hist_mean_delay    -- avg delay in this month
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nSTEP 2 -- Building historical features (from training fold only)")

col_idx = {name: i for i, name in enumerate(feature_names)}

carrier_cols  = [c for c in feature_names if c.startswith("OP_CARRIER_")]
origin_cols   = [c for c in feature_names if c.startswith("origin_city_")]
dest_cols     = [c for c in feature_names if c.startswith("destination_city_")]
dep_slot_cols = [c for c in feature_names
                 if c.startswith("CRS_DEP_TIME_") and c in col_idx]
month_cols    = [c for c in feature_names
                 if c.startswith("MONTH_") and c in col_idx]


def _compute_group_stats(X_raw, y_total, y_status, group_cols, col_idx,
                          global_mean, global_rate):
    """
    For each column in group_cols (one-hot group), compute:
      mean_delay : average total_delay of DELAYED rows in that group
      delay_rate : P(delayed) in that group
    Returns two dicts  {col_name: value}.
    """
    mean_lut = {}
    rate_lut = {}
    for col in group_cols:
        if col not in col_idx:
            continue
        ci = col_idx[col]
        mask = X_raw[:, ci] == 1
        if mask.sum() < 10:          # too few rows -- use global fallback
            continue
        delayed_mask = mask & (y_status == 1)
        mean_lut[col] = (float(y_total[delayed_mask].mean())
                         if delayed_mask.sum() > 0 else global_mean)
        rate_lut[col] = float(y_status[mask].mean())
    return mean_lut, rate_lut


def _apply_hist_features(X_raw, fit_data):
    """
    Vectorised lookup: for each row, extract the one active column in
    each one-hot group and look up its pre-computed statistic.
    Returns array shape (n, 7).
    """
    fd  = fit_data
    ci  = fd["col_idx"]
    gm  = fd["global_mean"]
    gr  = fd["global_rate"]
    n   = len(X_raw)
    hist = np.full((n, 7), fill_value=gm, dtype=np.float32)

    def _group_features(group_cols, mean_lut, rate_lut, out_mean_col, out_rate_col):
        valid = [(c, ci[c]) for c in group_cols if c in ci]
        if not valid:
            return
        cols_list, idxs = zip(*valid)
        mean_arr = np.array([mean_lut.get(c, gm) for c in cols_list])
        rate_arr = np.array([rate_lut.get(c, gr) for c in cols_list])
        X_group  = X_raw[:, list(idxs)]           # (n, g)
        row_sums = X_group.sum(axis=1)             # (n,)
        m_feat   = X_group @ mean_arr              # (n,)
        r_feat   = X_group @ rate_arr              # (n,)
        active   = row_sums > 0
        hist[:, out_mean_col] = np.where(active, m_feat, gm)
        if out_rate_col is not None:
            hist[:, out_rate_col] = np.where(active, r_feat, gr)

    _group_features(fd["carrier_cols"],  fd["carrier_mean"], fd["carrier_rate"], 0, 1)
    _group_features(fd["origin_cols"],   fd["origin_mean"],  fd["origin_rate"],  2, 3)
    _group_features(fd["dest_cols"],     fd["dest_mean"],    fd["dest_rate"],    4, None)
    _group_features(fd["dep_slot_cols"], fd["dep_mean"],     fd["dep_rate"],     5, None)
    _group_features(fd["month_cols"],    fd["month_mean"],   fd["month_rate"],   6, None)
    return hist


# Compute global fallbacks from training rows
delayed_tr_mask = y_tr_status == 1
global_mean_delay = float(y_tr_total[delayed_tr_mask].mean())
global_delay_rate = float(y_tr_status.mean())

fit_data = {
    "global_mean":   global_mean_delay,
    "global_rate":   global_delay_rate,
    "col_idx":       col_idx,
    "carrier_cols":  carrier_cols,
    "origin_cols":   origin_cols,
    "dest_cols":     dest_cols,
    "dep_slot_cols": dep_slot_cols,
    "month_cols":    month_cols,
}
# Compute per-group stats from training fold only
fit_data["carrier_mean"], fit_data["carrier_rate"] = _compute_group_stats(
    X_tr_raw, y_tr_total, y_tr_status, carrier_cols, col_idx,
    global_mean_delay, global_delay_rate)
fit_data["origin_mean"], fit_data["origin_rate"] = _compute_group_stats(
    X_tr_raw, y_tr_total, y_tr_status, origin_cols, col_idx,
    global_mean_delay, global_delay_rate)
fit_data["dest_mean"], fit_data["dest_rate"] = _compute_group_stats(
    X_tr_raw, y_tr_total, y_tr_status, dest_cols, col_idx,
    global_mean_delay, global_delay_rate)
fit_data["dep_mean"], fit_data["dep_rate"] = _compute_group_stats(
    X_tr_raw, y_tr_total, y_tr_status, dep_slot_cols, col_idx,
    global_mean_delay, global_delay_rate)
fit_data["month_mean"], fit_data["month_rate"] = _compute_group_stats(
    X_tr_raw, y_tr_total, y_tr_status, month_cols, col_idx,
    global_mean_delay, global_delay_rate)

# Apply to train and test (no leakage: test uses stats computed on train)
X_tr_hist = _apply_hist_features(X_tr_raw, fit_data)
X_te_hist = _apply_hist_features(X_te_raw, fit_data)

# Combine: 203 pre-flight + 7 historical = 210 features
X_tr = np.hstack([X_tr_raw, X_tr_hist])
X_te = np.hstack([X_te_raw, X_te_hist])
print(f"  Combined feature shape: {X_tr.shape}  "
      f"(203 pre-flight + 7 historical)")

# Carrier-level stats for display
print("\n  Carrier historical delay stats (from training fold):")
for col, mean_d in sorted(fit_data["carrier_mean"].items(),
                           key=lambda x: -x[1])[:5]:
    rate = fit_data["carrier_rate"].get(col, 0)
    print(f"    {col.replace('OP_CARRIER_',''):25s}: "
          f"mean delay={mean_d:.0f} min, delay_rate={rate:.2%}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. FILTER TO DELAYED FLIGHTS -- regression trains on delayed only
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tr_delayed = y_tr_status == 1
te_delayed = y_te_status == 1

X_tr_d = X_tr[tr_delayed];    X_te_d = X_te[te_delayed]
y_tr_d = y_tr_total[tr_delayed];  y_te_d = y_te_total[te_delayed]
y_tr_carrier_d = y_tr_carrier[tr_delayed]; y_te_carrier_d = y_te_carrier[te_delayed]
y_tr_weather_d = y_tr_weather[tr_delayed]; y_te_weather_d = y_te_weather[te_delayed]
y_tr_nas_d     = y_tr_nas[tr_delayed];     y_te_nas_d     = y_te_nas[te_delayed]
y_tr_late_d    = y_tr_late_ac[tr_delayed]; y_te_late_d    = y_te_late_ac[te_delayed]

print(f"\n  Delayed rows -- train: {tr_delayed.sum():,}   test: {te_delayed.sum():,}")

# Log1p-transform: handles right skew (skew 4.7 -> ~0.5)
y_tr_log = np.log1p(y_tr_d)
print(f"  Target after log1p: mean={y_tr_log.mean():.2f}, "
      f"std={y_tr_log.std():.2f}, skew={pd.Series(y_tr_log).skew():.2f}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 5. MAIN POINT-ESTIMATE REGRESSOR
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nSTEP 3 -- Main HistGradientBoostingRegressor (point estimate)")
print("         Training on 210 features with early stopping...")

reg_main = HistGradientBoostingRegressor(
    loss              = "squared_error",
    max_iter          = 600,
    learning_rate     = 0.04,
    max_depth         = 8,
    min_samples_leaf  = 20,
    l2_regularization = 0.1,
    max_features      = 0.8,          # stochastic: use 80% of features per split
    early_stopping    = True,
    validation_fraction = 0.1,
    n_iter_no_change  = 40,
    random_state      = RANDOM_STATE,
)
reg_main.fit(X_tr_d, y_tr_log)

preds_log = reg_main.predict(X_te_d)
preds     = np.expm1(np.clip(preds_log, 0, np.log1p(800)))
preds     = np.clip(preds, 1, 800)

mae   = mean_absolute_error(y_te_d, preds)
medae = median_absolute_error(y_te_d, preds)
r2    = r2_score(y_te_d, preds)
print(f"  MAE:       {mae:.1f} min")
print(f"  Median AE: {medae:.1f} min")
print(f"  R²:        {r2:.3f}")

joblib.dump(reg_main, MODEL_DIR / "delay_regressor.pkl")
print("  Saved -> model/delay_regressor.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 6. QUANTILE REGRESSORS  (10th & 90th pct -> 80% prediction interval)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nSTEP 4 -- Quantile regressors  (p10, p90)")

common_q_params = dict(
    max_iter         = 400,
    learning_rate    = 0.05,
    max_depth        = 6,
    min_samples_leaf = 30,
    max_features     = 0.8,
    random_state     = RANDOM_STATE,
)

reg_p10 = HistGradientBoostingRegressor(loss="quantile", quantile=0.10,
                                         **common_q_params)
reg_p90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.90,
                                         **common_q_params)
reg_p10.fit(X_tr_d, y_tr_log)
reg_p90.fit(X_tr_d, y_tr_log)

p10_preds = np.expm1(np.clip(reg_p10.predict(X_te_d), 0, None))
p90_preds = np.expm1(np.clip(reg_p90.predict(X_te_d), 0, None))

coverage = np.mean((y_te_d >= p10_preds) & (y_te_d <= p90_preds))
interval_width = (p90_preds - p10_preds).mean()
print(f"  80% interval coverage : {coverage*100:.1f}%  (target ~ 80%)")
print(f"  Mean interval width   : {interval_width:.0f} min")

joblib.dump(reg_p10, MODEL_DIR / "delay_regressor_p10.pkl")
joblib.dump(reg_p90, MODEL_DIR / "delay_regressor_p90.pkl")
print("  Saved -> model/delay_regressor_p10.pkl")
print("  Saved -> model/delay_regressor_p90.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 7. PER-TYPE REGRESSORS
#    Each predicts its own delay type (carrier / weather / NAS /
#    late_aircraft) in log-space.  At inference the four predictions
#    are rescaled to sum to the main point estimate, giving a per-flight
#    breakdown that actually varies with the input.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print("\nSTEP 5 -- Per-type regressors  (carrier / weather / NAS / late_aircraft)")

type_configs = [
    ("carrier",      y_tr_carrier_d, y_te_carrier_d),
    ("weather",      y_tr_weather_d, y_te_weather_d),
    ("nas",          y_tr_nas_d,     y_te_nas_d),
    ("late_aircraft",y_tr_late_d,    y_te_late_d),
]

type_regressors = {}
for name, y_tr_type, y_te_type in type_configs:
    rg = HistGradientBoostingRegressor(
        loss              = "squared_error",
        max_iter          = 300,
        learning_rate     = 0.05,
        max_depth         = 6,
        min_samples_leaf  = 20,
        l2_regularization = 0.1,
        max_features      = 0.8,
        random_state      = RANDOM_STATE,
    )
    rg.fit(X_tr_d, np.log1p(y_tr_type))
    type_preds = np.expm1(np.clip(rg.predict(X_te_d), 0, None))
    mae_t  = mean_absolute_error(y_te_type, type_preds)
    medae_t = median_absolute_error(y_te_type, type_preds)
    pct_nonzero = (y_te_type > 0).mean()
    print(f"  {name:20s}: MAE={mae_t:.1f} min | "
          f"Median AE={medae_t:.1f} | "
          f"active in {pct_nonzero*100:.0f}% of delayed flights")
    type_regressors[name] = rg

joblib.dump(type_regressors, MODEL_DIR / "delay_type_regressors.pkl")
print("  Saved -> model/delay_type_regressors.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 8. SAVE HISTORICAL FEATURE LOOKUP  (used by predict.py at inference)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
joblib.dump(fit_data, MODEL_DIR / "hist_feature_lookup.pkl")
print("\nSaved -> model/hist_feature_lookup.pkl")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 9. SUMMARY
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print()
print("=" * 65)
print("REGRESSION TRAINING COMPLETE")
print("=" * 65)
print(f"  Main regressor:  MAE={mae:.1f} min  |  Median AE={medae:.1f} min  |  R²={r2:.3f}")
print(f"  80% PI coverage: {coverage*100:.1f}%")
print(f"  Feature count:   {X_tr.shape[1]}  (203 pre-flight + 7 historical)")
print()
print("  Model artifacts saved:")
print("    model/delay_regressor.pkl")
print("    model/delay_regressor_p10.pkl")
print("    model/delay_regressor_p90.pkl")
print("    model/delay_type_regressors.pkl")
print("    model/hist_feature_lookup.pkl")
print("=" * 65)
