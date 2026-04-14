import React, { useEffect, useState } from 'react'
import { getModelInfo } from '../api'
import { Database, GitBranch, Cpu, Globe, RefreshCw, Layers, BarChart2 } from 'lucide-react'

export default function About() {
  const [info, setInfo] = useState(null)
  useEffect(() => { getModelInfo().then(setInfo).catch(() => {}) }, [])

  return (
    <div className="max-w-4xl mx-auto px-4 py-12 space-y-10">
      <div>
        <h1 className="text-3xl font-bold mb-2">About this Project</h1>
        <p className="text-slate-400">
          Master's Thesis · 21MA23002 · US Domestic Flight Delay Prediction
        </p>
      </div>

      {/* ── Classification pipeline ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Cpu size={18} className="text-blue-400" /> Classification Pipeline
        </h2>
        <p className="text-slate-400 text-sm">
          Binary prediction (delayed / on-time) using a calibrated LightGBM classifier trained on
          86,478 BTS On-Time Performance records.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {[
            { label: 'Raw Data',            sub: '86,478 flights · 219 cols'      },
            { label: 'Pre-flight Features', sub: '203 cols · no leakage'          },
            { label: '70/10/20 Split',      sub: 'train / val / test · seed=42'   },
            { label: 'FeatureEngineer',     sub: '+15 engineered → 218 features'  },
            { label: 'LGBMClassifier',      sub: '1,000 trees · early stopping'   },
            { label: 'Calibration',         sub: 'CalibratedClassifierCV · prefit'},
            { label: 'Prediction',          sub: '91.34% accuracy · AUC 0.973'    },
          ].map((step, i, arr) => (
            <React.Fragment key={i}>
              <div className="bg-slate-800 rounded-xl px-4 py-3 text-center min-w-[130px]">
                <p className="font-medium text-white text-xs">{step.label}</p>
                <p className="text-slate-500 text-xs mt-0.5">{step.sub}</p>
              </div>
              {i < arr.length - 1 && <span className="text-slate-600 text-lg">→</span>}
            </React.Fragment>
          ))}
        </div>
        <p className="text-slate-500 text-xs leading-relaxed">
          Calibration uses <code className="text-blue-300 bg-slate-800 px-1 rounded">CalibratedClassifierCV(cv='prefit', method='isotonic')</code>.
          The classifier is already trained; calibration fits only the isotonic mapping on the held-out
          validation set — no refitting of 1,000 trees.
        </p>
      </div>

      {/* ── Regression pipeline ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <BarChart2 size={18} className="text-purple-400" /> Regression Pipeline
        </h2>
        <p className="text-slate-400 text-sm">
          Predicts delay minutes (with uncertainty bounds) for flights classified as delayed.
          Trained only on the delayed subset to avoid prediction collapse to the mean.
        </p>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {[
            { label: 'Delayed Flights',     sub: 'delayed subset only'              },
            { label: '218 Features',        sub: 'same FeatureEngineer transform'   },
            { label: 'Point Regressor',     sub: 'LGBMReg · log1p target'          },
            { label: 'Quantile Regressors', sub: 'p10 + p90 → 80% PI'             },
            { label: 'Type Regressors ×4',  sub: 'carrier · weather · NAS · late'  },
            { label: 'Output',              sub: 'MAE 30.8 min · R² 0.704'         },
          ].map((step, i, arr) => (
            <React.Fragment key={i}>
              <div className="bg-slate-800 rounded-xl px-4 py-3 text-center min-w-[130px]">
                <p className="font-medium text-white text-xs">{step.label}</p>
                <p className="text-slate-500 text-xs mt-0.5">{step.sub}</p>
              </div>
              {i < arr.length - 1 && <span className="text-slate-600 text-lg">→</span>}
            </React.Fragment>
          ))}
        </div>
        <p className="text-slate-500 text-xs leading-relaxed">
          Delay minutes use a log1p transform before training (distribution skewness ≈ 4.66).
          Per-type regressors (carrier, weather, NAS, late aircraft) are rescaled so their
          breakdown sums exactly to the point estimate.
        </p>
      </div>

      {/* ── Data leakage prevention ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Database size={18} className="text-amber-400" /> Data Leakage Prevention
        </h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          Many delay models accidentally use post-flight features — values only available
          <em> after</em> the flight has landed (actual taxi time, carrier delay minutes, etc.).
          This model uses only <strong className="text-white">pre-flight features</strong>:
          scheduled times, distance, airline, city, and weather at origin/destination.
          This makes predictions genuinely useful at booking or check-in time.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-red-950/50 border border-red-900 rounded-lg p-3">
            <p className="text-red-400 font-medium mb-1">Excluded — post-flight</p>
            <p className="text-slate-500">TAXI_OUT, WHEELS_OFF, ACTUAL_ELAPSED_TIME,
              CARRIER_DELAY, NAS_DELAY, AIR_TIME, TAXI_IN…</p>
          </div>
          <div className="bg-green-950/50 border border-green-900 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Used — pre-flight</p>
            <p className="text-slate-500">Distance, scheduled dep/arr times, airline,
              origin/dest city, month, day, live weather condition…</p>
          </div>
        </div>
        <p className="text-slate-500 text-xs">
          The <code className="text-blue-300 bg-slate-800 px-1 rounded">FeatureEngineer</code> is
          fitted on the training fold only and saved as <code className="text-blue-300 bg-slate-800 px-1 rounded">feature_engineering.pkl</code>.
          Validation and test sets are transformed with <code className="text-slate-400 bg-slate-800 px-1 rounded">.transform()</code> only,
          preventing any leakage of historical statistics.
        </p>
      </div>

      {/* ── Incremental learning ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <RefreshCw size={18} className="text-green-400" /> Incremental Learning
        </h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          Actual outcomes submitted via <code className="text-blue-300 bg-slate-800 px-1 rounded">POST /feedback</code> accumulate
          in <code className="text-slate-400 bg-slate-800 px-1 rounded">data/feedback.csv</code>.
          Every night at 2 AM UTC, a GitHub Actions workflow runs <code className="text-slate-400 bg-slate-800 px-1 rounded">train_incremental.py</code>:
        </p>
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {[
            { label: 'Check rows',       sub: '≥ 500 feedback rows?'           },
            { label: 'Warm-start LGBM',  sub: 'init_model = lgbm_clf.pkl'      },
            { label: 'Re-calibrate',     sub: 'CalibratedClassifierCV · prefit' },
            { label: 'Safety gate',      sub: 'accuracy drop > 2% → rollback'  },
            { label: 'Commit + push',    sub: 'new .pkl files → Render redeploy'},
          ].map((step, i, arr) => (
            <React.Fragment key={i}>
              <div className="bg-slate-800 rounded-lg px-3 py-2 text-center min-w-[110px]">
                <p className="font-medium text-white">{step.label}</p>
                <p className="text-slate-500 mt-0.5">{step.sub}</p>
              </div>
              {i < arr.length - 1 && <span className="text-slate-600">→</span>}
            </React.Fragment>
          ))}
        </div>
        <p className="text-slate-500 text-xs">
          Warm-starting adds new trees on top of the existing model without discarding
          prior knowledge. The safety gate evaluates accuracy on the held-out test set
          before committing — if performance regresses by more than 2%, the old model is kept.
        </p>
      </div>

      {/* ── Why LightGBM ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <GitBranch size={18} className="text-purple-400" /> Why LightGBM?
        </h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          The original Sprint 1 approach used DBSCAN clustering to split flights into ~600 micro-clusters,
          then trained a separate Random Forest per cluster. This was expensive (507 models on ~135 rows
          each) and fragile (small clusters, high variance). Accuracy was 73.43%.
        </p>
        <p className="text-slate-400 text-sm leading-relaxed">
          Sprint 7 replaced all of that with a single <strong className="text-white">LGBMClassifier</strong> on
          all 218 features. LightGBM handles heterogeneity natively via feature interactions at each
          split — no manual clustering required. Accuracy jumped to <strong className="text-white">91.34%</strong> with
          ROC-AUC <strong className="text-white">0.973</strong>.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs mt-2">
          <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3">
            <p className="text-red-400 font-medium mb-1">Old — DBSCAN + RandomForest</p>
            <p className="text-slate-500">507 per-cluster models · avg 135 rows/cluster ·
              KNN assigner · 73.43% accuracy</p>
          </div>
          <div className="bg-slate-800/60 border border-slate-700 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">New — LightGBM</p>
            <p className="text-slate-500">1 calibrated classifier · 218 features ·
              warm-start incremental · 91.34% accuracy</p>
          </div>
        </div>
      </div>

      {/* ── Real-time data sources ── */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Globe size={18} className="text-green-400" /> Real-time Data Sources
        </h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-white font-medium mb-1">OpenSky Network</p>
            <p className="text-slate-400">Free, no API key. Live ADS-B aircraft positions,
              callsigns, altitude, velocity, and heading. Used to populate the Live Map.</p>
          </div>
          <div>
            <p className="text-white font-medium mb-1">Open-Meteo</p>
            <p className="text-slate-400">Free, no API key. Current weather (temperature,
              humidity, wind, condition text) at any lat/lon. Used for weather features
              in the predict form and live map predictions.</p>
          </div>
        </div>
      </div>

      {/* ── Live model stats ── */}
      {info && (
        <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
          <h2 className="text-lg font-semibold mb-4">Live Model Stats</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm">
            <div>
              <p className="text-2xl font-bold text-blue-400">{info.overall_accuracy_pct}</p>
              <p className="text-slate-500 mt-1">Test Accuracy</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-purple-400">{info.roc_auc}</p>
              <p className="text-slate-500 mt-1">ROC-AUC</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-amber-400">{info.regression_mae} min</p>
              <p className="text-slate-500 mt-1">Regression MAE</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-green-400">{info.n_features}</p>
              <p className="text-slate-500 mt-1">Features Used</p>
            </div>
          </div>
          {info.incremental_updates > 0 && (
            <p className="text-center text-slate-600 text-xs mt-4">
              {info.incremental_updates} incremental update{info.incremental_updates !== 1 ? 's' : ''} applied ·
              {info.feedback_rows_used} feedback rows used
            </p>
          )}
        </div>
      )}
    </div>
  )
}
