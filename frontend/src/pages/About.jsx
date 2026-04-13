import React, { useEffect, useState } from 'react'
import { getModelInfo } from '../api'
import { Database, GitBranch, Cpu, Globe } from 'lucide-react'

export default function About() {
  const [info, setInfo] = useState(null)
  useEffect(() => { getModelInfo().then(setInfo).catch(() => {}) }, [])

  return (
    <div className="max-w-4xl mx-auto px-4 py-12 space-y-10">
      <div>
        <h1 className="text-3xl font-bold mb-2">About this Project</h1>
        <p className="text-slate-400">
          Master's Thesis Project · Student 21MA23002 · Flight Delay Prediction
        </p>
      </div>

      {/* Pipeline diagram */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-4">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Cpu size={18} className="text-blue-400"/> ML Pipeline
        </h2>
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {[
            { label: 'Raw Data', sub: '86,478 flights · 219 cols' },
            { label: 'Pre-flight Features', sub: '203 cols (no leakage)' },
            { label: 'StandardScaler', sub: 'zero mean, unit var' },
            { label: 'ANOVA SelectKBest', sub: 'top 20 features' },
            { label: 'DBSCAN Clustering', sub: '612 clusters' },
            { label: 'Per-cluster RF', sub: '419 models' },
            { label: 'Prediction', sub: '71.74% accuracy' },
          ].map((step, i, arr) => (
            <React.Fragment key={i}>
              <div className="bg-slate-800 rounded-xl px-4 py-3 text-center min-w-[120px]">
                <p className="font-medium text-white text-xs">{step.label}</p>
                <p className="text-slate-500 text-xs mt-0.5">{step.sub}</p>
              </div>
              {i < arr.length - 1 && <span className="text-slate-600 text-lg">→</span>}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Why no leakage */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Database size={18} className="text-amber-400"/> Data Leakage Prevention
        </h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          Many flight delay models accidentally use post-flight features — information
          that is only available <em>after</em> the flight has landed (e.g. actual taxi time,
          carrier delay minutes). This model uses only <strong className="text-white">pre-flight features</strong>:
          scheduled times, distance, airline, city, weather at origin and destination.
          This makes predictions genuinely useful at booking or check-in time.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs">
          <div className="bg-red-950/50 border border-red-900 rounded-lg p-3">
            <p className="text-red-400 font-medium mb-1">Excluded (post-flight)</p>
            <p className="text-slate-500">TAXI_OUT, WHEELS_OFF, ACTUAL_ELAPSED_TIME,
            CARRIER_DELAY, NAS_DELAY, AIR_TIME…</p>
          </div>
          <div className="bg-green-950/50 border border-green-900 rounded-lg p-3">
            <p className="text-green-400 font-medium mb-1">Used (pre-flight)</p>
            <p className="text-slate-500">Distance, scheduled times, airline, origin/dest city,
            humidity, temperature, weather condition…</p>
          </div>
        </div>
      </div>

      {/* DBSCAN trick */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <GitBranch size={18} className="text-purple-400"/> Cluster-then-Classify
        </h2>
        <p className="text-slate-400 text-sm leading-relaxed">
          Instead of training one global model, flights are first grouped by similarity
          using <strong className="text-white">DBSCAN</strong> (density-based clustering).
          Each cluster gets its own <strong className="text-white">Random Forest</strong> trained
          on its specific flight patterns. This captures heterogeneity in the data — a red-eye
          transcontinental flight has very different delay drivers than a short regional hop.
        </p>
        <p className="text-slate-400 text-sm leading-relaxed">
          Since DBSCAN has no <code className="text-blue-300 bg-slate-800 px-1 rounded">.predict()</code> for
          new points, a <strong className="text-white">KNN classifier</strong> is trained
          post-clustering to route new flights to the nearest cluster.
          Points in DBSCAN noise (cluster = −1) fall back to a global HistGradientBoosting model.
        </p>
      </div>

      {/* Real-time data */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-3">
        <h2 className="text-lg font-semibold flex items-center gap-2">
          <Globe size={18} className="text-green-400"/> Real-time Data Sources
        </h2>
        <div className="grid md:grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-white font-medium mb-1">OpenSky Network</p>
            <p className="text-slate-400">Free, no API key. Provides live ADS-B aircraft positions,
            callsigns, altitude, velocity, and heading for flights worldwide.</p>
          </div>
          <div>
            <p className="text-white font-medium mb-1">Open-Meteo</p>
            <p className="text-slate-400">Free, no API key. Provides current weather (temperature,
            humidity, wind, condition) at any lat/lon coordinate.</p>
          </div>
        </div>
      </div>

      {info && (
        <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
          <h2 className="text-lg font-semibold mb-3">Live Model Stats</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center text-sm">
            <div><p className="text-2xl font-bold text-blue-400">{info.overall_accuracy_pct}</p>
              <p className="text-slate-500">Test Accuracy</p></div>
            <div><p className="text-2xl font-bold text-purple-400">{info.n_clusters.toLocaleString()}</p>
              <p className="text-slate-500">DBSCAN Clusters</p></div>
            <div><p className="text-2xl font-bold text-green-400">{info.n_cluster_models}</p>
              <p className="text-slate-500">Trained Models</p></div>
            <div><p className="text-2xl font-bold text-amber-400">20</p>
              <p className="text-slate-500">ANOVA Features</p></div>
          </div>
        </div>
      )}
    </div>
  )
}
