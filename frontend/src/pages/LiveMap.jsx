import React, { useState, useEffect, useCallback, useRef, Suspense } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'
import { getLiveFlights } from '../api'
import { RefreshCw, AlertTriangle, CheckCircle, Wifi, Globe2, Map, Plane } from 'lucide-react'

// ── Lazy-load the heavy globe ──────────────────────────────────────────────
const GlobeGL = React.lazy(() => import('react-globe.gl'))

// ── Globe wrapper ──────────────────────────────────────────────────────────
function GlobeView({ flights, onSelect }) {
  const globeRef = useRef()

  useEffect(() => {
    if (globeRef.current) {
      globeRef.current.pointOfView({ lat: 38, lng: -96, altitude: 1.6 }, 800)
    }
  }, [])

  const points = flights.map(f => ({
    ...f,
    lat  : f.lat,
    lng  : f.lon,
    color: f.prediction?.delayed ? '#ef4444' : '#22c55e',
    size : f.prediction?.delayed ? 0.45 : 0.28,
    label: `${f.callsign || '?'} · ${f.prediction?.delayed ? 'Delay Risk' : 'On Track'} · ${Math.round((f.prediction?.probability ?? 0.5) * 100)}%`,
  }))

  return (
    <Suspense
      fallback={
        <div className="absolute inset-0 flex items-center justify-center text-slate-500 text-sm">
          Loading 3D globe…
        </div>
      }
    >
      <GlobeGL
        ref={globeRef}
        width={undefined}
        height={undefined}
        backgroundColor="rgba(2,6,23,1)"
        globeImageUrl="https://unpkg.com/three-globe/example/img/earth-night.jpg"
        atmosphereColor="#3b82f6"
        atmosphereAltitude={0.12}
        pointsData={points}
        pointLat="lat"
        pointLng="lng"
        pointColor="color"
        pointAltitude={0.01}
        pointRadius="size"
        pointLabel="label"
        onPointClick={p => onSelect(p)}
        pointsMerge={false}
        style={{ width: '100%', height: '100%' }}
      />
    </Suspense>
  )
}

// ── Loading overlay shown over the map while fetching ─────────────────────
function LoadingOverlay({ isFirstLoad }) {
  return (
    <div
      className="absolute inset-0 flex flex-col items-center justify-center"
      style={{ background: 'rgba(2,6,23,0.92)', backdropFilter: 'blur(4px)', zIndex: 9999 }}
    >
      <div className="flex flex-col items-center gap-5">
        <div className="relative w-20 h-20">
          <span className="absolute inset-0 rounded-full bg-blue-500/20 animate-ping" />
          <span
            className="absolute inset-3 rounded-full bg-blue-400/15 animate-ping"
            style={{ animationDelay: '0.5s' }}
          />
          <div
            className="relative w-20 h-20 rounded-full flex items-center justify-center"
            style={{ border: '1px solid rgba(59,130,246,0.4)', background: 'rgba(59,130,246,0.08)' }}
          >
            <Plane size={28} className="text-blue-400" style={{ transform: 'rotate(45deg)' }} />
          </div>
        </div>
        <div className="text-center">
          <p className="text-white font-semibold text-sm tracking-wide">Fetching live flights…</p>
          <p className="text-slate-500 text-xs mt-1.5 max-w-[220px] leading-relaxed">
            {isFirstLoad
              ? 'First load may take ~30 s while the backend wakes up'
              : 'Refreshing flight positions'}
          </p>
        </div>
      </div>
    </div>
  )
}

// ── Main ───────────────────────────────────────────────────────────────────
export default function LiveMap() {
  const [flights,      setFlights]      = useState([])
  const [loading,      setLoading]      = useState(false)
  const [lastFetch,    setLastFetch]    = useState(null)
  const [error,        setError]        = useState(null)
  const [selected,     setSelected]     = useState(null)
  const [viewMode,     setViewMode]     = useState('map')
  const [search,       setSearch]       = useState('')
  const [riskFilter,   setRiskFilter]   = useState('all')
  const [filterOrigin, setFilterOrigin] = useState('')
  const [filterDest,   setFilterDest]   = useState('')

  const fetchFlights = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const data = await getLiveFlights(60)
      setFlights(data.flights || [])
      setLastFetch(new Date())
    } catch {
      setError('Could not load flights. The server may be starting up or OpenSky may be rate-limited — try again in 30 s.')
    } finally {
      setLoading(false)
    }
  }, [])

  // Fetch on mount
  useEffect(() => { fetchFlights() }, [fetchFlights])

  // Auto-refresh every 5 minutes
  useEffect(() => {
    const id = setInterval(fetchFlights, 5 * 60 * 1000)
    return () => clearInterval(id)
  }, [fetchFlights])

  const visibleFlights = flights.filter(f => {
    if (search       && !f.callsign?.toLowerCase().includes(search.toLowerCase()))        return false
    if (riskFilter === 'delayed' && !f.prediction?.delayed)                               return false
    if (riskFilter === 'ontime'  &&  f.prediction?.delayed)                               return false
    if (filterOrigin && !f.origin?.toLowerCase().includes(filterOrigin.toLowerCase()))    return false
    if (filterDest   && !f.destination?.toLowerCase().includes(filterDest.toLowerCase())) return false
    return true
  })

  return (
    <div className="flex flex-col" style={{ height: 'calc(100vh - 4rem)', background: '#020617' }}>

      {/* ── Header bar ── */}
      <div
        className="flex items-center justify-between gap-4 flex-shrink-0 px-4 py-3"
        style={{ background: 'rgba(15,23,42,0.9)', borderBottom: '1px solid rgba(255,255,255,0.06)', backdropFilter: 'blur(10px)' }}
      >
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
            <Wifi size={14} className="text-blue-400" />
          </div>
          <span className="font-semibold text-sm text-white">Live Flights</span>
          {lastFetch && (
            <span className="text-slate-500 text-xs hidden sm:block">
              Updated {lastFetch.toLocaleTimeString()}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          {/* City-pair filters */}
          <input
            type="text"
            value={filterOrigin}
            onChange={e => setFilterOrigin(e.target.value)}
            placeholder="From city…"
            className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 border border-white/10
                       text-slate-200 placeholder-slate-500 outline-none focus:border-blue-500/60 w-24"
          />
          <input
            type="text"
            value={filterDest}
            onChange={e => setFilterDest(e.target.value)}
            placeholder="To city…"
            className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 border border-white/10
                       text-slate-200 placeholder-slate-500 outline-none focus:border-blue-500/60 w-24"
          />
          {/* Callsign search */}
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Callsign…"
            className="text-xs px-2.5 py-1 rounded-lg bg-slate-800/80 border border-white/10
                       text-slate-200 placeholder-slate-500 outline-none focus:border-blue-500/60 w-24"
          />
          {/* Risk filter */}
          <div
            className="flex items-center rounded-xl p-0.5 gap-0.5"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            {[
              { id: 'all',     label: 'All'       },
              { id: 'delayed', label: '⚠ At-Risk'  },
              { id: 'ontime',  label: '✓ On-Track' },
            ].map(f => (
              <button
                key={f.id}
                onClick={() => setRiskFilter(f.id)}
                className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-200"
                style={riskFilter === f.id ? {
                  background: 'linear-gradient(135deg,#2563eb,#4f46e5)',
                  color: 'white',
                } : { color: '#94a3b8' }}
              >
                {f.label}
              </button>
            ))}
          </div>
          {/* Flight counts */}
          <span className="flex items-center gap-1 text-xs text-red-400 font-medium">
            <AlertTriangle size={11} /> {visibleFlights.filter(f => f.prediction?.delayed).length} at-risk
          </span>
          <span className="flex items-center gap-1 text-xs text-green-400 font-medium">
            <CheckCircle size={11} /> {visibleFlights.filter(f => !f.prediction?.delayed).length} on-track
          </span>
          {/* View toggle */}
          <div
            className="flex items-center rounded-xl p-0.5 gap-0.5"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            {[
              { id: 'map',   icon: <Map    size={13} />, label: 'Map'   },
              { id: 'globe', icon: <Globe2 size={13} />, label: 'Globe' },
            ].map(v => (
              <button
                key={v.id}
                onClick={() => setViewMode(v.id)}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-200"
                style={viewMode === v.id ? {
                  background: 'linear-gradient(135deg,#2563eb,#4f46e5)',
                  color: 'white',
                  boxShadow: '0 0 12px rgba(99,102,241,0.4)',
                } : { color: '#94a3b8' }}
              >
                {v.icon} {v.label}
              </button>
            ))}
          </div>
          {/* Refresh */}
          <button
            onClick={fetchFlights}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-300
                       hover:text-white disabled:opacity-40 transition-all duration-200"
            style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)' }}
          >
            <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error strip */}
      {error && (
        <div
          className="px-4 py-2 text-red-400 text-xs flex-shrink-0"
          style={{ background: 'rgba(239,68,68,0.08)', borderBottom: '1px solid rgba(239,68,68,0.2)' }}
        >
          {error}
        </div>
      )}

      {/* ── Map + sidebar ── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Main view */}
        <div className="flex-1 relative overflow-hidden">

          {/* Loading overlay — covers the entire map area */}
          {loading && <LoadingOverlay isFirstLoad={!lastFetch} />}

          {/* Empty state when no flights loaded yet */}
          {!loading && flights.length === 0 && !error && (
            <div
              className="absolute inset-0 flex flex-col items-center justify-center z-10"
              style={{ background: 'rgba(2,6,23,0.6)' }}
            >
              <Plane size={32} className="text-slate-600 mb-3" style={{ transform: 'rotate(45deg)' }} />
              <p className="text-slate-400 text-sm font-medium">No flight data</p>
              <p className="text-slate-600 text-xs mt-1">Click Refresh to load live flights</p>
            </div>
          )}

          {viewMode === 'map' ? (
            <MapContainer
              center={[39, -98]}
              zoom={4}
              style={{ height: '100%', width: '100%' }}
            >
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='&copy; OpenStreetMap'
              />
              {visibleFlights.map(f => {
                const isDelayed = f.prediction?.delayed
                const prob      = f.prediction?.probability ?? 0.5
                return (
                  <CircleMarker
                    key={f.icao24}
                    center={[f.lat, f.lon]}
                    radius={isDelayed ? 7 : 5}
                    pathOptions={{
                      color      : isDelayed ? '#ef4444' : '#22c55e',
                      fillColor  : isDelayed ? '#ef444490' : '#22c55e90',
                      fillOpacity: 0.85,
                      weight     : 1.5,
                    }}
                    eventHandlers={{ click: () => setSelected(f) }}
                  >
                    <Popup>
                      <div className="text-slate-800 text-xs space-y-1 min-w-[150px]">
                        <p className="font-bold text-sm">{f.callsign}</p>
                        <p>Status:{' '}
                          <span className={isDelayed ? 'text-red-600 font-medium' : 'text-green-600 font-medium'}>
                            {isDelayed ? 'Delay Risk' : 'On Track'}
                          </span>
                        </p>
                        <p>Delay prob: <b>{Math.round(prob * 100)}%</b></p>
                        <p>Alt: {Math.round(f.altitude_m)} m</p>
                        <p>Speed: {Math.round(f.velocity_ms * 3.6)} km/h</p>
                        {f.weather && <p>Temp: {f.weather.temperature_celsius}°C</p>}
                      </div>
                    </Popup>
                  </CircleMarker>
                )
              })}
            </MapContainer>
          ) : (
            <div style={{ width: '100%', height: '100%' }}>
              <GlobeView flights={visibleFlights} onSelect={setSelected} />
            </div>
          )}
        </div>

        {/* Sidebar */}
        <div
          className="w-64 overflow-y-auto flex-shrink-0 hidden lg:flex flex-col"
          style={{ background: 'rgba(15,23,42,0.95)', borderLeft: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div
            className="px-3 py-2.5 text-[10px] text-slate-500 font-semibold uppercase tracking-widest flex-shrink-0"
            style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}
          >
            {visibleFlights.length} / {flights.length} Aircraft
          </div>

          {!loading && flights.length === 0 && (
            <p className="p-4 text-slate-500 text-sm text-center">No data — click Refresh.</p>
          )}

          {visibleFlights.map(f => {
            const isDelayed  = f.prediction?.delayed
            const prob       = f.prediction?.probability ?? 0.5
            const isSelected = selected?.icao24 === f.icao24
            return (
              <div
                key={f.icao24}
                onClick={() => setSelected(f)}
                className="p-3 cursor-pointer transition-all duration-150 flex-shrink-0"
                style={{
                  borderBottom: '1px solid rgba(255,255,255,0.04)',
                  background  : isSelected ? 'rgba(37,99,235,0.12)' : undefined,
                  borderLeft  : isSelected ? '2px solid #3b82f6' : '2px solid transparent',
                }}
                onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = 'rgba(255,255,255,0.03)' }}
                onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = undefined }}
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm text-white font-medium">{f.callsign || '—'}</span>
                  <span
                    className="text-xs font-bold tabular-nums"
                    style={{ color: isDelayed ? '#f87171' : '#4ade80' }}
                  >
                    {Math.round(prob * 100)}%
                  </span>
                </div>
                <div className="flex items-center gap-1.5 mt-1">
                  <div
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: isDelayed ? '#ef4444' : '#22c55e' }}
                  />
                  <span className="text-[11px] text-slate-500 truncate">
                    {isDelayed ? 'Delay Risk' : 'On Track'}
                    {f.weather?.condition_text ? ` · ${f.weather.condition_text}` : ''}
                  </span>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* ── Selected flight detail panel ── */}
      {selected && (
        <div
          className="flex-shrink-0 px-4 py-3 flex items-center justify-between gap-4"
          style={{ background: 'rgba(15,23,42,0.95)', borderTop: '1px solid rgba(255,255,255,0.06)' }}
        >
          <div className="flex items-center gap-3">
            <div
              className="w-2 h-2 rounded-full flex-shrink-0"
              style={{ backgroundColor: selected.prediction?.delayed ? '#ef4444' : '#22c55e' }}
            />
            <span className="font-mono font-bold text-white text-sm">{selected.callsign || '—'}</span>
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded-full"
              style={selected.prediction?.delayed
                ? { background: 'rgba(239,68,68,0.15)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)' }
                : { background: 'rgba(34,197,94,0.12)', color: '#4ade80', border: '1px solid rgba(34,197,94,0.25)' }}
            >
              {selected.prediction?.delayed ? 'Delay Risk' : 'On Track'}
            </span>
          </div>
          <div className="flex items-center gap-6 text-xs text-slate-400">
            <span>Prob: <b className="text-white">{Math.round((selected.prediction?.probability ?? 0.5) * 100)}%</b></span>
            <span>Alt: <b className="text-white">{Math.round(selected.altitude_m)} m</b></span>
            <span>Speed: <b className="text-white">{Math.round(selected.velocity_ms * 3.6)} km/h</b></span>
            {selected.weather && <span>Temp: <b className="text-white">{selected.weather.temperature_celsius}°C</b></span>}
          </div>
          <button
            onClick={() => setSelected(null)}
            className="text-slate-600 hover:text-slate-400 text-xs transition-colors ml-auto"
          >
            ✕
          </button>
        </div>
      )}

      {/* ── Legend ── */}
      <div
        className="flex-shrink-0 px-4 py-2 flex items-center gap-6 text-xs text-slate-500"
        style={{ background: 'rgba(2,6,23,0.95)', borderTop: '1px solid rgba(255,255,255,0.04)' }}
      >
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-red-500 inline-block" />
          Delay Risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full bg-green-500 inline-block" />
          On Track
        </span>
        <span className="ml-auto text-slate-700 hidden sm:block">
          OpenSky · Open-Meteo · LightGBM
        </span>
      </div>
    </div>
  )
}
