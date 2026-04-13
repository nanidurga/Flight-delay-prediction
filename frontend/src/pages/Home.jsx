import React, { useState } from 'react'
import { predictFlight } from '../api'
import {
  PlaneTakeoff, PlaneLanding, Clock, AlertTriangle,
  CheckCircle, Loader, Zap, Wind, Droplets, ThumbsUp,
  Info, TrendingUp, Shield, Calendar, ArrowRight,
} from 'lucide-react'

// ── City / carrier lists ───────────────────────────────────────────────────
const ALL_CITIES = [
  'atlanta','austin','baltimore','boston','buffalo','burlington',
  'charlotte','chicago','cincinnati','cleveland','columbus','dallas',
  'dallas-fort worth','dayton','denver','detroit','fort lauderdale',
  'fort myers','greensboro','honolulu','houston','indianapolis',
  'jacksonville','kansas city','las vegas','los angeles','louisville',
  'madison','memphis','miami','milwaukee','minneapolis','nashville',
  'new orleans','new york','newark','norfolk','oakland','omaha','daytona beach',
  'orlando','panama city','philadelphia','phoenix','pittsburgh',
  'portland','raleigh-durham','richmond','rochester','sacramento',
  'salt lake city','san antonio','san diego','san francisco','san jose',
  'san juan','sarasota','seattle','st. louis','syracuse','tampa',
  'washington','west palm beach',
]

const CARRIERS = [
  'Allegiant Air','American Airlines','Delta Airlines','Endeavor Air',
  'Envoy Air','ExpressJet','Frontier Airlines','Hawaiian Airlines',
  'JetBlue Airways','Mesa Airline','PSA Airlines','Republic Airways',
  'SkyWest Airlines','Southwest Airlines','Spirit Airlines',
  'United Airlines','Virgin America',
]

const DISTANCES = {
  'new york|chicago': 790, 'new york|los angeles': 2475, 'new york|miami': 1280,
  'new york|boston': 215, 'new york|washington': 230, 'new york|atlanta': 865,
  'chicago|los angeles': 2020, 'chicago|miami': 1380, 'chicago|houston': 1090,
  'chicago|dallas': 920, 'los angeles|san francisco': 380, 'los angeles|seattle': 1140,
  'los angeles|denver': 1020, 'dallas|houston': 240, 'dallas|miami': 1310,
  'atlanta|miami': 660, 'atlanta|washington': 640, 'boston|washington': 440,
  'miami|orlando': 235, 'miami|tampa': 280, 'seattle|portland': 175,
  'san francisco|seattle': 810, 'denver|las vegas': 760, 'houston|dallas': 240,
  'philadelphia|washington': 140, 'nashville|atlanta': 250, 'charlotte|washington': 390,
}

const AIRPORT_COORDS = {
  'new york':       { lat: 40.6413, lon: -73.7781 },
  'los angeles':    { lat: 33.9425, lon: -118.408  },
  'chicago':        { lat: 41.9742, lon: -87.9073  },
  'miami':          { lat: 25.7959, lon: -80.2870  },
  'dallas':         { lat: 32.8998, lon: -97.0403  },
  'dallas-fort worth': { lat: 32.8998, lon: -97.0403 },
  'houston':        { lat: 29.9902, lon: -95.3368  },
  'atlanta':        { lat: 33.6407, lon: -84.4277  },
  'seattle':        { lat: 47.4502, lon: -122.309  },
  'denver':         { lat: 39.8561, lon: -104.676  },
  'boston':         { lat: 42.3656, lon: -71.0096  },
  'san francisco':  { lat: 37.6213, lon: -122.379  },
  'washington':     { lat: 38.9531, lon: -77.4565  },
  'las vegas':      { lat: 36.0840, lon: -115.154  },
  'phoenix':        { lat: 33.4373, lon: -112.008  },
  'orlando':        { lat: 28.4294, lon: -81.3089  },
  'philadelphia':   { lat: 39.8744, lon: -75.2424  },
  'minneapolis':    { lat: 44.8848, lon: -93.2223  },
  'san diego':      { lat: 32.7338, lon: -117.190  },
  'tampa':          { lat: 27.9755, lon: -82.5332  },
  'portland':       { lat: 45.5898, lon: -122.592  },
  'charlotte':      { lat: 35.2140, lon: -80.9431  },
  'nashville':      { lat: 36.1263, lon: -86.6774  },
  'salt lake city': { lat: 40.7884, lon: -111.978  },
  'kansas city':    { lat: 39.2976, lon: -94.7139  },
  'memphis':        { lat: 35.0424, lon: -89.9767  },
  'new orleans':    { lat: 29.9934, lon: -90.2580  },
  'baltimore':      { lat: 39.1754, lon: -76.6682  },
  'san jose':       { lat: 37.3626, lon: -121.929  },
  'oakland':        { lat: 37.7213, lon: -122.221  },
  'austin':         { lat: 30.1975, lon: -97.6664  },
  'san antonio':    { lat: 29.5337, lon: -98.4698  },
  'jacksonville':   { lat: 30.4941, lon: -81.6879  },
  'indianapolis':   { lat: 39.7173, lon: -86.2944  },
  'columbus':       { lat: 39.9980, lon: -82.8919  },
  'detroit':        { lat: 42.2162, lon: -83.3554  },
  'cleveland':      { lat: 41.4117, lon: -81.8498  },
  'pittsburgh':     { lat: 40.4915, lon: -80.2329  },
  'richmond':       { lat: 37.5052, lon: -77.3197  },
  'norfolk':        { lat: 36.8973, lon: -76.0179  },
  'raleigh-durham': { lat: 35.8801, lon: -78.7880  },
  'buffalo':        { lat: 42.9405, lon: -78.7322  },
  'sacramento':     { lat: 38.6954, lon: -121.591  },
  'honolulu':       { lat: 21.3245, lon: -157.925  },
  'san juan':       { lat: 18.4394, lon: -66.0018  },
  'newark':         { lat: 40.6895, lon: -74.1745  },
  'fort lauderdale':{ lat: 26.0726, lon: -80.1527  },
  'west palm beach':{ lat: 26.6832, lon: -80.0956  },
  'fort myers':     { lat: 26.5362, lon: -81.7552  },
  'dayton':         { lat: 39.9024, lon: -84.2194  },
  'louisville':     { lat: 38.1744, lon: -85.7360  },
  'milwaukee':      { lat: 42.9472, lon: -87.8966  },
  'omaha':          { lat: 41.3032, lon: -95.8941  },
  'cincinnati':     { lat: 39.0488, lon: -84.6678  },
  'st. louis':      { lat: 38.7487, lon: -90.3700  },
  'greensboro':     { lat: 36.0978, lon: -79.9373  },
  'rochester':      { lat: 43.1189, lon: -77.6724  },
  'panama city':    { lat: 30.2121, lon: -85.6828  },
  'sarasota':       { lat: 27.3954, lon: -82.5543  },
  'daytona beach':  { lat: 29.1799, lon: -81.0581  },
  'madison':        { lat: 43.1399, lon: -89.3375  },
  'burlington':     { lat: 44.4719, lon: -73.1533  },
  'syracuse':       { lat: 43.1112, lon: -76.1063  },
}

function haversineDistanceMiles(c1, c2) {
  const R = 3958.8
  const dLat = (c2.lat - c1.lat) * Math.PI / 180
  const dLon = (c2.lon - c1.lon) * Math.PI / 180
  const a = Math.sin(dLat / 2) ** 2 +
    Math.cos(c1.lat * Math.PI / 180) * Math.cos(c2.lat * Math.PI / 180) *
    Math.sin(dLon / 2) ** 2
  return Math.round(R * 2 * Math.asin(Math.sqrt(a)))
}

function getApproxDistance(origin, dest) {
  const known = DISTANCES[`${origin}|${dest}`] || DISTANCES[`${dest}|${origin}`]
  if (known) return { miles: known, estimated: false }
  const c1 = AIRPORT_COORDS[origin], c2 = AIRPORT_COORDS[dest]
  if (c1 && c2) return { miles: haversineDistanceMiles(c1, c2), estimated: true }
  return null
}

const PRESETS = [
  { label: 'NYC → Chicago', origin: 'new york',    dest: 'chicago',     carrier: 'American Airlines',  distance: 790,  duration: 130, dep: 8  },
  { label: 'LA → NYC',      origin: 'los angeles', dest: 'new york',    carrier: 'Delta Airlines',     distance: 2475, duration: 315, dep: 7  },
  { label: 'MIA → NYC',     origin: 'miami',       dest: 'new york',    carrier: 'JetBlue Airways',    distance: 1280, duration: 190, dep: 6  },
  { label: 'ATL → ORD',     origin: 'atlanta',     dest: 'chicago',     carrier: 'Southwest Airlines', distance: 720,  duration: 120, dep: 10 },
  { label: 'DAL → HOU',     origin: 'dallas',      dest: 'houston',     carrier: 'Southwest Airlines', distance: 240,  duration: 55,  dep: 9  },
]

const HOURS = Array.from({ length: 24 }, (_, i) => ({
  value: i,
  label: i === 0 ? '12:00 AM' : i < 12 ? `${i}:00 AM` : i === 12 ? '12:00 PM' : `${i - 12}:00 PM`,
}))

const MONTHS = [
  'January','February','March','April','May','June',
  'July','August','September','October','November','December',
]

// ── Time utilities ─────────────────────────────────────────────────────────
function formatTime(hour, minute = 0) {
  const h12  = hour % 12 || 12
  const ampm = hour < 12 ? 'AM' : 'PM'
  const mm   = String(minute).padStart(2, '0')
  return `${h12}:${mm} ${ampm}`
}

function addMinsToHour(baseHour, addMin) {
  const total   = baseHour * 60 + addMin
  const hour    = Math.floor(total / 60) % 24
  const minute  = total % 60
  const nextDay = total >= 24 * 60
  return { hour, minute, nextDay }
}

// ── Gauge ──────────────────────────────────────────────────────────────────
function ProbGauge({ prob }) {
  const pct    = Math.round(prob * 100)
  const color  = prob > 0.6 ? '#ef4444' : prob > 0.4 ? '#f59e0b' : '#22c55e'
  const glow   = prob > 0.6 ? '0 0 16px rgba(239,68,68,0.5)' : prob > 0.4 ? '0 0 16px rgba(245,158,11,0.5)' : '0 0 16px rgba(34,197,94,0.5)'
  const angle  = Math.PI * (1 - prob)
  const nx     = 90 + 62 * Math.cos(angle)
  const ny     = 90 - 62 * Math.sin(angle)
  const filled = prob * 251.3

  return (
    <div className="flex flex-col items-center">
      <svg width="190" height="105" viewBox="0 0 200 110">
        <path d="M15,95 A85,85 0 0,1 185,95" fill="none" stroke="#1e293b" strokeWidth="18" strokeLinecap="round"/>
        <path d="M15,95 A85,85 0 0,1 185,95"
          fill="none" stroke={color} strokeWidth="18" strokeLinecap="round"
          strokeDasharray={`${filled} 300`}
          style={{ filter: glow, transition: 'stroke-dasharray 0.6s ease, stroke 0.4s' }}
        />
        <text x="18"  y="108" fill="#22c55e" fontSize="9" fontWeight="600">LOW</text>
        <text x="86"  y="20"  fill="#f59e0b" fontSize="9" fontWeight="600" textAnchor="middle">MED</text>
        <text x="172" y="108" fill="#ef4444" fontSize="9" fontWeight="600" textAnchor="end">HIGH</text>
        <line x1="100" y1="95" x2={nx} y2={ny}
          stroke="white" strokeWidth="2.5" strokeLinecap="round"
          style={{ transition: 'x2 0.6s ease, y2 0.6s ease' }}
        />
        <circle cx="100" cy="95" r="5" fill="white"/>
        <text x="100" y="80" textAnchor="middle" fill="white" fontSize="21" fontWeight="700">{pct}%</text>
      </svg>
      <p className="text-xs text-slate-500 -mt-1">delay probability</p>
    </div>
  )
}

// ── Flight Timeline (time variable visualization) ──────────────────────────
function FlightTimeline({ depHour, arrHour, duration, delayMin, delayed }) {
  const durH   = Math.floor(duration / 60)
  const durM   = duration % 60
  const durationLabel = durH > 0 ? `${durH}h ${durM}m` : `${durM}m`

  const scheduled = formatTime(arrHour)
  const { hour: dh, minute: dm, nextDay } = addMinsToHour(arrHour, delayMin || 0)
  const delayedArr = formatTime(dh, dm)

  return (
    <div
      className="rounded-2xl border p-4"
      style={{ background: 'rgba(15,23,42,0.8)', borderColor: 'rgba(255,255,255,0.06)' }}
    >
      <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest mb-3">
        Flight Timeline
      </p>

      {/* Departure → Arrival row */}
      <div className="flex items-center gap-2">
        {/* DEP */}
        <div className="text-center min-w-[68px]">
          <p className="text-base font-bold text-white leading-tight">{formatTime(depHour)}</p>
          <p className="text-[10px] text-slate-500 mt-0.5">Departure</p>
        </div>

        {/* Line + icon + duration */}
        <div className="flex-1 flex flex-col items-center gap-0.5">
          <div className="w-full flex items-center gap-1">
            <div className="flex-1 h-px bg-gradient-to-r from-slate-700 via-blue-700/40 to-slate-700"/>
            <div className="w-5 h-5 rounded-full bg-blue-600/20 border border-blue-500/30 flex items-center justify-center flex-shrink-0">
              <PlaneTakeoff size={10} className="text-blue-400"/>
            </div>
            <div className="flex-1 h-px bg-gradient-to-r from-slate-700 via-blue-700/40 to-slate-700"/>
          </div>
          <span className="text-[10px] text-slate-500">{durationLabel}</span>
        </div>

        {/* Scheduled ARR */}
        <div className="text-center min-w-[68px]">
          <p className={`text-base font-bold leading-tight transition-all ${delayed && delayMin > 0 ? 'line-through text-slate-600' : 'text-white'}`}>
            {scheduled}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">Scheduled</p>
        </div>
      </div>

      {/* Delayed arrival */}
      {delayed && delayMin > 0 && (
        <div
          className="mt-3 pt-3 flex items-center justify-between rounded-xl px-3 py-2.5"
          style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.2)' }}
        >
          <div className="flex items-center gap-2">
            <AlertTriangle size={13} className="text-red-400 flex-shrink-0"/>
            <div>
              <p className="text-[10px] text-slate-400">Expected arrival</p>
              <p className="text-xs text-slate-500">{nextDay ? 'next day' : 'same day'}</p>
            </div>
          </div>
          <div className="text-right">
            <p className="text-lg font-black text-red-400 leading-tight">
              {delayedArr}
            </p>
            <p className="text-[10px] text-orange-400 font-medium">+{delayMin} min late</p>
          </div>
        </div>
      )}

      {/* On-time message */}
      {!delayed && (
        <div
          className="mt-3 pt-3 flex items-center justify-between rounded-xl px-3 py-2"
          style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}
        >
          <div className="flex items-center gap-2">
            <CheckCircle size={13} className="text-green-400"/>
            <span className="text-xs text-slate-400">Arrives on schedule</span>
          </div>
          <p className="text-base font-bold text-green-400">{scheduled}</p>
        </div>
      )}
    </div>
  )
}

// ── Category Badge ─────────────────────────────────────────────────────────
function CategoryBadge({ category }) {
  const styles = {
    minor:       'bg-yellow-900/60 text-yellow-300 border-yellow-700/50',
    moderate:    'bg-orange-900/60 text-orange-300 border-orange-700/50',
    significant: 'bg-red-900/60    text-red-300    border-red-700/50',
    severe:      'bg-rose-950/60   text-rose-300   border-rose-600/50',
  }
  const cls = styles[category] || 'bg-slate-800 text-slate-300 border-slate-600'
  return (
    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border capitalize ${cls}`}>
      {category}
    </span>
  )
}

// ── Confidence Badge ───────────────────────────────────────────────────────
function ConfidenceBadge({ confidence }) {
  const styles = {
    high:   'bg-blue-900/60   text-blue-300   border-blue-700/50',
    medium: 'bg-purple-900/60 text-purple-300 border-purple-700/50',
    low:    'bg-slate-800/60  text-slate-400  border-slate-600/50',
  }
  const cls   = styles[confidence] || 'bg-slate-800 text-slate-300 border-slate-600'
  const icons = { high: '●●●', medium: '●●○', low: '●○○' }
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border capitalize ${cls}`}>
      <span className="tracking-widest text-xs opacity-70">{icons[confidence] || '●○○'}</span>
      {confidence} confidence
    </span>
  )
}

// ── Delay Breakdown ────────────────────────────────────────────────────────
function DelayBreakdown({ breakdown }) {
  if (!breakdown) return null
  const segments = [
    { key: 'late_aircraft', label: 'Late Aircraft', color: '#f87171' },
    { key: 'carrier',       label: 'Carrier',       color: '#fb923c' },
    { key: 'nas',           label: 'Air Traffic',   color: '#facc15' },
    { key: 'weather',       label: 'Weather',       color: '#60a5fa' },
  ]
  const total = segments.reduce((s, seg) => s + (breakdown[seg.key] || 0), 0)
  if (total === 0) return null

  return (
    <div
      className="rounded-2xl border p-5"
      style={{ background: 'rgba(15,23,42,0.8)', borderColor: 'rgba(255,255,255,0.06)' }}
    >
      <p className="text-sm font-semibold text-slate-200 mb-4 flex items-center gap-2">
        <TrendingUp size={14} className="text-orange-400"/>
        Delay Breakdown
      </p>

      {/* Stacked bar */}
      <div className="flex h-4 rounded-lg overflow-hidden mb-4 gap-px">
        {segments.map(seg => {
          const val = breakdown[seg.key] || 0
          const pct = total > 0 ? (val / total) * 100 : 0
          if (pct === 0) return null
          return (
            <div key={seg.key}
              style={{ width: `${pct}%`, backgroundColor: seg.color, transition: 'width 0.7s ease' }}
              className="h-full first:rounded-l-lg last:rounded-r-lg"
              title={`${seg.label}: ${val} min`}
            />
          )
        })}
      </div>

      <div className="space-y-2.5">
        {segments.map(seg => {
          const val = breakdown[seg.key] || 0
          const pct = total > 0 ? (val / total) * 100 : 0
          return (
            <div key={seg.key} className="flex items-center gap-3">
              <div className="w-24 text-xs text-slate-400 text-right flex-shrink-0">{seg.label}</div>
              <div className="flex-1 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                <div
                  style={{ width: `${pct}%`, backgroundColor: seg.color, transition: 'width 0.8s ease' }}
                  className="h-full rounded-full"
                />
              </div>
              <div className="w-12 text-xs text-slate-300 font-medium flex-shrink-0 tabular-nums">
                {val} min
              </div>
            </div>
          )
        })}
      </div>

      <div className="flex flex-wrap gap-3 mt-4 pt-3 border-t border-slate-800/60">
        {segments.map(seg => (
          <div key={seg.key} className="flex items-center gap-1.5 text-xs text-slate-400">
            <div style={{ backgroundColor: seg.color }} className="w-2 h-2 rounded-full flex-shrink-0"/>
            {seg.label}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Tips ───────────────────────────────────────────────────────────────────
function TipsSection({ result }) {
  const { delay_category, delayed } = result
  const tipsByCategory = {
    severe:      [
      { icon: 'alert', text: 'Very high delay risk — consider rebooking or an alternate route.' },
      { icon: 'clock', text: 'Build in at least 2-hour connection buffer if connecting.' },
      { icon: 'weather', text: 'Check weather forecasts at both airports.' },
      { icon: 'info', text: "Arrive early and have your airline's customer service number ready." },
    ],
    significant: [
      { icon: 'wind',  text: 'Consider an earlier flight — morning slots typically have lower delays.' },
      { icon: 'clock', text: 'Build in a 90-min connection buffer if connecting.' },
      { icon: 'weather', text: 'Check weather at both airports before departing.' },
      { icon: 'info', text: "Sign up for flight status alerts via your airline's app." },
    ],
    moderate:    [
      { icon: 'wind',  text: 'Morning flights on this route are generally more reliable.' },
      { icon: 'clock', text: 'Allow a 60-min connection buffer just in case.' },
      { icon: 'info',  text: 'Monitor flight status the night before and morning of travel.' },
    ],
    minor:       [
      { icon: 'check', text: 'Low delay expected — this looks like a reliable flight.' },
      { icon: 'info',  text: 'Still check real-time status on the day of travel.' },
      { icon: 'check', text: 'A 45-min connection buffer should be sufficient.' },
    ],
  }
  const onTimeTips = [
    { icon: 'check', text: "Great news — this flight looks on track." },
    { icon: 'check', text: 'Morning slots on this route are historically reliable.' },
    { icon: 'info',  text: 'Still verify real-time status on departure day.' },
  ]
  const tips   = !delayed ? onTimeTips : (tipsByCategory[delay_category] || tipsByCategory.moderate)
  const iconMap = {
    alert:   <AlertTriangle size={11} className="text-red-400 mt-0.5 flex-shrink-0"/>,
    wind:    <Wind          size={11} className="text-amber-400 mt-0.5 flex-shrink-0"/>,
    weather: <Droplets      size={11} className="text-blue-400 mt-0.5 flex-shrink-0"/>,
    clock:   <Clock         size={11} className="text-slate-400 mt-0.5 flex-shrink-0"/>,
    check:   <CheckCircle   size={11} className="text-green-400 mt-0.5 flex-shrink-0"/>,
    info:    <Info          size={11} className="text-slate-400 mt-0.5 flex-shrink-0"/>,
  }

  return (
    <div
      className="rounded-2xl border p-5"
      style={{ background: 'rgba(15,23,42,0.8)', borderColor: 'rgba(255,255,255,0.06)' }}
    >
      <p className="text-sm font-semibold text-slate-200 mb-3 flex items-center gap-2">
        <ThumbsUp size={14} className="text-blue-400"/>
        {delayed ? 'Tips to Manage Your Delay' : 'Travel Tips'}
      </p>
      <ul className="space-y-2 text-xs text-slate-400">
        {tips.map((tip, i) => (
          <li key={i} className="flex gap-2">
            {iconMap[tip.icon] || iconMap.info}
            {tip.text}
          </li>
        ))}
      </ul>
    </div>
  )
}

// ── Main ───────────────────────────────────────────────────────────────────
export default function Home() {
  const today = new Date()

  const [form, setForm] = useState({
    origin_city      : 'new york',
    dest_city        : 'chicago',
    carrier          : 'Southwest Airlines',
    distance         : 790,
    crs_elapsed_time : 130,
    dep_hour         : 8,
    arr_hour         : 10,
    month            : today.getMonth() + 1,
    day              : today.getDate(),
    is_weekend       : [0, 6].includes(today.getDay()),
    origin_iata      : 'JFK',
    dest_iata        : 'ORD',
  })
  const [result,        setResult]        = useState(null)
  const [loading,       setLoading]       = useState(false)
  const [error,         setError]         = useState(null)
  const [distEstimated, setDistEstimated] = useState(false)

  function set(k, v) { setForm(f => ({ ...f, [k]: v })) }

  function applyPreset(p) {
    setResult(null)
    setForm(f => ({
      ...f,
      origin_city      : p.origin,
      dest_city        : p.dest,
      carrier          : p.carrier,
      distance         : p.distance,
      crs_elapsed_time : p.duration,
      dep_hour         : p.dep,
      arr_hour         : Math.round(p.dep + p.duration / 60),
    }))
  }

  function handleDateChange(e) {
    const d = new Date(e.target.value)
    if (isNaN(d)) return
    set('month', d.getMonth() + 1)
    set('day',   d.getDate())
    set('is_weekend', [0, 6].includes(d.getDay()))
  }

  function handleOriginChange(city) {
    const dist = getApproxDistance(city, form.dest_city)
    setDistEstimated(dist?.estimated ?? false)
    setForm(f => ({
      ...f,
      origin_city     : city,
      ...(dist && {
        // (distance / 7) + 30 ≈ minutes at ~420 mph ground speed + 30 min taxi/climb
        distance        : dist.miles,
        crs_elapsed_time: Math.round(dist.miles / 7 + 30),
      }),
    }))
  }

  function handleDestChange(city) {
    const dist = getApproxDistance(form.origin_city, city)
    setDistEstimated(dist?.estimated ?? false)
    setForm(f => ({
      ...f,
      dest_city       : city,
      ...(dist && {
        distance        : dist.miles,
        crs_elapsed_time: Math.round(dist.miles / 7 + 30),
      }),
    }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try {
      const res = await predictFlight({
        ...form,
        distance         : parseFloat(form.distance),
        crs_elapsed_time : parseFloat(form.crs_elapsed_time),
        dep_hour         : parseInt(form.dep_hour),
        arr_hour         : parseInt(form.arr_hour),
        month            : parseInt(form.month),
        day              : parseInt(form.day),
      })
      setResult(res)
    } catch {
      setError('Cannot reach the API. Run: python -m uvicorn api.main:app --port 8000')
    } finally {
      setLoading(false)
    }
  }

  const inputCls = [
    'w-full rounded-xl px-3 py-2.5 text-sm text-white',
    'transition-all duration-200 outline-none',
    'focus:ring-1 focus:ring-blue-500/40 focus:border-blue-500/60',
  ].join(' ')

  const inputStyle = {
    background  : 'rgba(15,23,42,0.9)',
    border      : '1px solid rgba(255,255,255,0.07)',
  }

  const labelCls = 'block text-xs font-medium text-slate-400 mb-1.5'

  return (
    <div
      className="min-h-screen"
      style={{
        background: 'radial-gradient(ellipse 80% 60% at 50% -10%, rgba(37,99,235,0.12) 0%, transparent 60%), #020617',
      }}
    >
      <div className="max-w-6xl mx-auto px-4 py-10">

        {/* ── Hero ── */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium mb-4
                          text-blue-300 border border-blue-500/20"
               style={{ background: 'rgba(37,99,235,0.1)' }}>
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse"/>
            ML-powered · 86k flights trained · 91.34% accuracy
          </div>
          <h1 className="text-4xl sm:text-5xl font-black tracking-tight mb-3 leading-tight">
            Will your flight be{' '}
            <span
              style={{
                background: 'linear-gradient(90deg,#60a5fa,#a78bfa)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              delayed?
            </span>
          </h1>
          <p className="text-slate-500 text-sm max-w-md mx-auto">
            LightGBM pipeline · 218 features · ROC-AUC 0.973 · Real-time weather
          </p>
        </div>

        {/* ── Presets ── */}
        <div className="flex flex-wrap gap-2 justify-center mb-8">
          <span className="text-xs text-slate-600 self-center mr-1 font-medium">Try:</span>
          {PRESETS.map(p => (
            <button key={p.label} type="button" onClick={() => applyPreset(p)}
              className="px-3 py-1.5 text-xs rounded-full font-medium
                         text-slate-300 hover:text-white transition-all duration-200
                         hover:scale-105 active:scale-95
                         flex items-center gap-1.5"
              style={{
                background   : 'rgba(30,41,59,0.8)',
                border       : '1px solid rgba(255,255,255,0.07)',
                boxShadow    : '0 1px 3px rgba(0,0,0,0.3)',
              }}
            >
              <Zap size={11} className="text-blue-400"/>
              {p.label}
            </button>
          ))}
        </div>

        <div className="grid lg:grid-cols-5 gap-6">

          {/* ── FORM (3 cols) ── */}
          <form onSubmit={handleSubmit}
            className="lg:col-span-3 rounded-2xl p-6 space-y-5"
            style={{
              background  : 'rgba(15,23,42,0.7)',
              border      : '1px solid rgba(255,255,255,0.07)',
              backdropFilter: 'blur(12px)',
              boxShadow   : '0 4px 32px rgba(0,0,0,0.4)',
            }}
          >
            {/* Route */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>
                  <span className="flex items-center gap-1"><PlaneTakeoff size={11}/> From</span>
                </label>
                <select className={inputCls} style={inputStyle} value={form.origin_city}
                  onChange={e => handleOriginChange(e.target.value)}>
                  {ALL_CITIES.map(c => (
                    <option key={c} value={c}>{c.replace(/\b\w/g, l => l.toUpperCase())}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className={labelCls}>
                  <span className="flex items-center gap-1"><PlaneLanding size={11}/> To</span>
                </label>
                <select className={inputCls} style={inputStyle} value={form.dest_city}
                  onChange={e => handleDestChange(e.target.value)}>
                  {ALL_CITIES.map(c => (
                    <option key={c} value={c}>{c.replace(/\b\w/g, l => l.toUpperCase())}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Route pill */}
            <div
              className="flex items-center justify-between px-4 py-2.5 rounded-xl text-sm"
              style={{ background: 'rgba(37,99,235,0.08)', border: '1px solid rgba(37,99,235,0.2)' }}
            >
              <span className="text-white font-semibold capitalize">{form.origin_city}</span>
              <div className="flex items-center gap-1.5 text-slate-500 text-xs">
                <div className="w-8 h-px bg-slate-700"/>
                <ArrowRight size={12} className="text-blue-400"/>
                <div className="w-8 h-px bg-slate-700"/>
              </div>
              <span className="text-white font-semibold capitalize">{form.dest_city}</span>
              <span className="text-slate-500 text-xs tabular-nums">
                {form.distance} mi{distEstimated && <span className="ml-1 text-slate-600">(est.)</span>}
              </span>
            </div>

            {/* Airline */}
            <div>
              <label className={labelCls}>Airline</label>
              <select className={inputCls} style={inputStyle} value={form.carrier}
                onChange={e => set('carrier', e.target.value)}>
                {CARRIERS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>

            {/* Distance + Duration */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>Distance (miles)</label>
                <input type="number" className={inputCls} style={inputStyle}
                  value={form.distance} onChange={e => set('distance', e.target.value)}
                  min="50" max="5000"/>
              </div>
              <div>
                <label className={labelCls}>
                  <span className="flex items-center gap-1"><Clock size={11}/> Duration (min)</span>
                </label>
                <input type="number" className={inputCls} style={inputStyle}
                  value={form.crs_elapsed_time}
                  onChange={e => set('crs_elapsed_time', e.target.value)} min="30" max="720"/>
              </div>
            </div>

            {/* Dep + Arr time */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>Departure Time</label>
                <select className={inputCls} style={inputStyle} value={form.dep_hour}
                  onChange={e => set('dep_hour', parseInt(e.target.value))}>
                  {HOURS.map(h => <option key={h.value} value={h.value}>{h.label}</option>)}
                </select>
              </div>
              <div>
                <label className={labelCls}>Scheduled Arrival</label>
                <select className={inputCls} style={inputStyle} value={form.arr_hour}
                  onChange={e => set('arr_hour', parseInt(e.target.value))}>
                  {HOURS.map(h => <option key={h.value} value={h.value}>{h.label}</option>)}
                </select>
              </div>
            </div>

            {/* Date */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className={labelCls}>Flight Date</label>
                <input type="date" className={inputCls} style={inputStyle}
                  defaultValue={`${today.getFullYear()}-${String(today.getMonth()+1).padStart(2,'0')}-${String(today.getDate()).padStart(2,'0')}`}
                  onChange={handleDateChange}/>
              </div>
              <div>
                <label className={labelCls}>Date Info</label>
                <div className={`${inputCls} text-slate-400`} style={inputStyle}>
                  {MONTHS[form.month - 1]} · Day {form.day}
                  {form.is_weekend && <span className="ml-2 text-amber-400 text-xs">(weekend)</span>}
                </div>
              </div>
            </div>

            {/* IATA codes */}
            <div
              className="rounded-xl p-4 space-y-3"
              style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
            >
              <p className="text-xs text-slate-500">
                Optional: IATA codes fetch live weather at each airport
              </p>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className={labelCls}>Origin IATA</label>
                  <input className={`${inputCls} uppercase tracking-widest font-mono`}
                    style={inputStyle} placeholder="e.g. JFK" maxLength={4}
                    value={form.origin_iata}
                    onChange={e => set('origin_iata', e.target.value.toUpperCase())}/>
                </div>
                <div>
                  <label className={labelCls}>Destination IATA</label>
                  <input className={`${inputCls} uppercase tracking-widest font-mono`}
                    style={inputStyle} placeholder="e.g. ORD" maxLength={4}
                    value={form.dest_iata}
                    onChange={e => set('dest_iata', e.target.value.toUpperCase())}/>
                </div>
              </div>
            </div>

            {/* Submit */}
            <button type="submit" disabled={loading}
              className="w-full py-3.5 rounded-xl font-bold text-white text-sm
                         disabled:opacity-50 active:scale-[0.98]
                         flex items-center justify-center gap-2 transition-all duration-200"
              style={{
                background  : loading ? 'rgba(37,99,235,0.5)' : 'linear-gradient(135deg,#2563eb,#4f46e5)',
                boxShadow   : loading ? 'none' : '0 0 24px rgba(79,70,229,0.35), 0 4px 12px rgba(0,0,0,0.3)',
              }}
            >
              {loading
                ? <><Loader size={16} className="animate-spin"/> Analysing flight...</>
                : <><Zap size={16}/> Predict Delay Risk</>}
            </button>

            {error && (
              <p className="text-red-400 text-sm text-center rounded-xl px-4 py-3"
                 style={{ background: 'rgba(239,68,68,0.08)', border: '1px solid rgba(239,68,68,0.25)' }}>
                {error}
              </p>
            )}
          </form>

          {/* ── RESULT PANEL (2 cols) ── */}
          <div className="lg:col-span-2 flex flex-col gap-4">

            {/* Placeholder */}
            {!result && !loading && (
              <div
                className="flex-1 rounded-2xl flex flex-col items-center justify-center text-center p-10 gap-4"
                style={{
                  background: 'rgba(15,23,42,0.5)',
                  border: '1px dashed rgba(255,255,255,0.08)',
                }}
              >
                <div
                  className="w-16 h-16 rounded-2xl flex items-center justify-center"
                  style={{ background: 'rgba(37,99,235,0.08)', border: '1px solid rgba(37,99,235,0.15)' }}
                >
                  <PlaneTakeoff size={28} className="text-blue-500/40"/>
                </div>
                <div>
                  <p className="text-slate-400 text-sm font-medium">Ready to predict</p>
                  <p className="text-slate-600 text-xs mt-1">Fill in flight details and click Predict</p>
                </div>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div
                className="flex-1 rounded-2xl flex flex-col items-center justify-center gap-3 py-16"
                style={{ background: 'rgba(15,23,42,0.7)', border: '1px solid rgba(255,255,255,0.06)' }}
              >
                <Loader size={34} className="animate-spin text-blue-400"/>
                <p className="text-slate-500 text-sm">Running prediction model...</p>
              </div>
            )}

            {/* Result */}
            {result && (
              <>
                {/* Status Banner */}
                <div
                  className="rounded-2xl p-5 flex flex-col items-center gap-3"
                  style={result.delayed ? {
                    background: 'linear-gradient(135deg, rgba(127,29,29,0.3), rgba(127,29,29,0.1))',
                    border    : '1px solid rgba(239,68,68,0.3)',
                    boxShadow : '0 0 40px rgba(239,68,68,0.08)',
                  } : {
                    background: 'linear-gradient(135deg, rgba(20,83,45,0.3), rgba(20,83,45,0.1))',
                    border    : '1px solid rgba(34,197,94,0.3)',
                    boxShadow : '0 0 40px rgba(34,197,94,0.06)',
                  }}
                >
                  {/* Headline */}
                  <div className={`flex items-center gap-2 text-xl font-black tracking-tight
                    ${result.delayed ? 'text-red-400' : 'text-green-400'}`}>
                    {result.delayed
                      ? <><AlertTriangle size={21}/> Likely Delayed</>
                      : <><CheckCircle   size={21}/> Likely On-Time</>}
                  </div>

                  {/* Delay quantity */}
                  {result.delayed ? (
                    <div className="text-center">
                      <p className="text-5xl font-black text-white tracking-tight leading-none tabular-nums">
                        ~{result.expected_delay_min}
                        <span className="text-xl font-semibold text-slate-400 ml-1">min</span>
                      </p>
                      {result.delay_range && (
                        <p className="text-slate-500 text-xs mt-1 font-medium tabular-nums">
                          Range: {result.delay_range}
                        </p>
                      )}
                    </div>
                  ) : (
                    <div className="text-center">
                      <p className="text-3xl font-black text-green-400 tracking-tight">On Schedule</p>
                      <p className="text-slate-500 text-xs mt-1">No significant delay expected</p>
                    </div>
                  )}

                  {/* Badges */}
                  <div className="flex flex-wrap gap-2 justify-center">
                    {result.delayed && result.delay_category && (
                      <CategoryBadge category={result.delay_category}/>
                    )}
                    <ConfidenceBadge confidence={result.confidence}/>
                  </div>

                  {/* Gauge */}
                  <ProbGauge prob={result.probability}/>

                  {/* Verdict */}
                  <p className="text-slate-300 text-xs text-center leading-relaxed w-full rounded-xl px-4 py-2.5"
                     style={{ background: 'rgba(0,0,0,0.2)', border: '1px solid rgba(255,255,255,0.05)' }}>
                    {result.verdict}
                  </p>
                </div>

                {/* Flight Timeline — time variable visualization */}
                <FlightTimeline
                  depHour  = {form.dep_hour}
                  arrHour  = {form.arr_hour}
                  duration = {form.crs_elapsed_time}
                  delayMin = {result.expected_delay_min}
                  delayed  = {result.delayed}
                />

                {/* Breakdown */}
                {result.delayed && result.delay_breakdown && (
                  <DelayBreakdown breakdown={result.delay_breakdown}/>
                )}

                {/* Meta strip */}
                <div className="grid grid-cols-2 gap-3">
                  {[
                    { icon: <Shield size={10}/>, label: 'Model', value: result.model_used, cls: 'text-slate-300' },
                    { icon: <Calendar size={10}/>, label: 'Cluster',
                      value: result.cluster === -1 ? 'Global' : `#${result.cluster}`, cls: 'text-purple-400' },
                  ].map(card => (
                    <div key={card.label}
                      className="rounded-xl p-3.5 text-center"
                      style={{ background: 'rgba(15,23,42,0.8)', border: '1px solid rgba(255,255,255,0.06)' }}
                    >
                      <p className="text-xs text-slate-500 mb-1 flex items-center justify-center gap-1">
                        {card.icon} {card.label}
                      </p>
                      <p className={`font-semibold text-sm capitalize ${card.cls}`}>{card.value}</p>
                    </div>
                  ))}
                </div>

                {/* Tips */}
                <TipsSection result={result}/>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
