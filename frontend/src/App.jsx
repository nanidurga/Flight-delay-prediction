import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home'
import Dashboard from './pages/Dashboard'
import LiveMap from './pages/LiveMap'
import About from './pages/About'

export default function App() {
  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          <Route path="/"          element={<Home />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/live"      element={<LiveMap />} />
          <Route path="/about"     element={<About />} />
        </Routes>
      </main>
      <footer
        className="text-center text-slate-600 text-xs py-4"
        style={{ borderTop: '1px solid rgba(255,255,255,0.04)', background: 'rgba(2,6,23,0.9)' }}
      >
        MTP · Flight Delay Predictor · 21MA23002 · Cluster-then-Classify Pipeline
      </footer>
    </div>
  )
}
