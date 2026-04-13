import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { PlaneTakeoff, Menu, X } from 'lucide-react'

const links = [
  { to: '/',          label: 'Predict'   },
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/live',      label: 'Live Map'  },
  { to: '/about',     label: 'About'     },
]

export default function Navbar() {
  const [open,    setOpen]    = useState(false)
  const [hovered, setHovered] = useState(null)

  return (
    <nav
      className="sticky top-0 z-50 border-b border-white/5"
      style={{ background: 'rgba(2,6,23,0.88)', backdropFilter: 'blur(20px)' }}
    >
      <div className="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">

        {/* ── Logo ── */}
        <NavLink to="/" className="group flex items-center gap-2.5 select-none">
          <div
            className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-300
                       group-hover:scale-110 group-hover:rotate-[-6deg]"
            style={{
              background: 'linear-gradient(135deg, #3b82f6, #6366f1)',
              boxShadow: '0 0 18px rgba(99,102,241,0.45)',
            }}
          >
            <PlaneTakeoff size={16} className="text-white" />
          </div>
          <span className="font-bold text-[15px] tracking-tight">
            <span className="text-white">Flight</span>
            <span
              style={{
                background: 'linear-gradient(90deg,#60a5fa,#a78bfa)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              Sense
            </span>
          </span>
        </NavLink>

        {/* ── Desktop pill nav ── */}
        <div
          className="hidden md:flex items-center gap-0.5 p-1 rounded-2xl border border-white/5"
          style={{ background: 'rgba(15,23,42,0.7)' }}
        >
          {links.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === '/'}
              onMouseEnter={() => setHovered(l.to)}
              onMouseLeave={() => setHovered(null)}
              className={({ isActive }) =>
                'relative px-4 py-1.5 rounded-xl text-sm font-medium select-none outline-none ' +
                'transition-colors duration-150 ' +
                (isActive
                  ? 'text-white'
                  : hovered === l.to
                  ? 'text-slate-100'
                  : 'text-slate-400')
              }
            >
              {({ isActive }) => (
                <>
                  {/* Active gradient pill */}
                  {isActive && (
                    <span
                      className="absolute inset-0 rounded-xl pointer-events-none"
                      style={{
                        background: 'linear-gradient(135deg,#2563eb,#1d4ed8)',
                        boxShadow:
                          '0 0 22px rgba(37,99,235,0.4), inset 0 1px 0 rgba(255,255,255,0.1)',
                      }}
                    />
                  )}
                  {/* Hover background */}
                  {!isActive && hovered === l.to && (
                    <span
                      className="absolute inset-0 rounded-xl pointer-events-none transition-opacity duration-150"
                      style={{ background: 'rgba(255,255,255,0.06)' }}
                    />
                  )}
                  <span className="relative z-10">{l.label}</span>
                </>
              )}
            </NavLink>
          ))}
        </div>

        {/* ── Mobile toggle ── */}
        <button
          className="md:hidden w-9 h-9 flex items-center justify-center rounded-xl
                     bg-slate-800/80 text-slate-400 hover:text-white hover:bg-slate-700
                     transition-all duration-200 border border-white/5"
          onClick={() => setOpen(o => !o)}
          aria-label="Toggle menu"
        >
          <span
            className="transition-transform duration-250"
            style={{ display: 'block', transform: open ? 'rotate(90deg)' : 'rotate(0deg)' }}
          >
            {open ? <X size={17} /> : <Menu size={17} />}
          </span>
        </button>
      </div>

      {/* ── Mobile dropdown (animated height) ── */}
      <div
        className="md:hidden overflow-hidden transition-all duration-300 ease-in-out"
        style={{ maxHeight: open ? '240px' : '0px' }}
      >
        <div className="border-t border-white/5 px-4 pb-4 pt-2 flex flex-col gap-1">
          {links.map(l => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.to === '/'}
              onClick={() => setOpen(false)}
              className={({ isActive }) =>
                'px-4 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ' +
                (isActive
                  ? 'text-white shadow-md shadow-blue-900/40'
                  : 'text-slate-400 hover:text-white hover:bg-slate-800')
              }
              style={({ isActive }) =>
                isActive
                  ? { background: 'linear-gradient(135deg,#2563eb,#1d4ed8)' }
                  : {}
              }
            >
              {l.label}
            </NavLink>
          ))}
        </div>
      </div>
    </nav>
  )
}
