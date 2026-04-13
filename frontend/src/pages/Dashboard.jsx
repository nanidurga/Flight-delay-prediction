import React, { useEffect, useState } from 'react'
import { getStats, getModelInfo } from '../api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, PieChart, Pie, Cell, Legend,
} from 'recharts'
import { TrendingUp, Layers, Cpu, Target } from 'lucide-react'

const COLORS = ['#3b82f6', '#ef4444', '#f59e0b', '#22c55e', '#8b5cf6', '#ec4899']

export default function Dashboard() {
  const [stats, setStats]     = useState(null)
  const [info,  setInfo]      = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getStats(), getModelInfo()])
      .then(([s, i]) => { setStats(s); setInfo(i) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="flex items-center justify-center h-96 text-slate-400">
      Loading dashboard...
    </div>
  )

  const pieData = [
    { name: 'Delayed',  value: stats.delayed_pct  },
    { name: 'On-Time',  value: stats.on_time_pct  },
  ]

  return (
    <div className="max-w-7xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-3xl font-bold mb-1">Dashboard</h1>
        <p className="text-slate-400">Model performance and historical delay patterns</p>
      </div>

      {/* Model stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard icon={<Target className="text-blue-400"/>}
          label="Test Accuracy" value={info.overall_accuracy_pct} />
        <StatCard icon={<Layers className="text-purple-400"/>}
          label="DBSCAN Clusters" value={info.n_clusters.toLocaleString()} />
        <StatCard icon={<Cpu className="text-green-400"/>}
          label="Cluster Models" value={info.n_cluster_models.toLocaleString()} />
        <StatCard icon={<TrendingUp className="text-amber-400"/>}
          label="Training Flights" value="86,478" />
      </div>

      {/* Charts row 1 */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Monthly delay chart */}
        <ChartCard title="Delay Rate by Month">
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={stats.delay_by_month}>
              <XAxis dataKey="month" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis domain={[35, 65]} tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                labelStyle={{ color: '#94a3b8' }}
                formatter={v => [`${v}%`, 'Delay Rate']}
              />
              <Line type="monotone" dataKey="delay_rate_pct" stroke="#3b82f6"
                strokeWidth={2} dot={{ fill: '#3b82f6', r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Delay by time of day */}
        <ChartCard title="Delay Rate by Time of Day">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={stats.delay_by_time_of_day} barCategoryGap="30%">
              <XAxis dataKey="slot" tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <YAxis domain={[30, 70]} tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={v => [`${v}%`, 'Delay Rate']}
              />
              <Bar dataKey="delay_rate_pct" radius={[6,6,0,0]}>
                {stats.delay_by_time_of_day.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Charts row 2 */}
      <div className="grid md:grid-cols-3 gap-6">
        {/* Airline delay rates */}
        <div className="md:col-span-2">
          <ChartCard title="Delay Rate by Airline">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={stats.top_delay_carriers} layout="vertical" barCategoryGap="25%">
                <XAxis type="number" domain={[40, 60]} tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
                <YAxis type="category" dataKey="carrier" tick={{ fill: '#94a3b8', fontSize: 11 }} width={130} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  formatter={v => [`${v}%`, 'Delay Rate']}
                />
                <Bar dataKey="delay_rate_pct" radius={[0,6,6,0]} fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Overall pie */}
        <ChartCard title="Overall Split">
          <ResponsiveContainer width="100%" height={240}>
            <PieChart>
              <Pie data={pieData} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                dataKey="value" paddingAngle={4}>
                {pieData.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? '#ef4444' : '#22c55e'} />
                ))}
              </Pie>
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 13 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                formatter={v => [`${v}%`]}
              />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Weather vs delay */}
      <ChartCard title="Delay Rate by Weather Condition">
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={stats.top_delay_conditions} barCategoryGap="25%">
            <XAxis dataKey="condition" tick={{ fill: '#94a3b8', fontSize: 11 }} />
            <YAxis domain={[30, 75]} tick={{ fill: '#94a3b8', fontSize: 11 }} unit="%" />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
              formatter={v => [`${v}%`, 'Delay Rate']}
            />
            <Bar dataKey="delay_rate_pct" radius={[6,6,0,0]}>
              {stats.top_delay_conditions.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Selected features */}
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6">
        <h3 className="font-semibold text-slate-300 mb-4">Top 20 ANOVA Features</h3>
        <div className="flex flex-wrap gap-2">
          {info.selected_features.map(f => (
            <span key={f} className="bg-slate-800 text-slate-300 text-xs px-3 py-1 rounded-full">
              {f}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

function StatCard({ icon, label, value }) {
  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5 flex items-center gap-4">
      <div className="p-2 bg-slate-800 rounded-lg">{icon}</div>
      <div>
        <p className="text-xs text-slate-500">{label}</p>
        <p className="text-xl font-bold text-white">{value}</p>
      </div>
    </div>
  )
}

function ChartCard({ title, children }) {
  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 p-5">
      <h3 className="font-semibold text-slate-300 mb-4">{title}</h3>
      {children}
    </div>
  )
}
