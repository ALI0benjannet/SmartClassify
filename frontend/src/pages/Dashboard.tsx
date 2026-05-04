import { useEffect, useState, useCallback } from 'react'
import { getStats, getTrainingStatus, type Stats, type TrainingStatus } from '../api/client'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'

// ─── Training status card ────────────────────────────────────────────────────

function TrainingStatusCard({ ts }: { ts: TrainingStatus | null }) {
  if (!ts) return null

  const cfg: Record<string, { bg: string; text: string; dot: string; icon: string; label: string }> = {
    idle:     { bg: 'from-slate-500/15 to-slate-600/5', text: 'text-slate-300', dot: 'bg-slate-400', icon: '💤', label: 'En attente' },
    training: { bg: 'from-blue-500/20 to-blue-600/5',   text: 'text-blue-300',  dot: 'bg-blue-400 animate-pulse', icon: '⚙️', label: 'Entraînement…' },
    done:     { bg: 'from-emerald-500/20 to-emerald-600/5', text: 'text-emerald-300', dot: 'bg-emerald-400', icon: '✅', label: 'Terminé' },
    error:    { bg: 'from-red-500/20 to-red-600/5',     text: 'text-red-300',   dot: 'bg-red-400', icon: '❌', label: 'Erreur' },
  }
  const c = cfg[ts.status] ?? cfg.idle

  const next = ts.auto_retrain_every - (ts.ingested_count % ts.auto_retrain_every)
  const progressPct = Math.round(
    ((ts.ingested_count % ts.auto_retrain_every) / ts.auto_retrain_every) * 100
  )

  return (
    <div className={`glass rounded-2xl p-6 bg-gradient-to-br ${c.bg} mb-8`}>
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-5">
        <div className="flex items-center gap-3">
          <span className="text-3xl">{c.icon}</span>
          <div>
            <p className="text-xs font-bold uppercase tracking-widest text-slate-400">Auto-Retrain</p>
            <div className="flex items-center gap-2 mt-1">
              <span className={`w-2.5 h-2.5 rounded-full ${c.dot}`} />
              <span className={`font-extrabold text-lg ${c.text}`}>{c.label}</span>
            </div>
          </div>
        </div>
        <div className="text-right">
          <p className="text-3xl font-extrabold text-white">{ts.ingested_count}</p>
          <p className="text-xs text-slate-400">données ingérées</p>
        </div>
      </div>

      {/* Progress bar toward next retrain */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>Prochain réentraînement dans <strong className="text-white">{next}</strong> donnée{next > 1 ? 's' : ''}</span>
          <span>{ts.ingested_count % ts.auto_retrain_every}/{ts.auto_retrain_every}</span>
        </div>
        <div className="h-2 rounded-full bg-white/10">
          <div
            className="h-2 rounded-full bg-gradient-to-r from-indigo-500 to-blue-400 transition-all duration-500"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>

      {/* Metrics after last retrain */}
      {ts.metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 mt-3">
          {Object.entries(ts.metrics).map(([k, v]) => (
            <div key={k} className="bg-white/5 rounded-xl p-2.5">
              <p className="text-xs text-slate-400 capitalize">{k.replace(/_/g, ' ')}</p>
              <p className="font-bold text-white text-sm">
                {v < 1 ? `${(v * 100).toFixed(1)}%` : v.toFixed(3)}
              </p>
            </div>
          ))}
        </div>
      )}

      {ts.last_trained_at && (
        <p className="text-xs text-slate-500 mt-3">
          Dernier entraînement : {new Date(ts.last_trained_at).toLocaleString('fr-FR')}
          {ts.trigger_count != null && ` (après ${ts.trigger_count} enregistrements)`}
        </p>
      )}
      {ts.error && (
        <p className="text-xs text-red-400 mt-2 font-mono">{ts.error}</p>
      )}
    </div>
  )
}

// ─── Stat card ───────────────────────────────────────────────────────────────

function StatCard({
  label, value, icon, color = 'blue', sub,
}: {
  label: string; value: string | number; icon: string; color?: string; sub?: string
}) {
  const gradients: Record<string, string> = {
    blue: 'from-blue-500/20 to-blue-600/5',
    purple: 'from-purple-500/20 to-purple-600/5',
    emerald: 'from-emerald-500/20 to-emerald-600/5',
    sky: 'from-sky-500/20 to-sky-600/5',
    pink: 'from-pink-500/20 to-pink-600/5',
    amber: 'from-amber-500/20 to-amber-600/5',
  }
  return (
    <div className={`glass rounded-2xl p-5 bg-gradient-to-br ${gradients[color] ?? gradients.blue}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">{label}</p>
          <p className="text-3xl font-extrabold text-white">{value}</p>
          {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
        </div>
        <span className="text-2xl">{icon}</span>
      </div>
    </div>
  )
}

// ─── Status dot ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const up = status === 'up'
  return (
    <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold border ${
      up ? 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30'
         : 'bg-red-500/15 text-red-300 border-red-500/30'
    }`}>
      <span className={`w-2 h-2 rounded-full ${up ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
      {up ? 'En ligne' : status}
    </span>
  )
}

// ─── Custom tooltip for recharts ─────────────────────────────────────────────

function CustomTooltip({ active, payload, label }: {
  active?: boolean; payload?: { value: number }[]; label?: string
}) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-4 py-2 text-sm">
      <p className="font-bold text-white">{label}</p>
      <p className="text-blue-300">{payload[0].value}</p>
    </div>
  )
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null)
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date())

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [data, ts] = await Promise.all([getStats(), getTrainingStatus()])
      setStats(data)
      setTrainingStatus(ts)
      setLastRefresh(new Date())
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 15000)
    return () => clearInterval(id)
  }, [refresh])

  const chartData = stats
    ? [
        { name: 'Runs', value: stats.runs },
        { name: 'Métriques', value: stats.metrics },
        { name: 'Traces', value: stats.traces },
        { name: 'Datasets', value: stats.datasets },
        { name: 'Inputs', value: stats.inputs },
        { name: 'Éval.', value: stats.evaluation_datasets },
      ]
    : []

  return (
    <div className="max-w-7xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4 mb-10">
        <div>
          <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-blue-500/15 text-blue-300 border border-blue-500/20 mb-3">
            <span className="w-2 h-2 bg-blue-400 rounded-full animate-pulse" />
            Tableau de bord en direct
          </span>
          <h1 className="text-4xl font-extrabold text-white leading-tight">
            Smart<span className="text-blue-400">Classify</span> Dashboard
          </h1>
          <p className="text-slate-400 mt-2 text-sm">
            Données en temps réel depuis l'API et la base MLflow.
            Mise à jour toutes les 15 secondes.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">
            Dernière maj : {lastRefresh.toLocaleTimeString('fr-FR')}
          </span>
          <button onClick={refresh} className="btn-gradient text-sm py-2 px-4">
            {loading ? '⟳ Chargement…' : '⟳ Actualiser'}
          </button>
        </div>
      </div>

      {error && (
        <div className="glass rounded-2xl p-4 mb-6 border-red-500/30 bg-red-500/10 text-red-300 text-sm">
          ⚠️ Impossible de joindre l'API : {error}
          <br />
          <span className="text-red-400/70 text-xs">
            Assurez-vous que l'API tourne sur le port 8002 (<code>make api</code> ou <code>make docker-run</code>).
          </span>
        </div>
      )}

      {/* Training status */}
      <TrainingStatusCard ts={trainingStatus} />

      {/* Status row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="glass rounded-2xl p-5 col-span-2 md:col-span-1">
          <p className="text-xs text-slate-400 uppercase font-semibold tracking-wider mb-2">API</p>
          {stats ? <StatusBadge status={stats.api_status} /> : <span className="text-slate-500 text-sm">—</span>}
        </div>
        <div className="glass rounded-2xl p-5 col-span-2 md:col-span-1">
          <p className="text-xs text-slate-400 uppercase font-semibold tracking-wider mb-2">MLflow</p>
          {stats ? <StatusBadge status={stats.mlflow_status} /> : <span className="text-slate-500 text-sm">—</span>}
        </div>
        <div className="glass rounded-2xl p-5 col-span-2">
          <p className="text-xs text-slate-400 uppercase font-semibold tracking-wider mb-1">Dernier run</p>
          {stats?.latest_run_id ? (
            <>
              <p className="text-xs font-mono text-white truncate">{stats.latest_run_id}</p>
              <span className={`inline-block mt-1 text-xs font-bold px-2 py-0.5 rounded-full ${
                stats.latest_run_status === 'FINISHED'
                  ? 'bg-emerald-500/15 text-emerald-300'
                  : 'bg-slate-500/15 text-slate-300'
              }`}>
                {stats.latest_run_status}
              </span>
            </>
          ) : (
            <p className="text-slate-500 text-sm">Aucun run enregistré</p>
          )}
        </div>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <StatCard label="Runs MLflow" value={stats?.runs ?? '—'} icon="🏃" color="blue" sub="Entraînements enregistrés" />
        <StatCard label="Métriques" value={stats?.metrics ?? '—'} icon="📏" color="purple" sub="Accuracy, F1, etc." />
        <StatCard label="Traces HTTP" value={stats?.traces ?? '—'} icon="🔍" color="sky" sub="Requêtes observées" />
        <StatCard label="Datasets" value={stats?.datasets ?? '—'} icon="🗄️" color="emerald" sub="Datasets entraînement" />
        <StatCard label="Inputs" value={stats?.inputs ?? '—'} icon="📥" color="amber" sub="Inputs loggés" />
        <StatCard label="Éval. datasets" value={stats?.evaluation_datasets ?? '—'} icon="🧪" color="pink" sub="Datasets d'évaluation" />
      </div>

      {/* Chart */}
      <div className="glass rounded-2xl p-6 mb-8">
        <h2 className="text-lg font-bold text-white mb-6">Vue d'ensemble MLflow</h2>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={chartData} margin={{ top: 0, right: 0, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} axisLine={false} tickLine={false} />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            <Bar dataKey="value" radius={[8, 8, 0, 0]}
              fill="url(#barGrad)" />
            <defs>
              <linearGradient id="barGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6366f1" />
                <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.7} />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Quick actions */}
      <div className="glass rounded-2xl p-6">
        <h2 className="text-lg font-bold text-white mb-4">Accès rapide</h2>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <a href="http://localhost:5000" target="_blank" rel="noreferrer"
            className="flex items-center gap-3 p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-blue-500/30 transition-all group">
            <span className="text-2xl">📈</span>
            <div>
              <p className="font-semibold text-white text-sm group-hover:text-blue-300 transition-colors">MLflow UI</p>
              <p className="text-xs text-slate-500">localhost:5000</p>
            </div>
          </a>
          <a href="/api/docs" target="_blank" rel="noreferrer"
            className="flex items-center gap-3 p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-purple-500/30 transition-all group">
            <span className="text-2xl">📖</span>
            <div>
              <p className="font-semibold text-white text-sm group-hover:text-purple-300 transition-colors">Swagger UI</p>
              <p className="text-xs text-slate-500">Documentation API</p>
            </div>
          </a>
          <a href="/predict"
            className="flex items-center gap-3 p-4 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 hover:border-emerald-500/30 transition-all group">
            <span className="text-2xl">🔬</span>
            <div>
              <p className="font-semibold text-white text-sm group-hover:text-emerald-300 transition-colors">Tester le modèle</p>
              <p className="text-xs text-slate-500">Faire une prédiction</p>
            </div>
          </a>
        </div>
      </div>
    </div>
  )
}
