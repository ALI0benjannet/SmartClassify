import { useState } from 'react'
import { retrain, type RetrainInput, type RetrainResult } from '../api/client'

const DEFAULT_VALUES: RetrainInput = {
  data_path: 'archive (1)/Obesity_Dataset.arff',
  model_path: 'artifacts/obesity_model.joblib',
  test_size: 0.2,
  random_state: 42,
}

export default function Retrain() {
  const [values, setValues] = useState<RetrainInput>(DEFAULT_VALUES)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<RetrainResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await retrain(values)
      setResult(res)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-amber-500/15 text-amber-300 border border-amber-500/20 mb-3">
          ⚙️ Pipeline MLflow
        </span>
        <h1 className="text-4xl font-extrabold text-white">Réentraînement du modèle</h1>
        <p className="text-slate-400 mt-2 text-sm max-w-lg">
          Déclenche un nouveau cycle d'entraînement Random Forest sur les données d'origine.
          Les métriques sont automatiquement loggées dans MLflow.
        </p>
      </div>

      {/* Warning */}
      <div className="glass rounded-2xl p-4 mb-6 border-amber-500/20 bg-amber-500/8">
        <div className="flex items-start gap-3">
          <span className="text-xl mt-0.5">⚠️</span>
          <div className="text-sm text-amber-200/80">
            <p className="font-semibold text-amber-300 mb-0.5">Attention</p>
            Cette action remplace le modèle actuel par un nouveau modèle entraîné.
            Le fichier <code className="text-amber-200 bg-amber-900/30 px-1 rounded">obesity_model.joblib</code> sera écrasé.
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="glass rounded-2xl p-6 mb-5">
          <h2 className="text-base font-bold text-amber-300 mb-4 flex items-center gap-2">
            📂 Chemins des fichiers
          </h2>
          <div className="grid grid-cols-1 gap-4">
            <label className="flex flex-col gap-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                Chemin du dataset
              </span>
              <input
                type="text"
                className="field-input font-mono text-sm"
                value={values.data_path}
                onChange={(e) => setValues({ ...values, data_path: e.target.value })}
                required
              />
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                Chemin du modèle (sortie)
              </span>
              <input
                type="text"
                className="field-input font-mono text-sm"
                value={values.model_path}
                onChange={(e) => setValues({ ...values, model_path: e.target.value })}
                required
              />
            </label>
          </div>
        </div>

        <div className="glass rounded-2xl p-6 mb-5">
          <h2 className="text-base font-bold text-amber-300 mb-4 flex items-center gap-2">
            🎛️ Hyperparamètres
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <label className="flex flex-col gap-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                Taille du jeu de test
              </span>
              <input
                type="number"
                className="field-input"
                value={values.test_size}
                onChange={(e) => setValues({ ...values, test_size: Number(e.target.value) })}
                min={0.05}
                max={0.5}
                step={0.05}
                required
              />
              <span className="text-xs text-slate-500">Entre 0.05 et 0.50</span>
            </label>
            <label className="flex flex-col gap-1.5">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                Random state
              </span>
              <input
                type="number"
                className="field-input"
                value={values.random_state}
                onChange={(e) => setValues({ ...values, random_state: Number(e.target.value) })}
                min={0}
                required
              />
              <span className="text-xs text-slate-500">Reproductibilité</span>
            </label>
          </div>
        </div>

        <div className="flex gap-3">
          <button type="submit" className="btn-gradient flex-1 py-4 text-base" disabled={loading}>
            {loading
              ? <span className="flex items-center justify-center gap-2">
                  <span className="inline-block w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Entraînement en cours…
                </span>
              : '⚙️ Lancer le réentraînement'
            }
          </button>
          <button
            type="button"
            onClick={() => { setValues(DEFAULT_VALUES); setResult(null); setError(null) }}
            className="py-4 px-6 rounded-2xl font-bold text-slate-400 hover:text-white border border-white/10 hover:bg-white/5 transition-all text-sm"
          >
            Réinitialiser
          </button>
        </div>
      </form>

      {error && (
        <div className="glass rounded-2xl p-4 mt-6 border-red-500/30 bg-red-500/10 text-red-300 text-sm">
          ⚠️ Erreur : {error}
        </div>
      )}

      {result && (
        <div className="glass rounded-2xl p-6 mt-6 bg-gradient-to-br from-emerald-500/15 to-emerald-600/5">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-3xl">✅</span>
            <div>
              <h3 className="text-lg font-bold text-white">Entraînement terminé</h3>
              <p className="text-xs text-slate-400 font-mono">{result.model_path}</p>
            </div>
          </div>
          <p className="text-sm text-slate-300 mb-5">{result.message}</p>

          <h4 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-3">Métriques</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {Object.entries(result.metrics).map(([key, val]) => (
              <div key={key} className="bg-white/5 rounded-xl p-3">
                <p className="text-xs text-slate-400 capitalize mb-0.5">
                  {key.replace(/_/g, ' ')}
                </p>
                <p className="text-xl font-extrabold text-white">
                  {typeof val === 'number' ? (val < 1 ? `${(val * 100).toFixed(1)}%` : val.toFixed(3)) : val}
                </p>
              </div>
            ))}
          </div>

          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-2 mt-5 text-sm font-semibold text-blue-300 hover:text-white transition-colors"
          >
            📈 Voir dans MLflow →
          </a>
        </div>
      )}
    </div>
  )
}
