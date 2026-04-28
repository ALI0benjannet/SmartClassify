import { useState } from 'react'
import { predict, type PredictionInput, type PredictionResult } from '../api/client'

// ─── Field metadata ───────────────────────────────────────────────────────────

const FIELDS: {
  key: keyof PredictionInput
  label: string
  type: 'select' | 'number'
  options?: { value: number; label: string }[]
  min?: number
  max?: number
  step?: number
  placeholder?: string
  required?: boolean
  section: string
}[] = [
  // Section : Profil
  {
    key: 'Sex', label: 'Sexe', type: 'select', section: 'Profil personnel',
    options: [{ value: 1, label: 'Femme' }, { value: 2, label: 'Homme' }], required: true,
  },
  {
    key: 'Age', label: 'Âge (ans)', type: 'number', section: 'Profil personnel',
    min: 1, max: 120, placeholder: 'Ex : 25', required: true,
  },
  {
    key: 'Height', label: 'Taille (cm)', type: 'number', section: 'Profil personnel',
    min: 50, max: 250, placeholder: 'Ex : 170', required: true,
  },
  {
    key: 'Weight', label: 'Poids (kg) — optionnel', type: 'number', section: 'Profil personnel',
    min: 1, step: 0.1, placeholder: 'Ex : 70', required: false,
  },
  // Section : Alimentation
  {
    key: 'Overweight_Obese_Family', label: 'Antécédents familiaux d\'obésité', type: 'select',
    section: 'Habitudes alimentaires',
    options: [{ value: 1, label: 'Non' }, { value: 2, label: 'Oui' }], required: true,
  },
  {
    key: 'Consumption_of_Fast_Food', label: 'Consommation de fast food', type: 'select',
    section: 'Habitudes alimentaires',
    options: [
      { value: 1, label: 'Non' },
      { value: 2, label: 'Parfois' },
      { value: 3, label: 'Souvent' },
      { value: 4, label: 'Toujours' },
    ], required: true,
  },
  {
    key: 'Frequency_of_Consuming_Vegetables', label: 'Consommation de légumes', type: 'select',
    section: 'Habitudes alimentaires',
    options: [
      { value: 1, label: 'Jamais' },
      { value: 2, label: 'Parfois' },
      { value: 3, label: 'Toujours' },
    ], required: true,
  },
  {
    key: 'Number_of_Main_Meals_Daily', label: 'Nombre de repas principaux / jour', type: 'select',
    section: 'Habitudes alimentaires',
    options: [
      { value: 1, label: '1 repas' },
      { value: 2, label: '2 repas' },
      { value: 3, label: '3 repas' },
      { value: 4, label: '4 repas ou plus' },
    ], required: true,
  },
  {
    key: 'Food_Intake_Between_Meals', label: 'Grignotage entre les repas', type: 'select',
    section: 'Habitudes alimentaires',
    options: [
      { value: 1, label: 'Non' },
      { value: 2, label: 'Parfois' },
      { value: 3, label: 'Souvent' },
      { value: 4, label: 'Toujours' },
    ], required: true,
  },
  {
    key: 'Calculation_of_Calorie_Intake', label: 'Calcul des calories', type: 'select',
    section: 'Habitudes alimentaires',
    options: [{ value: 1, label: 'Non' }, { value: 2, label: 'Oui' }], required: true,
  },
  // Section : Mode de vie
  {
    key: 'Smoking', label: 'Tabagisme', type: 'select', section: 'Mode de vie',
    options: [{ value: 1, label: 'Non' }, { value: 2, label: 'Oui' }], required: true,
  },
  {
    key: 'Liquid_Intake_Daily', label: 'Hydratation quotidienne', type: 'select',
    section: 'Mode de vie',
    options: [
      { value: 1, label: 'Moins de 1 L' },
      { value: 2, label: 'Entre 1 L et 2 L' },
      { value: 3, label: 'Plus de 2 L' },
    ], required: true,
  },
  {
    key: 'Physical_Excercise', label: 'Activité physique / semaine', type: 'select',
    section: 'Mode de vie',
    options: [
      { value: 1, label: 'Aucune' },
      { value: 2, label: '1 – 2 jours' },
      { value: 3, label: '2 – 4 jours' },
      { value: 4, label: '4 – 5 jours' },
    ], required: true,
  },
  {
    key: 'Schedule_Dedicated_to_Technology', label: 'Temps écran / jour', type: 'select',
    section: 'Mode de vie',
    options: [
      { value: 1, label: '0 – 2 heures' },
      { value: 2, label: '3 – 5 heures' },
      { value: 3, label: 'Plus de 5 heures' },
    ], required: true,
  },
  {
    key: 'Type_of_Transportation_Used', label: 'Transport principal', type: 'select',
    section: 'Mode de vie',
    options: [
      { value: 1, label: 'Voiture' },
      { value: 2, label: 'Moto' },
      { value: 3, label: 'Vélo' },
      { value: 4, label: 'Transport en commun' },
      { value: 5, label: 'À pied' },
    ], required: true,
  },
]

const SECTIONS = [...new Set(FIELDS.map((f) => f.section))]

// ─── Result card ──────────────────────────────────────────────────────────────

const CLASS_COLORS: Record<number, { badge: string; bar: string; bg: string }> = {
  1: { badge: 'badge-blue', bar: 'bg-blue-400', bg: 'from-blue-500/20 to-blue-600/5' },
  2: { badge: 'badge-green', bar: 'bg-emerald-400', bg: 'from-emerald-500/20 to-emerald-600/5' },
  3: { badge: 'badge-yellow', bar: 'bg-yellow-400', bg: 'from-yellow-500/20 to-yellow-600/5' },
  4: { badge: 'badge-red', bar: 'bg-red-400', bg: 'from-red-500/20 to-red-600/5' },
}

const CLASS_ICONS: Record<number, string> = { 1: '🪶', 2: '✅', 3: '⚠️', 4: '🚨' }

function ResultCard({ result }: { result: PredictionResult }) {
  const colors = CLASS_COLORS[result.predicted_class] ?? CLASS_COLORS[2]
  const confidence = Math.round(result.confidence * 100)

  return (
    <div className={`glass rounded-2xl p-6 bg-gradient-to-br ${colors.bg} mt-8`}>
      <h3 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-4">
        Résultat de la prédiction
      </h3>

      <div className="flex flex-col sm:flex-row sm:items-center gap-4 mb-6">
        <span className="text-5xl">{CLASS_ICONS[result.predicted_class]}</span>
        <div>
          <p className="text-2xl font-extrabold text-white">{result.predicted_label}</p>
          <p className="text-sm text-slate-400 mt-0.5">
            Source : <span className="text-slate-300 font-semibold">{result.decision_source}</span>
          </p>
          {result.reason && (
            <p className="text-xs text-slate-500 mt-1 italic">{result.reason}</p>
          )}
        </div>
      </div>

      {/* Confidence bar */}
      <div className="mb-6">
        <div className="flex justify-between text-xs text-slate-400 mb-1.5">
          <span>Confiance du modèle</span>
          <span className="font-bold text-white">{confidence}%</span>
        </div>
        <div className="h-2.5 rounded-full bg-white/10">
          <div
            className={`h-2.5 rounded-full ${colors.bar} transition-all duration-700`}
            style={{ width: `${confidence}%` }}
          />
        </div>
      </div>

      {/* Details grid */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-white/5 rounded-xl p-3">
          <p className="text-slate-400 text-xs mb-0.5">Classe prédite</p>
          <p className="font-bold text-white">{result.predicted_class} — {result.predicted_label}</p>
        </div>
        <div className="bg-white/5 rounded-xl p-3">
          <p className="text-slate-400 text-xs mb-0.5">Modèle RF seul</p>
          <p className="font-bold text-white">{result.model_predicted_class} — {result.model_predicted_label}</p>
        </div>
        {result.bmi != null && (
          <div className="bg-white/5 rounded-xl p-3">
            <p className="text-slate-400 text-xs mb-0.5">IMC calculé</p>
            <p className="font-bold text-white">{result.bmi.toFixed(1)} kg/m²</p>
          </div>
        )}
        {result.bmi_class != null && (
          <div className="bg-white/5 rounded-xl p-3">
            <p className="text-slate-400 text-xs mb-0.5">Classe IMC</p>
            <p className="font-bold text-white">{result.bmi_class}</p>
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

type FormValues = Partial<Record<keyof PredictionInput, string>>

export default function Predict() {
  const [values, setValues] = useState<FormValues>({})
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  const set = (key: keyof PredictionInput, val: string) =>
    setValues((v) => ({ ...v, [key]: val }))

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const payload: PredictionInput = {
        Sex: Number(values.Sex),
        Age: Number(values.Age),
        Height: Number(values.Height),
        Weight: values.Weight ? Number(values.Weight) : undefined,
        Overweight_Obese_Family: Number(values.Overweight_Obese_Family),
        Consumption_of_Fast_Food: Number(values.Consumption_of_Fast_Food),
        Frequency_of_Consuming_Vegetables: Number(values.Frequency_of_Consuming_Vegetables),
        Number_of_Main_Meals_Daily: Number(values.Number_of_Main_Meals_Daily),
        Food_Intake_Between_Meals: Number(values.Food_Intake_Between_Meals),
        Smoking: Number(values.Smoking),
        Liquid_Intake_Daily: Number(values.Liquid_Intake_Daily),
        Calculation_of_Calorie_Intake: Number(values.Calculation_of_Calorie_Intake),
        Physical_Excercise: Number(values.Physical_Excercise),
        Schedule_Dedicated_to_Technology: Number(values.Schedule_Dedicated_to_Technology),
        Type_of_Transportation_Used: Number(values.Type_of_Transportation_Used),
      }
      const res = await predict(payload)
      setResult(res)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const fillExample = () => {
    setValues({
      Sex: '2', Age: '21', Height: '170', Weight: '70',
      Overweight_Obese_Family: '2', Consumption_of_Fast_Food: '2',
      Frequency_of_Consuming_Vegetables: '3', Number_of_Main_Meals_Daily: '2',
      Food_Intake_Between_Meals: '2', Smoking: '2', Liquid_Intake_Daily: '2',
      Calculation_of_Calorie_Intake: '2', Physical_Excercise: '3',
      Schedule_Dedicated_to_Technology: '3', Type_of_Transportation_Used: '4',
    })
  }

  return (
    <div className="max-w-5xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-bold bg-purple-500/15 text-purple-300 border border-purple-500/20 mb-3">
          🔬 Modèle Random Forest
        </span>
        <h1 className="text-4xl font-extrabold text-white">Prédiction d'obésité</h1>
        <p className="text-slate-400 mt-2 text-sm max-w-xl">
          Remplissez le formulaire ci-dessous. Le modèle prédit la classe de corpulence à partir de vos données
          alimentaires et comportementales.
        </p>
      </div>

      <form onSubmit={handleSubmit}>
        {SECTIONS.map((section) => {
          const sectionFields = FIELDS.filter((f) => f.section === section)
          return (
            <div key={section} className="glass rounded-2xl p-6 mb-5">
              <h2 className="text-base font-bold text-blue-300 mb-4 flex items-center gap-2">
                {section === 'Profil personnel' && '👤'}
                {section === 'Habitudes alimentaires' && '🥗'}
                {section === 'Mode de vie' && '🏃'}
                {section}
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {sectionFields.map((field) => (
                  <label key={field.key} className="flex flex-col gap-1.5">
                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wide">
                      {field.label}
                      {field.required && <span className="text-red-400 ml-1">*</span>}
                    </span>
                    {field.type === 'select' ? (
                      <select
                        className="field-input"
                        value={values[field.key] ?? ''}
                        onChange={(e) => set(field.key, e.target.value)}
                        required={field.required}
                      >
                        <option value="" disabled>— Choisir —</option>
                        {field.options!.map((o) => (
                          <option key={o.value} value={o.value}>{o.label}</option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        className="field-input"
                        value={values[field.key] ?? ''}
                        onChange={(e) => set(field.key, e.target.value)}
                        min={field.min}
                        max={field.max}
                        step={field.step ?? 1}
                        placeholder={field.placeholder}
                        required={field.required}
                      />
                    )}
                  </label>
                ))}
              </div>
            </div>
          )
        })}

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-3 mt-2">
          <button type="submit" className="btn-gradient flex-1 py-4 text-base" disabled={loading}>
            {loading ? '⟳ Analyse en cours…' : '🔬 Lancer la prédiction'}
          </button>
          <button
            type="button"
            onClick={fillExample}
            className="py-4 px-6 rounded-2xl font-bold text-slate-300 hover:text-white border border-white/15 hover:bg-white/8 transition-all text-sm"
          >
            Remplir un exemple
          </button>
          <button
            type="button"
            onClick={() => { setValues({}); setResult(null); setError(null) }}
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

      {result && <ResultCard result={result} />}
    </div>
  )
}
