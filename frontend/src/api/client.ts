// ─── Types ───────────────────────────────────────────────────────────────────

export interface PredictionInput {
  Sex: number
  Age: number
  Height: number
  Weight?: number
  Overweight_Obese_Family: number
  Consumption_of_Fast_Food: number
  Frequency_of_Consuming_Vegetables: number
  Number_of_Main_Meals_Daily: number
  Food_Intake_Between_Meals: number
  Smoking: number
  Liquid_Intake_Daily: number
  Calculation_of_Calorie_Intake: number
  Physical_Excercise: number
  Schedule_Dedicated_to_Technology: number
  Type_of_Transportation_Used: number
}

export interface PredictionResult {
  predicted_class: number
  predicted_label: string
  confidence: number
  model_predicted_class: number
  model_predicted_label: string
  bmi?: number
  bmi_class?: number
  decision_source: string
  reason?: string
}

export interface Stats {
  api_status: string
  mlflow_status: string
  latest_run_id: string | null
  latest_run_status: string | null
  runs: number
  metrics: number
  datasets: number
  inputs: number
  evaluation_datasets: number
  traces: number
}

export interface RetrainInput {
  data_path: string
  model_path: string
  test_size: number
  random_state: number
}

export interface RetrainResult {
  message: string
  model_path: string
  metrics: Record<string, number>
}

// ─── Base URL ────────────────────────────────────────────────────────────────

const BASE = '/api'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} — ${text}`)
  }
  return res.json() as Promise<T>
}

// ─── Endpoints ───────────────────────────────────────────────────────────────

export const getStats = (): Promise<Stats> => request<Stats>('/stats')

export const predict = (data: PredictionInput): Promise<PredictionResult> =>
  request<PredictionResult>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })

export const retrain = (data: RetrainInput): Promise<RetrainResult> =>
  request<RetrainResult>('/retrain', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })

export interface TrainingStatus {
  status: 'idle' | 'training' | 'done' | 'error'
  ingested_count: number
  last_trained_at: string | null
  trigger_count: number | null
  metrics: Record<string, number> | null
  error: string | null
  auto_retrain_every: number
}

export const getTrainingStatus = (): Promise<TrainingStatus> =>
  request<TrainingStatus>('/training-status')
