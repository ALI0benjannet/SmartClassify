import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Dashboard from './pages/Dashboard'
import Predict from './pages/Predict'
import Retrain from './pages/Retrain'

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Navbar />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/retrain" element={<Retrain />} />
          </Routes>
        </main>
        <footer className="border-t border-white/8 py-4 px-6 text-center text-xs text-slate-600">
          Obesity MLOps Studio — API sur{' '}
          <a href="http://localhost:8001/docs" target="_blank" rel="noreferrer"
            className="text-slate-500 hover:text-slate-400 transition-colors">
            localhost:8001
          </a>
        </footer>
      </div>
    </BrowserRouter>
  )
}
