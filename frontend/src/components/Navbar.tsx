import { NavLink } from 'react-router-dom'

const links = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/predict', label: 'Prédiction', icon: '🔬' },
  { to: '/retrain', label: 'Réentraînement', icon: '⚙️' },
]

export default function Navbar() {
  return (
    <nav className="sticky top-0 z-50 w-full border-b border-white/10 backdrop-blur-xl"
      style={{ background: 'rgba(7, 16, 31, 0.85)' }}>
      <div className="max-w-7xl mx-auto px-6 flex items-center gap-8 h-16">
        {/* Logo */}
        <div className="flex items-center gap-3 mr-4">
          <div className="w-8 h-8 rounded-xl flex items-center justify-center text-lg"
            style={{ background: 'linear-gradient(135deg, #6366f1, #3b82f6)' }}>
            🫀
          </div>
          <span className="font-extrabold text-white text-lg tracking-tight hidden sm:block">
            Smart<span className="text-blue-400">Classify</span>
          </span>
        </div>

        {/* Nav links */}
        <div className="flex items-center gap-1 flex-1">
          {links.map(({ to, label, icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-150 ${
                  isActive
                    ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-white/8'
                }`
              }
            >
              <span className="text-base">{icon}</span>
              <span className="hidden md:block">{label}</span>
            </NavLink>
          ))}
        </div>

        {/* External links */}
        <div className="flex items-center gap-2">
          <a
            href="http://localhost:5000"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-semibold text-slate-400 hover:text-white px-3 py-2 rounded-xl hover:bg-white/8 transition-all hidden sm:flex items-center gap-1.5"
          >
            <span>📈</span> MLflow
          </a>
          <a
            href="/api/docs"
            target="_blank"
            rel="noreferrer"
            className="text-xs font-semibold text-slate-400 hover:text-white px-3 py-2 rounded-xl hover:bg-white/8 transition-all hidden sm:flex items-center gap-1.5"
          >
            <span>📖</span> Swagger
          </a>
        </div>
      </div>
    </nav>
  )
}
