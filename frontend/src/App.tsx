import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { getMe } from './api/auth'
import type { UserResponse } from './api/types'
import AuthPage from './pages/AuthPage'
import DashboardPage from './pages/DashboardPage'
import NotFoundPage from './pages/NotFoundPage'
import StoryBoardPage from './pages/StoryBoardPage'

type Phase = 'loading' | 'auth' | 'transitioning' | 'dashboard'

export default function App() {
  const [phase, setPhase] = useState<Phase>('loading')
  const [user, setUser] = useState<UserResponse | null>(null)
  const authWrapRef = useRef<HTMLDivElement>(null)
  const timersRef = useRef<number[]>([])
  const navigate = useNavigate()
  const location = useLocation()

  useEffect(() => {
    let active = true
    getMe().then(currentUser => {
      if (!active) return
      setUser(currentUser)
      setPhase(currentUser ? 'dashboard' : 'auth')
    })
    return () => { active = false }
  }, [])

  useEffect(() => () => {
    timersRef.current.forEach(window.clearTimeout)
    document.body.classList.remove('lumen-exiting')
  }, [])

  const handleAuth = useCallback((authenticatedUser: UserResponse) => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    setUser(authenticatedUser)
    setPhase('transitioning')

    requestAnimationFrame(() => {
      document.body.classList.add('lumen-exiting')
      const wrapper = authWrapRef.current
      const screen = document.querySelector<HTMLElement>('.lumen-screen')
      if (!reducedMotion && wrapper && screen) {
        const bounds = screen.getBoundingClientRect()
        const scale = Math.max(window.innerWidth / bounds.width, window.innerHeight / bounds.height) * 1.04
        wrapper.style.transformOrigin = `${bounds.left + bounds.width / 2}px ${bounds.top + bounds.height / 2}px`
        requestAnimationFrame(() => {
          wrapper.style.transition = 'transform 1.35s cubic-bezier(.65,0,.35,1)'
          wrapper.style.transform = `scale(${scale.toFixed(4)})`
        })
      }
    })

    const fadeDelay = reducedMotion ? 200 : 1000
    const finishDelay = reducedMotion ? 600 : 1700
    timersRef.current.push(window.setTimeout(() => {
      const wrapper = authWrapRef.current
      if (wrapper) {
        wrapper.style.transition = wrapper.style.transition
          ? `${wrapper.style.transition}, opacity 0.6s ease`
          : 'opacity 0.6s ease'
        wrapper.style.opacity = '0'
      }
    }, fadeDelay))
    timersRef.current.push(window.setTimeout(() => {
      document.body.classList.remove('lumen-exiting')
      setPhase('dashboard')
      navigate('/dashboard', { replace: true })
    }, finishDelay))
  }, [navigate])

  const handleLogout = useCallback(() => {
    setUser(null)
    setPhase('auth')
    document.body.classList.remove('lumen-exiting')
    navigate('/login', { replace: true })
  }, [navigate])

  if (phase === 'loading') return <AppSpinner />

  const dashboard = user
    ? <DashboardPage user={user} onLogout={handleLogout} />
    : null

  return (
    <Routes>
      <Route path="/" element={<Navigate to={user ? '/dashboard' : '/login'} replace />} />
      <Route
        path="/login"
        element={user && phase !== 'transitioning'
          ? <Navigate to="/dashboard" replace />
          : (
            <>
              {phase === 'transitioning' && user && (
                <div className="route-transition-underlay" aria-hidden="true">{dashboard}</div>
              )}
              <div
                key="auth-overlay"
                ref={authWrapRef}
                className="route-auth-screen"
                aria-hidden={phase === 'transitioning' ? true : undefined}
                style={{ pointerEvents: phase === 'transitioning' ? 'none' : undefined }}
              >
                <AuthPage onAuth={phase === 'auth' ? handleAuth : noop} />
              </div>
            </>
          )}
      />
      <Route
        path="/dashboard"
        element={user ? dashboard : <Navigate to="/login" replace state={{ from: location }} />}
      />
      <Route
        path="/story-board"
        element={user ? <StoryBoardPage /> : <Navigate to="/login" replace state={{ from: location }} />}
      />
      <Route path="*" element={<NotFoundPage />} />
    </Routes>
  )
}

function AppSpinner() {
  return (
    <div className="route-loading" role="status" aria-label="Loading your studio">
      <span />
    </div>
  )
}

function noop() {}
