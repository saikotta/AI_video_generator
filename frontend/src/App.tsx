/**
 * App.tsx — Root component + login→dashboard transition orchestration.
 *
 * Phase machine:
 *   loading      → getMe() in flight
 *   auth         → no session; AuthPage shown
 *   transitioning→ auth succeeded; AuthPage kept alive as overlay while
 *                  DashboardPage mounts underneath and starts fetching data
 *   dashboard    → overlay gone; dashboard is the only thing rendered
 *
 * Transition sequence (Part B of spec):
 *   0ms    auth form fades (lumen-exiting CSS class on body)
 *   0ms    login wrapper scales forward (push-in, 1.35s cubic-bezier(.65,0,.35,1))
 *   1000ms login wrapper opacity → 0 (600ms fade)
 *   1050ms dashboard entrance animation starts
 *   1700ms login overlay removed from DOM
 *
 * Rules:
 *  - Session already exists on load → dashboard directly, no animation.
 *  - Sign out → auth page, no animation.
 *  - Auth fails → error stays in AuthPage, no transition.
 *  - prefers-reduced-motion → simple 200ms fade, no push-in, no entrance.
 *  - Data fetch starts as soon as DashboardPage mounts (950ms before overlay gone).
 */

import { useState, useEffect, useRef } from 'react'
import { getMe } from './api/auth'
import type { UserResponse } from './api/types'
import AuthPage from './pages/AuthPage'
import DashboardPage from './pages/DashboardPage'

type Phase = 'loading' | 'auth' | 'transitioning' | 'dashboard'

function prefersReducedMotion(): boolean {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

export default function App() {
  const [phase, setPhase]             = useState<Phase>('loading')
  const [user,  setUser]              = useState<UserResponse | null>(null)
  const [runEntrance, setRunEntrance] = useState(false)
  const authWrapRef                   = useRef<HTMLDivElement>(null)

  /* ── Session check on mount ── */
  useEffect(() => {
    getMe()
      .then(u  => { setUser(u); setPhase(u ? 'dashboard' : 'auth') })
      .catch(() => setPhase('auth'))
  }, [])

  /* ── Called by AuthPage after login/register succeeds ── */
  function handleAuth(u: UserResponse) {
    const reduced = prefersReducedMotion()

    // Mount DashboardPage (data fetch begins) + flip overlay to transition mode
    setUser(u)
    setPhase('transitioning')

    // Kick off visual exit in next animation frame (DOM must be ready)
    requestAnimationFrame(() => {
      // 1. Apply CSS exit transitions to login elements
      document.body.classList.add('lumen-exiting')

      // 2. Scale the login wrapper forward (camera push-in)
      //    Transform-origin = centre of the cinema screen element.
      //    Target scale fills the viewport edge-to-edge.
      if (!reduced) {
        const wrap     = authWrapRef.current
        const screenEl = document.querySelector<HTMLElement>('.lumen-screen')

        if (wrap && screenEl) {
          const sr    = screenEl.getBoundingClientRect()
          const cx    = sr.left + sr.width  / 2
          const cy    = sr.top  + sr.height / 2
          const scale = (
            Math.max(window.innerWidth / sr.width, window.innerHeight / sr.height) * 1.04
          ).toFixed(4)

          wrap.style.transformOrigin = `${cx}px ${cy}px`

          // Nested rAF: let the browser commit the transformOrigin first
          requestAnimationFrame(() => {
            wrap.style.transition = `transform 1.35s cubic-bezier(.65,0,.35,1)`
            wrap.style.transform  = `scale(${scale})`
          })
        }
      }
    })

    // Timings: full animation vs reduced-motion fast path
    const fadeAt  = reduced ? 200  : 1000   // ms: start fading login overlay
    const clearAt = reduced ? 600  : 1700   // ms: remove overlay from DOM

    // Fade login overlay out; start dashboard entrance
    setTimeout(() => {
      const wrap = authWrapRef.current
      if (wrap) {
        // Append opacity transition (transform transition may already be set)
        const existing = wrap.style.transition
        wrap.style.transition = existing
          ? `${existing}, opacity 0.6s ease`
          : 'opacity 0.6s ease'
        wrap.style.opacity = '0'
      }
      // Entrance animation only when motion is allowed
      setRunEntrance(!reduced)
    }, fadeAt)

    // Remove overlay and clean up
    setTimeout(() => {
      setPhase('dashboard')
      document.body.classList.remove('lumen-exiting')
    }, clearAt)
  }

  /* ── Sign-out: go back to auth with no animation ── */
  function handleLogout() {
    setUser(null)
    setRunEntrance(false)
    setPhase('auth')
    // Clean up any lingering body class (edge case: logout during transition)
    document.body.classList.remove('lumen-exiting')
  }

  /* ── Loading screen ── */
  if (phase === 'loading') {
    return (
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        height: '100vh', background: '#0c0a07',
      }}>
        <AppSpinner size={38} />
      </div>
    )
  }

  /* ── Render ── */
  return (
    <>
      {/*
       * Dashboard — mounted as soon as user is set.
       * Stays mounted through the transition (user remains non-null).
       * Immediately starts fetching projects in the background.
       */}
      {user && (
        <DashboardPage
          user={user}
          onLogout={handleLogout}
          runEntrance={runEntrance}
        />
      )}

      {/*
       * Auth / login overlay.
       * During 'auth' phase: sole visible UI, normal pointer-events.
       * During 'transitioning': sits on top (z 900), pointer-events none,
       *   scale + opacity transitions applied via authWrapRef.
       * The SAME AuthPage instance persists through both phases so the
       * projector is in its post-animation state when the push-in fires.
       */}
      {(phase === 'auth' || phase === 'transitioning') && (
        <div
          ref={authWrapRef}
          aria-hidden={phase === 'transitioning' ? true : undefined}
          style={{
            position:     'fixed',
            inset:         0,
            zIndex:        phase === 'transitioning' ? 900 : 1,
            pointerEvents: phase === 'transitioning' ? 'none' : undefined,
          }}
        >
          {/*
           * onAuth: only wired during 'auth' phase.
           * During 'transitioning' it's a no-op to prevent double-fire.
           */}
          <AuthPage onAuth={phase === 'auth' ? handleAuth : _noop} />
        </div>
      )}
    </>
  )
}

/* eslint-disable-next-line @typescript-eslint/no-empty-function */
function _noop() {}

/* ── App-level loading spinner (dark bg) ── */
function AppSpinner({ size = 24 }: { size?: number }) {
  return (
    <svg
      width={size} height={size} viewBox="0 0 24 24" fill="none"
      style={{ animation: 'spin 0.8s linear infinite' }}
      aria-label="Loading"
      role="img"
    >
      <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
      <circle cx="12" cy="12" r="10" stroke="#3a352f" strokeWidth="3" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="#F8F7F3" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}
