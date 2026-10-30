/**
 * AuthPage.tsx — Cinematic Lumen login page
 *
 * Layout (top → bottom, vertically centred):
 *   • White cinema screen  ← login / register card
 *   • Warm light beam (trapezoid)
 *   • Film projector (SVG)
 *
 * The 10-second intro shot zooms in on the projector, ignites the lamp,
 * then pulls back to reveal the full card.  Form elements fly from the
 * lens to their final positions via the Web Animations API.
 *
 * All real auth logic from the original component is preserved.
 */

import { useState, useEffect, useRef } from 'react'
import { login, register } from '../api/auth'
import type { UserResponse } from '../api/types'
import { ApiError } from '../api/client'

/* ─────────────────────────────────────────────────────────────
   Types
───────────────────────────────────────────────────────────── */
interface Props {
  onAuth: (user: UserResponse) => void
}

/* ─────────────────────────────────────────────────────────────
   Component
───────────────────────────────────────────────────────────── */
export default function AuthPage({ onAuth }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail]       = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw]     = useState(false)
  const [error, setError]       = useState('')
  const [loading, setLoading]   = useState(false)
  const [status, setStatus]     = useState('')

  const camRef      = useRef<HTMLDivElement>(null)
  const lensRef     = useRef<HTMLDivElement>(null)
  const projSvgRef  = useRef<SVGSVGElement>(null)

  /* ── Auth handler ── */
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setError('')
    setStatus('')
    if (!email.trim()) { setError('Please enter your email address.'); return }
    if (!password)     { setError('Please enter your password.');      return }
    setLoading(true)
    try {
      if (mode === 'login') {
        const user = await login(email, password)
        onAuth(user)
      } else {
        await register(email, password)
        const user = await login(email, password)
        onAuth(user)
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong.')
    } finally {
      setLoading(false)
    }
  }

  function switchMode() {
    setMode(m => m === 'login' ? 'register' : 'login')
    setError('')
    setStatus('')
  }

  /* ── Cinematic intro (runs once on mount) ── */
  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const cam     = camRef.current
    const lens    = lensRef.current
    const projSvg = projSvgRef.current
    if (!cam || !lens || !projSvg) return

    /* Spawn glowing letter motes inside the beam */
    const beamWrap = document.getElementById('lumen-beam-wrap')
    if (beamWrap) {
      const CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
      for (let i = 0; i < 34; i++) {
        const m  = document.createElement('span')
        m.className = 'lumen-mote'
        m.textContent = CHARS[Math.floor(Math.random() * 26)]
        m.style.setProperty('--x0',  (44 + Math.random() * 12) + '%')
        m.style.setProperty('--x1',  (8  + Math.random() * 84) + '%')
        m.style.setProperty('--r0',  (Math.random() * 50  - 25)  + 'deg')
        m.style.setProperty('--r1',  (Math.random() * 220 - 110) + 'deg')
        m.style.setProperty('--sz',  (12 + Math.random() * 18)   + 'px')
        m.style.setProperty('--dur', (5  + Math.random() * 5)    + 's')
        m.style.setProperty('--del', (3.8 + Math.random() * 5.5) + 's')
        beamWrap.appendChild(m)
      }
    }

    if (reduced) return   /* skip animation, CSS shows finished state */

    /* ── Measure positions at rest (camera animation paused at end) ── */
    cam.style.animationPlayState = 'paused'
    cam.style.animationDelay     = '-9999s'

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        const projR = projSvg.getBoundingClientRect()
        const camR  = cam.getBoundingClientRect()

        const projCY = projR.top  + projR.height / 2
        const oyPct  = ((projCY - camR.top) / camR.height * 100).toFixed(2) + '%'
        const tyPx   = (window.innerHeight / 2 - projCY).toFixed(1) + 'px'
        const s      = Math.max(1.8, Math.min(3.6, window.innerWidth * 0.70 / projR.width)).toFixed(3)

        cam.style.setProperty('--oy', oyPct)
        cam.style.setProperty('--ty', tyPx)
        cam.style.setProperty('--s',  s)

        /* Restore camera animation */
        cam.style.animationPlayState = ''
        cam.style.animationDelay     = '0s'

        /* ── Fly-in form elements ── */
        if (!Element.prototype.animate) return
        /* Measure lens position (still at rest) */
        cam.style.animationPlayState = 'paused'
        cam.style.animationDelay     = '-9999s'

        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            const lR  = lens.getBoundingClientRect()
            const lx  = lR.left + lR.width  / 2
            const ly  = lR.top  + lR.height / 2

            const flyEls = document.querySelectorAll<HTMLElement>('.lumen-fly')
            const f0 = 'invert(1) blur(8px) drop-shadow(0 0 12px rgba(255,236,180,.95))'
            const f1 = 'invert(1) blur(2px) drop-shadow(0 0 8px rgba(255,230,160,.70))'
            const f2 = 'invert(0) blur(0px) drop-shadow(0 0 0 rgba(0,0,0,0))'

            flyEls.forEach((el, idx) => {
              const r   = el.getBoundingClientRect()
              const dx  = lx - (r.left + r.width  / 2)
              const dy  = ly - (r.top  + r.height / 2)
              const rot = (idx % 2 === 0 ? 1 : -1) * (5 + idx * 1.8)

              el.animate(
                [
                  { transform: `translate(${dx}px,${dy}px) scale(.05) rotate(${rot}deg)`,    opacity: 0, filter: f0 },
                  { transform: `translate(${dx*.9}px,${dy*.9}px) scale(.07) rotate(${rot*.8}deg)`, opacity: 1, filter: f0, offset: 0.08 },
                  { transform: `translate(${dx*.28}px,${dy*.28}px) scale(.78) rotate(${rot/3}deg)`, filter: f1, opacity: 1, offset: 0.60 },
                  { transform: 'translate(0,0) scale(1) rotate(0deg)', opacity: 1, filter: f2 },
                ],
                {
                  duration: 1700,
                  delay:    6700 + idx * 170,
                  easing:   'cubic-bezier(.22,.78,.24,1)',
                  fill:     'both',
                }
              )
            })

            /* Restore camera */
            cam.style.animationPlayState = ''
            cam.style.animationDelay     = '0s'

            /* Hard-reveal safety net */
            setTimeout(() => {
              document.querySelectorAll<HTMLElement>('.lumen-fly').forEach(el => {
                if (parseFloat(window.getComputedStyle(el).opacity) < 0.5) {
                  el.style.opacity   = '1'
                  el.style.filter    = 'none'
                  el.style.transform = 'none'
                }
              })
            }, 12500)
          })
        })
      })
    })

    /* Outer hard-reveal if launchElements failed entirely */
    const outerFallback = setTimeout(() => {
      document.querySelectorAll<HTMLElement>('.lumen-fly').forEach(el => {
        el.style.opacity   = '1'
        el.style.filter    = 'none'
        el.style.transform = 'none'
      })
    }, 12000)

    return () => clearTimeout(outerFallback)
  }, [])   // run once

  /* ─────────────────────────────────────────────────────────
     Render
  ───────────────────────────────────────────────────────── */
  return (
    <>
      {/* ── injected styles (scoped to lumen- prefix) ── */}
      <style>{LUMEN_CSS}</style>

      <div className="lumen-room" aria-hidden="true" />

      <button
        className="lumen-replay"
        type="button"
        aria-label="Replay intro animation"
        onClick={() => window.location.reload()}
      >
        ↺ Replay intro
      </button>

      <div className="lumen-vp">
        <div className="lumen-cam" id="lumen-cam" ref={camRef}>
          <main className="lumen-stage">

            {/* ════ SCREEN / CARD ════ */}
            <section className="lumen-screen" aria-label={mode === 'login' ? 'Sign in to Lumen' : 'Create your Lumen account'}>

              {/* Brand */}
              <div className="lumen-brand lumen-fly">
                <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true" focusable={false}>
                  <circle cx="13" cy="13" r="11.5" fill="none" stroke="#1b1814" strokeWidth="1.7"/>
                  <polygon points="10.5,8.5 10.5,17.5 19,13" fill="#1b1814"/>
                </svg>
                Lumen
              </div>

              {/* Heading */}
              <h1 className="lumen-fly">
                {mode === 'login' ? 'Welcome back' : 'Create account'}
              </h1>

              {/* Subtext */}
              <p className="lumen-sub lumen-fly">
                {mode === 'login'
                  ? 'Sign in to keep turning ideas into video.'
                  : 'Join Lumen and start generating AI-powered video.'}
              </p>

              <form onSubmit={handleSubmit} noValidate>
                {/* Email */}
                <div className="lumen-field lumen-fly">
                  <label htmlFor="lumen-email">Email address</label>
                  <input
                    id="lumen-email"
                    type="email"
                    value={email}
                    onChange={e => setEmail(e.target.value)}
                    placeholder="you@studio.com"
                    autoComplete="email"
                    spellCheck={false}
                    autoCapitalize="none"
                    required
                  />
                </div>

                {/* Password */}
                <div className="lumen-field lumen-fly">
                  <label htmlFor="lumen-pw">Password</label>
                  <div className="lumen-pw-wrap">
                    <input
                      id="lumen-pw"
                      type={showPw ? 'text' : 'password'}
                      value={password}
                      onChange={e => setPassword(e.target.value)}
                      placeholder="Your password"
                      autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
                      minLength={mode === 'register' ? 8 : undefined}
                      required
                    />
                    <button
                      type="button"
                      className="lumen-toggle"
                      aria-label={showPw ? 'Hide password' : 'Show password'}
                      aria-pressed={showPw}
                      onClick={() => setShowPw(v => !v)}
                    >
                      {showPw ? 'Hide' : 'Show'}
                    </button>
                  </div>
                </div>

                {/* Forgot (login only) */}
                {mode === 'login' && (
                  <div className="lumen-row-end lumen-fly">
                    <a href="#">Forgot password?</a>
                  </div>
                )}

                {/* Error */}
                {error && (
                  <p role="alert" className="lumen-error lumen-fly">{error}</p>
                )}

                {/* Submit */}
                <div className="lumen-fly">
                  <button
                    type="submit"
                    className="lumen-btn lumen-btn-primary"
                    disabled={loading}
                    aria-busy={loading}
                  >
                    {loading
                      ? (mode === 'login' ? 'Signing in…' : 'Creating account…')
                      : (mode === 'login' ? 'Sign in' : 'Create account')}
                  </button>
                </div>
              </form>

              {/* Divider */}
              <div className="lumen-or lumen-fly" aria-hidden="true">or</div>

              {/* Google */}
              <div className="lumen-fly">
                <button type="button" className="lumen-btn lumen-btn-ghost" aria-label="Continue with Google">
                  <GoogleIcon />
                  Continue with Google
                </button>
              </div>

              {/* Switch mode */}
              <p className="lumen-foot lumen-fly">
                {mode === 'login' ? 'New to Lumen?' : 'Already have an account?'}{' '}
                <button type="button" className="lumen-link-btn" onClick={switchMode}>
                  {mode === 'login' ? 'Create an account' : 'Sign in'}
                </button>
              </p>

              {/* Status */}
              {status && (
                <p className="lumen-status" role="status" aria-live="polite">{status}</p>
              )}

            </section>
            {/* end screen */}

            {/* ════ BEAM ════ */}
            <div className="lumen-beam-wrap" id="lumen-beam-wrap" aria-hidden="true">
              <div className="lumen-beam" />
              <div className="lumen-beam-core" />
            </div>

            {/* ════ PROJECTOR ════ */}
            <div className="lumen-projw" aria-hidden="true" role="img" aria-label="Film projector">
              <div className="lumen-lensglow" ref={lensRef} />

              <svg
                className="lumen-proj"
                ref={projSvgRef}
                viewBox="0 0 260 152"
                xmlns="http://www.w3.org/2000/svg"
              >
                <defs>
                  <linearGradient id="lg-body" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0"   stopColor="#3d3832"/>
                    <stop offset=".45" stopColor="#2a2620"/>
                    <stop offset="1"   stopColor="#14120f"/>
                  </linearGradient>
                  <linearGradient id="lg-lens" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0"   stopColor="#0f0e0c"/>
                    <stop offset=".4"  stopColor="#534d44"/>
                    <stop offset=".62" stopColor="#6a6358"/>
                    <stop offset="1"   stopColor="#0f0e0c"/>
                  </linearGradient>
                  <radialGradient id="lg-reel" cx="50%" cy="50%" r="50%">
                    <stop offset="0" stopColor="#3a3630"/>
                    <stop offset="1" stopColor="#1a1815"/>
                  </radialGradient>
                </defs>

                {/* Film strips */}
                <path d="M72 82 Q90 78 110 70"  stroke="#1a1710" strokeWidth="4" fill="none" strokeLinecap="round"/>
                <path d="M188 82 Q170 78 150 70" stroke="#1a1710" strokeWidth="4" fill="none" strokeLinecap="round"/>

                {/* Reels */}
                <g className="lumen-reel">
                  <circle cx="72" cy="48" r="34" fill="url(#lg-reel)" stroke="#3a3530" strokeWidth="1.5"/>
                  <circle cx="72" cy="48" r="10" fill="#141210" stroke="#2e2a24" strokeWidth="1"/>
                  <line x1="72" y1="16" x2="72" y2="80" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="40" y1="48" x2="104" y2="48" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="50" y1="24" x2="94" y2="72" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="94" y1="24" x2="50" y2="72" stroke="#2a2620" strokeWidth="2"/>
                  <circle cx="72" cy="48" r="34" fill="none" stroke="#4a4540" strokeWidth="1"/>
                </g>
                <g className="lumen-reel">
                  <circle cx="188" cy="48" r="34" fill="url(#lg-reel)" stroke="#3a3530" strokeWidth="1.5"/>
                  <circle cx="188" cy="48" r="10" fill="#141210" stroke="#2e2a24" strokeWidth="1"/>
                  <line x1="188" y1="16" x2="188" y2="80" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="156" y1="48" x2="220" y2="48" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="166" y1="24" x2="210" y2="72" stroke="#2a2620" strokeWidth="2"/>
                  <line x1="210" y1="24" x2="166" y2="72" stroke="#2a2620" strokeWidth="2"/>
                  <circle cx="188" cy="48" r="34" fill="none" stroke="#4a4540" strokeWidth="1"/>
                </g>

                {/* Lens barrel */}
                <rect x="110" y="4" width="40" height="72" rx="4" fill="url(#lg-lens)"/>
                <rect x="105" y="2" width="50" height="11" rx="4" fill="#2a2722" stroke="#5a544b"/>
                <ellipse cx="130" cy="3.5" rx="20" ry="4" fill="#fff4dc"/>
                <rect x="112" y="16" width="36" height="8" rx="2" fill="#1e1c18" stroke="#3a3530"/>
                <rect x="112" y="30" width="36" height="8" rx="2" fill="#1e1c18" stroke="#3a3530"/>
                <rect x="112" y="44" width="36" height="8" rx="2" fill="#1e1c18" stroke="#3a3530"/>
                <ellipse cx="130" cy="68" rx="15" ry="7" fill="#0a0908" stroke="#3a3530"/>
                <ellipse cx="130" cy="68" rx="9"  ry="4.5" fill="#fff8e0" opacity=".18"/>

                {/* Body */}
                <rect x="22" y="74" width="216" height="68" rx="11" fill="url(#lg-body)" stroke="#4a443c" strokeWidth="1.5"/>
                <path d="M30 74 h200" stroke="#5a5448" strokeWidth="1" opacity=".5"/>

                {/* Vents */}
                <g stroke="#0d0c0a" strokeWidth="2.5" strokeLinecap="round">
                  <line x1="158" y1="90"  x2="222" y2="90"/>
                  <line x1="158" y1="100" x2="222" y2="100"/>
                  <line x1="158" y1="110" x2="222" y2="110"/>
                  <line x1="158" y1="120" x2="222" y2="120"/>
                </g>

                {/* Film gate slot */}
                <rect x="122" y="90" width="26" height="7" rx="2" fill="#6d655a"/>

                {/* Pilot light */}
                <circle className="lumen-pilot" cx="50" cy="100" r="5" fill="#ff9d42"/>
                <circle cx="50" cy="100" r="8" fill="none" stroke="#3a2a10" strokeWidth="1.5"/>

                {/* Rocker switch housing */}
                <rect x="84" y="88" width="30" height="26" rx="5" fill="#0d0c0a" stroke="#4a443c"/>
                {/* Lever */}
                <g className="lumen-lever">
                  <rect x="95" y="82" width="8" height="26" rx="4" fill="#d4cabb" stroke="#8a8272" strokeWidth="1"/>
                  <line x1="98" y1="88"  x2="100" y2="88"  stroke="#a09890" strokeWidth="1"/>
                  <line x1="98" y1="94"  x2="100" y2="94"  stroke="#a09890" strokeWidth="1"/>
                  <line x1="98" y1="100" x2="100" y2="100" stroke="#a09890" strokeWidth="1"/>
                </g>

                {/* Label plate */}
                <rect x="36" y="122" width="38" height="16" rx="2" fill="#0d0c0a" stroke="#2a2520" strokeWidth=".8"/>
                <text x="55" y="134" fontFamily="monospace" fontSize="5" fill="#5a5448" textAnchor="middle">LMN-16</text>

                {/* Feet */}
                <rect x="44"  y="140" width="30" height="9" rx="3" fill="#0e0d0b" stroke="#2a2520" strokeWidth=".8"/>
                <rect x="186" y="140" width="30" height="9" rx="3" fill="#0e0d0b" stroke="#2a2520" strokeWidth=".8"/>
              </svg>
            </div>
            {/* end projector */}

          </main>
        </div>
      </div>
    </>
  )
}

/* ─────────────────────────────────────────────────────────────
   Helper components
───────────────────────────────────────────────────────────── */
function GoogleIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true" focusable={false}>
      <path d="M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z" fill="#4285F4"/>
      <path d="M9 18c2.43 0 4.467-.806 5.956-2.18l-2.909-2.259c-.806.54-1.837.86-3.047.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z" fill="#34A853"/>
      <path d="M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z" fill="#FBBC05"/>
      <path d="M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 6.29C4.672 4.163 6.656 3.58 9 3.58z" fill="#EA4335"/>
    </svg>
  )
}

/* ─────────────────────────────────────────────────────────────
   All Lumen CSS (injected via <style> so it doesn't bleed into
   the rest of the app and doesn't require a separate .css file)
───────────────────────────────────────────────────────────── */
const LUMEN_CSS = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Instrument+Serif&display=swap');

/* ── reset for lumen scope ── */
.lumen-vp, .lumen-vp * { box-sizing: border-box; }

/* ── full-page black backdrop ── */
.lumen-vp {
  position: fixed;
  inset: 0;
  overflow: hidden;
  background: #000;
  z-index: 0;
}

/* ── ambient wall glow ── */
.lumen-room {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 1;
  background: radial-gradient(ellipse 60% 55% at 50% 38%,
    rgba(255,235,190,.10), transparent 70%);
  animation: l-fadeIn 1.8s ease 4.2s both;
}

/* ── replay button ── */
.lumen-replay {
  position: fixed;
  z-index: 200;
  right: max(14px, env(safe-area-inset-right, 14px));
  top: calc(env(safe-area-inset-top, 0px) + 12px);
  background: rgba(18,15,12,.72);
  border: 1px solid #2e2820;
  color: #8a7f72;
  font-family: "DM Sans", system-ui, sans-serif;
  font-size: .76rem;
  border-radius: 999px;
  padding: .38rem .85rem;
  cursor: pointer;
  backdrop-filter: blur(6px);
  -webkit-backdrop-filter: blur(6px);
  animation: l-fadeIn 1s ease 11s both;
  transition: color .18s, border-color .18s;
}
.lumen-replay:hover { color: #ffeec5; border-color: #5a5048; }

/* ── camera wrapper ── */
.lumen-cam {
  position: relative;
  min-height: 100dvh;
  transform-origin: 50% var(--oy, 90%);
  will-change: transform;
  animation: l-camShot 7.8s cubic-bezier(.25,.1,.25,1) 0s both;
}

/* ── stage column ── */
.lumen-stage {
  position: relative;
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding: clamp(12px,3.5vh,32px) 0 clamp(20px,4vh,48px);
  font-family: "DM Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* ─────────────────────────────────────────────────────
   SCREEN
───────────────────────────────────────────────────── */
.lumen-screen {
  position: relative;
  z-index: 10;
  width: min(92vw, 440px);
  background: radial-gradient(ellipse 80% 60% at 50% 38%,
    #ffffff 0%, #fffdf8 52%, #ede5d5 100%);
  border-radius: 4px;
  padding: clamp(22px,4.5vh,44px) clamp(20px,6vw,42px);
  box-shadow:
    0 0 0 7px #0b0a08,
    0 0 0 9px #22201c,
    0 0 0 10px #0b0a08,
    0 0 80px 14px rgba(255,240,205,.28),
    inset 0 0 80px rgba(140,110,50,.12);
  animation: l-screenOn 1.9s linear 4.1s both;
}

/* Film grain */
.lumen-screen::before {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.88' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.85 0'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)'/%3E%3C/svg%3E");
  opacity: .09;
  mix-blend-mode: multiply;
  animation: l-grain .65s steps(1) infinite;
}

/* Occasional flicker */
.lumen-screen::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  background: #fff;
  opacity: 0;
  animation: l-flicker 6s steps(1) 11s infinite;
}

/* ─────────────────────────────────────────────────────
   FORM ELEMENTS
───────────────────────────────────────────────────── */
.lumen-brand {
  display: flex;
  align-items: center;
  gap: .55rem;
  font-weight: 600;
  font-size: 1rem;
  letter-spacing: .01em;
  color: #1b1814;
}

.lumen-stage h1 {
  font-family: "Instrument Serif", Georgia, serif;
  font-weight: 400;
  font-size: clamp(1.9rem, 7.5vw, 2.7rem);
  line-height: 1.08;
  margin: clamp(14px,3vh,26px) 0 .4rem;
  color: #1b1814;
}

.lumen-sub {
  margin: 0 0 clamp(14px,3.5vh,26px);
  color: #6d655a;
  font-size: .95rem;
  line-height: 1.5;
}

.lumen-field {
  margin-bottom: .9rem;
  position: relative;
}

.lumen-field label {
  display: block;
  font-size: .82rem;
  font-weight: 500;
  color: #1b1814;
  margin-bottom: .38rem;
}

.lumen-field input {
  display: block;
  width: 100%;
  font-family: "DM Sans", system-ui, sans-serif;
  font-size: .95rem;
  color: #1b1814;
  background: #fff;
  border: 1.5px solid #d9d2c4;
  border-radius: 7px;
  padding: .76rem .88rem;
  transition: border-color .18s, box-shadow .18s;
  -webkit-appearance: none;
}
.lumen-field input::placeholder { color: #b5ac9e; }
.lumen-field input:focus {
  outline: none;
  border-color: #7a7060;
  box-shadow: 0 0 0 3px rgba(110,96,76,.16);
}

.lumen-pw-wrap { position: relative; }
.lumen-pw-wrap input { padding-right: 4.6rem; }

.lumen-toggle {
  position: absolute;
  right: .4rem;
  top: 50%;
  transform: translateY(-50%);
  border: 0;
  background: none;
  font-family: "DM Sans", system-ui, sans-serif;
  font-size: .8rem;
  font-weight: 500;
  color: #6d655a;
  padding: .42rem .55rem;
  cursor: pointer;
  border-radius: 5px;
  line-height: 1;
}
.lumen-toggle:hover { color: #1b1814; }

.lumen-row-end {
  display: flex;
  justify-content: flex-end;
  margin: -.15rem 0 1.1rem;
  font-size: .84rem;
}
.lumen-row-end a {
  color: #1b1814;
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}

.lumen-error {
  padding: .6rem .8rem;
  background: #fff0f0;
  color: #b91c1c;
  border-radius: 6px;
  font-size: .84rem;
  margin-bottom: .8rem;
}

.lumen-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: .55rem;
  width: 100%;
  font-family: "DM Sans", system-ui, sans-serif;
  font-weight: 600;
  font-size: .96rem;
  border-radius: 7px;
  padding: .82rem 1rem;
  cursor: pointer;
  transition: background .18s, transform .14s;
  border: 1.5px solid transparent;
}
.lumen-btn:active { transform: scale(.985); }

.lumen-btn-primary {
  background: #1b1814;
  color: #fffdf8;
  border-color: #1b1814;
}
.lumen-btn-primary:hover:not([disabled]) { background: #2e2a24; }
.lumen-btn-primary[disabled] { opacity: .72; cursor: progress; }

.lumen-btn-ghost {
  background: #fff;
  color: #1b1814;
  border-color: #d9d2c4;
}
.lumen-btn-ghost:hover { background: #f5f0e6; }

.lumen-or {
  display: flex;
  align-items: center;
  gap: .85rem;
  color: #6d655a;
  font-size: .8rem;
  margin: 1.05rem 0;
}
.lumen-or::before, .lumen-or::after {
  content: "";
  flex: 1;
  height: 1px;
  background: #d9d2c4;
}

.lumen-foot {
  margin: 1.15rem 0 0;
  font-size: .88rem;
  color: #6d655a;
}

.lumen-link-btn {
  background: none;
  border: none;
  font-family: "DM Sans", system-ui, sans-serif;
  font-size: inherit;
  font-weight: 600;
  color: #1b1814;
  padding: 0;
  cursor: pointer;
  text-decoration: underline;
  text-underline-offset: 2px;
}

.lumen-status {
  margin: .75rem 0 0;
  font-size: .84rem;
  color: #6d655a;
}

/* ── Fly-in CSS fallback (12 s safety net) ── */
.lumen-fly {
  animation: l-flyReveal .01s linear 12s both;
}
@keyframes l-flyReveal {
  to { opacity: 1; filter: none; transform: none; }
}

/* ─────────────────────────────────────────────────────
   BEAM
───────────────────────────────────────────────────── */
.lumen-beam-wrap {
  position: relative;
  z-index: 5;
  width: min(92vw, 440px);
  flex: 1 1 auto;
  min-height: 80px;
  margin-bottom: -5px;
  overflow: hidden;
}

.lumen-beam {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse 45% 100% at 50% 100%,
      rgba(255,254,245,.6), transparent 72%),
    linear-gradient(to top,
      rgba(255,240,205,.92),
      rgba(255,240,205,.30) 55%,
      rgba(255,240,205,.42));
  clip-path: polygon(43% 100%, 57% 100%, 100% 0%, 0% 0%);
  animation:
    l-beamGrow   1.6s cubic-bezier(.2,.7,.2,1) 2.8s both,
    l-beamIgnite 1.6s linear                   2.8s both,
    l-beamShimmer 3.5s steps(1)                4.5s infinite;
}

.lumen-beam-core {
  position: absolute;
  inset: 0;
  background: linear-gradient(to top,
    rgba(255,255,255,.28), rgba(255,255,255,.04) 60%, transparent);
  clip-path: polygon(47% 100%, 53% 100%, 62% 0%, 38% 0%);
  animation:
    l-beamGrow   1.6s cubic-bezier(.2,.7,.2,1) 2.8s both,
    l-beamIgnite 1.6s linear                   2.8s both;
}

/* ─────────────────────────────────────────────────────
   MOTES (glowing letters)
───────────────────────────────────────────────────── */
.lumen-mote {
  position: absolute;
  left: var(--x0, 50%);
  bottom: 0;
  font-family: "Instrument Serif", Georgia, serif;
  font-size: var(--sz, 16px);
  line-height: 1;
  color: rgba(255,248,220,.9);
  text-shadow:
    0 0  5px rgba(255,236,180,.95),
    0 0 14px rgba(255,215,130,.70),
    0 0 30px rgba(255,200, 90,.35);
  opacity: 0;
  pointer-events: none;
  animation: l-moteRise linear var(--dur, 6s) var(--del, 4s) infinite;
  will-change: transform, opacity;
}

/* ─────────────────────────────────────────────────────
   PROJECTOR
───────────────────────────────────────────────────── */
.lumen-projw {
  position: relative;
  z-index: 20;
  line-height: 0;
  flex-shrink: 0;
}

.lumen-lensglow {
  position: absolute;
  left: 50%;
  top: 0%;
  width: 160px;
  height: 160px;
  margin: -80px 0 0 -80px;
  border-radius: 50%;
  pointer-events: none;
  background: radial-gradient(circle,
    rgba(255,240,205,.98) 0%,
    rgba(255,240,205,.30) 38%,
    transparent 68%);
  animation: l-lensGlow 4s linear 0s both;
}

.lumen-proj {
  height: clamp(72px, 14vh, 130px);
  width: auto;
  display: block;
  animation:
    l-projAppear 4.2s linear 0s both,
    l-projHum .16s steps(1) 2.8s infinite;
}

.lumen-reel {
  transform-box: fill-box;
  transform-origin: center;
  animation: l-reelSpin 1.4s linear 2.8s infinite;
}

.lumen-lever {
  transform-box: fill-box;
  transform-origin: 50% 100%;
  transform: rotate(28deg);
  animation: l-leverFlip .22s cubic-bezier(.3,1.55,.5,1) 2.4s both;
}

.lumen-pilot {
  animation: l-pilotGlow 3s linear 0s both;
}

/* ─────────────────────────────────────────────────────
   KEYFRAMES
───────────────────────────────────────────────────── */
@keyframes l-fadeIn  { from { opacity: 0; } }

@keyframes l-camShot {
  0%    { opacity: 0; transform: translate(0, var(--ty,-32vh)) scale(var(--s,3))      rotate(.7deg); }
  12.8% { opacity: 0; transform: translate(0, var(--ty,-32vh)) scale(var(--s,3))      rotate(.7deg); }
  16%   { opacity: 1; transform: translate(0, var(--ty,-32vh)) scale(calc(var(--s,3)*1.02)) rotate(.6deg); }
  30%   { opacity: 1; transform: translate(0, var(--ty,-32vh)) scale(calc(var(--s,3)*1.05)) rotate(.6deg); }
  31%   { opacity: 1; transform: translate(1px, calc(var(--ty,-32vh) + 4px)) scale(calc(var(--s,3)*1.05)) rotate(.3deg); }
  32.5% { opacity: 1; transform: translate(0, var(--ty,-32vh)) scale(calc(var(--s,3)*1.06)) rotate(.7deg); }
  35%   { opacity: 1; transform: translate(0, var(--ty,-32vh)) scale(calc(var(--s,3)*1.08)) rotate(.6deg); }
  51%   { opacity: 1; transform: translate(0, var(--ty,-32vh)) scale(calc(var(--s,3)*1.08)) rotate(.6deg); animation-timing-function: cubic-bezier(.55,0,.18,1); }
  100%  { opacity: 1; transform: translate(0, 0) scale(1) rotate(0deg); }
}

@keyframes l-screenOn {
  0%  { opacity: 0; }
  14% { opacity: .55; }
  22% { opacity: .10; }
  40% { opacity: .82; }
  52% { opacity: .38; }
  68% { opacity: .90; }
  82% { opacity: .80; }
  100%{ opacity: 1; }
}

@keyframes l-grain {
  0%  { background-position: 0 0; }
  25% { background-position: -44px 32px; }
  50% { background-position:  36px -52px; }
  75% { background-position: -22px -18px; }
}

@keyframes l-flicker {
  0%, 90%  { opacity: 0; }
  91%      { opacity: .05; }
  92%      { opacity: 0; }
  95%      { opacity: .07; }
  96%,100% { opacity: 0; }
}

@keyframes l-beamGrow   { from { clip-path: polygon(50% 100%,50% 100%,50% 100%,50% 100%); } }
@keyframes l-beamIgnite {
  0%  { opacity: 0; }  8% { opacity: .82; }
  16% { opacity: .10; } 28% { opacity: .97; }
  40% { opacity: .38; } 58% { opacity: 1; }
  72% { opacity: .78; } 100% { opacity: 1; }
}
@keyframes l-beamShimmer {
  0%,100% { opacity: 1; } 18% { opacity: .93; } 20% { opacity: 1; }
  57% { opacity: .90; } 59% { opacity: 1; }
}

@keyframes l-moteRise {
  0%   { left: var(--x0); bottom: 0%;   transform: scale(.28) rotate(var(--r0)); opacity: 0; }
  10%  { opacity: .88; }
  80%  { opacity: .70; }
  100% { left: var(--x1); bottom: 100%; transform: scale(1.7) rotate(var(--r1)); opacity: 0; }
}

@keyframes l-lensGlow {
  0%,64% { opacity: 0; } 67% { opacity: 1; }
  73%    { opacity: .12; } 82% { opacity: 1; } 100% { opacity: .92; }
}

@keyframes l-projAppear {
  0%,26%  { opacity: 0; filter: brightness(.45); }
  50%,58% { opacity: .38; filter: brightness(.5); }
  80%,100%{ opacity: 1; filter: brightness(1); }
}
@keyframes l-projHum {
  0%  { transform: translate(0,0); }
  50% { transform: translate(.5px,.6px); }
}
@keyframes l-pilotGlow { 0%,59% { opacity: .10; } 60%,100% { opacity: 1; } }
@keyframes l-leverFlip  { from { transform: rotate(-32deg); } }
@keyframes l-reelSpin   { to { transform: rotate(360deg); } }

/* ── Reduced motion: show finished state immediately ── */
@media (prefers-reduced-motion: reduce) {
  .lumen-cam     { animation: none !important; }
  .lumen-fly     { animation: none !important; opacity: 1 !important; }
  .lumen-screen,
  .lumen-beam, .lumen-beam-core,
  .lumen-proj, .lumen-lensglow, .lumen-room,
  .lumen-screen::before, .lumen-screen::after { animation: none !important; }
  .lumen-reel, .lumen-pilot, .lumen-lever,
  .lumen-replay, .lumen-mote { animation: none !important; }
  .lumen-lever { transform: rotate(28deg); }
  .lumen-pilot { opacity: 1; }
}

/* ── Responsive ── */
@media (max-width: 400px) {
  .lumen-screen { padding: 18px 16px; }
  .lumen-stage h1 { font-size: 1.75rem; }
}
`
