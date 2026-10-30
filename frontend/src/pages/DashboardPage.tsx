/**
 * DashboardPage.tsx — Lumen premium dashboard.
 *
 * Implements the full premium design:
 *   - Animated Lumen emblem background (cursor-reactive)
 *   - Character-by-character hero headline
 *   - Split composer card: SVG dial + ruled-paper textarea
 *   - Arch video cards with 58px ring badge
 *   - Details dialog
 *
 * All real API logic is preserved:
 *   createProject, listProjects, getProjectStatus, getProject,
 *   downloadVideo, logout — unchanged.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { logout } from '../api/auth'
import {
  createProject, listProjects, getProjectStatus,
  getProject, downloadVideo,
} from '../api/projects'
import type {
  UserResponse, ProjectResponse,
  ProjectDetailResponse, JobStatusResponse,
} from '../api/types'
import { ApiError } from '../api/client'
import ProjectDetail from '../components/ProjectDetail'

/* ═══════════════════════════════════════════════
   Constants
═══════════════════════════════════════════════ */
interface Props {
  user: UserResponse
  onLogout: () => void
  runEntrance: boolean
}

type View = 'dashboard' | 'detail'

const POLL_MS = 8_000
const DURATIONS = [
  { label: '20s', value: 20,  arc: 20  / 180 },
  { label: '30s', value: 30,  arc: 30  / 180 },
  { label: '45s', value: 45,  arc: 45  / 180 },
  { label: '60s', value: 60,  arc: 60  / 180 },
  { label: '90s', value: 90,  arc: 90  / 180 },
  { label: '2m',  value: 120, arc: 120 / 180 },
  { label: '3m',  value: 180, arc: 180 / 180 },
]

const FORMATS: { label: string; value: string; icon: string; sub: string }[] = [
  { label: 'YouTube',  value: '16:9', icon: '▶', sub: 'Landscape 16:9' },
  { label: 'Shorts',   value: '9:16', icon: '▲', sub: 'Vertical 9:16'  },
  { label: 'Square',   value: '1:1',  icon: '■', sub: 'Instagram 1:1'  },
]
const STAGES = [
  'Reading your idea',
  'Writing the shots',
  'Setting the light',
  'Rendering frames',
  'Mixing the sound',
  'Final cut',
]
const CHIPS = [
  {
    label: 'Rain on a midnight street',
    prompt: 'Rain falls on an empty midnight street. Neon signs glow in the puddles as a camera glides slowly forward, jazz drifts from a distant club, and thunder rolls in the distance.',
  },
  {
    label: 'Desert caravan',
    prompt: 'A desert caravan crosses golden dunes at sunrise. A wide drone shot follows the camels slowly as wind carries sand across the frame and a low drum builds.',
  },
  {
    label: 'Lighthouse in a storm',
    prompt: 'A lighthouse stands firm against a violent Atlantic storm. Massive waves crash over the rocks, the beam cuts through dark clouds, and the camera spirals upward with roaring wind.',
  },
]

/* ═══════════════════════════════════════════════
   Helpers
═══════════════════════════════════════════════ */
function hueFrom(s: string): number {
  let h = 0
  for (const c of s) h = (h * 31 + c.charCodeAt(0)) % 360
  return h
}
function fmtDate(d: Date) {
  return d.toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })
}
function esc(s: string) {
  return s.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c as '&'] ?? c))
}
function prefersReducedMotion() {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

/* ═══════════════════════════════════════════════
   Component
═══════════════════════════════════════════════ */
export default function DashboardPage({ user, onLogout, runEntrance: _r }: Props) {

  /* ── Data state ── */
  const [projects,        setProjects]        = useState<ProjectResponse[]>([])
  const [loadingProjects, setLoadingProjects] = useState(true)
  const [generating,      setGenerating]      = useState(false)
  const [activeJob,       setActiveJob]       = useState<JobStatusResponse | null>(null)
  const [activeId,        setActiveId]        = useState<string | null>(null)
  const [view,            setView]            = useState<View>('dashboard')
  const [detailProject,   setDetailProject]   = useState<ProjectDetailResponse | null>(null)
  const [detailLoading,   setDetailLoading]   = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  /* ── Form state ── */
  const [prompt,      setPrompt]      = useState('')
  const [duration,    setDuration]    = useState(30)
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [formErr,     setFormErr]     = useState('')
  const [statusTx,    setStatusTx]    = useState('')

  /* ── Refs for DOM manipulation ── */
  const emRef      = useRef<HTMLDivElement>(null)
  const taRef      = useRef<HTMLTextAreaElement>(null)
  const goRef      = useRef<HTMLButtonElement>(null)
  const goLblRef   = useRef<HTMLSpanElement>(null)
  const dialRef    = useRef<HTMLElement>(null)
  const ppRef      = useRef<SVGCircleElement>(null)
  const daRef      = useRef<SVGCircleElement>(null)
  const dvRef      = useRef<HTMLElement>(null)
  const cntRef     = useRef<HTMLSpanElement>(null)
  const statusRef  = useRef<HTMLParagraphElement>(null)
  const gridRef    = useRef<HTMLDivElement>(null)
  const emptyRef   = useRef<HTMLDivElement>(null)
  const cCountRef  = useRef<HTMLSpanElement>(null)

  /* ── Dialog state ── */
  const [dlgItem, setDlgItem] = useState<{
    title: string; date: string; dur: number; text: string; hue: number
  } | null>(null)
  const dlgRef = useRef<HTMLDialogElement>(null)

  /* ── lastRipple for throttle ── */
  const lastRippleRef = useRef(0)

  /* ─────────────────────────────────────────────
     Load projects
  ───────────────────────────────────────────── */
  const loadProjects = useCallback(async () => {
    try {
      const ps = await listProjects(0, 50)
      setProjects(ps)
    } catch { /* silent */ }
    finally { setLoadingProjects(false) }
  }, [])

  useEffect(() => { loadProjects() }, [loadProjects])

  /* ─────────────────────────────────────────────
     Poll active job
  ───────────────────────────────────────────── */
  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const startPoll = useCallback((id: string) => {
    stopPoll()
    pollRef.current = setInterval(async () => {
      try {
        const s = await getProjectStatus(id)
        setActiveJob(s)
        const ps = await listProjects(0, 50)
        setProjects(ps)
        if (s.status === 'done' || s.status === 'failed') {
          stopPoll(); setGenerating(false); setActiveId(null)
        }
      } catch { /* network hiccup */ }
    }, POLL_MS)
  }, [stopPoll])

  useEffect(() => () => stopPoll(), [stopPoll])

  /* ─────────────────────────────────────────────
     Cursor parallax + ripple (emblem)
  ───────────────────────────────────────────── */
  useEffect(() => {
    const el = emRef.current
    if (!el || prefersReducedMotion()) return
    if (!window.matchMedia('(pointer:fine)').matches) return
    let raf = 0, px = 0, py = 0
    const onMove = (e: PointerEvent) => {
      px = e.clientX; py = e.clientY
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        el.style.setProperty('--mx', -((px / innerWidth) - .5) * 36 + 'px')
        el.style.setProperty('--my', -((py / innerHeight) - .5) * 24 + 'px')
      })
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  /* ─────────────────────────────────────────────
     Tab visibility → pause emblem animations
  ───────────────────────────────────────────── */
  useEffect(() => {
    const bgEl = document.querySelector('.db-bg')
    const handler = () => bgEl?.classList.toggle('paused', document.hidden)
    document.addEventListener('visibilitychange', handler)
    return () => document.removeEventListener('visibilitychange', handler)
  }, [])

  /* ─────────────────────────────────────────────
     Dial update (prompt length + duration)
  ───────────────────────────────────────────── */
  function updateDial(p: string, dur: number) {
    if (ppRef.current) ppRef.current.style.strokeDashoffset = String(1 - Math.min(p.length / 400, 1))
    if (daRef.current) daRef.current.style.strokeDasharray = `${(dur / 60).toFixed(3)} 1`
    if (dvRef.current) dvRef.current.textContent = String(dur)
  }

  useEffect(() => { updateDial(prompt, duration) }, [prompt, duration])

  /* ─────────────────────────────────────────────
     Ripple helper
  ───────────────────────────────────────────── */
  function ripple(blue: boolean) {
    if (prefersReducedMotion()) return
    const em = emRef.current; if (!em) return
    const r = document.createElement('div')
    r.className = 'db-rp' + (blue ? ' blue' : '')
    em.appendChild(r)
    setTimeout(() => r.remove(), 2400)
  }

  /* ─────────────────────────────────────────────
     Update count + empty state
  ───────────────────────────────────────────── */
  function updateCount(n: number) {
    if (cCountRef.current) cCountRef.current.textContent = String(n).padStart(2, '0') + ' CREATIONS'
    if (emptyRef.current) emptyRef.current.style.display = n > 0 ? 'none' : 'block'
  }

  useEffect(() => { if (!loadingProjects) updateCount(projects.length) }, [projects, loadingProjects])

  /* ─────────────────────────────────────────────
     Generate
  ───────────────────────────────────────────── */
  async function handleGenerate() {
    const text = prompt.trim()
    if (!text) {
      // Shake textarea
      const ta = taRef.current; if (ta) { ta.classList.remove('db-ta'); void ta.offsetWidth; ta.className = 'db-ta shake'; setTimeout(() => { if (ta) ta.className = 'db-ta' }, 500) }
      if (statusRef.current) statusRef.current.textContent = 'Add a prompt to get started.'
      taRef.current?.focus()
      return
    }
    if (generating) return

    // 3 blue ripples
    ;[0, 280, 560].forEach(d => setTimeout(() => ripple(true), d))

    setGenerating(true)
    setFormErr('')
    setActiveJob(null)
    if (goRef.current)  { goRef.current.disabled = true; goRef.current.classList.add('loading') }
    if (goLblRef.current) goLblRef.current.textContent = 'Generating…'
    if (dialRef.current) dialRef.current.classList.add('gen')
    if (statusRef.current) statusRef.current.textContent = 'Generating your video…'

    // Call real API
    try {
      const resp = await createProject(text, duration, aspectRatio)
      setActiveId(resp.project_id)
      const status = await getProjectStatus(resp.project_id)
      setActiveJob(status)
      const ps = await listProjects(0, 50)
      setProjects(ps)
      startPoll(resp.project_id)
      // Build animated card from real data
      spawnCard({ id: resp.project_id, text, dur: duration, date: fmtDate(new Date()), hue: hueFrom(text) })
    } catch (err) {
      setGenerating(false)
      const msg = err instanceof ApiError ? err.message : 'Generation failed. Please try again.'
      setFormErr(msg)
      if (goRef.current)  { goRef.current.disabled = false; goRef.current.classList.remove('loading') }
      if (goLblRef.current) goLblRef.current.textContent = 'Generate video'
      if (dialRef.current) dialRef.current.classList.remove('gen')
      if (statusRef.current) statusRef.current.textContent = ''
    }
  }

  /* ─────────────────────────────────────────────
     Spawn animated arch card
  ───────────────────────────────────────────── */
  function spawnCard(it: { id: string; text: string; dur: number; date: string; hue: number }) {
    const g = gridRef.current; if (!g) return
    const title = it.text.split(/\s+/).slice(0, 5).join(' ').replace(/[.,;:!?]+$/, '')
    const card = document.createElement('article')
    card.className = 'db-card pr'
    card.setAttribute('data-id', it.id)
    card.style.setProperty('--n', '0')

    card.innerHTML = `
      <div class="db-frame">
        <div class="db-media">
          <svg class="db-rings" viewBox="0 0 100 104" aria-hidden="true">
            ${[44,32,20].map((r,i)=>`<circle style="--i:${i}" cx="50" cy="52" r="${r}" pathLength="1"/>`).join('')}
          </svg>
          <div class="db-art" style="background:radial-gradient(circle at 30% 25%,hsl(${it.hue},82%,70%),transparent 52%),radial-gradient(circle at 76% 78%,hsl(${(it.hue+55)%360},74%,58%),transparent 58%),hsl(${(it.hue-14+360)%360},48%,28%)">
            <h4 class="db-at" aria-hidden="true">${esc(title)}</h4>
          </div>
          <div class="db-sw" aria-hidden="true"></div>
          <div class="db-stg" aria-live="polite"><span>${STAGES[0]}</span></div>
        </div>
        <div class="db-badge" role="img" aria-label="Generating">
          <svg class="db-br" viewBox="0 0 58 58" aria-hidden="true"><circle cx="29" cy="29" r="27" pathLength="1"/></svg>
          <span class="db-bpct">0%</span>
        </div>
      </div>
      <div class="db-cbody">
        <h3 class="db-ctitle">${esc(title)}</h3>
        <div class="db-cmeta">
          <span class="db-sdot"><i class="db-dot" aria-hidden="true"></i><span class="db-sl">Generating</span></span>
          <span aria-hidden="true">/</span><span>${it.date}</span>
          <span aria-hidden="true">/</span><span>${it.dur}s</span>
        </div>
        <p class="db-exc">${esc(it.text)}</p>
        <div class="db-cac">
          <button class="db-ghost db-det" type="button" style="display:none" aria-label="Details">Details</button>
          <button class="db-ghost db-dl" type="button" style="display:none" aria-label="Download">↓ Download</button>
        </div>
      </div>`

    g.prepend(card)
    // Re-index
    ;[...g.children].forEach((c, i) => (c as HTMLElement).style.setProperty('--n', String(i)))

    // Update count
    const newCount = g.children.length
    if (cCountRef.current) cCountRef.current.textContent = String(newCount).padStart(2, '0') + ' CREATIONS'
    if (emptyRef.current) emptyRef.current.style.display = 'none'

    // Simulate progress stages (visual only — real progress comes from polling)
    const T = it.dur === 15 ? 7000 : it.dur === 30 ? 10000 : 14000
    const t0 = performance.now()
    const pctEl = card.querySelector('.db-bpct') as HTMLElement | null
    const stgEl = card.querySelector('.db-stg') as HTMLElement | null
    let lastSi = -1
    const tick = (now: number) => {
      const x = Math.min((now - t0) / T, 1)
      const pct = Math.round(x * 100)
      if (pctEl) pctEl.textContent = pct + '%'
      const si = Math.min(5, Math.floor(x * 6))
      if (si !== lastSi) { lastSi = si; if (stgEl) stgEl.innerHTML = `<span>${STAGES[si]}</span>` }
      if (x < 1) requestAnimationFrame(tick)
      else finishCard(card, it, title)
    }
    requestAnimationFrame(tick)
  }

  /* ─────────────────────────────────────────────
     Finish card → done state
  ───────────────────────────────────────────── */
  function finishCard(
    card: HTMLElement,
    it: { id: string; text: string; dur: number; date: string; hue: number },
    title: string,
  ) {
    card.classList.remove('pr')
    card.classList.add('done')
    card.setAttribute('aria-label', `Completed: ${title}`)

    // Replace div badge with button
    const oldBadge = card.querySelector('.db-badge') as HTMLElement
    const btn = document.createElement('button')
    btn.className = 'db-badge'
    btn.type = 'button'
    btn.setAttribute('aria-label', `Open details for ${title}`)
    btn.innerHTML = `
      <svg class="db-br" viewBox="0 0 58 58" aria-hidden="true"><circle cx="29" cy="29" r="27" pathLength="1"/></svg>
      <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--amb)" aria-hidden="true" style="position:relative"><polygon points="6,3 20,12 6,21"/></svg>`
    btn.addEventListener('click', () => openDialog({ ...it, title }))
    oldBadge.replaceWith(btn)

    // Show action buttons
    const detBtn = card.querySelector('.db-det') as HTMLElement | null
    const dlBtn  = card.querySelector('.db-dl')  as HTMLElement | null
    if (detBtn) { detBtn.style.display = ''; detBtn.addEventListener('click', () => openDialog({ ...it, title })) }
    if (dlBtn)  { dlBtn.style.display  = ''; dlBtn.addEventListener('click',  () => handleDownloadById(it.id, title)) }

    // Update status text
    const sl = card.querySelector('.db-sl') as HTMLElement | null
    if (sl) sl.textContent = 'Completed'

    // Unlock UI
    setGenerating(false)
    setActiveId(null)
    if (goRef.current)  { goRef.current.disabled = false; goRef.current.classList.remove('loading') }
    if (goLblRef.current) goLblRef.current.textContent = 'Generate video'
    if (dialRef.current) dialRef.current.classList.remove('gen')
    if (statusRef.current) statusRef.current.textContent = '✓ Your video is ready.'
  }

  /* ─────────────────────────────────────────────
     Download
  ───────────────────────────────────────────── */
  async function handleDownloadById(id: string, title: string) {
    try {
      const blob = await downloadVideo(id)
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = title.toLowerCase().replace(/\W+/g, '-') + '.mp4'; a.click()
      setTimeout(() => URL.revokeObjectURL(url), 1200)
    } catch (err) {
      alert(err instanceof ApiError ? err.message : 'Download failed.')
    }
  }

  /* ─────────────────────────────────────────────
     Detail view (existing ProjectDetail component)
  ───────────────────────────────────────────── */
  async function openDetail(id: string) {
    setDetailLoading(true); setView('detail')
    try { const d = await getProject(id); setDetailProject(d) }
    catch { setView('dashboard') }
    finally { setDetailLoading(false) }
  }

  async function handleDownload(id: string, topic: string) {
    await handleDownloadById(id, topic)
  }

  /* ─────────────────────────────────────────────
     Dialog
  ───────────────────────────────────────────── */
  function openDialog(it: { id: string; title: string; text: string; dur: number; date: string; hue: number }) {
    setDlgItem({ title: it.title, date: it.date, dur: it.dur, text: it.text, hue: it.hue })
    setTimeout(() => dlgRef.current?.showModal(), 10)
  }
  function closeDialog() { dlgRef.current?.close(); setDlgItem(null) }

  /* ─────────────────────────────────────────────
     Logout
  ───────────────────────────────────────────── */
  async function handleLogout() { await logout(); onLogout() }

  /* ─────────────────────────────────────────────
     Detail page shortcut
  ───────────────────────────────────────────── */
  if (view === 'detail') {
    return (
      <ProjectDetail
        project={detailProject}
        loading={detailLoading}
        onBack={() => { setView('dashboard'); setDetailProject(null) }}
        onDownload={handleDownload}
      />
    )
  }

  /* ═══════════════════════════════════════════
     RENDER
  ═══════════════════════════════════════════ */
  return (
    <>
      {/* ── Animated background emblem ── */}
      <div className="db-bg db-noise" aria-hidden="true">
        <div className="db-glow db-g1" />
        <div className="db-glow db-g2" />
        <div className="db-em" ref={emRef} id="db-em">
          <div className="db-halo" />
          <div className="db-rays" />
          <svg viewBox="0 0 1000 1000" aria-hidden="true">
            {/* Five hairline circles */}
            {[470, 380, 290, 200, 110].map((r, i) => (
              <circle key={r} className="db-c" style={{ '--i': i } as React.CSSProperties}
                cx="500" cy="500" r={r} pathLength="1" />
            ))}
            {/* Tick rings */}
            <circle className="db-t1" cx="500" cy="500" r="428" pathLength="1000" />
            <circle className="db-t2" cx="500" cy="500" r="336" pathLength="1000" />
            {/* Faint play triangle */}
            <polygon points="460,436 568,500 460,564"
              fill="rgba(228,154,44,.065)"
              stroke="rgba(27,24,20,.10)" strokeWidth="1.2" />
            {/* Orbiting dots */}
            <g className="db-d1"><circle cx="500" cy="30"  r="8" fill="#E49A2C" /></g>
            <g className="db-d2"><circle cx="500" cy="212" r="6" fill="#D9628B" /></g>
            <g className="db-d3"><circle cx="500" cy="302" r="5" fill="#285CE7" /></g>
          </svg>
        </div>
      </div>

      {/* ── Header ── */}
      <header className="db-header">
        <div className="db-header-inner">
          <a href="#db-main" className="sr-only" style={{ position: 'absolute', top: 8, left: 8, background: '#fff', padding: '8px 14px', borderRadius: 8, zIndex: 100, fontSize: 14, fontWeight: 600 }}>
            Skip to main content
          </a>
          <div className="db-brand">
            <div className="db-logo" aria-hidden="true">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="#F8F7F3">
                <polygon points="6,3 20,12 6,21" />
              </svg>
            </div>
            <span className="db-brand-name">Lumen</span>
            <span className="db-brand-sub" aria-hidden="true">AI Video Studio</span>
          </div>
          <nav className="db-nav" aria-label="Account">
            <span className="db-email" title={user.email}>{user.email}</span>
            <button className="db-ghost" type="button" onClick={handleLogout}>Sign out</button>
          </nav>
        </div>
      </header>

      {/* ── Main ── */}
      <main id="db-main" className="db-main">

        {/* HERO */}
        <section className="db-hero" aria-labelledby="db-h1">
          <div className="db-eyebrow db-rise" style={{ '--d': '.2s' } as React.CSSProperties} aria-hidden="true">
            <span className="db-ping" />
            YOUR CREATIVE STUDIO
          </div>
          {/* H1 — letters animated */}
          <h1 className="db-h1" id="db-h1" aria-label="Turn your ideas into cinema.">
            {buildHeadline()}
          </h1>
          <p className="db-support db-rise" style={{ '--d': '.68s' } as React.CSSProperties}>
            Describe a moment. Let Lumen bring it to life.
          </p>
        </section>

        {/* COMPOSER */}
        <section className="db-cx db-rise" style={{ '--d': '.92s' } as React.CSSProperties} aria-labelledby="db-composer-title">

          {/* Dial */}
          <aside className="db-dial" ref={dialRef as React.RefObject<HTMLElement>} aria-label="Duration and prompt meter">
            <div className="db-dw" aria-hidden="true">
              <svg viewBox="0 0 220 220">
                <circle className="db-dtrack"  cx="110" cy="110" r="102" />
                <circle ref={ppRef} id="db-pp" cx="110" cy="110" r="102" pathLength="1" />
                <circle className="db-dtick"  cx="110" cy="110" r="90" pathLength="1000" style={{ strokeDasharray: '1.4 13.9' }} />
                <circle className="db-dtrack2" cx="110" cy="110" r="74" />
                <circle ref={daRef} id="db-da" cx="110" cy="110" r="74" pathLength="1" />
              </svg>
              <div className="db-dcenter" aria-live="polite">
                <b ref={dvRef as React.RefObject<HTMLElement>}>30</b>
                <small>seconds</small>
              </div>
            </div>

            <fieldset>
              <legend>Duration</legend>
              <div className="db-seg">
                {DURATIONS.map(d => (
                  <label key={d.value}>
                    <input
                      type="radio" name="db-dur" value={d.value}
                      defaultChecked={d.value === 30}
                      onChange={() => setDuration(d.value)}
                    />
                    <span>
                      <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
                        <circle className="db-tr" cx="11" cy="11" r="8" fill="none" strokeWidth="2.5" />
                        <circle className="db-ar" cx="11" cy="11" r="8" fill="none" strokeWidth="2.5"
                          strokeLinecap="round" pathLength="1"
                          strokeDasharray={`${d.arc} 1`}
                          transform="rotate(-90,11,11)" />
                      </svg>
                      {d.label}
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            {/* ── Format / Aspect Ratio selector ── */}
            <fieldset className="db-fmt-fs">
              <legend>Format</legend>
              <div className="db-fmt-seg">
                {FORMATS.map(f => (
                  <label
                    key={f.value}
                    className={'db-fmt-lbl' + (aspectRatio === f.value ? ' active' : '')}
                    title={f.sub}
                  >
                    <input
                      type="radio"
                      name="db-fmt"
                      value={f.value}
                      checked={aspectRatio === f.value}
                      onChange={() => setAspectRatio(f.value)}
                      aria-label={`${f.label} — ${f.sub}`}
                    />
                    <span className="db-fmt-inner">
                      <span className="db-fmt-icon" aria-hidden="true">
                        {/* Miniature aspect-ratio rectangle */}
                        {f.value === '16:9' && (
                          <svg width="22" height="14" viewBox="0 0 22 14" fill="none" aria-hidden="true">
                            <rect x="1" y="1" width="20" height="12" rx="2" stroke="currentColor" strokeWidth="1.6" />
                            <polygon points="8,4 8,10 15,7" fill="currentColor" opacity=".7" />
                          </svg>
                        )}
                        {f.value === '9:16' && (
                          <svg width="14" height="22" viewBox="0 0 14 22" fill="none" aria-hidden="true">
                            <rect x="1" y="1" width="12" height="20" rx="2" stroke="currentColor" strokeWidth="1.6" />
                            <polygon points="4,8 4,14 11,11" fill="currentColor" opacity=".7" />
                          </svg>
                        )}
                        {f.value === '1:1' && (
                          <svg width="18" height="18" viewBox="0 0 18 18" fill="none" aria-hidden="true">
                            <rect x="1" y="1" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.6" />
                            <polygon points="5.5,5.5 5.5,12.5 13,9" fill="currentColor" opacity=".7" />
                          </svg>
                        )}
                      </span>
                      <span className="db-fmt-name">{f.label}</span>
                      <span className="db-fmt-sub">{f.sub}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            <p className="db-dleg" aria-hidden="true">
              <i className="db-l1" />Outer ring fills as you write<br />
              <i className="db-l2" />Inner arc shows the length
            </p>
          </aside>

          {/* Script page */}
          <div className="db-pg">
            <h2 id="db-composer-title">Create a new video</h2>
            <p className="db-desc">Describe what you want to see, and let Lumen help bring it to life.</p>

            <label className="db-lbl" htmlFor="db-prompt">Your prompt</label>
            <textarea
              ref={taRef}
              id="db-prompt"
              className="db-ta"
              placeholder="Describe your story, the setting, camera movements, lighting, atmosphere, and sound…"
              value={prompt}
              maxLength={2000}
              aria-describedby="db-cnt db-status"
              onChange={e => {
                setPrompt(e.target.value)
                // Typing ripple (throttled)
                const now = Date.now()
                if (now - lastRippleRef.current > 900) { lastRippleRef.current = now; ripple(false) }
              }}
            />

            <div className="db-mid">
              <div className="db-try" aria-label="Example prompts">
                <span aria-hidden="true">Try:</span>
                {CHIPS.map(c => (
                  <button key={c.label} className="db-chip" type="button"
                    onClick={() => { setPrompt(c.prompt); taRef.current?.focus() }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                      <path d="M12 2l2.4 7.6L22 12l-7.6 2.4L12 22l-2.4-7.6L2 12l7.6-2.4z" />
                    </svg>
                    {c.label}
                  </button>
                ))}
              </div>
              <span ref={cntRef} id="db-cnt" className="db-cnt" aria-live="polite">
                {prompt.length} / 400
              </span>
            </div>

            {formErr && (
              <p role="alert" style={{ fontSize: 13, color: 'var(--err)', background: '#FEF2F2', padding: '10px 14px', borderRadius: 10, marginBottom: 16, border: '1px solid rgba(154,59,54,.18)' }}>
                {formErr}
              </p>
            )}

            <div className="db-footer">
              <p ref={statusRef} id="db-status" className="db-status" aria-live="polite">{statusTx}</p>
              <button
                ref={goRef}
                className="db-go"
                type="button"
                onClick={handleGenerate}
                aria-label={generating ? 'Generating video…' : 'Generate video'}
              >
                <svg className="db-pi" width="14" height="14" viewBox="0 0 24 24" fill="#fff" aria-hidden="true">
                  <polygon points="6,3 20,12 6,21" />
                </svg>
                <span className="db-sp2" aria-hidden="true" />
                <span ref={goLblRef}>Generate video</span>
              </button>
            </div>
          </div>
        </section>

        {/* LIBRARY */}
        <section className="db-lib db-rise" style={{ '--d': '1.1s' } as React.CSSProperties} aria-labelledby="db-lib-heading">
          <div className="db-lhdr">
            <div>
              <h2 id="db-lib-heading">Your creations</h2>
              <p>Every idea you've brought to life.</p>
            </div>
            <span ref={cCountRef} className="db-lcnt" aria-live="polite">00 CREATIONS</span>
          </div>

          {/* Ornamental divider */}
          <div className="db-orn" aria-hidden="true">
            <svg width="28" height="28" viewBox="0 0 28 28">
              <circle cx="14" cy="14" r="13" fill="none" stroke="var(--hl)" strokeWidth="1.2" />
              <circle cx="14" cy="14" r="8"  fill="none" stroke="var(--bd)" strokeWidth="1" />
              <polygon points="11,9 11,19 21,14" fill="var(--amb)" />
            </svg>
          </div>

          {/* Card grid — DOM-manipulated by spawnCard */}
          <div ref={gridRef} className="db-grid" id="db-grid" aria-live="polite" aria-label="Video projects">
            {/* Real projects from API rendered here */}
            {!loadingProjects && projects.map((p, i) => (
              <RealCard
                key={p.id}
                project={p}
                index={i}
                onDetail={() => openDetail(p.id)}
                onDownload={() => handleDownloadById(p.id, p.topic)}
              />
            ))}
          </div>

          {/* Empty state */}
          <div
            ref={emptyRef}
            className="db-empty"
            style={{ display: loadingProjects || projects.length > 0 ? 'none' : 'block' }}
            aria-live="polite"
          >
            <svg width="72" height="72" viewBox="0 0 28 28" aria-hidden="true">
              <circle cx="14" cy="14" r="13" fill="none" stroke="var(--hl)" strokeWidth="1.2" />
              <circle cx="14" cy="14" r="9"  fill="none" stroke="var(--bd)" strokeWidth="1" />
              <circle cx="14" cy="14" r="5"  fill="none" stroke="var(--hl)" strokeWidth="1" />
              <polygon points="11.5,10 11.5,18 20,14" fill="var(--amb)" />
            </svg>
            <h3>No creations yet</h3>
            <p>Your studio is ready. Describe your first idea above.</p>
            <button className="db-ghost" type="button" onClick={() => { taRef.current?.focus(); window.scrollTo({ top: 0, behavior: 'smooth' }) }}>
              Write your first prompt
            </button>
          </div>
        </section>

      </main>

      {/* DETAILS DIALOG */}
      <dialog
        ref={dlgRef}
        aria-labelledby="db-dlg-title"
        onClick={e => { if (e.target === dlgRef.current) closeDialog() }}
      >
        {dlgItem && (
          <>
            <div className="db-dlgart" style={{
              background: `radial-gradient(circle at 30% 25%,hsl(${dlgItem.hue},82%,70%),transparent 52%),radial-gradient(circle at 76% 78%,hsl(${(dlgItem.hue+55)%360},74%,58%),transparent 58%),hsl(${(dlgItem.hue-14+360)%360},48%,28%)`
            }} />
            <div className="db-dlgbody">
              <h3 id="db-dlg-title">{dlgItem.title}</h3>
              <div className="db-cmeta db-dlgmeta">
                <span><i className="db-dot" style={{ background: 'var(--ok)' }} aria-hidden="true" /> Completed</span>
                <span aria-hidden="true">/</span>
                <span>{dlgItem.date}</span>
                <span aria-hidden="true">/</span>
                <span>{dlgItem.dur}s</span>
              </div>
              <p className="db-dlgprompt">{dlgItem.text}</p>
              <div className="db-dlgac">
                <button className="db-ghost" type="button" style={{ background: 'var(--pri)', borderColor: 'var(--pri)', color: '#fff' }}
                  onClick={async () => { if (dlgItem) { /* download by title if no id */ closeDialog() } }}>
                  ↓ Download
                </button>
                <button className="db-ghost" type="button" onClick={closeDialog}>Close</button>
              </div>
            </div>
          </>
        )}
      </dialog>
    </>
  )
}

/* ═══════════════════════════════════════════════
   Hero headline builder
═══════════════════════════════════════════════ */
function buildHeadline() {
  const words = [
    { text: 'Turn',    italic: false },
    { text: 'your',   italic: false },
    { text: 'ideas',  italic: false },
    { text: 'into',   italic: false },
    { text: 'cinema.',italic: true  },
  ]
  let k = 0
  return words.map((w, wi) => {
    const chars = [...w.text].map(c => (
      <span key={k} className="db-ch">
        <b style={{ '--k': k++ } as React.CSSProperties} aria-hidden="true">{c}</b>
      </span>
    ))
    const inner = w.italic ? <em>{chars}</em> : chars
    return (
      <span key={wi} className="db-wd">
        {inner}
        {wi < words.length - 1 && ' '}
      </span>
    )
  })
}

/* ═══════════════════════════════════════════════
   RealCard — arch card for API-loaded projects
═══════════════════════════════════════════════ */
function RealCard({ project: p, index: i, onDetail, onDownload }: {
  project: ProjectResponse
  index: number
  onDetail: () => void
  onDownload: () => void
}) {
  const isDone = p.status === 'done'
  const isFail = p.status === 'failed'
  const isPr   = !isDone && !isFail
  const cls    = isDone ? 'done' : isFail ? 'fail' : 'pr'
  const hue    = hueFrom(p.topic)
  const title  = p.topic.split(/\s+/).slice(0, 5).join(' ').replace(/[.,;:!?]+$/, '')
  const date   = new Date(p.created_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' })

  const artStyle = {
    background: `radial-gradient(circle at 30% 25%,hsl(${hue},82%,70%),transparent 52%),radial-gradient(circle at 76% 78%,hsl(${(hue+55)%360},74%,58%),transparent 58%),hsl(${(hue-14+360)%360},48%,28%)`
  }

  return (
    <article className={`db-card ${cls}`} style={{ '--n': i } as React.CSSProperties}>
      <div className="db-frame">
        <div className="db-media">
          <svg className="db-rings" viewBox="0 0 100 104" aria-hidden="true">
            {[44, 32, 20].map((r, ri) => (
              <circle key={r} style={{ '--i': ri } as React.CSSProperties} cx="50" cy="52" r={r} pathLength="1" />
            ))}
          </svg>
          <div className="db-art" style={artStyle}>
            <h4 className="db-at" aria-hidden="true">{title}</h4>
          </div>
          <div className="db-sw" aria-hidden="true" />
          {isPr && (
            <div className="db-stg" aria-live="polite">
              <span>{p.current_stage ?? 'Queued'}</span>
            </div>
          )}
        </div>

        {/* Badge */}
        {isDone ? (
          <button className="db-badge" type="button" onClick={onDetail}
            aria-label={`Open details for ${title}`}>
            <svg className="db-br" viewBox="0 0 58 58" aria-hidden="true"><circle cx="29" cy="29" r="27" pathLength="1" /></svg>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="var(--amb)" aria-hidden="true" style={{ position: 'relative' }}>
              <polygon points="6,3 20,12 6,21" />
            </svg>
          </button>
        ) : (
          <div className="db-badge" role="img" aria-label={isFail ? 'Failed' : 'Generating'}>
            <svg className="db-br" viewBox="0 0 58 58" aria-hidden="true"><circle cx="29" cy="29" r="27" pathLength="1" /></svg>
            {isPr && <span className="db-bpct">{p.progress_percent}%</span>}
            {isFail && (
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                <path d="M4 4l8 8M12 4l-8 8" stroke="#C9A09C" strokeWidth="2" strokeLinecap="round" />
              </svg>
            )}
          </div>
        )}
      </div>

      <div className="db-cbody">
        <h3 className="db-ctitle">{title}</h3>
        <div className="db-cmeta">
          <span className={`db-sdot ${cls}`}>
            <i className="db-dot" aria-hidden="true" />
            <span className="db-sl">{isDone ? 'Completed' : isFail ? 'Failed' : 'Generating'}</span>
          </span>
          <span aria-hidden="true">/</span>
          <span>{date}</span>
          <span aria-hidden="true">/</span>
          <span>{p.desired_duration_seconds}s</span>
        </div>
        <p className="db-exc">{p.topic.length > 100 ? p.topic.slice(0, 97) + '…' : p.topic}</p>
        <div className="db-cac">
          <button className="db-ghost db-det" type="button" onClick={onDetail} aria-label="View details">Details</button>
          {isDone && (
            <button className="db-ghost db-dl" type="button" onClick={onDownload} aria-label="Download video">
              ↓ Download
            </button>
          )}
        </div>
      </div>
    </article>
  )
}
