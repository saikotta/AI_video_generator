import { useCallback, useEffect, useRef, useState } from 'react'
import { logout } from '../api/auth'
import { createProject, downloadVideo, getProject, getProjectStatus, listProjects } from '../api/projects'
import type { ProjectDetailResponse, ProjectResponse, UserResponse } from '../api/types'
import { ApiError } from '../api/client'
import ProjectDetail from '../components/ProjectDetail'
import cerevynCss from '../styles/cerevyn-dashboard.css?raw'

interface Props {
  user: UserResponse
  onLogout: () => void
}

const DURATIONS = [15, 30, 60]
const FORMATS = [
  { name: 'Shorts', ratio: '9:16', duration: 60 },
  { name: 'Reels', ratio: '9:16', duration: 30 },
  { name: 'Ad', ratio: '1:1', duration: 15 },
  { name: 'YouTube', ratio: '16:9', duration: 60 },
]
const IDEAS = [
  { label: 'Moon garden', icon: '✦', prompt: 'A lone astronaut tends a glowing garden on the rim of a quiet moon.' },
  { label: 'Neon alley', icon: '☂', prompt: 'A rain-soaked Tokyo alley at night, neon reflections, a cat watches a stranger pass.' },
  { label: 'Desert caravan', icon: '☀', prompt: 'Golden hour over a desert caravan, slow dolly shot, dust catching the light.' },
]
const LENS = <svg viewBox="0 0 200 200" aria-hidden="true"><circle cx="100" cy="100" r="78"/><circle cx="100" cy="100" r="58"/><circle cx="100" cy="100" r="38"/><path d="M100 2v44M100 154v44M2 100h44M154 100h44"/></svg>

function projectStatus(project: ProjectResponse) {
  if (project.status === 'done') return { text: '● COMPLETE', className: 'done' }
  if (project.status === 'failed') return { text: '× FAILED', className: 'fail' }
  return { text: `● ${project.status.replace(/_/g, ' ').toUpperCase()}`, className: 'proc' }
}

export default function DashboardPage({ user, onLogout }: Props) {
  const [projects, setProjects] = useState<ProjectResponse[]>([])
  const [loadingProjects, setLoadingProjects] = useState(true)
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(30)
  const [aspectRatio, setAspectRatio] = useState('16:9')
  const [generating, setGenerating] = useState(false)
  const [activeProjectId, setActiveProjectId] = useState<string | null>(null)
  const [error, setError] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [detailProject, setDetailProject] = useState<ProjectDetailResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const promptRef = useRef<HTMLTextAreaElement>(null)
  const activeProjectIdRef = useRef(activeProjectId)
  activeProjectIdRef.current = activeProjectId

  const refreshProjects = useCallback(async () => {
    try { setProjects(await listProjects(0, 100)) }
    catch { setError('Could not load your projects. Please refresh and try again.') }
    finally { setLoadingProjects(false) }
  }, [])

  useEffect(() => { void refreshProjects() }, [refreshProjects])

  useEffect(() => {
    const timer = window.setInterval(async () => {
      try {
        const latest = await listProjects(0, 100)
        setProjects(latest)
        const activeId = activeProjectIdRef.current
        if (!activeId) return
        const job = await getProjectStatus(activeId)
        if (job.status === 'done' || job.status === 'failed') {
          setActiveProjectId(null)
          setGenerating(false)
          setStatusMessage(job.status === 'done' ? 'Your film is ready.' : job.error_message || 'Film generation failed.')
          setProjects(await listProjects(0, 100))
        } else {
          setStatusMessage(job.current_stage || 'Your film is being created…')
        }
      } catch { /* keep the last project state when the connection is briefly unavailable */ }
    }, 8000)
    return () => window.clearInterval(timer)
  }, [])

  async function handleGenerate() {
    const topic = prompt.trim()
    if (!topic) {
      setError('Write a scene before generating your film.')
      promptRef.current?.focus()
      return
    }
    if (generating) return

    setError('')
    setStatusMessage('Preparing your film…')
    setGenerating(true)
    try {
      const created = await createProject(topic, duration, aspectRatio)
      setPrompt('')
      const [job, updatedProjects] = await Promise.all([
        getProjectStatus(created.project_id),
        listProjects(0, 100),
      ])
      setProjects(updatedProjects)
      if (job.status === 'done' || job.status === 'failed') {
        setGenerating(false)
        setStatusMessage(job.status === 'done' ? 'Your film is ready.' : job.error_message || 'Film generation failed.')
      } else {
        setActiveProjectId(created.project_id)
        setStatusMessage(job.current_stage || 'Your film is being created…')
      }
    } catch (cause) {
      setGenerating(false)
      setStatusMessage('')
      setError(cause instanceof ApiError ? cause.message : 'Film generation failed. Please try again.')
    }
  }

  async function openDetails(id: string) {
    setDetailLoading(true)
    try { setDetailProject(await getProject(id)) }
    catch (cause) { setError(cause instanceof ApiError ? cause.message : 'Could not load project details.') }
    finally { setDetailLoading(false) }
  }

  async function handleDownload(id: string, title: string) {
    try {
      const blob = await downloadVideo(id)
      const objectUrl = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = objectUrl
      link.download = `${title.toLowerCase().replace(/[^a-z0-9]+/g, '-') || 'cerevyn-film'}.mp4`
      link.click()
      window.setTimeout(() => URL.revokeObjectURL(objectUrl), 10_000)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : 'Download failed. Please try again.')
    }
  }

  async function handleLogout() {
    try { await logout() } finally { onLogout() }
  }

  if (detailProject || detailLoading) {
    return <ProjectDetail project={detailProject} loading={detailLoading} onBack={() => setDetailProject(null)} onDownload={handleDownload} />
  }

  const activeDurationAngle = duration === 15 ? '-46deg' : duration === 60 ? '46deg' : '0deg'

  return (
    <div className="cerevyn-dashboard">
      <style>{cerevynCss}</style>
      <div className="bg" aria-hidden="true">
        <div className="pool p1" />
        <div className="pool p2" />
        <div className="optics">
          <svg viewBox="0 0 400 400" aria-hidden="true">
            <g className="sp"><circle cx="200" cy="200" r="190"/><circle cx="200" cy="200" r="150" strokeDasharray="2 6"/><circle cx="200" cy="200" r="110"/><path d="M200 10A190 190 0 0 1 390 200" strokeWidth="1.6"/></g>
            <g className="sp r"><circle cx="200" cy="200" r="170" strokeDasharray="40 14"/><circle cx="200" cy="200" r="70"/><circle cx="200" cy="130" r="22"/><circle cx="200" cy="270" r="22"/><circle cx="130" cy="200" r="22"/><circle cx="270" cy="200" r="22"/></g>
          </svg>
          <div className="sw" />
        </div>
        <div id="pts">
          {Array.from({ length: 28 }, (_, index) => (
            <i key={index} className={`pt${index % 4 === 0 ? ' g' : ''}`} style={{ left: `${(index * 37) % 100}%`, top: `${(index * 61) % 100}%`, width: `${index % 4 === 0 ? 3 : 2}px`, height: `${index % 4 === 0 ? 3 : 2}px`, '--d': `${4 + (index % 7)}s`, '--x': `${(index % 3 - 1) * 30}px`, '--y': `${(index % 5 - 2) * 24}px` } as React.CSSProperties} />
          ))}
        </div>
      </div>

      <main>
        <nav>
          <div className="logo sf">CEREVYN<b>.</b></div>
          <div className="nav-right">
            <span className="nav-email" title={user.email}>{user.email}</span>
            <button className="btn-out" type="button" onClick={handleLogout}>Sign out</button>
          </div>
        </nav>

        <section className="hero">
          <div className="eb"><i />YOUR CREATIVE STUDIO</div>
          <h1 className="sf"><span className="l">Turn your</span><span className="l">ideas into</span><br /><em>cinema.</em></h1>
          <p className="sub">Write the scene. CEREVYN shoots, assembles and renders it.</p>
        </section>

        <section className={`comp${generating ? ' gen' : ''}`} aria-labelledby="composerTitle">
          <div className="lens">
            <div className="bez" />
            <div className="lens-glint" />
            <div className="durs" role="group" aria-label="Video duration">
              {DURATIONS.map((seconds, index) => (
                <button key={seconds} type="button" aria-pressed={duration === seconds} data-d={seconds} style={{ '--a': index === 0 ? '-46deg' : index === 2 ? '46deg' : '0deg' } as React.CSSProperties} onClick={() => setDuration(seconds)}>{seconds}s</button>
              ))}
            </div>
            <div className="mk" style={{ '--ma': activeDurationAngle } as React.CSSProperties} />
            <div className="fadel">FADE IN</div>
            <textarea ref={promptRef} id="pr" maxLength={500} value={prompt} onChange={event => setPrompt(event.target.value)} placeholder="Write the opening scene of your film…" aria-label="Video prompt" aria-describedby="composerTitle" />
            <div className="cnt"><span>{prompt.length}</span> / 500</div>
          </div>

          <button className="go" type="button" onClick={handleGenerate} disabled={generating} aria-label="Generate video">
            <svg className="apert" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="10"/><path d="M12 2l4 7M22 12l-8 0M18 20l-4-7M6 20l4-7M2 12l8 0M6 4l4 7"/></svg>
            <span className="spin" aria-hidden="true" />
          </button>
          <div className="gl2" id="composerTitle">{generating ? 'Creating your film…' : 'Generate Video'}</div>
          <div className="err" role="alert" aria-live="polite">{error || statusMessage}</div>

          <div className="sg" role="list" aria-label="Prompt ideas">
            {IDEAS.map(idea => <button key={idea.label} type="button" role="listitem" data-t={idea.prompt} onClick={() => { setPrompt(idea.prompt); setError(''); promptRef.current?.focus() }}><span>{idea.icon}</span>{idea.label}</button>)}
          </div>

          <div className="fmt-row" role="group" aria-label="Video format">
            {FORMATS.map(format => (
              <button key={format.name} className="fmt-card" type="button" aria-pressed={aspectRatio === format.ratio && duration === format.duration} title={`${format.name} video`} onClick={() => { setAspectRatio(format.ratio); setDuration(format.duration) }}>
                <div className="fmt-icon"><FormatIcon name={format.name} /></div>
                <span className="fmt-name">{format.name}</span>
                <span className="fmt-detail">{format.ratio} · {format.duration}s</span>
                {format.name === 'Ad' && <span className="fmt-badge">NEW</span>}
              </button>
            ))}
          </div>
        </section>

        <section className="gal" aria-labelledby="galTitle">
          <div className="gh"><h2 className="sf" id="galTitle">Your <em>creations</em></h2><span className="tag" id="gc">{projects.length} films</span></div>
          <div className="grid video-card-grid" id="grid" aria-live="polite">
            {projects.map(project => <GalleryCard key={project.id} project={project} onDetails={() => void openDetails(project.id)} onDownload={() => void handleDownload(project.id, project.topic)} />)}
            {!loadingProjects && projects.length === 0 && <div className="empty">{LENS}<p>Your archive awaits its first film</p></div>}
            {loadingProjects && <div className="empty"><p>Loading your films…</p></div>}
          </div>
        </section>
        <p className="foot">CEREVYN · AI Video Studio</p>
      </main>
    </div>
  )
}

function FormatIcon({ name }: { name: string }) {
  if (name === 'Shorts') return <svg width="20" height="32" viewBox="0 0 20 32" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="1" y="1" width="18" height="30" rx="3"/><circle cx="10" cy="26" r="1.5" fill="currentColor" stroke="none"/></svg>
  if (name === 'Reels') return <svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="1" y="1" width="20" height="20" rx="5"/><circle cx="11" cy="11" r="4.5"/><circle cx="16.5" cy="5.5" r="1.2" fill="currentColor" stroke="none"/></svg>
  if (name === 'Ad') return <svg width="26" height="26" viewBox="0 0 26 26" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="1" y="1" width="24" height="24" rx="4"/><path d="M7 13h12M13 7v12" strokeLinecap="round"/></svg>
  return <svg width="32" height="22" viewBox="0 0 32 22" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true"><rect x="1" y="1" width="30" height="20" rx="3"/><path d="M12 7l9 4-9 4V7z" fill="currentColor" stroke="none"/></svg>
}

function GalleryCard({ project, onDetails, onDownload }: { project: ProjectResponse; onDetails: () => void; onDownload: () => void }) {
  const status = projectStatus(project)
  const done = project.status === 'done'
  const failed = project.status === 'failed'
  const title = (project.topic || 'Untitled').replace(/[.,;:!?]+$/, '').slice(0, 42)
  const date = project.created_at ? new Date(project.created_at).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' }) : '—'
  return (
    <div className="cw video-card">
      <div className="card video-card-shell">
        <div className="post video-card-preview" role="img" aria-label={`${status.text}: ${title}`}>
          <span className={`badge ${status.className}`}>{status.text}</span>{LENS}
        </div>
        <div className="stub video-card-info">
          <div className="ti">
            <b title={project.topic}>{title}</b>
            <small>{date} · {project.desired_duration_seconds}s · {project.aspect_ratio}</small>
            {!done && !failed && <><div className="pbar"><div className="pbar-fill" style={{ width: `${project.progress_percent}%` }} /></div><div className="stage-label">{project.current_stage || 'Queued'}</div></>}
            {failed && <div className="stage-label" style={{ color: '#c07070' }}>{project.error_message || 'Render failed'}</div>}
          </div>
          <div className="acts">
            <button className="bt pri" type="button" onClick={onDetails}>Details</button>
            {done && <button className="bt" type="button" onClick={onDownload}>Download</button>}
          </div>
        </div>
      </div>
    </div>
  )
}
