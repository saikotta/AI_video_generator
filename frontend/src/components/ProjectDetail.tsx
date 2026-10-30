import { useState } from 'react'
import type { ProjectDetailResponse } from '../api/types'
import StatusBadge from './StatusBadge'
import ProgressBar from './ProgressBar'

interface Props {
  project: ProjectDetailResponse | null
  loading: boolean
  onBack: () => void
  onDownload: (id: string, topic: string) => void
}

export default function ProjectDetail({ project: p, loading, onBack, onDownload }: Props) {
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [downloading, setDownloading] = useState(false)

  async function handleDownload() {
    if (!p) return
    setDownloading(true)
    try {
      onDownload(p.id, p.topic)
    } finally {
      setDownloading(false)
    }
  }

  function fmt(iso: string | null) {
    if (!iso) return '—'
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  }

  return (
    <div style={s.page}>
      <header style={s.header}>
        <div style={s.headerInner}>
          <button onClick={onBack} style={s.backBtn} aria-label="Back to dashboard">
            ← Back
          </button>
          <h1 style={s.title}>Project details</h1>
        </div>
      </header>

      <main style={s.main}>
        {loading && <p style={s.muted}>Loading…</p>}
        {!loading && !p && <p style={s.muted}>Project not found.</p>}
        {!loading && p && (
          <>
            {/* Meta */}
            <div style={s.card}>
              <div style={s.metaRow}>
                <StatusBadge status={p.status} />
                <span style={s.duration}>{p.desired_duration_seconds}s requested</span>
              </div>
              <h2 style={s.topic}>{p.topic}</h2>

              {p.status !== 'done' && p.status !== 'failed' && (
                <div style={s.progress}>
                  <div style={s.progressHeader}>
                    <span style={s.stage}>{p.current_stage ?? 'Queued'}</span>
                    <span style={s.pct}>{p.progress_percent}%</span>
                  </div>
                  <ProgressBar value={p.progress_percent} />
                </div>
              )}

              {p.status === 'failed' && p.error_message && (
                <div style={s.errorBox} role="alert">
                  <strong>Error:</strong> {p.error_message}
                </div>
              )}

              <div style={s.timestamps}>
                <span>Created: {fmt(p.created_at)}</span>
                {p.completed_at && <span>Completed: {fmt(p.completed_at)}</span>}
              </div>

              {p.status === 'done' && (
                <div style={s.actionRow}>
                  <button
                    onClick={handleDownload}
                    disabled={downloading}
                    style={s.downloadBtn}
                    aria-label="Download MP4 video"
                  >
                    {downloading ? 'Downloading…' : '↓ Download MP4'}
                  </button>
                </div>
              )}
            </div>

            {/* Scenes */}
            {p.scenes.length > 0 && (
              <div style={s.card}>
                <h3 style={s.sectionHeading}>Scenes ({p.scenes.length})</h3>
                <div style={s.sceneList}>
                  {p.scenes.map((sc) => (
                    <div key={sc.id} style={s.scene}>
                      <div style={s.sceneHeader}>
                        <span style={s.sceneNum}>Scene {sc.scene_order}</span>
                        {sc.duration_seconds != null && (
                          <span style={s.sceneDur}>{sc.duration_seconds.toFixed(1)}s</span>
                        )}
                        <StatusBadge status={sc.status} />
                      </div>
                      <p style={s.sceneText}>{sc.scene_text}</p>
                      {sc.visual_prompt && (
                        <p style={s.visualPrompt}>🎬 {sc.visual_prompt}</p>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  page: { minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' },
  header: {
    background: 'var(--surface)', borderBottom: '1px solid var(--border)',
    position: 'sticky', top: 0, zIndex: 10,
  },
  headerInner: {
    maxWidth: '960px', margin: '0 auto', padding: '0 24px',
    height: '56px', display: 'flex', alignItems: 'center', gap: '16px',
  },
  backBtn: {
    background: 'none', fontSize: '14px', fontWeight: 600,
    color: 'var(--primary)', cursor: 'pointer',
  },
  title: { fontSize: '17px', fontWeight: 700 },
  main: { flex: 1, maxWidth: '960px', margin: '0 auto', padding: '28px 24px', width: '100%', display: 'flex', flexDirection: 'column', gap: '20px' },
  card: {
    background: 'var(--surface)', borderRadius: 'var(--radius)',
    boxShadow: 'var(--shadow)', padding: '24px', display: 'flex', flexDirection: 'column', gap: '14px',
  },
  metaRow: { display: 'flex', gap: '12px', alignItems: 'center' },
  duration: { fontSize: '13px', color: 'var(--text-muted)' },
  topic: { fontSize: '18px', fontWeight: 700, lineHeight: 1.4 },
  progress: { display: 'flex', flexDirection: 'column', gap: '8px' },
  progressHeader: { display: 'flex', justifyContent: 'space-between' },
  stage: { fontSize: '14px', fontWeight: 600, color: 'var(--primary)' },
  pct: { fontSize: '14px', fontWeight: 600, color: 'var(--primary)' },
  errorBox: {
    padding: '12px', background: 'var(--danger-light)', color: 'var(--danger)',
    borderRadius: 'var(--radius-sm)', fontSize: '14px',
  },
  timestamps: { display: 'flex', gap: '20px', flexWrap: 'wrap', fontSize: '13px', color: 'var(--text-muted)' },
  actionRow: { paddingTop: '4px' },
  downloadBtn: {
    padding: '10px 20px', background: 'var(--primary)', color: '#fff',
    borderRadius: 'var(--radius-sm)', fontSize: '15px', fontWeight: 600,
  },
  sectionHeading: { fontSize: '16px', fontWeight: 700 },
  sceneList: { display: 'flex', flexDirection: 'column', gap: '12px' },
  scene: {
    padding: '14px', border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius-sm)', display: 'flex', flexDirection: 'column', gap: '8px',
  },
  sceneHeader: { display: 'flex', gap: '10px', alignItems: 'center' },
  sceneNum: { fontSize: '13px', fontWeight: 700, color: 'var(--text-secondary)' },
  sceneDur: { fontSize: '12px', color: 'var(--text-muted)' },
  sceneText: { fontSize: '14px', lineHeight: 1.5, color: 'var(--text)' },
  visualPrompt: { fontSize: '13px', color: 'var(--text-secondary)', fontStyle: 'italic' },
  muted: { color: 'var(--text-muted)', fontSize: '14px' },
}
