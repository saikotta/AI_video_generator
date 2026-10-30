/**
 * ProjectCard.tsx — Redesigned Lumen project card.
 *
 * Logic is identical to original:
 *   - isDone / isFailed / inProgress states.
 *   - onClick → openDetail, onDownload only when done.
 *   - isActive ring highlight (active generation job).
 *
 * Visual additions:
 *   - 16:9 warm placeholder thumbnail (no thumbnail_url in API).
 *   - Hover state: border darkens, shadow lifts, thumbnail zooms to 1.05
 *     (clipped to its container). Text and buttons stay still. 220ms ease.
 *   - Status chip (StatusBadge), duration badge, timestamp, title excerpt.
 *   - In-progress: ProgressBar + stage label.
 *   - Failed: muted error message.
 */

import { useState } from 'react'
import type { ProjectResponse } from '../api/types'
import StatusBadge from './StatusBadge'
import ProgressBar from './ProgressBar'

interface Props {
  project:    ProjectResponse
  isActive:   boolean
  onClick:    () => void
  onDownload: () => void
}

export default function ProjectCard({ project: p, isActive, onClick, onDownload }: Props) {
  const [hovered, setHovered] = useState(false)

  const isDone     = p.status === 'done'
  const isFailed   = p.status === 'failed'
  const inProgress = !isDone && !isFailed

  function fmt(iso: string) {
    return new Date(iso).toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    })
  }

  function fmtDuration(s: number) {
    return s >= 60 ? `${s / 60}min` : `${s}s`
  }

  /* Dynamic card styles based on hover / active */
  const cardStyle: React.CSSProperties = {
    ...s.card,
    borderColor: isActive
      ? 'var(--primary)'
      : hovered
      ? 'var(--border-hover)'
      : 'var(--border)',
    boxShadow: isActive
      ? '0 0 0 2px rgba(40,92,231,0.18), var(--shadow)'
      : hovered
      ? 'var(--shadow-hover)'
      : 'var(--shadow)',
    transition: 'border-color 220ms ease, box-shadow 220ms ease',
  }

  return (
    <article
      style={cardStyle}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {/* ── 16:9 Thumbnail ── */}
      <div style={s.thumbContainer}>
        <div
          style={{
            ...s.thumb,
            transform: hovered ? 'scale(1.05)' : 'scale(1)',
            transition: 'transform 220ms ease',
          }}
        >
          <ThumbnailPlaceholder done={isDone} failed={isFailed} />
        </div>

        {/* Duration pill overlay */}
        <span style={s.durationPill} aria-label={`Duration: ${fmtDuration(p.desired_duration_seconds)}`}>
          {fmtDuration(p.desired_duration_seconds)}
        </span>
      </div>

      {/* ── Card body ── */}
      <div style={s.body}>

        {/* Status + date */}
        <div style={s.topRow}>
          <StatusBadge status={p.status} />
          <span style={s.date}>{fmt(p.created_at)}</span>
        </div>

        {/* Title / prompt excerpt */}
        <p
          style={s.topic}
          title={p.topic}
          onClick={onClick}
          tabIndex={0}
          role="button"
          aria-label={`Open project: ${p.topic}`}
          onKeyDown={e => e.key === 'Enter' && onClick()}
        >
          {p.topic.length > 90 ? p.topic.slice(0, 87) + '…' : p.topic}
        </p>

        {/* In-progress: stage + progress bar */}
        {inProgress && (
          <div style={s.progressWrap}>
            <div style={s.stageRow}>
              <span style={s.stage}>{p.current_stage ?? 'Queued'}</span>
              <span style={s.pct}>{p.progress_percent}%</span>
            </div>
            <ProgressBar value={p.progress_percent} />
          </div>
        )}

        {/* Failed error */}
        {isFailed && p.error_message && (
          <p style={s.errorMsg} title={p.error_message}>
            {p.error_message.length > 80
              ? p.error_message.slice(0, 77) + '…'
              : p.error_message}
          </p>
        )}

        {/* Actions */}
        <div style={s.footer}>
          <button
            type="button"
            style={s.detailBtn}
            onClick={onClick}
            aria-label="View project details"
          >
            Details
          </button>
          {isDone && (
            <button
              type="button"
              style={s.downloadBtn}
              onClick={onDownload}
              aria-label="Download MP4"
            >
              ↓ Download
            </button>
          )}
        </div>

      </div>
    </article>
  )
}

/* ── Thumbnail placeholder ── */
function ThumbnailPlaceholder({ done, failed }: { done: boolean; failed: boolean }) {
  const bg = failed
    ? 'linear-gradient(135deg, #fdf0f0 0%, #f9e5e5 100%)'
    : done
    ? 'linear-gradient(135deg, #eef4f0 0%, #e2ede6 100%)'
    : 'linear-gradient(135deg, #F3EBDD 0%, #EDE3D4 50%, #E8DDD0 100%)'

  const iconColor = failed ? '#C4BAB0' : done ? '#B8D4C0' : '#C4BAB0'

  return (
    <div style={{ ...s.placeholder, background: bg }}>
      {/* Film frame / play icon */}
      <svg
        width="36" height="36" viewBox="0 0 36 36" fill="none"
        aria-hidden="true" style={{ opacity: 0.8 }}
      >
        {done ? (
          /* Completed: play triangle */
          <>
            <rect x="2"  y="6"  width="32" height="24" rx="4" stroke={iconColor} strokeWidth="1.5" fill="none" />
            <polygon points="13,12 13,24 25,18" fill={iconColor} />
          </>
        ) : failed ? (
          /* Failed: X mark */
          <>
            <rect x="2"  y="6"  width="32" height="24" rx="4" stroke={iconColor} strokeWidth="1.5" fill="none" />
            <path d="M13 13l10 10M23 13l-10 10" stroke={iconColor} strokeWidth="1.8" strokeLinecap="round" />
          </>
        ) : (
          /* Processing or queued: film reel */
          <>
            <rect x="2"  y="6"  width="32" height="24" rx="4" stroke={iconColor} strokeWidth="1.5" fill="none" />
            <rect x="2"  y="9"  width="6"  height="4"  rx="1" fill={iconColor} />
            <rect x="2"  y="16" width="6"  height="4"  rx="1" fill={iconColor} />
            <rect x="2"  y="23" width="6"  height="4"  rx="1" fill={iconColor} />
            <rect x="28" y="9"  width="6"  height="4"  rx="1" fill={iconColor} />
            <rect x="28" y="16" width="6"  height="4"  rx="1" fill={iconColor} />
            <rect x="28" y="23" width="6"  height="4"  rx="1" fill={iconColor} />
            <circle cx="18" cy="18" r="5"  stroke={iconColor} strokeWidth="1.5" fill="none" />
          </>
        )}
      </svg>
    </div>
  )
}

/* ── Styles ── */
const s: Record<string, React.CSSProperties> = {
  card: {
    background: 'var(--card)',
    borderRadius: 'var(--radius)',
    border: '1px solid var(--border)',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
  },

  /* Thumbnail */
  thumbContainer: {
    position: 'relative',
    width: '100%',
    aspectRatio: '16 / 9',
    overflow: 'hidden',          /* clips the thumbnail zoom */
    flexShrink: 0,
  },
  thumb: {
    position: 'absolute',
    inset: 0,
    width: '100%',
    height: '100%',
    transformOrigin: 'center center',
    willChange: 'transform',
  },
  placeholder: {
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  durationPill: {
    position: 'absolute',
    bottom: '8px',
    right: '8px',
    background: 'rgba(27,24,20,0.65)',
    color: '#fff',
    fontSize: '11px',
    fontWeight: 600,
    letterSpacing: '0.02em',
    padding: '3px 8px',
    borderRadius: '99px',
    backdropFilter: 'blur(4px)',
    WebkitBackdropFilter: 'blur(4px)',
  },

  /* Body */
  body: {
    padding: '16px',
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
    flex: 1,
  },
  topRow: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: '8px',
  },
  date: {
    fontSize: '11px',
    color: 'var(--text-muted)',
    flexShrink: 0,
  },
  topic: {
    fontSize: '14px',
    fontWeight: 500,
    lineHeight: 1.45,
    color: 'var(--text)',
    cursor: 'pointer',
    wordBreak: 'break-word',
  },
  progressWrap: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  stageRow: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  stage: {
    fontSize: '12px',
    color: 'var(--primary)',
    fontWeight: 500,
  },
  pct: {
    fontSize: '12px',
    color: 'var(--primary)',
    fontWeight: 600,
  },
  errorMsg: {
    fontSize: '12px',
    color: 'var(--error)',
    background: 'var(--error-bg)',
    padding: '6px 8px',
    borderRadius: '6px',
    lineHeight: 1.4,
  },
  footer: {
    display: 'flex',
    gap: '8px',
    marginTop: 'auto',
    paddingTop: '4px',
  },
  detailBtn: {
    padding: '6px 12px',
    fontSize: '12px',
    fontWeight: 500,
    background: 'none',
    border: '1px solid var(--border)',
    borderRadius: '8px',
    color: 'var(--muted)',
    cursor: 'pointer',
    transition: 'border-color 0.15s, color 0.15s',
  },
  downloadBtn: {
    padding: '6px 12px',
    fontSize: '12px',
    fontWeight: 600,
    background: 'var(--primary)',
    color: '#fff',
    borderRadius: '8px',
    cursor: 'pointer',
    border: 'none',
    transition: 'background 0.15s',
  },
}
