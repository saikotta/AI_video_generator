/**
 * GenerateForm.tsx — Redesigned to Lumen spec.
 *
 * Logic is identical to the original:
 *   - Same DURATION_OPTIONS (exact backend-supported values).
 *   - Same validation (topic.trim().length < 3).
 *   - Same disabled/loading states block duplicate submits.
 *   - Prompt is never cleared on error or loading.
 *
 * Visual changes:
 *   - Labelled textarea, 130px min-height, resizable, warm border + focus ring.
 *   - Placeholder per spec.
 *   - Character counter.
 *   - Duration selector styled warmly, same 7 durations.
 *   - "Generate video" button with play icon, primary blue.
 *   - Desktop: selector + button in one row. Mobile: stacked, full-width button.
 */

import { useState } from 'react'

interface Props {
  disabled: boolean
  onSubmit: (topic: string, duration: number) => void
  error: string
}

/* Exact durations the backend supports — do not add or remove. */
const DURATION_OPTIONS = [
  { label: '20 seconds', value: 20  },
  { label: '30 seconds', value: 30  },
  { label: '45 seconds', value: 45  },
  { label: '60 seconds', value: 60  },
  { label: '90 seconds', value: 90  },
  { label: '2 minutes',  value: 120 },
  { label: '3 minutes',  value: 180 },
]

export default function GenerateForm({ disabled, onSubmit, error }: Props) {
  const [topic,    setTopic]    = useState('')
  const [duration, setDuration] = useState(30)

  // Focus state for styling (inline styles can't do :focus-visible)
  const [topicFocused,    setTopicFocused]    = useState(false)
  const [durationFocused, setDurationFocused] = useState(false)
  const [btnHovered,      setBtnHovered]      = useState(false)

  const charCount = topic.length
  const canSubmit = !disabled && topic.trim().length >= 3

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!canSubmit) return
    // Prompt is NOT cleared — user keeps their text on error / loading
    onSubmit(topic.trim(), duration)
  }

  /* Dynamic styles */
  const textareaStyle: React.CSSProperties = {
    ...s.textarea,
    borderColor: topicFocused
      ? 'var(--primary)'
      : 'var(--border)',
    boxShadow: topicFocused
      ? '0 0 0 3px rgba(40,92,231,0.12)'
      : 'none',
    opacity:    disabled ? 0.6 : 1,
    cursor:     disabled ? 'not-allowed' : 'text',
  }

  const selectStyle: React.CSSProperties = {
    ...s.select,
    borderColor: durationFocused ? 'var(--primary)' : 'var(--border)',
    boxShadow:   durationFocused ? '0 0 0 3px rgba(40,92,231,0.12)' : 'none',
    opacity:     disabled ? 0.6 : 1,
    cursor:      disabled ? 'not-allowed' : 'pointer',
  }

  const btnStyle: React.CSSProperties = {
    ...s.btn,
    background: disabled
      ? '#8caaf4'
      : btnHovered
      ? 'var(--primary-hover)'
      : 'var(--primary)',
    cursor: disabled ? 'not-allowed' : 'pointer',
    transform: btnHovered && !disabled ? 'translateY(-1px)' : 'none',
    boxShadow: btnHovered && !disabled
      ? '0 4px 12px rgba(40,92,231,0.32)'
      : 'none',
  }

  return (
    <form onSubmit={handleSubmit} style={s.form} noValidate>

      {/* ── Textarea ── */}
      <div style={s.fieldGroup}>
        <label htmlFor="gf-topic" style={s.label}>
          Video description
        </label>
        <textarea
          id="gf-topic"
          value={topic}
          onChange={e => setTopic(e.target.value)}
          onFocus={() => setTopicFocused(true)}
          onBlur={() => setTopicFocused(false)}
          placeholder="Describe your story, the setting, camera movements, lighting, atmosphere, and sound…"
          disabled={disabled}
          style={textareaStyle}
          aria-describedby="gf-topic-hint"
          aria-required="true"
          aria-invalid={topic.trim().length > 0 && topic.trim().length < 3 ? 'true' : undefined}
          required
        />
        <div style={s.textareaFooter} id="gf-topic-hint">
          <span
            style={{
              ...s.charCount,
              color: charCount === 0
                ? 'var(--text-muted)'
                : charCount < 3
                ? 'var(--error)'
                : 'var(--muted)',
            }}
          >
            {charCount > 0 && charCount < 3
              ? 'Add a few more words…'
              : `${charCount} characters`}
          </span>
        </div>
      </div>

      {/* ── Duration + Button row ── */}
      <div style={s.row}>

        {/* Duration selector */}
        <div style={s.durationGroup}>
          <label htmlFor="gf-duration" style={s.label}>
            Duration
          </label>
          <select
            id="gf-duration"
            value={duration}
            onChange={e => setDuration(Number(e.target.value))}
            onFocus={() => setDurationFocused(true)}
            onBlur={() => setDurationFocused(false)}
            disabled={disabled}
            style={selectStyle}
            aria-label="Select video duration"
          >
            {DURATION_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>

        {/* Generate button */}
        <div style={s.btnGroup}>
          {/* Invisible label so the button column aligns with the selector column */}
          <span style={s.hiddenLabel} aria-hidden="true">‍</span>
          <button
            id="gf-submit"
            type="submit"
            disabled={!canSubmit}
            aria-busy={disabled}
            aria-label={disabled ? 'Generating video…' : 'Generate video'}
            style={btnStyle}
            onMouseEnter={() => setBtnHovered(true)}
            onMouseLeave={() => setBtnHovered(false)}
            onFocus={() => setBtnHovered(true)}
            onBlur={() => setBtnHovered(false)}
          >
            {disabled ? <SpinnerIcon /> : <PlayIcon />}
            {disabled ? 'Generating…' : 'Generate video'}
          </button>
        </div>
      </div>

      {/* ── Error ── */}
      {error && (
        <p role="alert" aria-live="assertive" style={s.error}>
          {error}
        </p>
      )}

    </form>
  )
}

/* ── Icon helpers ── */
function PlayIcon() {
  return (
    <svg
      width="15" height="15" viewBox="0 0 15 15" fill="none"
      style={{ flexShrink: 0 }}
      aria-hidden="true"
    >
      <polygon points="3,1.5 3,13.5 13.5,7.5" fill="white" />
    </svg>
  )
}

function SpinnerIcon() {
  return (
    <svg
      width="16" height="16" viewBox="0 0 24 24" fill="none"
      style={{ animation: 'spin 0.8s linear infinite', flexShrink: 0 }}
      aria-hidden="true"
    >
      <circle cx="12" cy="12" r="10" stroke="rgba(255,255,255,0.3)" strokeWidth="3" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="white" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

/* ── Styles ── */
const s: Record<string, React.CSSProperties> = {
  form: {
    display: 'flex',
    flexDirection: 'column',
    gap: '20px',
  },
  fieldGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '7px',
  },
  label: {
    fontSize: '13px',
    fontWeight: 600,
    color: 'var(--text)',
    letterSpacing: '0.01em',
  },
  hiddenLabel: {
    fontSize: '13px',
    display: 'block',
    marginBottom: '7px',
    visibility: 'hidden',
  },
  textarea: {
    width: '100%',
    minHeight: '130px',
    padding: '13px 14px',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    fontSize: '15px',
    color: 'var(--text)',
    background: 'var(--bg)',
    resize: 'vertical',
    lineHeight: 1.55,
    transition: 'border-color 0.15s, box-shadow 0.15s, opacity 0.15s',
  },
  textareaFooter: {
    display: 'flex',
    justifyContent: 'flex-end',
  },
  charCount: {
    fontSize: '12px',
    transition: 'color 0.15s',
  },
  /* Desktop: selector + button side by side. Mobile: wrap to stack. */
  row: {
    display: 'flex',
    gap: '16px',
    alignItems: 'flex-end',
    flexWrap: 'wrap',
  },
  durationGroup: {
    display: 'flex',
    flexDirection: 'column',
    gap: '7px',
    flex: '0 0 auto',
  },
  select: {
    padding: '11px 14px',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius-sm)',
    fontSize: '14px',
    fontWeight: 500,
    color: 'var(--text)',
    background: 'var(--card)',
    transition: 'border-color 0.15s, box-shadow 0.15s, opacity 0.15s',
    appearance: 'auto',
    minWidth: '150px',
  },
  btnGroup: {
    display: 'flex',
    flexDirection: 'column',
    flex: '1 1 160px',
  },
  btn: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: '9px',
    padding: '12px 22px',
    background: 'var(--primary)',
    color: '#fff',
    borderRadius: 'var(--radius-sm)',
    fontSize: '15px',
    fontWeight: 600,
    whiteSpace: 'nowrap',
    transition: 'background 0.15s, transform 0.12s, box-shadow 0.15s, opacity 0.15s',
    width: '100%',
  },
  error: {
    padding: '12px 14px',
    background: 'var(--error-bg)',
    color: 'var(--error)',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid rgba(185,28,28,0.18)',
    fontSize: '14px',
    lineHeight: 1.5,
  },
}
