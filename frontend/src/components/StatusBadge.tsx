/**
 * StatusBadge.tsx — Updated to warm Lumen palette.
 * Colors are muted and harmonious (no loud saturated hues).
 */

interface Props { status: string }

const STATUS_MAP: Record<string, { label: string; bg: string; color: string }> = {
  queued:             { label: 'Queued',          bg: '#F3F0E9', color: '#6D655A' },
  generating_script:  { label: 'Writing script',  bg: '#EBF0FD', color: '#285CE7' },
  splitting_scenes:   { label: 'Splitting scenes', bg: '#EBF0FD', color: '#285CE7' },
  generating_visuals: { label: 'Sourcing visuals', bg: '#EBF0FD', color: '#285CE7' },
  generating_voice:   { label: 'Narrating',        bg: '#FEF9EC', color: '#92650A' },
  assembling:         { label: 'Assembling',        bg: '#F3EBFD', color: '#7C3AED' },
  done:               { label: 'Completed',         bg: '#ECFDF5', color: '#276749' },
  failed:             { label: 'Failed',            bg: '#FEF2F2', color: '#B91C1C' },
}

export default function StatusBadge({ status }: Props) {
  const cfg = STATUS_MAP[status] ?? { label: status, bg: '#F3F0E9', color: '#6D655A' }
  return (
    <span style={{
      display: 'inline-block',
      padding: '3px 9px',
      borderRadius: '99px',
      fontSize: '11px',
      fontWeight: 600,
      letterSpacing: '0.02em',
      background: cfg.bg,
      color: cfg.color,
      flexShrink: 0,
    }}>
      {cfg.label}
    </span>
  )
}
