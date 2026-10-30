interface Props {
  value: number // 0–100
}

export default function ProgressBar({ value }: Props) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div
      role="progressbar"
      aria-valuenow={pct}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-label={`Progress: ${pct}%`}
      style={s.track}
    >
      <div
        style={{
          ...s.fill,
          width: `${pct}%`,
          transition: pct === 0 ? 'none' : 'width 0.6s ease',
        }}
      />
    </div>
  )
}

const s: Record<string, React.CSSProperties> = {
  track: {
    height: '8px', background: '#dbeafe',
    borderRadius: '99px', overflow: 'hidden',
  },
  fill: {
    height: '100%', background: 'var(--primary)',
    borderRadius: '99px',
  },
}
