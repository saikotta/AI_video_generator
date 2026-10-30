import { Link } from 'react-router-dom'

export default function StoryBoardPage() {
  return (
    <main className="route-message">
      <p className="route-eyebrow">CEREVYN STUDIO</p>
      <h1>Story board</h1>
      <p>This project does not currently include a storyboard page.</p>
      <Link to="/dashboard" className="route-home-link">Return to dashboard</Link>
    </main>
  )
}
