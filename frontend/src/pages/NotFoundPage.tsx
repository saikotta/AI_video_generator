import { Link } from 'react-router-dom'

export default function NotFoundPage() {
  return (
    <main className="route-message">
      <p className="route-eyebrow">CEREVYN STUDIO</p>
      <h1>Page not found</h1>
      <p>That address does not match a page in the studio.</p>
      <Link to="/" className="route-home-link">Go to the studio</Link>
    </main>
  )
}
