/**
 * API client — thin wrapper around fetch that:
 *  - Forwards the JWT Bearer token stored in localStorage.
 *  - Handles 401 by clearing the token and reloading.
 *  - Throws ApiError with the server detail message on non-2xx responses.
 */

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

const TOKEN_KEY = 'aivg_token'

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token)
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY)
}

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
  isDownload = false,
): Promise<T> {
  const headers: Record<string, string> = {}
  const token = getToken()
  const hadToken = token !== null          // capture BEFORE any clearToken()
  if (token) headers['Authorization'] = `Bearer ${token}`
  if (body !== undefined) headers['Content-Type'] = 'application/json'

  const resp = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
    credentials: 'include',
  })

  if (resp.status === 401) {
    clearToken()
    // Only reload if we had an active token that was just rejected (expired session).
    // On the initial /auth/me probe there is no token yet — just throw so the
    // caller can handle it (getMe returns null, app shows login page).
    if (hadToken) {
      window.location.reload()
    }
    throw new ApiError(401, 'Unauthorized')
  }

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const err = await resp.json()
      if (typeof err?.detail === 'string') {
        detail = err.detail
      } else if (Array.isArray(err?.detail)) {
        // FastAPI validation errors: [{msg, loc, type}, ...]
        detail = err.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ')
      } else if (err?.detail) {
        detail = JSON.stringify(err.detail)
      }
    } catch {}
    throw new ApiError(resp.status, detail)
  }

  if (isDownload) return resp.blob() as unknown as T
  if (resp.status === 204) return undefined as unknown as T
  return resp.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body: unknown) => request<T>('POST', path, body),
  download: (path: string) => request<Blob>('GET', path, undefined, true),
}
