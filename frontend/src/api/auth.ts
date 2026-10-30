import { api, setToken, getToken, clearToken } from './client'
import type { UserResponse } from './types'

/**
 * Login: POSTs credentials, then fetches the JWT from /auth/me via cookie,
 * and stores it in localStorage as a Bearer token for subsequent API calls.
 *
 * The backend sets an httponly cookie on login. We immediately call /auth/me
 * with credentials:include so the cookie is sent, which validates the session
 * and confirms the user is authenticated.
 */
export async function login(email: string, password: string): Promise<UserResponse> {
  // Step 1: login — backend sets httponly cookie AND returns token in body
  const resp = await api.post<{ message: string; token?: string }>('/auth/login', { email, password })
  // Step 2: store the Bearer token so all subsequent requests work across
  // origins (Vite dev server → backend proxy gap)
  if (resp.token) {
    setToken(resp.token)
  }
  // Step 3: fetch and return the user record
  const user = await api.get<UserResponse>('/auth/me')
  return user
}

export async function register(email: string, password: string): Promise<UserResponse> {
  return api.post<UserResponse>('/auth/register', { email, password })
}

export async function logout(): Promise<void> {
  try {
    await api.post<{ message: string }>('/auth/logout', {})
  } finally {
    clearToken()
  }
}

export async function getMe(): Promise<UserResponse | null> {
  try {
    // Race the request against a 5-second timeout so a slow backend
    // never leaves the app stuck on the loading spinner.
    const timeout = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error('timeout')), 5000)
    )
    return await Promise.race([
      api.get<UserResponse>('/auth/me'),
      timeout,
    ])
  } catch {
    return null
  }
}
