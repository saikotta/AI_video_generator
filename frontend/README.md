# CEREVYN frontend

The frontend is a React 18 and TypeScript single-page application built with Vite. React Router renders the sign-in, dashboard, and storyboard route; project and authentication data continue to use the FastAPI API.

## Run locally

```sh
npm install
npm run dev
```

Vite listens on `http://localhost:5174`. The FastAPI backend should be running on `http://127.0.0.1:8001`; Vite proxies `/auth`, `/projects`, and `/health` to it.

## Routes

- `/login` — sign in or create an account
- `/dashboard` — authenticated video studio and project library
- `/story-board` — reserved route; this repository has no storyboard HTML page or storyboard feature yet

The root route redirects to `/dashboard` when authenticated and `/login` otherwise. All routes are handled by React Router, so Vite serves the app shell on direct route access and refresh.

## Build

```sh
npm run build
npm run preview
```

Page components live in `src/pages`, reusable UI in `src/components`, and the existing API client and endpoint functions in `src/api`.
