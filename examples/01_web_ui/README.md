pooolify Web UI (Vite + Tailwind)

A minimal web UI built with React (Vite) and TailwindCSS that calls the pooolify API.

Backend

- Start the example backend first:
  - cd pooolify/examples/00_simple_agent
  - uvicorn main:app --reload --port 8000

Frontend (Vite + Tailwind)

- cd pooolify/examples/01_web_ui
- npm install
- npm run dev

Open http://localhost:5173

Environment (optional)

- Set these as browser globals or via reverse proxy env-injection (optional):
  - POOOLIFY_API_BASE (default http://127.0.0.1:8000)
  - POOOLIFY_API_TOKEN
  - POOOLIFY_SESSION_ID (default demo)
  - (No model selection in UI; backend manager config decides)

Notes

- In dev, if API_TOKEN is unset in backend and APP_ENV=dev, auth is skipped.
- The UI posts to POST /v1/chat and polls GET /v1/sessions/{session_id}/conversation at a configurable interval.
