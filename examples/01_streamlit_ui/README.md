Streamlit UI (Example)

Run a minimal UI using only Python.

Backend

- Start the example backend first:
  - cd pooolify/examples/00_simple_agent
  - uvicorn main:app --reload --port 8000

Frontend (Streamlit)

- cd pooolify/examples/01_streamlit_ui
- pip install -r requirements.txt
- streamlit run app.py --server.address 0.0.0.0 --server.port 5173

Then open http://localhost:5173

Environment (optional)

- POOOLIFY_API_BASE (default http://127.0.0.1:8000)
- POOOLIFY_API_TOKEN
- POOOLIFY_SESSION_ID (default demo)
- POOOLIFY_MODEL (default gpt-5)

Notes

- In dev, if API_TOKEN is unset in backend, auth is skipped.
- The UI calls POST /v1/chat and polls GET /v1/sessions/{id}/conversation.
