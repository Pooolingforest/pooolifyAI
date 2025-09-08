Quickstart (scaffolded project)

1. Create .env

```
LLM_OPENAI_API_KEY=sk-...
APP_ENV=dev
```

2. Install and run the backend

```
uv sync
uv run uvicorn main:app --reload
```
