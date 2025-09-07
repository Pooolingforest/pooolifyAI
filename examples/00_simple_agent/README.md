00_simple_agent — Tools and env example

Files added

- tools/echo/index.py — simple echo tool
- tools/time/index.py — UTC time tool
- .env.example — environment template

Usage

1. Copy env and set your key

```
cp .env.example .env
export LLM_OPENAI_API_KEY=sk-...
```

2. Run the server (from repo root)

```
uvicorn pooolify.examples.00_simple_agent.main:app --reload
```

3. Call API

```
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","query":"Use tools if needed and greet me."}'
```

Notes

- Agents/tools are auto‑loaded from the folder structure. The sample `greeter` agent currently does not call tools by default; you can inject tools in `agent/greeter/index.py` by passing them via `tools`.
