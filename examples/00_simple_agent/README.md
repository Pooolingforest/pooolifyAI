00_simple_agent — Multi-agent pipeline (no external integrations)

Agents

- agent/action_extractor — extract actionable tasks
- agent/issue_extractor — extract open issues/risks
- agent/decision_extractor — extract explicit decisions
- agent/inference — infer owner/due when explicit
- agent/prioritizer — assign priorities/tags
- agent/formatter — produce final concise summary

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

3. Call API — provide meeting notes as the query

```
curl -X POST http://localhost:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","query":"Meeting notes: Alice will draft the spec by Friday. Backend migration risk flagged. Decision: use Postgres 16."}'
```

Notes

- Agents are auto‑loaded from the folder structure. The manager is configured to always schedule a 4‑step pipeline: parallel extractors → inference → prioritizer → formatter.
- No external tools are required for this example; all agents answer directly.
