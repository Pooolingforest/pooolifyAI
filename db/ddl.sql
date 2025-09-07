-- Sessions
CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  created_at TIMESTAMPTZ DEFAULT now(),
  metadata JSONB NULL
);

-- Messages
CREATE TABLE IF NOT EXISTS chat_messages (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  is_internal BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- LLM call costs
CREATE TABLE IF NOT EXISTS ai_call_costs (
  id BIGSERIAL PRIMARY KEY,
  request_id TEXT NOT NULL, session_id TEXT,
  provider TEXT NOT NULL, model TEXT NOT NULL,
  purpose TEXT NOT NULL,
  input_tokens INT, output_tokens INT, total_tokens INT,
  cost_amount NUMERIC(16,6), currency TEXT NOT NULL DEFAULT 'USD',
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Plan executions
CREATE TABLE IF NOT EXISTS plan_executions (
  id BIGSERIAL PRIMARY KEY,
  request_id TEXT NOT NULL UNIQUE, session_id TEXT NOT NULL,
  initial_plan JSONB, status TEXT NOT NULL DEFAULT 'running',
  started_at TIMESTAMPTZ DEFAULT now(), ended_at TIMESTAMPTZ,
  final_result JSONB
);

-- Tool calls
CREATE TABLE IF NOT EXISTS tool_calls (
  id BIGSERIAL PRIMARY KEY,
  request_id TEXT NOT NULL, agent_name TEXT NOT NULL, tool_name TEXT NOT NULL,
  input JSONB, output JSONB,
  success BOOLEAN, latency_ms INT,
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Message feedback
CREATE TABLE IF NOT EXISTS message_feedback (
  id BIGSERIAL PRIMARY KEY,
  message_id BIGINT NOT NULL REFERENCES chat_messages(id) ON DELETE SET NULL,
  rating SMALLINT, comment TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

