CHAT_SESSIONS = """
CREATE TABLE IF NOT EXISTS chat_sessions (
  id TEXT PRIMARY KEY,
  message_count INTEGER DEFAULT 0,
  last_message_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  metadata JSONB NULL
);
"""



AI_CALL_COSTS = """
CREATE TABLE IF NOT EXISTS ai_call_costs (
  id BIGSERIAL PRIMARY KEY,
  request_id TEXT NOT NULL, session_id TEXT,
  provider TEXT NOT NULL, model TEXT NOT NULL,
  purpose TEXT NOT NULL,
  input_tokens INT, output_tokens INT, total_tokens INT,
  cache_input_tokens INT,
  created_at TIMESTAMPTZ DEFAULT now()
);
"""

CONVERSATION_MESSAGES = """
CREATE TABLE IF NOT EXISTS conversation_messages (
  id BIGSERIAL PRIMARY KEY,
  session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
  message_id TEXT NOT NULL,
  message_data JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(session_id, message_id)
);
"""

ALL_DDL = [
    CHAT_SESSIONS,
    AI_CALL_COSTS,
    CONVERSATION_MESSAGES,
]

