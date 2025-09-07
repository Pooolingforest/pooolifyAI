import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  getConversation,
  isCompleted,
  postChat,
  type ConversationDTO,
} from "./api";

type Settings = {
  apiBase: string;
  token: string;
  sessionId: string;
  model: string;
  autoRefresh: boolean;
  refreshMs: number;
  showInternal: boolean;
};

const defaultSettings: Settings = {
  apiBase:
    (typeof window !== "undefined" && (window as any).POOOLIFY_API_BASE) ||
    "http://127.0.0.1:8000",
  token:
    (typeof window !== "undefined" && (window as any).POOOLIFY_API_TOKEN) || "",
  sessionId:
    (typeof window !== "undefined" && (window as any).POOOLIFY_SESSION_ID) ||
    "demo",
  model:
    (typeof window !== "undefined" && (window as any).POOOLIFY_MODEL) ||
    "gpt-5",
  autoRefresh: true,
  refreshMs: 1000,
  showInternal: false,
};

export default function App() {
  const [settings, setSettings] = useState<Settings>(defaultSettings);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [conv, setConv] = useState<ConversationDTO | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  const processing = useMemo(() => !!conv?.current_request_id, [conv]);

  const refreshConversation = useCallback(async () => {
    try {
      const data = await getConversation({
        apiBase: settings.apiBase,
        token: settings.token,
        sessionId: settings.sessionId,
      });
      setConv(data);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  }, [settings.apiBase, settings.token, settings.sessionId]);

  // initial fetch
  useEffect(() => {
    refreshConversation();
  }, [refreshConversation]);

  // auto refresh loop while processing
  useEffect(() => {
    if (!settings.autoRefresh || !processing) return;
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = window.setInterval(() => {
      refreshConversation();
    }, Math.max(250, settings.refreshMs));
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      timerRef.current = null;
    };
  }, [
    settings.autoRefresh,
    settings.refreshMs,
    processing,
    refreshConversation,
  ]);

  const send = useCallback(async () => {
    if (!prompt.trim()) return;
    setLoading(true);
    setError(null);
    try {
      await postChat({
        apiBase: settings.apiBase,
        token: settings.token,
        sessionId: settings.sessionId,
        query: prompt.trim(),
        model: settings.model,
      });
      // immediate polling until completion or short timeout (optimistic)
      const start = Date.now();
      while (Date.now() - start < 120_000) {
        const data = await getConversation({
          apiBase: settings.apiBase,
          token: settings.token,
          sessionId: settings.sessionId,
        });
        setConv(data);
        if (!data.current_request_id || isCompleted(data)) break;
        await new Promise((r) =>
          setTimeout(r, Math.max(250, settings.refreshMs))
        );
      }
    } catch (e: any) {
      setError(String(e?.message || e));
    } finally {
      setLoading(false);
      setPrompt("");
      refreshConversation();
    }
  }, [
    prompt,
    settings.apiBase,
    settings.token,
    settings.sessionId,
    settings.model,
    settings.refreshMs,
    refreshConversation,
  ]);

  return (
    <div className="min-h-full bg-gray-50 text-gray-900">
      <div className="mx-auto max-w-6xl p-4">
        <header className="mb-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold">pooolify – Web UI</h1>
          <a
            className="text-sm text-blue-600 hover:underline"
            href="https://github.com/Pooolingforest/pooolifyAI"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </header>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <aside className="md:col-span-1 space-y-3">
            <div className="rounded-lg border bg-white p-3 shadow-sm">
              <div className="mb-2 text-sm font-medium">Settings</div>
              <div className="space-y-2">
                <label className="block text-xs text-gray-600">API Base</label>
                <input
                  className="w-full rounded border px-2 py-1"
                  value={settings.apiBase}
                  onChange={(e) =>
                    setSettings((s) => ({ ...s, apiBase: e.target.value }))
                  }
                />
                <label className="block text-xs text-gray-600">API Token</label>
                <input
                  className="w-full rounded border px-2 py-1"
                  value={settings.token}
                  onChange={(e) =>
                    setSettings((s) => ({ ...s, token: e.target.value }))
                  }
                  placeholder="optional"
                />
                <label className="block text-xs text-gray-600">
                  Session ID
                </label>
                <input
                  className="w-full rounded border px-2 py-1"
                  value={settings.sessionId}
                  onChange={(e) =>
                    setSettings((s) => ({ ...s, sessionId: e.target.value }))
                  }
                />
                <label className="block text-xs text-gray-600">Model</label>
                <select
                  className="w-full rounded border px-2 py-1"
                  value={settings.model}
                  onChange={(e) =>
                    setSettings((s) => ({ ...s, model: e.target.value }))
                  }
                >
                  <option value="gpt-5">gpt-5</option>
                  <option value="gpt-5-high">gpt-5-high</option>
                </select>
                <div className="flex items-center justify-between pt-2">
                  <label className="flex items-center gap-2 text-xs">
                    <input
                      type="checkbox"
                      checked={settings.autoRefresh}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          autoRefresh: e.target.checked,
                        }))
                      }
                    />
                    Auto refresh
                  </label>
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-600">Interval</span>
                    <input
                      className="w-24 rounded border px-2 py-1"
                      type="number"
                      min={250}
                      max={5000}
                      step={100}
                      value={settings.refreshMs}
                      onChange={(e) =>
                        setSettings((s) => ({
                          ...s,
                          refreshMs: Number(e.target.value),
                        }))
                      }
                    />
                    <span className="text-xs">ms</span>
                  </div>
                </div>
                <label className="flex items-center gap-2 text-xs">
                  <input
                    type="checkbox"
                    checked={settings.showInternal}
                    onChange={(e) =>
                      setSettings((s) => ({
                        ...s,
                        showInternal: e.target.checked,
                      }))
                    }
                  />
                  Show internal thoughts
                </label>
                <button
                  className="mt-2 w-full rounded bg-gray-800 px-3 py-1 text-white hover:bg-gray-700"
                  onClick={refreshConversation}
                >
                  Refresh
                </button>
              </div>
            </div>
            {error && (
              <div className="rounded border border-red-300 bg-red-50 p-2 text-sm text-red-700">
                {error}
              </div>
            )}
          </aside>

          <main className="md:col-span-2 space-y-4">
            <section className="rounded-lg border bg-white p-3 shadow-sm">
              <div className="mb-2 text-sm font-medium">Conversation</div>
              {!conv ? (
                <div className="text-sm text-gray-600">
                  No conversation yet. Send a message.
                </div>
              ) : (
                <div className="space-y-2">
                  {conv.conversation.map((msg: any, idx: number) => {
                    const type = msg.type;
                    const role =
                      type === "MESSAGE_TYPE_HUMAN"
                        ? "You"
                        : type === "MESSAGE_TYPE_AI"
                        ? msg.agent || "AI"
                        : "System";
                    const content = msg.content || {};
                    const text =
                      content.answer ||
                      content.error ||
                      content.completion ||
                      "";
                    const thought = content.thought || "";
                    const plan = content.plan || "";
                    const route = content.route || "";
                    const decision = content.decision || "";
                    const time = msg.timestamp || "";
                    return (
                      <div key={idx} className="rounded border p-2">
                        <div className="mb-1 text-xs text-gray-600">
                          <span className="font-semibold">{role}</span> · {time}
                        </div>
                        {text && (
                          <div className="whitespace-pre-wrap text-sm">
                            {text}
                          </div>
                        )}
                        {settings.showInternal &&
                          (thought || plan || route || decision) && (
                            <details className="mt-2">
                              <summary className="cursor-pointer text-xs text-gray-600">
                                Internal
                              </summary>
                              {thought && (
                                <pre className="mt-1 overflow-auto rounded bg-gray-50 p-2 text-xs">
                                  {thought}
                                </pre>
                              )}
                              {plan && (
                                <pre className="mt-1 overflow-auto rounded bg-gray-50 p-2 text-xs">
                                  {plan}
                                </pre>
                              )}
                              {route && (
                                <pre className="mt-1 overflow-auto rounded bg-gray-50 p-2 text-xs">
                                  {route}
                                </pre>
                              )}
                              {decision && (
                                <pre className="mt-1 overflow-auto rounded bg-gray-50 p-2 text-xs">
                                  {decision}
                                </pre>
                              )}
                            </details>
                          )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            <section className="rounded-lg border bg-white p-3 shadow-sm">
              <div className="mb-2 text-sm font-medium">Compose</div>
              <textarea
                className="h-28 w-full resize-none rounded border p-2"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
              />
              <div className="mt-2 flex gap-2">
                <button
                  disabled={!prompt.trim() || loading}
                  onClick={send}
                  className="rounded bg-blue-600 px-3 py-1 text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {loading ? "Sending…" : "Send"}
                </button>
                <button
                  onClick={refreshConversation}
                  className="rounded bg-gray-200 px-3 py-1 hover:bg-gray-300"
                >
                  Refresh
                </button>
              </div>
              {processing && (
                <div className="mt-2 text-xs text-gray-600">
                  Processing… auto-refreshing
                </div>
              )}
            </section>
          </main>
        </div>

        <footer className="mt-6 text-center text-xs text-gray-500">
          Calls POST /v1/chat and polls GET /v1/sessions/{"{id}"}/conversation.
        </footer>
      </div>
    </div>
  );
}
