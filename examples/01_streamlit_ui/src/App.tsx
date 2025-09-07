import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
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
  const status = processing ? "streaming" : "idle";

  const lastAssistantMessageId = useMemo(() => {
    if (!conv?.conversation?.length) return null as string | null;
    for (let i = conv.conversation.length - 1; i >= 0; i--) {
      const msg = conv.conversation[i];
      if (msg?.type === "MESSAGE_TYPE_AI" && msg?.bubbleId)
        return msg.bubbleId as string;
    }
    return null as string | null;
  }, [conv]);

  const latestThought = useMemo(() => {
    if (!conv?.conversation?.length) return "";
    for (let i = conv.conversation.length - 1; i >= 0; i--) {
      const msg = conv.conversation[i];
      if (msg?.type === "MESSAGE_TYPE_AI" && msg?.content?.thought)
        return String(msg.content.thought);
    }
    return "";
  }, [conv]);

  const getAgentLabel = useCallback((agent?: string) => {
    if (!agent) return "AI";
    // simple prettifier; customize as needed
    return agent.replace(/_/g, " ").replace(/\b\w/g, (m) => m.toUpperCase());
  }, []);

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
                        {msg.agent && (
                          <div className="flex items-center gap-2 mb-2 pb-1 border-b">
                            <span className="inline-block h-2 w-2 rounded-full bg-purple-500" />
                            <span className="text-xs font-medium text-purple-700">
                              {getAgentLabel(msg.agent)}
                            </span>
                          </div>
                        )}
                        {text && (
                          <div>
                            <ReactMarkdown
                              remarkPlugins={[remarkGfm]}
                              components={{
                                a: (props) => (
                                  <a
                                    className="underline text-blue-600 hover:text-blue-500"
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    {...props}
                                  />
                                ),
                                p: (props) => (
                                  <p
                                    className="whitespace-pre-wrap leading-6"
                                    {...props}
                                  />
                                ),
                                ul: (props) => (
                                  <ul
                                    className="list-disc pl-5 space-y-1"
                                    {...props}
                                  />
                                ),
                                ol: (props) => (
                                  <ol
                                    className="list-decimal pl-5 space-y-1"
                                    {...props}
                                  />
                                ),
                                li: (props) => (
                                  <li className="leading-6" {...props} />
                                ),
                                pre: (props) => (
                                  <pre
                                    className="rounded bg-gray-900 text-white p-3 text-[12px] overflow-x-auto"
                                    {...props}
                                  />
                                ),
                                code: ({ inline, children, ...props }: any) =>
                                  inline ? (
                                    <code
                                      className="rounded bg-gray-200 px-1 py-0.5 text-[11px]"
                                      {...props}
                                    >
                                      {children}
                                    </code>
                                  ) : (
                                    <code {...props}>{children}</code>
                                  ),
                                table: (props) => (
                                  <table
                                    className="w-full text-left border-collapse"
                                    {...props}
                                  />
                                ),
                                thead: (props) => (
                                  <thead className="border-b" {...props} />
                                ),
                                th: (props) => (
                                  <th
                                    className="px-2 py-1 font-semibold"
                                    {...props}
                                  />
                                ),
                                td: (props) => (
                                  <td
                                    className="px-2 py-1 align-top"
                                    {...props}
                                  />
                                ),
                                blockquote: (props) => (
                                  <blockquote
                                    className="border-l-2 pl-3 text-gray-600"
                                    {...props}
                                  />
                                ),
                                hr: (props) => (
                                  <hr className="my-2" {...props} />
                                ),
                                h1: (props) => (
                                  <h1
                                    className="text-base font-bold"
                                    {...props}
                                  />
                                ),
                                h2: (props) => (
                                  <h2
                                    className="text-base font-bold"
                                    {...props}
                                  />
                                ),
                                h3: (props) => (
                                  <h3
                                    className="text-sm font-semibold"
                                    {...props}
                                  />
                                ),
                                h4: (props) => (
                                  <h4
                                    className="text-sm font-semibold"
                                    {...props}
                                  />
                                ),
                                h5: (props) => (
                                  <h5
                                    className="text-xs font-semibold"
                                    {...props}
                                  />
                                ),
                                h6: (props) => (
                                  <h6
                                    className="text-xs font-semibold"
                                    {...props}
                                  />
                                ),
                              }}
                            >
                              {text}
                            </ReactMarkdown>
                          </div>
                        )}
                        {status === "streaming" &&
                          msg.bubbleId === lastAssistantMessageId && (
                            <div className="mt-2 text-gray-600 text-sm">
                              <div className="flex items-center gap-2">
                                <span className="inline-block h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                                <span className="font-semibold">
                                  Thinking...
                                </span>
                              </div>
                              {latestThought && (
                                <div className="mt-2">
                                  <pre className="whitespace-pre-wrap text-xs text-gray-700 bg-gray-50 rounded p-2 max-h-40 overflow-y-auto">
                                    {latestThought}
                                  </pre>
                                </div>
                              )}
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
                        {/* Tool results */}
                        {content.tool_results &&
                          content.tool_results.length > 0 && (
                            <div className="mt-3 space-y-2">
                              {content.tool_results.map(
                                (tool: any, index: number) => (
                                  <div
                                    key={index}
                                    className="bg-gray-50 rounded-lg p-3 text-xs border"
                                  >
                                    <div className="flex items-center gap-2 mb-2">
                                      <span className="inline-block h-2 w-2 rounded-full bg-green-500" />
                                      <span className="font-semibold text-green-700">
                                        {tool.toolName || "Tool"} 실행
                                      </span>
                                      {tool.result?.latency_ms && (
                                        <span className="text-gray-500">
                                          ({tool.result.latency_ms}ms)
                                        </span>
                                      )}
                                    </div>
                                    {tool.result?.success &&
                                      tool.result?.output?.body && (
                                        <div className="space-y-2">
                                          {tool.result.output.url && (
                                            <div className="text-gray-600">
                                              <span className="font-medium">
                                                URL:
                                              </span>{" "}
                                              {tool.result.output.url}
                                            </div>
                                          )}
                                          <div className="bg-gray-100 rounded p-2">
                                            <div className="font-medium text-blue-700 mb-1">
                                              응답 데이터:
                                            </div>
                                            <pre className="text-xs text-gray-800 overflow-x-auto max-h-40">
                                              {JSON.stringify(
                                                tool.result.output.body,
                                                null,
                                                2
                                              )}
                                            </pre>
                                          </div>
                                        </div>
                                      )}
                                    {!tool.result?.success && (
                                      <div className="text-red-600">
                                        실행 실패:{" "}
                                        {tool.result?.error ||
                                          "알 수 없는 오류"}
                                      </div>
                                    )}
                                  </div>
                                )
                              )}
                            </div>
                          )}
                      </div>
                    );
                  })}
                </div>
              )}
            </section>

            {status === "streaming" && !lastAssistantMessageId && (
              <div className="text-gray-600 text-sm">
                <div className="flex items-center gap-2">
                  <span className="inline-block h-2 w-2 rounded-full bg-blue-500 animate-pulse" />
                  <span className="font-semibold">Thinking...</span>
                </div>
                {latestThought && (
                  <div className="mt-2">
                    <pre className="whitespace-pre-wrap text-xs text-gray-700 bg-gray-50 rounded p-2 max-h-40 overflow-y-auto">
                      {latestThought}
                    </pre>
                  </div>
                )}
              </div>
            )}

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
