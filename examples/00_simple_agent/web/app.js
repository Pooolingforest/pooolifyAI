"use strict";

(() => {
  /** DOM Elements */
  const $apiBase = document.getElementById("apiBase");
  const $apiToken = document.getElementById("apiToken");
  const $sessionId = document.getElementById("sessionId");
  const $autoRefresh = document.getElementById("autoRefresh");
  const $refreshMs = document.getElementById("refreshMs");
  const $showInternal = document.getElementById("showInternal");
  const $btnRefresh = document.getElementById("btnRefresh");
  const $btnManualRefresh = document.getElementById("btnManualRefresh");
  const $btnSend = document.getElementById("btnSend");
  const $prompt = document.getElementById("prompt");
  const $conversation = document.getElementById("conversation");
  const $errorBox = document.getElementById("errorBox");
  const $processingNote = document.getElementById("processingNote");
  const $healthDot = document.getElementById("healthDot");

  /** State */
  let currentConversation = null;
  let pollingTimer = null;
  let sending = false;
  let healthTimer = null;

  /** Utilities */
  function getSettings() {
    return {
      apiBase: (
        $apiBase.value ||
        window.POOOLIFY_API_BASE ||
        "http://127.0.0.1:8000"
      ).replace(/\/+$/, ""),
      token: $apiToken.value || window.POOOLIFY_API_TOKEN || "",
      sessionId: $sessionId.value || window.POOOLIFY_SESSION_ID || "demo",
      autoRefresh: !!$autoRefresh.checked,
      refreshMs: Math.max(250, Number($refreshMs.value || 1000) || 1000),
      showInternal: !!$showInternal.checked,
    };
  }

  function initInputsFromWindow() {
    $apiBase.value = window.POOOLIFY_API_BASE || "http://127.0.0.1:8000";
    $apiToken.value = window.POOOLIFY_API_TOKEN || "";
    $sessionId.value = window.POOOLIFY_SESSION_ID || "demo";
  }

  // Viewport height CSS var for responsive conversation area (mobile-friendly)
  function setViewportVar() {
    const vh = window.innerHeight * 0.01;
    document.documentElement.style.setProperty("--vh", `${vh}px`);
  }

  function setError(text) {
    if (!text) {
      $errorBox.classList.add("hidden");
      $errorBox.textContent = "";
    } else {
      $errorBox.textContent = String(text);
      $errorBox.classList.remove("hidden");
    }
  }

  function getAgentLabel(agent) {
    if (!agent) return "AI";
    return String(agent)
      .replace(/_/g, " ")
      .replace(/\b\w/g, (m) => m.toUpperCase());
  }

  function isCompleted(data) {
    const msgs = (data && data.conversation) || [];
    const last = msgs[msgs.length - 1];
    if (!last) return false;
    if (last.type === "MESSAGE_TYPE_COMPLETE") return true;
    const completion = String(
      (last.content && last.content.completion) || ""
    ).toUpperCase();
    return [
      "ORCHESTRATION_COMPLETED",
      "REQUEST_COMPLETED",
      "REQUEST_COMPLETED_WITH_ERROR",
    ].includes(completion);
  }

  function getLastAssistantMessageId(conv) {
    const list = (conv && conv.conversation) || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const msg = list[i];
      if (msg && msg.type === "MESSAGE_TYPE_AI" && msg.bubbleId)
        return String(msg.bubbleId);
    }
    return null;
  }

  function getLatestThought(conv) {
    const list = (conv && conv.conversation) || [];
    for (let i = list.length - 1; i >= 0; i--) {
      const msg = list[i];
      if (
        msg &&
        msg.type === "MESSAGE_TYPE_AI" &&
        msg.content &&
        msg.content.thought
      ) {
        return String(msg.content.thought);
      }
    }
    return "";
  }

  function markdownToHtml(md) {
    if (!md) return "";
    const html = window.marked ? window.marked.parse(md) : String(md);
    if (window.DOMPurify) {
      return window.DOMPurify.sanitize(html);
    }
    return html;
  }

  function el(tag, className, text) {
    const n = document.createElement(tag);
    if (className) n.className = className;
    if (text != null) n.textContent = String(text);
    return n;
  }

  function renderMessage(
    msg,
    status,
    lastAssistantId,
    latestThought,
    showInternal
  ) {
    const content = msg.content || {};
    const text = content.answer || content.error || content.completion || "";
    const thought = content.thought || "";
    const plan = content.plan || "";
    const route = content.route || "";
    const decision = content.decision || "";
    const time = msg.timestamp || "";

    const card = el("div", "rounded border border-gray-700 bg-gray-800 p-2");

    const meta = el("div", "mb-1 text-xs text-gray-400");
    const role =
      msg.type === "MESSAGE_TYPE_HUMAN"
        ? "You"
        : msg.type === "MESSAGE_TYPE_AI"
        ? msg.agent || "AI"
        : "System";
    meta.appendChild(el("span", "font-semibold", role));
    meta.appendChild(document.createTextNode(` · ${time}`));
    card.appendChild(meta);

    if (msg.agent) {
      const agentBar = el(
        "div",
        "flex items-center gap-2 mb-2 pb-1 border-b border-gray-700"
      );
      const dot = el("span", "inline-block h-2 w-2 rounded-full bg-purple-400");
      const label = el(
        "span",
        "text-xs font-medium text-purple-300",
        getAgentLabel(msg.agent)
      );
      agentBar.appendChild(dot);
      agentBar.appendChild(label);
      card.appendChild(agentBar);
    }

    if (text) {
      const textBox = el("div");
      const html = markdownToHtml(String(text));
      const mdContainer = el("div");
      mdContainer.innerHTML = html;
      // lightweight styling similar to React variant
      mdContainer.className = "prose prose-invert prose-sm max-w-none";
      textBox.appendChild(mdContainer);
      card.appendChild(textBox);
    }

    if (
      status === "streaming" &&
      msg.bubbleId &&
      String(msg.bubbleId) === String(lastAssistantId)
    ) {
      const think = el("div", "mt-2 text-gray-300 text-sm");
      const row = el("div", "flex items-center gap-2");
      row.appendChild(
        el(
          "span",
          "inline-block h-2 w-2 rounded-full bg-blue-500 animate-pulse"
        )
      );
      row.appendChild(el("span", "font-semibold", "Thinking..."));
      think.appendChild(row);
      if (latestThought) {
        const pre = el(
          "pre",
          "whitespace-pre-wrap text-xs text-gray-200 bg-gray-900 rounded p-2 max-h-40 overflow-y-auto"
        );
        pre.textContent = latestThought;
        think.appendChild(pre);
      }
      card.appendChild(think);
    }

    if (showInternal && (thought || plan || route || decision)) {
      const details = el("details", "mt-2");
      const summary = el(
        "summary",
        "cursor-pointer text-xs text-gray-300",
        "Internal"
      );
      details.appendChild(summary);
      if (thought) {
        const pre = el(
          "pre",
          "mt-1 overflow-auto rounded bg-gray-900 text-gray-200 p-2 text-xs"
        );
        pre.textContent = String(thought);
        details.appendChild(pre);
      }
      if (plan) {
        const pre = el(
          "pre",
          "mt-1 overflow-auto rounded bg-gray-900 text-gray-200 p-2 text-xs"
        );
        pre.textContent = String(plan);
        details.appendChild(pre);
      }
      if (route) {
        const pre = el(
          "pre",
          "mt-1 overflow-auto rounded bg-gray-900 text-gray-200 p-2 text-xs"
        );
        pre.textContent = String(route);
        details.appendChild(pre);
      }
      if (decision) {
        const pre = el(
          "pre",
          "mt-1 overflow-auto rounded bg-gray-900 text-gray-200 p-2 text-xs"
        );
        pre.textContent = String(decision);
        details.appendChild(pre);
      }
      card.appendChild(details);
    }

    if (content.tool_results && content.tool_results.length > 0) {
      const toolsWrap = el("div", "mt-3 space-y-2");
      for (const tool of content.tool_results) {
        const box = el(
          "div",
          "bg-gray-900 rounded-lg p-3 text-xs border border-gray-700"
        );
        const head = el("div", "flex items-center gap-2 mb-2");
        head.appendChild(
          el("span", "inline-block h-2 w-2 rounded-full bg-green-400")
        );
        const title = el(
          "span",
          "font-semibold text-green-300",
          `${tool.toolName || "Tool"} 실행`
        );
        head.appendChild(title);
        if (tool.result && tool.result.latency_ms) {
          head.appendChild(
            el("span", "text-gray-400", `(${tool.result.latency_ms}ms)`)
          );
        }
        box.appendChild(head);

        if (
          tool.result &&
          tool.result.success &&
          tool.result.output &&
          tool.result.output.body
        ) {
          const space = el("div", "space-y-2");
          if (tool.result.output.url) {
            const urlRow = el("div", "text-gray-300");
            const bold = el("span", "font-medium", "URL:");
            urlRow.appendChild(bold);
            urlRow.appendChild(document.createTextNode(" "));
            urlRow.appendChild(
              document.createTextNode(String(tool.result.output.url))
            );
            space.appendChild(urlRow);
          }
          const dataBox = el("div", "bg-gray-950 rounded p-2");
          dataBox.appendChild(
            el("div", "font-medium text-blue-300 mb-1", "응답 데이터:")
          );
          const pre = el(
            "pre",
            "text-xs text-gray-200 overflow-x-auto max-h-40"
          );
          pre.textContent = JSON.stringify(tool.result.output.body, null, 2);
          dataBox.appendChild(pre);
          space.appendChild(dataBox);
          box.appendChild(space);
        } else if (!tool.result || !tool.result.success) {
          box.appendChild(
            el(
              "div",
              "text-red-400",
              `실행 실패: ${
                tool.result && tool.result.error
                  ? tool.result.error
                  : "알 수 없는 오류"
              }`
            )
          );
        }
        toolsWrap.appendChild(box);
      }
      card.appendChild(toolsWrap);
    }

    return card;
  }

  function renderConversation(conv) {
    const settings = getSettings();
    const list = (conv && conv.conversation) || [];
    $conversation.innerHTML = "";

    if (!list.length) {
      const empty = el(
        "div",
        "text-sm text-gray-600",
        "No conversation yet. Send a message."
      );
      $conversation.appendChild(empty);
      return;
    }

    const processing = !!(conv && conv.current_request_id);
    const status = processing ? "streaming" : "idle";
    const lastAssistantId = getLastAssistantMessageId(conv);
    const latestThought = getLatestThought(conv);

    for (const msg of list) {
      $conversation.appendChild(
        renderMessage(
          msg,
          status,
          lastAssistantId,
          latestThought,
          settings.showInternal
        )
      );
    }

    // processing note
    if (processing && settings.autoRefresh) {
      $processingNote.classList.remove("hidden");
    } else {
      $processingNote.classList.add("hidden");
    }
  }

  async function getConversation({ apiBase, token, sessionId }) {
    const resp = await fetch(
      `${apiBase}/v1/sessions/${encodeURIComponent(sessionId)}/conversation`,
      {
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      }
    );
    // Ignore 404 (Session not found) as a non-error; show empty conversation instead
    if (resp.status === 404) {
      return { conversation: [], current_request_id: null };
    }
    if (!resp.ok) {
      const bodyText = await resp.text().catch(() => "");
      throw new Error(
        `GET conversation failed: ${resp.status} ${bodyText}`.trim()
      );
    }
    return await resp.json();
  }

  async function postChat({ apiBase, token, sessionId, query }) {
    const resp = await fetch(`${apiBase}/v1/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({ session_id: sessionId, query }),
    });
    if (!resp.ok) {
      throw new Error(
        `POST /v1/chat failed: ${resp.status} ${await resp.text()}`
      );
    }
    return await resp.json();
  }

  async function checkHealth() {
    try {
      const settings = getSettings();
      const resp = await fetch(`${settings.apiBase}/v1/healthz`, {
        headers: {
          "Content-Type": "application/json",
          ...(settings.token
            ? { Authorization: `Bearer ${settings.token}` }
            : {}),
        },
      });
      const ok = resp.ok;
      if ($healthDot) {
        $healthDot.classList.remove(
          "bg-gray-500",
          "bg-green-500",
          "bg-red-500"
        );
        $healthDot.classList.add(ok ? "bg-green-500" : "bg-red-500");
      }
    } catch (e) {
      if ($healthDot) {
        $healthDot.classList.remove("bg-gray-500", "bg-green-500");
        $healthDot.classList.add("bg-red-500");
      }
    }
  }

  async function refresh() {
    try {
      const settings = getSettings();
      const data = await getConversation({
        apiBase: settings.apiBase,
        token: settings.token,
        sessionId: settings.sessionId,
      });
      currentConversation = data;
      renderConversation(data);
      setError(null);
      managePolling();
    } catch (err) {
      setError(err && err.message ? err.message : String(err));
    }
  }

  function managePolling() {
    const settings = getSettings();
    const processing = !!(
      currentConversation && currentConversation.current_request_id
    );
    if (pollingTimer) {
      window.clearInterval(pollingTimer);
      pollingTimer = null;
    }
    if (settings.autoRefresh && processing) {
      pollingTimer = window.setInterval(() => {
        refresh();
      }, Math.max(250, settings.refreshMs));
    }

    // Health check interval follows refreshMs independently
    if (healthTimer) {
      window.clearInterval(healthTimer);
      healthTimer = null;
    }
    healthTimer = window.setInterval(() => {
      checkHealth().catch(() => {});
    }, Math.max(1000, settings.refreshMs));
  }

  async function handleSend() {
    const query = ($prompt.value || "").trim();
    if (!query || sending) return;
    sending = true;
    $btnSend.disabled = true;
    try {
      const settings = getSettings();
      await postChat({
        apiBase: settings.apiBase,
        token: settings.token,
        sessionId: settings.sessionId,
        query,
      });

      // Optimistic short polling
      const start = Date.now();
      while (Date.now() - start < 120000) {
        const data = await getConversation({
          apiBase: settings.apiBase,
          token: settings.token,
          sessionId: settings.sessionId,
        });
        currentConversation = data;
        renderConversation(data);
        if (!data.current_request_id || isCompleted(data)) break;
        await new Promise((r) =>
          setTimeout(r, Math.max(250, settings.refreshMs))
        );
      }
    } catch (err) {
      setError(err && err.message ? err.message : String(err));
    } finally {
      sending = false;
      $btnSend.disabled = false;
      $prompt.value = "";
      refresh();
    }
  }

  // Wire UI events
  $btnSend.addEventListener("click", handleSend);
  $btnRefresh.addEventListener("click", refresh);
  $btnManualRefresh.addEventListener("click", refresh);
  $autoRefresh.addEventListener("change", managePolling);
  $refreshMs.addEventListener("change", managePolling);

  // Configure marked
  if (window.marked) {
    window.marked.setOptions({ gfm: true, breaks: true });
  }

  // Init
  initInputsFromWindow();
  setViewportVar();
  window.addEventListener("resize", setViewportVar);
  refresh();
  checkHealth().catch(() => {});
})();
