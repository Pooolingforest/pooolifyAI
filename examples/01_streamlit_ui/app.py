from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
import streamlit as st


# -----------------------------
# Data models (lightweight)
# -----------------------------

@dataclass
class ToolResult:
    toolCallId: str
    toolName: str
    toolIndex: int
    rawArgs: str
    result: Dict[str, Any]


@dataclass
class MessageContent:
    answer: str = ""
    thought: str = ""
    plan: str = ""
    route: str = ""
    decision: str = ""
    error: str = ""
    completion: str = ""
    tool_results: Optional[List[ToolResult]] = None


@dataclass
class Message:
    type: str
    bubbleId: Optional[str] = None
    agent: Optional[str] = None
    timestamp: Optional[str] = None
    content: Optional[MessageContent] = None


@dataclass
class ConversationDTO:
    conversation: List[Message]
    session_id: str
    current_request_id: Optional[str]
    message_count: int


# -----------------------------
# API helpers
# -----------------------------

def _headers(token: str | None) -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def post_chat(api_base: str, token: Optional[str], session_id: str, query: str, model: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/chat"
    body = {"session_id": session_id, "query": query, "options": {"model": model}}
    resp = requests.post(url, json=body, headers=_headers(token), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"POST /v1/chat failed: {resp.status_code} {resp.text}")
    return resp.json()


def get_conversation(api_base: str, token: Optional[str], session_id: str) -> ConversationDTO:
    url = f"{api_base.rstrip('/')}/v1/sessions/{session_id}/conversation"
    resp = requests.get(url, headers=_headers(token), timeout=30)
    if not resp.ok:
        raise RuntimeError(f"GET conversation failed: {resp.status_code} {resp.text}")
    raw = resp.json()
    # return raw dict; UI will access dict keys directly to avoid strict typing issues
    return raw  # type: ignore[return-value]


def get_health(api_base: str) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/v1/healthz"
    resp = requests.get(url, timeout=10)
    if not resp.ok:
        raise RuntimeError(f"GET /v1/healthz failed: {resp.status_code}")
    return resp.json()


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="pooolify - Streamlit UI", layout="wide")

if "api_base" not in st.session_state:
    st.session_state.api_base = os.getenv("POOOLIFY_API_BASE", "http://127.0.0.1:8000")
if "api_token" not in st.session_state:
    st.session_state.api_token = os.getenv("POOOLIFY_API_TOKEN", "")
if "session_id" not in st.session_state:
    st.session_state.session_id = os.getenv("POOOLIFY_SESSION_ID", "demo")
if "model" not in st.session_state:
    st.session_state.model = os.getenv("POOOLIFY_MODEL", "gpt-5")
if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True
if "refresh_interval_ms" not in st.session_state:
    st.session_state.refresh_interval_ms = 1000
if "show_internal" not in st.session_state:
    st.session_state.show_internal = False
if "last_request_id" not in st.session_state:
    st.session_state.last_request_id = None


with st.sidebar:
    st.markdown("**pooolify - Streamlit UI**")

    st.session_state.api_base = st.text_input(
        "API Base",
        value=st.session_state.api_base,
        help="FastAPI base URL (examples/00_simple_agent uses 8000)",
    )
    st.session_state.api_token = st.text_input(
        "API Token",
        value=st.session_state.api_token,
        help="Leave empty in dev if backend allows",
        type="password",
    )
    st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.model = st.selectbox("Model", ["gpt-5", "gpt-5-high"], index=0 if st.session_state.model == "gpt-5" else 1)

    cols = st.columns(2)
    with cols[0]:
        st.session_state.auto_refresh = st.checkbox("Auto refresh while processing", value=st.session_state.auto_refresh)
        st.session_state.show_internal = st.checkbox("Show internal thoughts", value=st.session_state.show_internal)
    with cols[1]:
        st.session_state.refresh_interval_ms = st.number_input(
            "Refresh interval (ms)", min_value=500, max_value=5000, value=int(st.session_state.refresh_interval_ms), step=100
        )
        if st.button("Ping /healthz"):
            try:
                health = get_health(st.session_state.api_base)
                st.success(f"Health: {health}")
            except Exception as e:  # noqa: BLE001
                st.error(str(e))


st.title("pooolify – Chat Console")

# Fetch conversation on each run
conversation_error: Optional[str] = None
conversation_data: Optional[ConversationDTO] = None
try:
    conversation_data = get_conversation(
        api_base=st.session_state.api_base,
        token=st.session_state.api_token,
        session_id=st.session_state.session_id,
    )
except Exception as e:  # noqa: BLE001
    conversation_error = str(e)


processing = False
if conversation_data and isinstance(conversation_data, dict):
    processing = bool(conversation_data.get("current_request_id"))

if processing and st.session_state.auto_refresh:
    st.toast("Processing… auto-refreshing", icon="⏳")
    st.autorefresh(interval=st.session_state.refresh_interval_ms, key="auto_refresh_key")


# Messages panel
left, right = st.columns([2, 3])
with left:
    st.subheader("Conversation")
    if conversation_error:
        st.error(conversation_error)
    elif not conversation_data:
        st.info("No conversation yet. Send a message.")
    else:
        msgs: List[Dict[str, Any]] = conversation_data.get("conversation", [])  # type: ignore[assignment]
        for msg in msgs:
            msg_type = msg.get("type")
            role = "You" if msg_type == "MESSAGE_TYPE_HUMAN" else (msg.get("agent") or "AI") if msg_type == "MESSAGE_TYPE_AI" else "System"
            timestamp = msg.get("timestamp")
            content = msg.get("content") or {}
            answer = content.get("answer") or ""
            err = content.get("error") or ""
            completion = content.get("completion") or ""
            thought = content.get("thought") or ""
            plan = content.get("plan") or ""
            route = content.get("route") or ""
            decision = content.get("decision") or ""

            with st.container(border=True):
                st.markdown(f"**{role}** · {timestamp or ''}")
                text = answer or err or completion or ""
                if text:
                    st.write(text)
                if st.session_state.show_internal and any([thought, plan, route, decision]):
                    with st.expander("Internal (thought/plan/route/decision)"):
                        if thought:
                            st.markdown("- **thought**:")
                            st.code(thought)
                        if plan:
                            st.markdown("- **plan**:")
                            st.code(plan)
                        if route:
                            st.markdown("- **route**:")
                            st.code(route)
                        if decision:
                            st.markdown("- **decision**:")
                            st.code(decision)

with right:
    st.subheader("Compose")
    prompt = st.text_area("Message", value="안녕! 간단 테스트를 해보자.", height=140)

    send_col, refresh_col = st.columns(2)
    with send_col:
        if st.button("Send", type="primary", use_container_width=True, disabled=not bool(prompt.strip())):
            try:
                res = post_chat(
                    api_base=st.session_state.api_base,
                    token=st.session_state.api_token,
                    session_id=st.session_state.session_id,
                    query=prompt.strip(),
                    model=st.session_state.model,
                )
                st.session_state.last_request_id = res.get("request_id")
                st.success("Queued. Polling conversation…")
                time.sleep(0.2)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    with refresh_col:
        if st.button("Refresh", use_container_width=True):
            st.rerun()


st.caption(
    "This example calls POST /v1/chat and polls GET /v1/sessions/{id}/conversation. "
    "Toggle 'Show internal thoughts' to view the manager's reasoning stream if available."
)


