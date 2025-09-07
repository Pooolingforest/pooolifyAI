export type ConversationDTO = {
  conversation: any[];
  session_id: string;
  current_request_id?: string | null;
  message_count: number;
};

export async function postChat({
  apiBase,
  token,
  sessionId,
  query,
  model,
}: {
  apiBase: string;
  token?: string;
  sessionId: string;
  query: string;
  model: string;
}): Promise<{ status: string; session_id: string; request_id: string }> {
  const resp = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ session_id: sessionId, query, options: { model } }),
  });
  if (!resp.ok) {
    throw new Error(
      `POST /v1/chat failed: ${resp.status} ${await resp.text()}`
    );
  }
  return resp.json();
}

export async function getConversation({
  apiBase,
  token,
  sessionId,
}: {
  apiBase: string;
  token?: string;
  sessionId: string;
}): Promise<ConversationDTO> {
  const resp = await fetch(
    `${apiBase.replace(/\/+$/, "")}/v1/sessions/${sessionId}/conversation`,
    {
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
    }
  );
  if (!resp.ok) {
    throw new Error(
      `GET conversation failed: ${resp.status} ${await resp.text()}`
    );
  }
  return resp.json();
}

export function isCompleted(data: ConversationDTO): boolean {
  const msgs = data?.conversation || [];
  const last = msgs[msgs.length - 1];
  if (!last) return false;
  if (last.type === "MESSAGE_TYPE_COMPLETE") return true;
  const completion = (last.content?.completion || "").toUpperCase();
  return [
    "ORCHESTRATION_COMPLETED",
    "REQUEST_COMPLETED",
    "REQUEST_COMPLETED_WITH_ERROR",
  ].includes(completion);
}
