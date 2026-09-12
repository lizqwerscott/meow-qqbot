import type { ChatAuditEntry, ChatCompactionResult, ChatOptions, ChatSession, ChatSessionPage, StreamEvent, TurnPage } from "../contracts/chat";
import { isChatAuditEntry, isChatModelOption, isChatSession, isChatSessionPage, isTurnPage } from "../contracts/chat";

let csrfToken = "";

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { credentials: "same-origin" });
  if (!response.ok) {
    throw new Error(`请求失败 (${response.status})`);
  }
  return (await response.json()) as T;
}

async function getCsrfToken(): Promise<string> {
  if (csrfToken) return csrfToken;
  const payload = await getJson<{ token?: string }>("/api/csrf");
  if (!payload.token) throw new Error("无法获取 CSRF token");
  csrfToken = payload.token;
  return csrfToken;
}

async function postJson<T>(url: string, body: unknown): Promise<T> {
  const token = await getCsrfToken();
  const response = await fetch(url, {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : `请求失败 (${response.status})`,
    );
  }
  return (await response.json()) as T;
}

async function patchJson<T>(url: string, body: unknown): Promise<T> {
  const token = await getCsrfToken();
  const response = await fetch(url, {
    method: "PATCH",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : `请求失败 (${response.status})`,
    );
  }
  return (await response.json()) as T;
}

async function deleteJson<T>(url: string, body: unknown): Promise<T> {
  const token = await getCsrfToken();
  const response = await fetch(url, {
    method: "DELETE",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": token },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : `请求失败 (${response.status})`,
    );
  }
  return (await response.json()) as T;
}

export async function listChatSessions(
  cursor = "",
  limit = 50,
): Promise<ChatSessionPage> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (cursor) params.set("cursor", cursor);
  const payload = await getJson<unknown>(`/api/chat/sessions?${params.toString()}`);
  if (!isChatSessionPage(payload)) throw new Error("服务端返回了无效的会话列表");
  return payload;
}

export async function loadChatOptions(): Promise<ChatOptions> {
  const payload = await getJson<Partial<ChatOptions>>("/api/chat/options");
  const controls = payload.controls;
  if (!controls) throw new Error("服务端未返回聊天设置能力");
  const modelGroups = (payload.model_groups ?? []).filter(isChatModelOption);
  const validControl = (value: unknown): value is ChatOptions["controls"]["thinking_effort"] => {
    if (!value || typeof value !== "object") return false;
    const control = value as Partial<ChatOptions["controls"]["thinking_effort"]>;
    return typeof control.enabled === "boolean" && typeof control.default === "string" && typeof control.reason === "string" && (control.options === undefined || (Array.isArray(control.options) && control.options.every((option) => typeof option === "string")));
  };
  if (!validControl(controls.thinking_effort) || !validControl(controls.quick_mode) || !validControl(controls.context_compaction)) {
    throw new Error("服务端返回了无效的聊天设置能力");
  }
  return {
    model_groups: modelGroups,
    controls: {
      thinking_effort: controls.thinking_effort,
      quick_mode: controls.quick_mode,
      context_compaction: controls.context_compaction,
    },
  };
}

export async function listExternalChatSessions(limit = 100): Promise<ChatSessionPage> {
  const payload = await getJson<unknown>(`/api/chat/external-sessions?limit=${limit}`);
  if (!isChatSessionPage(payload)) throw new Error("服务端返回了无效的外部会话列表");
  return payload;
}

export async function loadChatSession(sessionId: string): Promise<ChatSession> {
  const payload = await getJson<{ session?: unknown }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}`,
  );
  if (!isChatSession(payload.session)) throw new Error("服务端返回了无效的会话");
  return payload.session;
}

export async function createChatSession(
  title = "新会话",
  mode: "agent" | "chat" = "agent",
): Promise<ChatSession> {
  const payload = await postJson<{ session: ChatSession }>("/api/chat/sessions", {
    title,
    mode,
  });
  return payload.session;
}

export async function renameChatSession(
  sessionId: string,
  title: string,
): Promise<ChatSession> {
  const payload = await patchJson<{ session: ChatSession }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}`,
    { title },
  );
  return payload.session;
}

export async function submitChatTurn(
  sessionId: string,
  content: string,
  requestId: string,
  mode?: "agent" | "chat",
  resources: Record<string, unknown>[] = [],
  modelGroup = "",
  reasoningEffort = "provider",
): Promise<{ turn_id: string }> {
  const payload = await postJson<{ receipt: { turn_id: string } }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/turns`,
    {
      content,
      request_id: requestId,
      resources,
      ...(mode ? { mode } : {}),
      ...(modelGroup && modelGroup !== "auto" ? { model_group: modelGroup } : {}),
      ...(reasoningEffort && reasoningEffort !== "provider" ? { reasoning_effort: reasoningEffort } : {}),
    },
  );
  return payload.receipt;
}

export async function uploadChatResource(
  sessionId: string,
  file: File,
): Promise<Record<string, unknown>> {
  const token = await getCsrfToken();
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/uploads`,
    {
      method: "POST",
      credentials: "same-origin",
      headers: { "X-CSRF-Token": token },
      body: form,
    },
  );
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : `上传失败 (${response.status})`,
    );
  }
  const payload = (await response.json()) as { resource?: Record<string, unknown> };
  if (!payload.resource) throw new Error("服务端未返回上传资源");
  return payload.resource;
}

export async function discardChatResource(
  sessionId: string,
  mediaUri: string,
  uploadId: string,
): Promise<void> {
  await deleteJson(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/uploads`,
    { media_uri: mediaUri, upload_id: uploadId },
  );
}

export async function loadChatAudit(
  sessionId: string,
  limit = 50,
): Promise<ChatAuditEntry[]> {
  const payload = await getJson<{ items?: unknown[] }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/audit?limit=${limit}`,
  );
  return (payload.items ?? []).filter(isChatAuditEntry);
}

export async function loadPendingChatApprovals(
  sessionId: string,
): Promise<Record<string, unknown>[]> {
  const payload = await getJson<{ items?: unknown[] }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/approvals`,
  );
  return (payload.items ?? []).filter(
    (item): item is Record<string, unknown> => Boolean(item && typeof item === "object"),
  );
}

export async function compactChatSession(
  sessionId: string,
): Promise<ChatCompactionResult> {
  const payload = await postJson<{ result?: unknown }>(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/compact`,
    {},
  );
  if (!payload.result || typeof payload.result !== "object") {
    throw new Error("服务端未返回压缩结果");
  }
  return payload.result as ChatCompactionResult;
}

export async function resolveChatApproval(
  sessionKey: string,
  decision: "allow-once" | "allow-always" | "deny",
): Promise<void> {
  await postJson("/api/chat/approvals/resolve", {
    session_key: sessionKey,
    decision,
  });
}

export function openChatEvents(
  sessionId: string,
  afterEventId: string,
  onEvent: (event: StreamEvent) => void,
): EventSource {
  const params = new URLSearchParams();
  if (afterEventId) params.set("after_event_id", afterEventId);
  if (!afterEventId) params.set("replay", "false");
  const suffix = params.size ? `?${params.toString()}` : "";
  const source = new EventSource(
    `/api/chat/sessions/${encodeURIComponent(sessionId)}/events${suffix}`,
    { withCredentials: true },
  );
  const eventTypes = [
    "session.ready",
    "session.updated",
    "turn.accepted",
    "message.created",
    "message.delta",
    "message.reset",
    "tool.started",
    "tool.updated",
    "tool.finished",
    "turn.completed",
    "turn.failed",
    "delivery.backpressure",
    "approval.requested",
    "context.compacted",
    "resync_required",
  ];
  for (const eventType of eventTypes) {
    source.addEventListener(eventType, (event) => {
      const message = event as MessageEvent<string>;
      try {
        onEvent(JSON.parse(message.data) as StreamEvent);
      } catch {
        onEvent({
          event_id: message.lastEventId,
          session_id: sessionId,
          turn_id: "",
          type: "resync_required",
          sequence: 0,
          occurred_at: Date.now() / 1000,
          payload: { reason: "invalid_event_payload" },
        });
      }
    });
  }
  return source;
}

export async function loadTurns(
  sessionId: string,
  options: { cutoff?: number; before?: number; limit?: number } = {},
): Promise<TurnPage> {
  const params = new URLSearchParams({ limit: String(options.limit ?? 30) });
  if (options.cutoff !== undefined) params.set("cutoff_sequence", String(options.cutoff));
  if (options.before !== undefined) {
    params.set("before_turn_sequence", String(options.before));
  }
  const page = await getJson<unknown>(
    `/api/sessions/${encodeURIComponent(sessionId)}/turns?${params.toString()}`,
  );
  if (!isTurnPage(page)) throw new Error("服务端返回了无效的历史数据");
  return page;
}
