export type ChatSession = {
  session_id: string;
  title: string;
  mode: string;
  created_at: number;
  updated_at: number;
  read_only?: boolean;
  channel?: string;
};

export type ChatSessionPage = {
  items: ChatSession[];
  has_more: boolean;
  next_cursor: string | null;
};

export type ChatModelOption = {
  id: string;
  label: string;
  model_count: number;
};

export type ChatControlOption = {
  enabled: boolean;
  default: string;
  reason: string;
  options?: string[];
};

export type ChatOptions = {
  model_groups: ChatModelOption[];
  controls: {
    thinking_effort: ChatControlOption;
    quick_mode: ChatControlOption;
    context_compaction: ChatControlOption;
  };
};

export type ContentBlock = {
  type: string;
  role?: string;
  sender_id?: string;
  text?: string;
  resource?: Record<string, unknown>;
  tool_calls?: unknown[];
  tool_call_id?: string;
  tool_name?: string;
  arguments?: unknown;
  result?: string;
  resources?: Record<string, unknown>[];
  status?: string;
  metadata?: Record<string, unknown>;
  approval?: Record<string, unknown>;
};

export type Turn = {
  turn_id: string;
  turn_sequence: number;
  created_at: number;
  status: string;
  turn_kind: string;
  blocks: ContentBlock[];
  metadata: Record<string, unknown>;
};

export type TurnPage = {
  session_id: string;
  items: Turn[];
  has_more: boolean;
  next_before_turn_sequence: number | null;
  cutoff_sequence: number;
  unstable_cursor: boolean;
};

export type StreamEvent = {
  event_id: string;
  session_id: string;
  turn_id: string;
  type: string;
  sequence: number;
  occurred_at: number;
  payload: Record<string, unknown>;
};

export type ChatAuditEntry = {
  action: string;
  details: Record<string, unknown>;
  created_at: number;
};

export type ChatCompactionResult = {
  changed: boolean;
  tier: number;
  operation: string;
  before_tokens: number;
  after_tokens: number;
  saved_tokens: number;
  reason: string;
  scope_count: number;
};

export function isChatSession(value: unknown): value is ChatSession {
  if (!value || typeof value !== "object") return false;
  const session = value as Partial<ChatSession>;
  return typeof session.session_id === "string" && typeof session.mode === "string";
}

export function isChatSessionPage(value: unknown): value is ChatSessionPage {
  if (!value || typeof value !== "object") return false;
  const page = value as Partial<ChatSessionPage>;
  return (
    Array.isArray(page.items) &&
    page.items.every(isChatSession) &&
    typeof page.has_more === "boolean" &&
    (page.next_cursor === null || typeof page.next_cursor === "string")
  );
}

export function isChatModelOption(value: unknown): value is ChatModelOption {
  if (!value || typeof value !== "object") return false;
  const option = value as Partial<ChatModelOption>;
  return (
    typeof option.id === "string" &&
    typeof option.label === "string" &&
    typeof option.model_count === "number"
  );
}

export function isTurnPage(value: unknown): value is TurnPage {
  if (!value || typeof value !== "object") return false;
  const page = value as Partial<TurnPage>;
  return Array.isArray(page.items) && typeof page.session_id === "string";
}

export function isChatAuditEntry(value: unknown): value is ChatAuditEntry {
  if (!value || typeof value !== "object") return false;
  const entry = value as Partial<ChatAuditEntry>;
  return (
    typeof entry.action === "string" &&
    typeof entry.created_at === "number" &&
    Boolean(entry.details && typeof entry.details === "object")
  );
}
