// @vitest-environment happy-dom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StreamEvent } from "../contracts/chat";

const mocks = vi.hoisted(() => ({
  compactChatSession: vi.fn(),
  createChatSession: vi.fn(),
  discardChatResource: vi.fn(),
  listChatSessions: vi.fn(),
  loadChatAudit: vi.fn(),
  loadChatOptions: vi.fn(),
  loadPendingChatApprovals: vi.fn(),
  loadTurns: vi.fn(),
  openChatEvents: vi.fn(),
  renameChatSession: vi.fn(),
  resolveChatApproval: vi.fn(),
  submitChatTurn: vi.fn(),
  uploadChatResource: vi.fn(),
}));

vi.mock("../api/chat-api", () => mocks);
import "./meow-chat-app";

const session = {
  session_id: "webui-session-1",
  title: "测试会话",
  mode: "agent",
  created_at: 1_756_000_000,
  updated_at: 1_756_000_000,
};

const emptyTurnPage = {
  session_id: session.session_id,
  items: [],
  has_more: false,
  next_before_turn_sequence: null,
  cutoff_sequence: 0,
  unstable_cursor: false,
};

const options = {
  model_groups: [{ id: "auto", label: "自动", model_count: 1 }],
  controls: {
    thinking_effort: {
      enabled: true,
      default: "provider",
      reason: "跟随模型配置",
      options: ["low", "medium", "high", "none"],
    },
    quick_mode: {
      enabled: false,
      default: "不可用",
      reason: "尚未实现",
    },
    context_compaction: {
      enabled: true,
      default: "可用",
      reason: "管理员可用",
    },
  },
};

async function settle(app: HTMLElement) {
  await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;
  await vi.waitFor(() => expect(app.shadowRoot?.querySelector("textarea")).not.toBeNull());
}

function makeEvent(type: string, sequence: number, payload: Record<string, unknown>): StreamEvent {
  return {
    event_id: `event-${sequence}`,
    session_id: session.session_id,
    turn_id: "turn-failed",
    type,
    sequence,
    occurred_at: 1_756_000_010,
    payload,
  };
}

beforeEach(() => {
  mocks.listChatSessions.mockResolvedValue({ items: [session], has_more: false, next_cursor: null });
  mocks.loadChatOptions.mockResolvedValue(options);
  mocks.loadTurns.mockResolvedValue(emptyTurnPage);
  mocks.loadPendingChatApprovals.mockResolvedValue([]);
  mocks.openChatEvents.mockReturnValue({ close: vi.fn(), onopen: null, onerror: null });
  mocks.submitChatTurn.mockResolvedValue({ turn_id: "turn-submitted" });
});

afterEach(() => {
  document.body.replaceChildren();
  vi.clearAllMocks();
});

describe("meow-chat-app", () => {
  it("submits composer text without navigating or losing the session", async () => {
    const app = document.createElement("meow-chat-app");
    document.body.append(app);
    await settle(app);

    const textarea = app.shadowRoot?.querySelector<HTMLTextAreaElement>("textarea");
    const form = app.shadowRoot?.querySelector("form");
    expect(textarea).not.toBeNull();
    expect(form).not.toBeNull();
    textarea!.value = "  hello from webui  ";
    textarea!.dispatchEvent(new Event("input", { bubbles: true }));
    form!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
    await vi.waitFor(() => expect(mocks.submitChatTurn).toHaveBeenCalledTimes(1));

    expect(mocks.submitChatTurn.mock.calls[0][0]).toBe(session.session_id);
    expect(mocks.submitChatTurn.mock.calls[0][1]).toBe("hello from webui");
    expect(mocks.submitChatTurn.mock.calls[0][3]).toBe("agent");
    expect(app.shadowRoot?.querySelector("aside")?.classList.contains("open")).toBe(false);
  });

  it("exposes a retry action after a failed turn", async () => {
    const app = document.createElement("meow-chat-app");
    document.body.append(app);
    await settle(app);
    const controller = app as unknown as {
      handleStreamEvent(event: StreamEvent): void;
    };
    controller.handleStreamEvent(
      makeEvent("turn.accepted", 1, {
        content: "retry this",
        resources: [],
        mode: "agent",
        model_group: "auto",
        reasoning_effort: "provider",
      }),
    );
    controller.handleStreamEvent(makeEvent("turn.failed", 2, { status: "failed" }));
    await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    const transcript = app.shadowRoot?.querySelector("meow-transcript") as
      | (HTMLElement & { updateComplete: Promise<unknown>; shadowRoot: ShadowRoot })
      | null;
    await transcript?.updateComplete;
    const retry = transcript?.shadowRoot.querySelector<HTMLButtonElement>(".retry-action");
    expect(retry).not.toBeNull();
    retry!.click();
    await vi.waitFor(() => expect(mocks.submitChatTurn).toHaveBeenCalledTimes(1));
    expect(mocks.submitChatTurn.mock.calls[0][1]).toBe("retry this");
  });

  it("opens and closes the mobile navigation drawer", async () => {
    const app = document.createElement("meow-chat-app");
    document.body.append(app);
    await settle(app);

    const menu = app.shadowRoot?.querySelector<HTMLButtonElement>(".menu-toggle");
    const aside = app.shadowRoot?.querySelector("aside");
    menu?.click();
    await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;
    expect(aside?.classList.contains("open")).toBe(true);
    menu?.click();
    await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;
    expect(aside?.classList.contains("open")).toBe(false);
  });

  it("does not allow an Agent session to switch back to Chat", async () => {
    const app = document.createElement("meow-chat-app");
    document.body.append(app);
    await settle(app);

    app.shadowRoot?.querySelector<HTMLButtonElement>(".settings-toggle")?.click();
    await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;
    const mode = app.shadowRoot?.querySelector<HTMLSelectElement>(
      'select[aria-label="本次消息模式"]',
    );
    expect(mode?.value).toBe("agent");
    expect(mode?.querySelector('option[value="chat"]')?.hasAttribute("disabled")).toBe(true);
    mode!.value = "chat";
    mode!.dispatchEvent(new Event("change", { bubbles: true }));
    await (app as unknown as { updateComplete: Promise<unknown> }).updateComplete;
    expect(mode?.value).toBe("agent");
  });
});
