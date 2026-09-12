import { expect, test, type Page, type Route } from "@playwright/test";

const session = {
  session_id: "webui-session-1",
  title: "浏览器验收会话",
  mode: "agent",
  created_at: 1_756_000_000,
  updated_at: 1_756_000_000,
};

const options = {
  model_groups: [{ id: "auto", label: "自动路由", model_count: 0 }],
  controls: {
    thinking_effort: {
      enabled: true,
      default: "provider",
      reason: "跟随模型配置",
      options: ["low", "medium", "high", "none"],
    },
    quick_mode: { enabled: false, default: "不可用", reason: "尚未实现" },
    context_compaction: { enabled: true, default: "可用", reason: "管理员可用" },
  },
};

const emptyTurnPage = {
  session_id: session.session_id,
  items: [],
  has_more: false,
  next_before_turn_sequence: null,
  cutoff_sequence: 0,
  unstable_cursor: false,
};

type MockWorkbenchOptions = {
  pendingApprovals?: Record<string, unknown>[];
  eventStream?: (route: Route) => Promise<void>;
  onApprovalDecision?: (body: unknown) => void;
};

async function mockWorkbench(page: Page, mockOptions: MockWorkbenchOptions = {}) {
  await page.route("**/api/chat/sessions?*", async (route) => {
    await route.fulfill({ json: { items: [session], has_more: false, next_cursor: null } });
  });
  await page.route("**/api/chat/options", async (route) => {
    await route.fulfill({ json: options });
  });
  await page.route("**/api/csrf", async (route) => {
    await route.fulfill({ json: { token: "e2e-csrf" } });
  });
  await page.route("**/api/sessions/*/turns?*", async (route) => {
    await route.fulfill({ json: emptyTurnPage });
  });
  await page.route("**/api/chat/sessions/*/approvals", async (route) => {
    await route.fulfill({ json: { items: mockOptions.pendingApprovals || [] } });
  });
  await page.route("**/api/chat/sessions/*/events*", async (route) => {
    if (mockOptions.eventStream) {
      await mockOptions.eventStream(route);
      return;
    }
    await route.fulfill({
      status: 200,
      headers: { "content-type": "text/event-stream" },
      body: "event: session.ready\ndata: {}\n\n",
    });
  });
  await page.route("**/api/chat/sessions/*/turns", async (route) => {
    await route.fulfill({ json: { receipt: { turn_id: "turn-e2e" } } });
  });
  await page.route("**/api/chat/approvals/resolve", async (route) => {
    mockOptions.onApprovalDecision?.(route.request().postDataJSON());
    await route.fulfill({ json: { ok: true } });
  });
}

test("loads the latest session and submits without navigation", async ({ page }) => {
  await mockWorkbench(page);
  await page.goto("./");

  await expect(page.locator("meow-chat-app")).toContainText("浏览器验收会话");
  const composer = page.locator("meow-chat-app textarea");
  await expect(composer).toBeVisible();
  await composer.fill("浏览器发送的消息");
  await page.locator("meow-chat-app form").press("Enter");

  await expect.poll(() => page.url()).toBe("http://127.0.0.1:4173/static/chat/");
  await expect(page.locator("meow-chat-app textarea")).toHaveValue("");
});

test("keeps Agent mode one-way in the browser UI", async ({ page }) => {
  await mockWorkbench(page);
  await page.goto("./");
  await page.locator("meow-chat-app .settings-toggle").click();

  const mode = page.locator('meow-chat-app select[aria-label="本次消息模式"]');
  await expect(mode).toHaveValue("agent");
  await expect(mode.locator('option[value="chat"]')).toHaveAttribute("disabled", "");
});

test("recovers from an SSE resync request and reopens the stream", async ({ page }) => {
  await page.addInitScript(() => {
    const sources: MockEventSource[] = [];
    class MockEventSource {
      static instances = sources;
      onopen: (() => void) | null = null;
      onerror: (() => void) | null = null;
      private listeners = new Map<string, ((event: MessageEvent<string>) => void)[]>();

      constructor(public readonly url: string) {
        sources.push(this);
        queueMicrotask(() => this.onopen?.());
      }

      addEventListener(type: string, listener: (event: MessageEvent<string>) => void) {
        this.listeners.set(type, [...(this.listeners.get(type) || []), listener]);
      }

      close() {}

      emit(type: string, payload: Record<string, unknown>) {
        const event = new MessageEvent(type, {
          data: JSON.stringify(payload),
          lastEventId: String(payload.event_id || ""),
        });
        for (const listener of this.listeners.get(type) || []) listener(event);
      }
    }
    (window as unknown as { __e2eEventSources: MockEventSource[] }).__e2eEventSources = sources;
    (window as unknown as { EventSource: typeof MockEventSource }).EventSource = MockEventSource;
  });
  await mockWorkbench(page);
  await page.goto("./");
  await expect(page.locator("meow-chat-app")).toContainText("浏览器验收会话");
  await expect.poll(() => page.evaluate(() => (window as any).__e2eEventSources.length)).toBe(1);

  await page.evaluate((sessionId) => {
    const source = (window as any).__e2eEventSources[0];
    source.emit("resync_required", {
      event_id: "event-resync",
      session_id: sessionId,
      turn_id: "",
      type: "resync_required",
      sequence: 2,
      occurred_at: 1_756_000_020,
      payload: { latest_event_id: "event-latest" },
    });
  }, session.session_id);

  await expect.poll(() => page.evaluate(() => (window as any).__e2eEventSources.length)).toBe(2);
});

test("reconnects the browser EventSource after a closed SSE response", async ({ page }) => {
  let eventConnections = 0;
  await mockWorkbench(page, {
    eventStream: async (route) => {
      eventConnections += 1;
      await route.fulfill({
        status: 200,
        headers: { "content-type": "text/event-stream", "retry": "10" },
        body: "retry: 10\nevent: session.ready\ndata: {}\n\n",
      });
    },
  });
  await page.goto("./");
  await expect(page.locator("meow-chat-app")).toContainText("浏览器验收会话");
  await expect.poll(() => eventConnections, { timeout: 10_000 }).toBeGreaterThan(1);
});

test("opens the mobile drawer and closes it after selecting a session", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 800 });
  await mockWorkbench(page);
  await page.goto("./");
  const app = page.locator("meow-chat-app");
  const aside = app.locator("aside");
  await expect(aside).not.toHaveClass(/open/);
  await app.locator(".menu-toggle").click();
  await expect(aside).toHaveClass(/open/);
  await app.locator(".session").click();
  await expect(aside).not.toHaveClass(/open/);
});

test("loads a pending approval and sends the administrator decision", async ({ page }) => {
  let decision: unknown;
  await mockWorkbench(page, {
    pendingApprovals: [{
      session_key: "approval-e2e",
      title: "需要管理员审批",
      description: "允许执行测试命令？",
      command_preview: "head -5 README.md",
      created_at: 1_756_000_010,
    }],
    onApprovalDecision: (body) => { decision = body; },
  });
  await page.goto("./");
  const app = page.locator("meow-chat-app");
  await expect(app).toContainText("需要管理员审批");
  await app.locator(".approval-actions button").first().click();
  await expect.poll(() => decision).toEqual({ session_key: "approval-e2e", decision: "allow-once" });
});
