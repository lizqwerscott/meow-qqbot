import { expect, test, type Page } from "@playwright/test";

async function mockBootstrap(page: Page) {
  await page.route("**/api/chat/sessions?*", async (route) => {
    await route.fulfill({ json: { items: [], has_more: false, next_cursor: null } });
  });
  await page.route("**/api/chat/options", async (route) => {
    await route.fulfill({
      json: {
        model_groups: [],
        controls: {
          thinking_effort: { enabled: false, default: "provider", reason: "测试", options: [] },
          quick_mode: { enabled: false, default: "不可用", reason: "测试" },
          context_compaction: { enabled: false, default: "不可用", reason: "测试" },
        },
      },
    });
  });
}

async function measureTranscript(page: Page, count: number) {
  return page.evaluate(async (turnCount: number) => {
    await customElements.whenDefined("meow-transcript");
    const transcript = document.createElement("meow-transcript") as any;
    transcript.style.cssText = "display:block;width:900px;height:640px";
    transcript.turns = Array.from({ length: turnCount }, (_, index) => ({
      turn_id: `perf-${index}`,
      turn_sequence: index + 1,
      created_at: 1_756_000_000 + index,
      status: "complete",
      turn_kind: "ai",
      blocks: [{ type: "text", role: "assistant", text: `消息 ${index}` }],
      metadata: {},
    }));
    transcript.loading = false;
    document.body.append(transcript);
    const started = performance.now();
    await transcript.updateComplete;
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    const elapsedMs = performance.now() - started;
    const history = transcript.shadowRoot.querySelector(".history");
    return {
      elapsedMs,
      articleCount: transcript.shadowRoot.querySelectorAll("article").length,
      scrollHeight: history?.scrollHeight || 0,
      totalTurns: turnCount,
    };
  }, count);
}

async function recordPerformance(result: Awaited<ReturnType<typeof measureTranscript>>) {
  test.info().annotations.push({ type: "performance", description: JSON.stringify(result) });
  await test.info().attach(`transcript-${result.totalTurns}-turns.json`, {
    body: JSON.stringify(result, null, 2),
    contentType: "application/json",
  });
  console.log(`transcript-performance ${JSON.stringify(result)}`);
}

test("virtualizes 1,000 turns", async ({ page }) => {
  await mockBootstrap(page);
  await page.goto("./");
  const result = await measureTranscript(page, 1_000);
  expect(result.articleCount).toBeLessThan(result.totalTurns);
  expect(result.articleCount).toBeGreaterThan(0);
  expect(result.scrollHeight).toBeGreaterThan(0);
  await recordPerformance(result);
});

test("virtualizes 10,000 turns without mounting the full transcript", async ({ page }) => {
  await mockBootstrap(page);
  await page.goto("./");
  const result = await measureTranscript(page, 10_000);
  expect(result.articleCount).toBeLessThan(100);
  expect(result.scrollHeight).toBeGreaterThan(0);
  await recordPerformance(result);
});

test("updates the virtual transcript height after an image loads", async ({ page }) => {
  await mockBootstrap(page);
  await page.goto("./");
  const result = await page.evaluate(async () => {
    await customElements.whenDefined("meow-transcript");
    const transcript = document.createElement("meow-transcript") as any;
    transcript.style.cssText = "display:block;width:900px;height:640px";
    transcript.turns = Array.from({ length: 200 }, (_, index) => ({
      turn_id: `dynamic-${index}`,
      turn_sequence: index + 1,
      created_at: 1_756_000_000 + index,
      status: "complete",
      turn_kind: "ai",
      blocks: [{ type: "text", role: "assistant", text: `消息 ${index}` }],
      metadata: {},
    }));
    document.body.append(transcript);
    await transcript.updateComplete;
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    const history = transcript.shadowRoot.querySelector(".history");
    const before = history.scrollHeight;
    transcript.turns = [{
      turn_id: "dynamic-0",
      turn_sequence: 1,
      created_at: 1_756_000_000,
      status: "complete",
      turn_kind: "ai",
      blocks: [{
        type: "image",
        role: "assistant",
        resource: {
          resource_type: "image",
          filename: "dynamic.svg",
          preview_url: "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='600'%3E%3Crect width='800' height='600' fill='%235865f2'/%3E%3C/svg%3E",
        },
      }],
      metadata: {},
    }, ...transcript.turns.slice(1)];
    await transcript.updateComplete;
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    const image = transcript.shadowRoot.querySelector("img") as HTMLImageElement;
    if (image && !image.complete) await new Promise((resolve) => image.addEventListener("load", resolve, { once: true }));
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    await new Promise((resolve) => requestAnimationFrame(() => resolve(undefined)));
    return { before, after: history.scrollHeight, imageHeight: image?.getBoundingClientRect().height || 0 };
  });

  expect(result.imageHeight).toBeGreaterThan(200);
  expect(result.after).toBeGreaterThan(result.before + 200);
});
