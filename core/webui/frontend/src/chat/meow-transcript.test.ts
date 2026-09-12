// @vitest-environment happy-dom

import { afterEach, describe, expect, it } from "vitest";
import type { Turn } from "../contracts/chat";
import "./meow-transcript";

const turns: Turn[] = [
  {
    turn_id: "turn-resource",
    turn_sequence: 1,
    created_at: 1_756_000_000,
    status: "complete",
    turn_kind: "ai",
    blocks: [
      {
        type: "emoji",
        role: "user",
        resource: {
          resource_type: "emoji",
          media_id: "emoji-1",
          filename: "笑脸",
        },
      },
    ],
    metadata: {},
  },
  {
    turn_id: "turn-approval",
    turn_sequence: 2,
    created_at: 1_756_000_001,
    status: "approval",
    turn_kind: "ai",
    blocks: [
      {
        type: "approval",
        role: "tool",
        approval: {
          session_key: "approval-1",
          title: "需要管理员审批",
          description: "允许执行测试命令？",
          command_preview: "head -5 README.md",
        },
      },
    ],
    metadata: {},
  },
];

afterEach(() => {
  document.body.replaceChildren();
});

describe("meow-transcript", () => {
  it("renders a resource-only message with its media fallback", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [turns[0]];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    const image = transcript.shadowRoot?.querySelector("img");
    expect(image?.getAttribute("src")).toBe("/media/emoji-1/content");
    expect(image?.getAttribute("alt")).toBe("笑脸");
    expect(transcript.shadowRoot?.querySelector(".empty")).toBeNull();
  });

  it("emits an approval decision from the approval card", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [turns[1]];
    transcript.loading = false;
    const decisions: unknown[] = [];
    transcript.addEventListener("approval-decision", (event) => {
      decisions.push((event as CustomEvent).detail);
    });
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    const buttons = transcript.shadowRoot?.querySelectorAll<HTMLButtonElement>(
      ".approval-actions button",
    );
    buttons?.[0]?.click();

    expect(decisions).toEqual([
      { sessionKey: "approval-1", decision: "allow-once" },
    ]);
  });

  it("blocks approval after a restart but keeps denial available", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [{
      ...turns[1],
      blocks: [{
        ...turns[1].blocks[0],
        approval: { ...turns[1].blocks[0].approval, recovery_state: "recovered" },
      }],
    }];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    const buttons = transcript.shadowRoot?.querySelectorAll<HTMLButtonElement>(
      ".approval-actions button",
    );
    expect(buttons?.[0]?.disabled).toBe(true);
    expect(buttons?.[1]?.disabled).toBe(true);
    expect(buttons?.[2]?.disabled).toBe(false);
    expect(transcript.shadowRoot?.textContent).toContain("进程已重启");
  });

  it("keeps an empty text block from hiding other resource content", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [
      {
        ...turns[0],
        blocks: [
          { type: "text", role: "user", text: "" },
          ...turns[0].blocks,
        ],
      },
    ];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelectorAll(".message").length).toBe(1);
    expect(transcript.shadowRoot?.querySelector("img")).not.toBeNull();
  });

  it("renders a safe card link and preserves unknown text blocks", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [
      {
        ...turns[0],
        blocks: [
          {
            type: "card",
            role: "assistant",
            text: "备用标题",
            metadata: {
              title: "参考资料",
              description: "打开文档",
              url: "https://example.com/docs",
            },
          },
          { type: "notice", role: "assistant", text: "原始通知" },
        ],
      },
    ];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelector(".card")?.textContent).toContain("参考资料");
    expect(transcript.shadowRoot?.querySelector(".card a")?.getAttribute("href")).toBe(
      "https://example.com/docs",
    );
    expect(transcript.shadowRoot?.textContent).toContain("原始通知");
  });

  it("renders tool arguments and result in an expandable card", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [{
      ...turns[0],
      blocks: [{
        type: "tool",
        role: "tool",
        tool_name: "send_emoji",
        status: "completed",
        arguments: { emoji_hash: "emoji-1" },
        result: '{"success":true}',
      }],
    }];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelector(".tool-card")?.textContent).toContain("send_emoji");
    expect(transcript.shadowRoot?.querySelector(".tool-detail")?.textContent).toContain("emoji-1");
    expect(transcript.shadowRoot?.querySelector(".tool-result")?.textContent).toContain("success");
  });

  it("merges a tool result into the matching trajectory step", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [{
      ...turns[0],
      blocks: [
        {
          type: "tool",
          role: "assistant",
          tool_call_id: "call-1",
          tool_name: "execute_command",
          arguments: { command: "ls" },
          status: "called",
          resources: [{ resource_type: "image", filename: "result.png", preview_url: "/result.png", mime_type: "image/png" }],
        },
        {
          type: "tool_result",
          role: "tool",
          tool_call_id: "call-1",
          tool_name: "execute_command",
          status: "completed",
          text: "README.md",
        },
      ],
    }];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelectorAll(".tool-card").length).toBe(1);
    expect(transcript.shadowRoot?.querySelector(".step-name")?.textContent).toContain("execute_command");
    expect(transcript.shadowRoot?.querySelector(".tool-result")?.textContent).toContain("README.md");
    expect(transcript.shadowRoot?.querySelector(".step-resources img")?.getAttribute("src")).toBe("/result.png");
  });

  it("renders image, audio, video, and file attachments", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [{
      ...turns[0],
      blocks: [
        { type: "image", role: "user", resource: { resource_type: "image", filename: "photo.png", preview_url: "/photo.png", download_url: "/photo.png?download=true", mime_type: "image/png" } },
        { type: "voice", role: "assistant", resource: { resource_type: "voice", filename: "voice.mp3", preview_url: "/voice.mp3", mime_type: "audio/mpeg" } },
        { type: "video", role: "assistant", resource: { resource_type: "video", filename: "clip.mp4", preview_url: "/clip.mp4", mime_type: "video/mp4" } },
        { type: "file", role: "assistant", resource: { resource_type: "file", filename: "report.pdf", download_url: "/report.pdf?download=true", mime_type: "application/pdf" } },
      ],
    }];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelector("img")?.getAttribute("src")).toBe("/photo.png");
    expect(transcript.shadowRoot?.querySelector("audio")?.getAttribute("src")).toBe("/voice.mp3");
    expect(transcript.shadowRoot?.querySelector("video")?.getAttribute("src")).toBe("/clip.mp4");
    expect(transcript.shadowRoot?.querySelector(".attachment-download")?.textContent).toContain("下载附件");
    expect(transcript.shadowRoot?.textContent).toContain("report.pdf");
  });

  it("labels each turn and participant and collapses reasoning by default", async () => {
    const transcript = document.createElement("meow-transcript") as HTMLElement & {
      turns: Turn[];
      loading: boolean;
    };
    transcript.turns = [{
      ...turns[0],
      turn_sequence: 7,
      blocks: [
        { type: "text", role: "user", sender_id: "user-42", text: "你好" },
        { type: "reasoning", role: "assistant", text: "先思考一下" },
        { type: "text", role: "assistant", text: "你好，我是助手" },
      ],
    }];
    transcript.loading = false;
    document.body.append(transcript);
    await (transcript as unknown as { updateComplete: Promise<unknown> }).updateComplete;

    expect(transcript.shadowRoot?.querySelector(".turn-label")?.textContent).toContain("第 7 轮");
    expect(transcript.shadowRoot?.querySelector(".user .message-label")?.textContent).toContain("user-42");
    expect(transcript.shadowRoot?.querySelector(".assistant .bubble:not(.reasoning-card) .message-label")?.textContent).toContain("助手");
    expect(transcript.shadowRoot?.querySelector(".reasoning-card")).not.toBeNull();
    expect(transcript.shadowRoot?.querySelector(".reasoning-card")?.hasAttribute("open")).toBe(false);
  });
});
