import { css, html, LitElement, type TemplateResult } from "lit";
import { customElement, property, state } from "lit/decorators.js";
import type { ContentBlock, Turn } from "../contracts/chat";
import {
  calculateVirtualRange,
  isFollowingTail,
  restoreScrollTop,
  VirtualHeightIndex,
} from "./transcript-state";

@customElement("meow-transcript")
export class MeowTranscript extends LitElement {
  @property({ attribute: false }) turns: Turn[] = [];
  @property({ attribute: false }) retryableTurnIds: Set<string> = new Set();
  @property({ type: Boolean }) loading = false;
  @property({ type: Boolean }) hasMore = false;
  private static readonly VIRTUAL_THRESHOLD = 120;
  private static readonly OVERSCAN = 8;
  private olderObserver: IntersectionObserver | undefined;
  private resizeObserver: ResizeObserver | undefined;
  private heightCache = new Map<string, number>();
  private heightIndex = new VirtualHeightIndex();
  @state() private viewportTop = 0;

  static styles = css`
    :host { display: block; height: 100%; min-height: 0; }
    .history { height: 100%; box-sizing: border-box; overflow-y: auto; max-width: 960px; margin: 0 auto; padding: 32px 32px 120px; }
    .spacer { width: 1px; pointer-events: none; }
    .turn { display: grid; gap: 14px; margin: 0 0 30px; padding: 0 8px 24px; border-bottom: 1px solid #e9ebf2; }
    .turn-meta { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: #a0a7b6; font-size: 11px; }
    .turn-label { color: #596174; font-weight: 700; }
    .turn-stream { display: grid; gap: 8px; position: relative; }
    .turn-stream::before { position: absolute; top: 12px; bottom: 12px; left: 9px; width: 1px; background: #e8ebf2; content: ""; }
    .message { display: flex; }
    .message.user { justify-content: flex-end; }
    .bubble { max-width: min(720px, 86%); border-radius: 18px; padding: 12px 16px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message-label { margin-bottom: 4px; color: #8b93a7; font-size: 11px; line-height: 1.2; }
    .user .message-label { color: rgba(255,255,255,.78); }
    .user .bubble { color: #fff; background: #5865f2; border-bottom-right-radius: 5px; }
    .assistant .bubble, .tool .bubble { background: #fff; border: 1px solid #e5e8f0; border-bottom-left-radius: 5px; }
    .card { display: grid; gap: 6px; }
    .card a { color: #4d5bd4; overflow-wrap: anywhere; text-decoration: none; }
    .trajectory-item { position: relative; z-index: 1; padding-left: 28px; }
    .trajectory-step { max-width: min(820px, 100%); border: 1px solid #e5e8f0; border-radius: 10px; background: #f8f9fc; color: #596174; }
    .trajectory-step summary { display: flex; align-items: center; gap: 9px; min-height: 38px; padding: 0 12px; cursor: pointer; list-style: none; }
    .trajectory-step summary::-webkit-details-marker { display: none; }
    .step-marker { display: inline-grid; flex: 0 0 18px; place-items: center; width: 18px; height: 18px; border-radius: 5px; color: #667085; background: #e7eaf1; font-size: 11px; }
    .step-kind { color: #596174; font-size: 12px; font-weight: 700; }
    .step-name { color: #737c90; font-size: 11px; }
    .step-summary { overflow: hidden; flex: 1; color: #7b8498; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
    .step-status { color: #8b93a7; font-size: 11px; white-space: nowrap; }
    .step-status.running { color: #4d63cf; }
    .step-status.failed, .step-status.blocked { color: #b14a4a; }
    .step-body { display: grid; gap: 8px; margin: 0 12px 12px; padding-top: 10px; border-top: 1px solid #e5e8f0; }
    .step-resources { display: grid; gap: 8px; }
    .tool-detail, .tool-result { max-height: 260px; overflow: auto; margin: 0; padding: 10px; border-radius: 8px; background: #fff; color: #515a70; font: 12px/1.5 ui-monospace, SFMono-Regular, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .tool-result { background: #292d3e; color: #f5f6fa; }
    .tool-note { color: #697389; font-size: 12px; }
    .reasoning-card { background: #fbfbfd; }
    .reasoning-summary { overflow: hidden; flex: 1; color: #8b93a7; font-size: 12px; font-weight: 400; text-overflow: ellipsis; white-space: nowrap; }
    .reasoning-body { white-space: pre-wrap; }
    .attachment { display: grid; gap: 8px; max-width: min(620px, 86%); border: 1px solid #e1e5ee; border-radius: 13px; padding: 9px; color: #596174; background: #fff; font-size: 13px; }
    .user .attachment { border-color: #cbd2ff; background: #f1f3ff; }
    .attachment-head { display: flex; align-items: center; gap: 8px; min-width: 0; }
    .attachment-icon { display: grid; flex: 0 0 30px; place-items: center; width: 30px; height: 30px; border-radius: 8px; background: #eef0f7; font-size: 16px; }
    .attachment-name { overflow: hidden; flex: 1; text-overflow: ellipsis; white-space: nowrap; }
    .attachment-meta { color: #8b93a7; font-size: 11px; }
    .attachment img { display: block; max-width: 100%; max-height: 420px; border-radius: 9px; object-fit: contain; }
    .attachment audio, .attachment video { display: block; max-width: 100%; }
    .attachment a { color: #4d5bd4; text-decoration: none; }
    .attachment-download { justify-self: start; border-radius: 7px; padding: 5px 8px; background: #eef0ff; font-size: 12px; }
    .approval { border-color: #e8c98b; background: #fffaf0; }
    .approval pre { max-width: 100%; overflow-x: auto; margin: 8px 0; padding: 8px; border-radius: 8px; background: #292d3e; color: #f5f6fa; white-space: pre-wrap; }
    .approval-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .approval-actions button { padding: 6px 10px; }
    .retry-action { margin-top: 8px; }
    .empty, .loading { padding: 64px 20px; color: #8b93a7; text-align: center; }
    button { border: 0; border-radius: 999px; padding: 8px 14px; color: #4d5bd4; background: #eef0ff; cursor: pointer; }
  `;

  firstUpdated() {
    this.setupOlderObserver();
    this.setupResizeObserver();
  }

  updated(changed: Map<string, unknown>) {
    if (changed.has("turns") || changed.has("viewportTop")) this.observeRenderedTurns();
    if (changed.has("hasMore") || changed.has("loading")) {
      if (!this.hasMore) {
        this.olderObserver?.disconnect();
        this.olderObserver = undefined;
      } else {
        this.olderObserver?.disconnect();
        this.olderObserver = undefined;
        this.setupOlderObserver();
      }
    }
  }

  private setupOlderObserver() {
    if (!this.hasMore || this.olderObserver) return;
    const history = this.renderRoot.querySelector(".history");
    const anchor = this.renderRoot.querySelector(".older-anchor");
    if (!history || !anchor || typeof IntersectionObserver === "undefined") return;
    this.olderObserver = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) this.loadOlder();
      },
      { root: history, rootMargin: "240px 0px 0px" },
    );
    this.olderObserver.observe(anchor);
  }

  disconnectedCallback() {
    this.olderObserver?.disconnect();
    this.olderObserver = undefined;
    this.resizeObserver?.disconnect();
    this.resizeObserver = undefined;
    super.disconnectedCallback();
  }

  private setupResizeObserver() {
    if (typeof ResizeObserver === "undefined") return;
    this.resizeObserver = new ResizeObserver((entries) => {
      let changed = false;
      for (const entry of entries) {
        const turnId = (entry.target as HTMLElement).dataset.turnId;
        if (!turnId) continue;
        const marginBottom = Number.parseFloat(
          getComputedStyle(entry.target).marginBottom,
        ) || 0;
        const nextHeight = Math.max(1, entry.contentRect.height + marginBottom);
        const previousHeight = this.heightCache.get(turnId);
        if (previousHeight === undefined) {
          this.heightCache.set(turnId, nextHeight);
          this.heightIndex.setMeasuredHeight(turnId, nextHeight);
          changed = true;
          continue;
        }
        if (Math.abs(previousHeight - nextHeight) < 1) {
          this.heightCache.set(turnId, nextHeight);
          this.heightIndex.setMeasuredHeight(turnId, nextHeight);
          continue;
        }
        const index = this.turns.findIndex((turn) => turn.turn_id === turnId);
        if (index >= 0 && index < this.virtualRange().start) {
          const history = this.renderRoot.querySelector(".history");
          if (history) history.scrollTop += nextHeight - previousHeight;
        }
        this.heightCache.set(turnId, nextHeight);
        this.heightIndex.setMeasuredHeight(turnId, nextHeight);
        changed = true;
      }
      if (changed) this.requestUpdate();
    });
    this.observeRenderedTurns();
  }

  private observeRenderedTurns() {
    if (!this.resizeObserver) return;
    this.resizeObserver.disconnect();
    for (const article of this.renderRoot.querySelectorAll<HTMLElement>("article[data-turn-id]")) {
      this.resizeObserver.observe(article);
    }
  }

  render() {
    const range = this.virtualRange();
    const visibleTurns = this.turns.slice(range.start, range.end);
    return html`
      <div class="history" @scroll=${this.handleScroll}>
        ${this.hasMore ? html`<div class="older-anchor"></div><button @click=${this.loadOlder}>↑ 加载更早消息</button>` : ""}
        ${this.loading ? html`<div class="loading">正在加载历史…</div>` : ""}
        ${this.turns.length === 0 && !this.loading
          ? html`<div class="empty">这个会话还没有消息</div>`
          : html`
            ${range.virtual ? html`<div class="spacer" style=${`height:${range.top}px`}></div>` : ""}
            ${(range.virtual ? visibleTurns : this.turns).map((turn) => this.renderTurn(turn))}
            ${range.virtual ? html`<div class="spacer" style=${`height:${range.bottom}px`}></div>` : ""}
          `}
      </div>
    `;
  }

  private renderTurn(turn: Turn): TemplateResult {
    return html`
      <article class="turn" data-turn-id=${turn.turn_id}>
        <div class="turn-meta"><span class="turn-label">第 ${turn.turn_sequence} 轮</span><span>${this.formatTime(turn.created_at)} · ${turn.status}</span></div>
        <div class="turn-stream">${this.renderTurnBlocks(turn.blocks)}</div>
        ${this.retryableTurnIds.has(turn.turn_id)
          ? html`<button class="retry-action" @click=${() => this.retryTurn(turn.turn_id)}>↻ 重试此消息</button>`
          : ""}
      </article>
    `;
  }

  private renderTurnBlocks(blocks: ContentBlock[]): TemplateResult[] {
    const results = new Map<string, ContentBlock>();
    for (const block of blocks) {
      if (block.type === "tool_result" && block.tool_call_id) {
        results.set(block.tool_call_id, block);
      }
    }
    const consumedResults = new Set<string>();
    return blocks.flatMap((block) => {
      if (block.type === "tool_result" && block.tool_call_id) {
        if (consumedResults.has(block.tool_call_id)) return [];
        return results.get(block.tool_call_id) === block ? [this.renderToolStep(block)] : [];
      }
      if (block.type === "tool") {
        const toolCallId = block.tool_call_id || this.toolCallId(block);
        const result = toolCallId ? results.get(toolCallId) : undefined;
        if (result) consumedResults.add(toolCallId);
        return [this.renderToolStep(block, result)];
      }
      return [this.renderBlock(block)];
    });
  }

  private renderBlock(block: ContentBlock): TemplateResult {
    const role = block.role === "user" ? "user" : block.role === "tool" ? "tool" : "assistant";
    const label = this.messageLabel(block);
    if (block.type === "text" && block.text) {
      return html`<div class="message ${role}"><div class="bubble"><div class="message-label">${label}</div>${block.text}</div></div>`;
    }
    if (block.type === "reasoning" && block.text) {
      const summary = block.text.split("\n").find((line) => line.trim()) || "思考内容";
      const status = block.status || "completed";
      return html`<div class="trajectory-item"><details class="trajectory-step reasoning-card" ?open=${status === "running" || status === "started"}>
        <summary><span class="step-marker">✦</span><span class="step-kind">Think</span><span class="reasoning-summary">${summary}</span><span class="step-status ${status}">${this.toolStatus(status)}</span></summary>
        <div class="step-body reasoning-body">${block.text}</div>
      </details></div>`;
    }
    if (block.type === "card") {
      const metadata = block.metadata || {};
      const title = typeof metadata.title === "string" && metadata.title
        ? metadata.title
        : block.text || "卡片消息";
      const description = typeof metadata.description === "string" ? metadata.description : "";
      const url = typeof metadata.url === "string" && /^https?:\/\//i.test(metadata.url)
        ? metadata.url
        : "";
      return html`<div class="message ${role}"><div class="bubble card">
        <div class="message-label">${label}</div>
        <strong>${title}</strong>
        ${description ? html`<div>${description}</div>` : ""}
        ${url ? html`<a href=${url} target="_blank" rel="noreferrer">${url}</a>` : ""}
      </div></div>`;
    }
    if (block.resource) {
      return this.renderResource(block, role, label);
    }
    if (block.type === "tool" || block.type === "tool_result") {
      return this.renderToolStep(block);
    }
    if (block.type === "approval" && block.approval) {
      const approval = block.approval;
      const sessionKey = typeof approval.session_key === "string" ? approval.session_key : "";
      const title = typeof approval.title === "string" ? approval.title : "需要管理员审批";
      const description = typeof approval.description === "string" ? approval.description : "";
      const command = typeof approval.command_preview === "string" ? approval.command_preview : "";
      const recovered = approval.recovery_state === "recovered";
      return html`<div class="trajectory-item"><div class="bubble approval">
        <strong>${title}</strong>
        ${description ? html`<div>${description}</div>` : ""}
        ${command ? html`<pre>${command}</pre>` : ""}
        ${recovered ? html`<div>进程已重启，原执行已停止，请重新发起操作。</div>` : ""}
        <div class="approval-actions">
          <button @click=${() => this.resolveApproval(sessionKey, "allow-once")} ?disabled=${!sessionKey || recovered}>允许一次</button>
          <button @click=${() => this.resolveApproval(sessionKey, "allow-always")} ?disabled=${!sessionKey || recovered}>始终允许</button>
          <button @click=${() => this.resolveApproval(sessionKey, "deny")} ?disabled=${!sessionKey}>拒绝</button>
        </div>
      </div></div>`;
    }
    if (block.text) {
      return html`<div class="message ${role}"><div class="bubble"><div class="message-label">${label}</div>${block.text}</div></div>`;
    }
    return html``;
  }

  private renderToolStep(block: ContentBlock, toolResult?: ContentBlock): TemplateResult {
    const toolName = block.tool_name || this.toolName(block);
    const resultStatus = toolResult?.status && toolResult.status !== "called" ? toolResult.status : "completed";
    const status = toolResult ? resultStatus : block.status || "running";
    const argumentsText = this.formatToolValue(block.arguments);
    const result = toolResult?.result || toolResult?.text || block.result || (block.type === "tool_result" ? block.text || "" : "");
    const summary = this.toolSummary(block);
    const resources = Array.isArray(block.resources) ? block.resources : [];
    return html`<div class="trajectory-item"><details class="trajectory-step tool-card" ?open=${status === "running" || status === "started" || status === "failed" || status === "blocked"}>
      <summary><span class="step-marker">${this.toolIcon(toolName)}</span><span class="step-kind">${this.toolKind(toolName)}</span><span class="step-name">${toolName}</span><span class="step-summary">${summary}</span><span class="step-status ${status}">${this.toolStatus(status)}</span></summary>
      <div class="step-body">
        ${argumentsText ? html`<pre class="tool-detail">${argumentsText}</pre>` : ""}
        ${result ? html`<pre class="tool-result">${result}</pre>` : ""}
        ${block.text && !result && block.type !== "tool_result" ? html`<div class="tool-note">${block.text}</div>` : ""}
        ${resources.length ? html`<div class="step-resources">${resources.map((resource) => this.renderResource({ type: String(resource.resource_type || "file"), role: "tool", resource }, "tool", toolName))}</div>` : ""}
      </div>
    </details></div>`;
  }

  private renderResource(block: ContentBlock, role: string, label: string): TemplateResult {
    const resource = block.resource || {};
    const mediaId = typeof resource.media_id === "string" ? resource.media_id : "";
    const fallbackUrl = mediaId ? `/media/${encodeURIComponent(mediaId)}/content` : "";
    const preview = typeof resource.preview_url === "string" && resource.preview_url ? resource.preview_url : fallbackUrl;
    const download = typeof resource.download_url === "string" && resource.download_url ? resource.download_url : (fallbackUrl ? `${fallbackUrl}?download=true` : "");
    const mimeType = typeof resource.mime_type === "string" ? resource.mime_type : "";
    const filename = typeof resource.filename === "string" && resource.filename ? resource.filename : block.type;
    const resourceType = typeof resource.resource_type === "string" ? resource.resource_type : block.type;
    const isImage = resource.is_image === true || resourceType === "image" || resourceType === "emoji" || mimeType.startsWith("image/");
    const isAudio = resource.is_audio === true || resourceType === "voice" || resourceType === "audio" || mimeType.startsWith("audio/");
    const isVideo = resource.is_video === true || resourceType === "video" || mimeType.startsWith("video/");
    const size = typeof resource.size === "number" ? this.formatBytes(resource.size) : "";
    const meta = [mimeType, size].filter(Boolean).join(" · ");
    const icon = isImage ? "▧" : isAudio ? "♫" : isVideo ? "▶" : "📎";
    return html`<div class="message ${role}"><div class="attachment">
      <div class="attachment-head"><span class="attachment-icon">${icon}</span><strong class="attachment-name" title=${filename}>${filename}</strong>${meta ? html`<span class="attachment-meta">${meta}</span>` : ""}</div>
      ${preview && isImage ? html`<a href=${download || preview} target="_blank" rel="noreferrer"><img src=${preview} alt=${filename} loading="lazy" /></a>`
        : preview && isAudio ? html`<audio controls preload="metadata" src=${preview}></audio>`
        : preview && isVideo ? html`<video controls preload="metadata" src=${preview}></video>`
        : ""}
      ${download ? html`<a class="attachment-download" href=${download} download=${filename}>下载附件</a>` : html`<span class="attachment-meta">${label}</span>`}
    </div></div>`;
  }

  private messageLabel(block: ContentBlock): string {
    if (block.role === "user") return block.sender_id ? `用户 · ${block.sender_id}` : "用户";
    if (block.role === "tool") return block.tool_name ? `工具 · ${block.tool_name}` : "工具";
    if (block.role === "system") return "系统";
    return "助手";
  }

  private toolName(block: ContentBlock): string {
    const calls = Array.isArray(block.tool_calls) ? block.tool_calls : [];
    const first = calls[0];
    if (first && typeof first === "object") {
      const functionValue = (first as Record<string, unknown>).function;
      if (functionValue && typeof functionValue === "object") {
        const name = (functionValue as Record<string, unknown>).name;
        if (typeof name === "string" && name) return name;
      }
    }
    return "工具调用";
  }

  private toolCallId(block: ContentBlock): string {
    const first = Array.isArray(block.tool_calls) ? block.tool_calls[0] : undefined;
    return first && typeof first === "object" && typeof (first as Record<string, unknown>).id === "string"
      ? String((first as Record<string, unknown>).id)
      : "";
  }

  private toolArguments(block: ContentBlock): Record<string, unknown> {
    if (block.arguments && typeof block.arguments === "object") return block.arguments as Record<string, unknown>;
    if (typeof block.arguments === "string") {
      try {
        const parsed = JSON.parse(block.arguments);
        return parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
      } catch { return {}; }
    }
    return {};
  }

  private toolKind(name: string): string {
    if (name.includes("command") || name === "bash" || name === "exec") return "Bash";
    if (name.includes("read")) return "Read";
    if (name.includes("write") || name.includes("edit") || name.includes("patch")) return "Edit";
    if (name.includes("search") || name.includes("grep")) return "Search";
    return "Tool";
  }

  private toolIcon(name: string): string {
    const kind = this.toolKind(name);
    return kind === "Bash" ? ">_" : kind === "Read" ? "▤" : kind === "Edit" ? "✎" : kind === "Search" ? "⌕" : "⚙";
  }

  private toolSummary(block: ContentBlock): string {
    const argumentsValue = this.toolArguments(block);
    const candidate = ["description", "command", "path", "file_path", "query", "url", "emoji_hash"]
      .map((key) => argumentsValue[key])
      .find((value) => typeof value === "string" && value.trim());
    if (typeof candidate === "string") return candidate.trim().replace(/\s+/g, " ").slice(0, 160);
    return block.tool_name || this.toolName(block);
  }

  private formatToolValue(value: unknown): string {
    if (value === undefined || value === null || value === "") return "";
    if (typeof value === "string") {
      try {
        return JSON.stringify(JSON.parse(value), null, 2);
      } catch {
        return value;
      }
    }
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value);
    }
  }

  private toolStatus(status: string): string {
    return ({ started: "准备中", running: "执行中", completed: "已完成", failed: "失败", blocked: "已阻止", cancelled: "已取消", called: "已调用" } as Record<string, string>)[status] || status;
  }

  private formatBytes(size: number): string {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    if (size < 1024 * 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    return `${(size / (1024 * 1024 * 1024)).toFixed(1)} GB`;
  }

  private resolveApproval(sessionKey: string, decision: "allow-once" | "allow-always" | "deny") {
    this.dispatchEvent(new CustomEvent("approval-decision", {
      bubbles: true,
      composed: true,
      detail: { sessionKey, decision },
    }));
  }

  private retryTurn(turnId: string) {
    this.dispatchEvent(new CustomEvent("retry-turn", {
      bubbles: true,
      composed: true,
      detail: { turnId },
    }));
  }

  private loadOlder = () => {
    if (this.loading || !this.hasMore) return;
    const history = this.renderRoot.querySelector(".history");
    this.dispatchEvent(new CustomEvent("load-older", {
      bubbles: true,
      composed: true,
      detail: history
        ? { scrollHeight: history.scrollHeight, scrollTop: history.scrollTop }
        : undefined,
    }));
  };

  private handleScroll = (event: Event) => {
    const history = event.currentTarget as HTMLElement;
    this.viewportTop = history.scrollTop;
    this.dispatchEvent(new CustomEvent("tail-change", {
      bubbles: true,
      composed: true,
      detail: {
        followingTail: isFollowingTail(
          history.scrollHeight,
          history.scrollTop,
          history.clientHeight,
        ),
      },
    }));
  };

  private virtualRange() {
    const history = this.renderRoot.querySelector(".history");
    const viewportHeight = history?.clientHeight || 720;
    return calculateVirtualRange(
      this.turns.map((turn) => turn.turn_id),
      this.viewportTop,
      viewportHeight,
      this.estimatedHeight(),
      (turnId) => this.heightCache.get(turnId),
      MeowTranscript.VIRTUAL_THRESHOLD,
      MeowTranscript.OVERSCAN,
      this.heightIndex,
    );
  }

  private estimatedHeight() {
    if (this.heightCache.size === 0) return 180;
    let total = 0;
    for (const height of this.heightCache.values()) total += height;
    return Math.max(96, total / this.heightCache.size);
  }

  scrollToBottom() {
    const history = this.renderRoot.querySelector(".history");
    if (history) history.scrollTop = history.scrollHeight;
  }

  restoreScroll(previous: { scrollHeight: number; scrollTop: number }) {
    const history = this.renderRoot.querySelector(".history");
    if (history) {
      history.scrollTop = restoreScrollTop(
        previous.scrollHeight,
        previous.scrollTop,
        history.scrollHeight,
      );
    }
  }

  private formatTime(timestamp: number): string {
    return new Date(timestamp * 1000).toLocaleString("zh-CN", { dateStyle: "short", timeStyle: "short" });
  }
}
