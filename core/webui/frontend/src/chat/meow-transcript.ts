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
    .history { height: 100%; box-sizing: border-box; overflow-y: auto; max-width: 860px; margin: 0 auto; padding: 32px 24px 120px; }
    .spacer { width: 1px; pointer-events: none; }
    .turn { display: grid; gap: 12px; margin: 0 0 28px; padding: 0 4px 20px; border-bottom: 1px solid #e9ebf2; }
    .turn-meta { display: flex; align-items: center; justify-content: center; gap: 8px; color: #8b93a7; font-size: 12px; }
    .turn-label { color: #596174; font-weight: 700; }
    .message { display: flex; }
    .message.user { justify-content: flex-end; }
    .bubble { max-width: min(720px, 86%); border-radius: 18px; padding: 12px 16px; line-height: 1.65; white-space: pre-wrap; overflow-wrap: anywhere; }
    .message-label { margin-bottom: 4px; color: #8b93a7; font-size: 11px; line-height: 1.2; }
    .user .message-label { color: rgba(255,255,255,.78); }
    .user .bubble { color: #fff; background: #5865f2; border-bottom-right-radius: 5px; }
    .assistant .bubble, .tool .bubble { background: #fff; border: 1px solid #e5e8f0; border-bottom-left-radius: 5px; }
    .card { display: grid; gap: 6px; }
    .card a { color: #4d5bd4; overflow-wrap: anywhere; text-decoration: none; }
    .approval { border-color: #e8c98b; background: #fffaf0; }
    .approval pre { max-width: 100%; overflow-x: auto; margin: 8px 0; padding: 8px; border-radius: 8px; background: #292d3e; color: #f5f6fa; white-space: pre-wrap; }
    .approval-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
    .approval-actions button { padding: 6px 10px; }
    .retry-action { margin-top: 8px; }
    .resource { border: 1px dashed #cbd2e1; color: #596174; font-size: 13px; }
    .resource img { display: block; max-width: min(520px, 100%); max-height: 420px; border-radius: 12px; object-fit: contain; }
    .resource audio, .resource video { display: block; max-width: min(520px, 100%); }
    .resource a { color: #4d5bd4; text-decoration: none; }
    .tool-card { display: grid; gap: 8px; max-width: min(720px, 86%); border-color: #dfe3ed; background: #f8f9fc; }
    .tool-card summary { display: flex; align-items: center; justify-content: space-between; gap: 16px; cursor: pointer; font-weight: 600; }
    .tool-card summary small { color: #7b8498; font-weight: 500; }
    .tool-detail, .tool-result { max-height: 260px; overflow: auto; margin: 0; padding: 10px; border-radius: 8px; background: #fff; color: #515a70; font: 12px/1.5 ui-monospace, SFMono-Regular, monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
    .tool-result { background: #292d3e; color: #f5f6fa; }
    .tool-note { color: #697389; font-size: 12px; }
    .reasoning-card { max-width: min(720px, 86%); border: 1px solid #e5e8f0; background: #f8f9fc; color: #596174; }
    .reasoning-card summary { display: flex; align-items: center; gap: 10px; cursor: pointer; font-weight: 600; }
    .reasoning-summary { overflow: hidden; color: #8b93a7; font-size: 12px; font-weight: 400; text-overflow: ellipsis; white-space: nowrap; }
    .reasoning-body { margin-top: 8px; padding-top: 8px; border-top: 1px solid #e5e8f0; white-space: pre-wrap; }
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
        ${turn.blocks.map((block) => this.renderBlock(block))}
        ${this.retryableTurnIds.has(turn.turn_id)
          ? html`<button class="retry-action" @click=${() => this.retryTurn(turn.turn_id)}>↻ 重试此消息</button>`
          : ""}
      </article>
    `;
  }

  private renderBlock(block: ContentBlock): TemplateResult {
    const role = block.role === "user" ? "user" : block.role === "tool" ? "tool" : "assistant";
    const label = this.messageLabel(block);
    if (block.type === "text" && block.text) {
      return html`<div class="message ${role}"><div class="bubble"><div class="message-label">${label}</div>${block.text}</div></div>`;
    }
    if (block.type === "reasoning" && block.text) {
      const summary = block.text.split("\n").find((line) => line.trim()) || "思考内容";
      return html`<div class="message assistant"><details class="bubble reasoning-card" ?open=${block.status === "running"}>
        <summary><span class="message-label">思考过程</span><span class="reasoning-summary">${summary}</span></summary>
        <div class="reasoning-body">${block.text}</div>
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
      const resource = block.resource;
      const mediaId = typeof resource.media_id === "string" ? resource.media_id : "";
      const fallbackUrl = mediaId ? `/media/${encodeURIComponent(mediaId)}/content` : "";
      const preview = typeof resource.preview_url === "string" && resource.preview_url ? resource.preview_url : fallbackUrl;
      const download = typeof resource.download_url === "string" && resource.download_url ? resource.download_url : (fallbackUrl ? `${fallbackUrl}?download=true` : "");
      const mimeType = typeof resource.mime_type === "string" ? resource.mime_type : "";
      const filename = typeof resource.filename === "string" ? resource.filename : block.type;
      return html`
        <div class="message ${role}">
          <div class="bubble resource">
            <div class="message-label">${label}</div>
            ${preview && (block.type === "image" || block.type === "emoji") ? html`<img src=${preview} alt=${filename} loading="lazy" />`
              : preview && (block.type === "voice" || block.type === "audio" || mimeType.startsWith("audio/")) ? html`<audio controls preload="metadata" src=${preview}></audio>`
              : preview && (block.type === "video" || mimeType.startsWith("video/")) ? html`<video controls preload="metadata" src=${preview}></video>`
              : download ? html`📎 <a href=${download} download=${filename}>${filename}</a>`
              : html`📎 ${filename}`}
          </div>
        </div>
      `;
    }
    if (block.type === "tool" || block.type === "tool_result") {
      const toolName = block.tool_name || this.toolName(block);
      const status = block.status || (block.type === "tool_result" ? "completed" : "running");
      const argumentsText = this.formatToolValue(block.arguments);
      const result = block.result || (block.type === "tool_result" ? block.text || "" : "");
      return html`<div class="message tool"><details class="bubble tool-card" ?open=${status === "running" || status === "started"}>
        <summary><span>${toolName}</span><small>${this.toolStatus(status)}</small></summary>
        ${argumentsText ? html`<pre class="tool-detail">${argumentsText}</pre>` : ""}
        ${result ? html`<pre class="tool-result">${result}</pre>` : ""}
        ${block.text && !result ? html`<div class="tool-note">${block.text}</div>` : ""}
      </details></div>`;
    }
    if (block.type === "approval" && block.approval) {
      const approval = block.approval;
      const sessionKey = typeof approval.session_key === "string" ? approval.session_key : "";
      const title = typeof approval.title === "string" ? approval.title : "需要管理员审批";
      const description = typeof approval.description === "string" ? approval.description : "";
      const command = typeof approval.command_preview === "string" ? approval.command_preview : "";
      const recovered = approval.recovery_state === "recovered";
      return html`<div class="message tool"><div class="bubble approval">
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
