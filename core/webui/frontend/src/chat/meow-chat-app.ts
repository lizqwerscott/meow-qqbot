import { css, html, LitElement } from "lit";
import { customElement, state } from "lit/decorators.js";
import { compactChatSession, createChatSession, discardChatResource, listChatSessions, listExternalChatSessions, loadChatAudit, loadChatOptions, loadChatSession, loadPendingChatApprovals, loadTurns, openChatEvents, renameChatSession, resolveChatApproval, submitChatTurn, uploadChatResource } from "../api/chat-api";
import type { ChatAuditEntry, ChatCompactionResult, ChatControlOption, ChatModelOption, ChatSession, ContentBlock, StreamEvent, Turn } from "../contracts/chat";
import "./meow-transcript";
import { createRequestId } from "./request-id";
import { mergeUniqueSessions } from "./transcript-state";

type OlderScroll = { scrollHeight: number; scrollTop: number };
type RetrySubmission = {
  content: string;
  resources: Record<string, unknown>[];
  mode: "agent" | "chat";
  modelGroup: string;
  reasoningEffort: string;
};

@customElement("meow-chat-app")
export class MeowChatApp extends LitElement {
  @state() private sessions: ChatSession[] = [];
  @state() private sessionsCursor: string | null = null;
  @state() private sessionsHasMore = false;
  @state() private sessionsLoadingMore = false;
  @state() private selectedSessionId = "";
  @state() private turns: Turn[] = [];
  @state() private cutoff: number | undefined;
  @state() private olderCursor: number | undefined;
  @state() private hasMore = false;
  @state() private loading = true;
  @state() private sending = false;
  @state() private uploading = false;
  @state() private uploads: Record<string, unknown>[] = [];
  @state() private removingUploads = new Set<string>();
  @state() private creating = false;
  @state() private connected = false;
  @state() private unread = 0;
  @state() private newSessionMode: "agent" | "chat" = "agent";
  @state() private composerMode: "agent" | "chat" = "agent";
  @state() private modelOptions: ChatModelOption[] = [];
  @state() private chatControls: Record<string, ChatControlOption> = {};
  @state() private modelGroup = "auto";
  @state() private reasoningEffort = "provider";
  @state() private editingTitle = false;
  @state() private titleDraft = "";
  @state() private sidebarOpen = false;
  @state() private auditOpen = false;
  @state() private auditLoading = false;
  @state() private auditEntries: ChatAuditEntry[] = [];
  @state() private auditError = "";
  @state() private compacting = false;
  @state() private compactionNotice = "";
  @state() private error = "";
  @state() private retrySubmissions: Record<string, RetrySubmission> = {};
  @state() private retryingTurnId = "";
  @state() private settingsOpen = false;
  private composerText = "";
  private eventSource: EventSource | undefined;
  private lastEventId = "";
  private latestSequence = 0;
  private followingTail = true;
  private hydrating = false;
  private pendingEvents: StreamEvent[] = [];

  static styles = css`
    :host { display: block; height: 100vh; color: #222738; background: #f7f8fc; font: 14px/1.5 Inter, system-ui, sans-serif; }
    .layout { display: grid; grid-template-columns: 360px minmax(0, 1fr); height: 100%; }
    aside { overflow-y: auto; padding: 22px 14px; background: #fff; border-right: 1px solid #e9ebf2; }
    h1 { margin: 0 10px 24px; font-size: 19px; }
    .new-session-row { display: flex; gap: 6px; margin: 0 10px 16px; }
    .new-session { flex: 1; border: 0; border-radius: 10px; padding: 10px 12px; color: #fff; background: #5865f2; cursor: pointer; }
    .mode-select { min-width: 76px; border: 1px solid #dfe3ed; border-radius: 10px; padding: 0 5px; color: #596174; background: #fff; }
    .new-session:disabled { cursor: wait; opacity: .65; }
    .session { display: block; width: 100%; border: 0; border-radius: 12px; padding: 11px 12px; color: inherit; background: transparent; text-align: left; cursor: pointer; }
    .session:hover, .session.active { background: #eef0ff; color: #4654c8; }
    .session-title { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .session-id { display: block; margin-top: 4px; color: #9aa2b4; font: 10px/1.35 ui-monospace, SFMono-Regular, monospace; overflow-wrap: anywhere; white-space: normal; word-break: break-all; }
    .session-mode { display: block; margin-top: 3px; color: #8b93a7; font-size: 12px; }
    .session-list-more { padding: 10px; color: #8b93a7; font-size: 12px; text-align: center; }
    .sidebar-section { margin: 20px 10px 7px; color: #8b93a7; font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; }
    .sidebar-links { display: grid; gap: 2px; margin: 0 4px; }
    .sidebar-links a { border-radius: 9px; padding: 8px 9px; color: #596174; text-decoration: none; }
    .sidebar-links a:hover { color: #4654c8; background: #eef0ff; }
    main { min-width: 0; min-height: 0; display: grid; grid-template-rows: auto minmax(0, 1fr) auto; }
    header { position: relative; display: flex; align-items: center; justify-content: space-between; padding: 18px 28px; background: rgba(247,248,252,.92); border-bottom: 1px solid #e9ebf2; }
    header strong { font-size: 16px; }
    .header-title, .header-controls, .mode-control { display: flex; align-items: center; gap: 8px; }
    .menu-toggle { display: none; border: 0; border-radius: 8px; padding: 5px 8px; color: #596174; background: #eef0f4; cursor: pointer; }
    .status { color: #8b93a7; font-size: 12px; }
    .status.connected { color: #32855d; }
    .status.disconnected { color: #ad6a30; }
    .edit-title, .title-action { border: 0; border-radius: 8px; padding: 4px 7px; color: #5865f2; background: #eef0ff; cursor: pointer; }
    .title-action.secondary { color: #70788c; background: #eef0f4; }
    .audit-toggle, .compact-button { border: 0; border-radius: 8px; padding: 5px 8px; color: #596174; background: #eef0f4; cursor: pointer; }
    .compact-button:hover { color: #4654c8; background: #eef0ff; }
    .compact-button:disabled { cursor: wait; opacity: .55; }
    .audit-toggle[aria-expanded="true"] { color: #4654c8; background: #eef0ff; }
    .settings-toggle { border: 0; border-radius: 8px; padding: 5px 8px; color: #596174; background: #eef0f4; cursor: pointer; }
    .settings-toggle[aria-expanded="true"] { color: #4654c8; background: #eef0ff; }
    .settings-panel { position: absolute; z-index: 6; top: calc(100% + 8px); right: 94px; width: min(330px, calc(100vw - 32px)); border: 1px solid #dfe3ed; border-radius: 12px; padding: 12px; background: #fff; box-shadow: 0 12px 28px rgba(39, 48, 85, .16); }
    .settings-heading { margin: 0 0 10px; color: #343b4e; font-size: 13px; }
    .settings-row { display: grid; gap: 4px; margin-top: 10px; color: #596174; font-size: 12px; }
    .settings-row select { border: 1px solid #dfe3ed; border-radius: 8px; padding: 6px 8px; color: #596174; background: #fff; font: inherit; }
    .settings-row select:disabled { color: #a1a8b8; background: #f5f6f9; }
    .settings-help { color: #8b93a7; font-size: 11px; }
    .settings-panel .compact-button { justify-self: start; margin-top: 12px; }
    .audit-panel { position: absolute; z-index: 5; top: calc(100% + 8px); right: 20px; width: min(360px, calc(100vw - 32px)); max-height: min(480px, calc(100vh - 130px)); overflow-y: auto; border: 1px solid #dfe3ed; border-radius: 12px; padding: 8px; background: #fff; box-shadow: 0 12px 28px rgba(39, 48, 85, .16); }
    .audit-heading { margin: 5px 7px 8px; color: #596174; font-size: 13px; }
    .audit-list { display: grid; gap: 4px; }
    .audit-entry { padding: 8px; border-radius: 8px; background: #f7f8fc; }
    .audit-action { display: block; color: #343b4e; font-size: 13px; font-weight: 600; }
    .audit-meta { display: block; margin-top: 2px; color: #8b93a7; font-size: 12px; }
    .audit-empty { padding: 14px 8px; color: #8b93a7; text-align: center; }
    .notice { margin: 10px 24px 0; padding: 9px 12px; border: 1px solid #dfe3ed; border-radius: 10px; color: #596174; background: #fff; font-size: 12px; }
    .title-input { width: min(360px, 45vw); border: 1px solid #cdd3e2; border-radius: 8px; padding: 6px 9px; color: inherit; background: #fff; font: inherit; }
    .mode-control { color: #8b93a7; font-size: 12px; }
    .mode-control select { border: 1px solid #dfe3ed; border-radius: 8px; padding: 5px 7px; color: #596174; background: #fff; }
    meow-transcript { min-height: 0; }
    .error { margin: 24px; padding: 14px 16px; border: 1px solid #f0b9b9; border-radius: 12px; color: #9b3838; background: #fff5f5; }
    .empty { display: grid; place-items: center; height: 100%; color: #8b93a7; }
    .readonly-notice { padding: 14px 24px 20px; color: #6f7890; background: #f7f8fc; font-size: 12px; text-align: center; }
    .composer-wrap { position: relative; padding: 12px 24px 20px; background: linear-gradient(transparent, #f7f8fc 18%); }
    .unread { position: absolute; left: 50%; top: -14px; transform: translateX(-50%); border: 0; border-radius: 999px; padding: 6px 12px; color: #4654c8; background: #eef0ff; box-shadow: 0 4px 16px rgba(63, 76, 170, .15); cursor: pointer; }
    form { display: flex; flex-direction: column; gap: 8px; max-width: 860px; margin: 0 auto; padding: 8px; border: 1px solid #dfe3ed; border-radius: 18px; background: #fff; box-shadow: 0 8px 24px rgba(39, 48, 85, .06); }
    .composer-row { display: flex; gap: 10px; align-items: flex-end; }
    .composer-tools { display: flex; align-items: center; gap: 8px; }
    .attach-button { border: 1px solid #dfe3ed; color: #596174; background: #fff; }
    .attach-button:hover { background: #f3f4f9; }
    .file-input { display: none; }
    .upload-list { display: flex; flex-wrap: wrap; gap: 6px; padding: 2px 4px 0; }
    .upload-chip { display: inline-flex; align-items: center; gap: 5px; max-width: 100%; border-radius: 999px; padding: 5px 8px 5px 10px; color: #596174; background: #f0f2f8; font-size: 12px; }
    .upload-chip span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .upload-chip button { border: 0; padding: 0 2px; color: #8b93a7; background: transparent; cursor: pointer; }
    textarea { flex: 1; min-height: 24px; max-height: 140px; resize: vertical; border: 0; outline: 0; padding: 8px 10px; color: inherit; background: transparent; font: inherit; }
    form button { align-self: flex-end; border: 0; border-radius: 12px; padding: 9px 16px; color: #fff; background: #5865f2; cursor: pointer; }
    form button:disabled { cursor: not-allowed; opacity: .5; }
    @media (max-width: 720px) { .layout { grid-template-columns: 1fr; } aside { display: none; position: fixed; z-index: 10; inset: 0 auto 0 0; width: min(360px, 82vw); box-shadow: 12px 0 32px rgba(39, 48, 85, .18); } aside.open { display: block; } header { padding: 14px 18px; } .menu-toggle { display: inline-flex; } .composer-wrap { padding: 10px 12px 14px; } .header-controls { gap: 4px; } .mode-control { font-size: 0; } .mode-control select { font-size: 12px; } .audit-panel { right: 12px; } }
  `;

  connectedCallback() { super.connectedCallback(); void this.loadSessions(); }

  disconnectedCallback() {
    if (!this.sending) this.discardDraftUploads(this.selectedSessionId, this.uploads);
    this.eventSource?.close();
    this.eventSource = undefined;
    super.disconnectedCallback();
  }

  render() {
    const session = this.sessions.find((item) => item.session_id === this.selectedSessionId);
    return html`
      <div class="layout">
        <aside class=${this.sidebarOpen ? "open" : ""} @scroll=${this.handleSidebarScroll}>
          <h1>🐱 猫猫聊天</h1>
          <div class="new-session-row">
            <button class="new-session" ?disabled=${this.creating || this.sending || this.uploading || this.removingUploads.size > 0} @click=${this.createSession}>${this.creating ? "正在创建…" : "+ 新建会话"}</button>
            <select class="mode-select" .value=${this.newSessionMode} ?disabled=${this.creating || this.sending || this.uploading || this.removingUploads.size > 0} @change=${this.changeNewSessionMode} aria-label="新会话模式">
              <option value="agent">Agent</option><option value="chat">Chat</option>
            </select>
          </div>
          ${this.renderSessionSection("WebUI 会话", this.sessions.filter((item) => !item.read_only), "暂无 WebUI 会话")}
          ${this.renderSessionSection("外部渠道（只读）", this.sessions.filter((item) => item.read_only), "暂无外部渠道消息")}
          ${this.sessionsHasMore ? html`<div class="session-list-more">${this.sessionsLoadingMore ? "正在加载更多…" : "继续下滑加载更多"}</div>` : ""}
          <div class="sidebar-section">会话记录</div>
          <nav class="sidebar-links" aria-label="会话记录">
            <a href="/sessions/archived">📦 归档会话</a>
            <a href="/sessions">🗂️ 全部会话记录</a>
          </nav>
          <div class="sidebar-section">管理</div>
          <nav class="sidebar-links" aria-label="管理功能">
            <a href="/routing">🧭 路由审计</a>
            <a href="/emojis">😊 表情资源</a>
            <a href="/learners/jargons">🧠 学习数据</a>
            <a href="/media">🖼️ 媒体管理</a>
          </nav>
          <div class="sidebar-section">系统</div>
          <nav class="sidebar-links" aria-label="系统功能">
            <a href="/status">📊 系统状态</a>
            <a href="/settings/engagement">🛡️ 运行时设置</a>
          </nav>
        </aside>
        <main>
          <header>
            <div class="header-title">
              <button class="menu-toggle" @click=${this.toggleSidebar} aria-label="打开会话与管理菜单">☰</button>
              ${this.editingTitle && session
                ? html`<input class="title-input" .value=${this.titleDraft} @input=${this.updateTitleDraft} @keydown=${this.handleTitleKeydown} aria-label="会话标题" />
                    <button class="title-action" @click=${this.saveTitle}>保存</button>
                    <button class="title-action secondary" @click=${this.cancelTitleEdit}>取消</button>`
                : html`<strong>${session?.title || "聊天工作台"}</strong>${session && !session.read_only ? html`<button class="edit-title" @click=${this.beginTitleEdit} aria-label="修改会话标题">✎</button>` : ""}`}
            </div>
            <div class="header-controls">
              ${session && !session.read_only ? html`<button class="settings-toggle" @click=${this.toggleSettings} aria-expanded=${String(this.settingsOpen)}>设置</button>` : ""}
              ${session && !session.read_only ? html`<button class="audit-toggle" @click=${this.toggleAudit} aria-expanded=${String(this.auditOpen)}>审计</button>` : ""}
              <span class="status ${this.connected ? "connected" : "disconnected"}">${session?.read_only ? "只读预览" : session ? (this.connected ? "● 实时连接" : "○ 正在连接") : "只读预览"}</span>
            </div>
            ${session && this.settingsOpen ? html`<section class="settings-panel" aria-label="聊天设置">
              <h2 class="settings-heading">本次会话设置</h2>
              <label class="settings-row">模式
                <select .value=${this.composerMode} @change=${this.changeComposerMode} aria-label="本次消息模式">
                  <option value="agent">Agent</option>
                  <option value="chat" ?disabled=${session.mode === "agent"}>Chat</option>
                </select>
              </label>
              ${this.modelOptions.length > 1 ? html`<label class="settings-row">模型组
                <select .value=${this.modelGroup} @change=${this.changeModelGroup} aria-label="本次消息模型">
                  ${this.modelOptions.map((option) => html`<option value=${option.id}>${option.label}</option>`)}
                </select>
              </label>` : ""}
              ${this.renderThinkingControl(this.chatControls.thinking_effort)}
              ${this.renderUnavailableControl("快速模式", this.chatControls.quick_mode)}
              ${this.chatControls.context_compaction?.enabled
                ? html`<button class="compact-button" @click=${this.compactContext} ?disabled=${this.compacting} title="压缩较早的模型上下文，不会删除聊天记录">${this.compacting ? "压缩中…" : "压缩上下文"}</button>`
                : html`<div class="settings-help">${this.chatControls.context_compaction?.reason || "上下文压缩不可用"}</div>`}
            </section>` : ""}
            ${session && this.auditOpen ? html`<section class="audit-panel" aria-label="会话审计">
              <h2 class="audit-heading">会话审计</h2>
              ${this.auditLoading ? html`<div class="audit-empty">正在加载…</div>` : this.auditError ? html`<div class="audit-empty">${this.auditError}</div>` : this.auditEntries.length ? html`<div class="audit-list">
                ${this.auditEntries.map((entry) => html`<div class="audit-entry">
                  <span class="audit-action">${this.auditActionLabel(entry.action)}</span>
                  <span class="audit-meta">${this.auditDescription(entry)}${this.auditDescription(entry) ? " · " : ""}${this.auditTimestamp(entry.created_at)}</span>
                </div>`)}
              </div>` : html`<div class="audit-empty">暂无审计记录</div>`}
            </section>` : ""}
          </header>
          ${this.compactionNotice ? html`<div class="notice">${this.compactionNotice}</div>` : ""}
          ${this.error ? html`<div class="error">${this.error}</div>` : session ? html`
            <meow-transcript .turns=${this.turns} .retryableTurnIds=${new Set(Object.keys(this.retrySubmissions))} .loading=${this.loading} .hasMore=${this.hasMore} @load-older=${this.loadOlder} @tail-change=${this.updateTail} @approval-decision=${this.resolveApproval} @retry-turn=${this.retryTurn}></meow-transcript>
            ${session.read_only ? html`<div class="readonly-notice">只读查看 ${session.channel || "外部渠道"} 会话，发送入口已关闭。</div>` : html`<div class="composer-wrap">
              ${this.unread > 0 ? html`<button class="unread" @click=${this.returnToTail}>${this.unread} 条新消息</button>` : ""}
              <form @submit=${this.submitTurn}>
                ${this.uploads.length ? html`<div class="upload-list" aria-label="待发送附件">
                  ${this.uploads.map((resource, index) => html`<span class="upload-chip">
                    <span title=${this.resourceFilename(resource)}>${this.resourceFilename(resource)}</span>
                    <button type="button" @click=${() => this.removeUpload(index)} ?disabled=${this.removingUploads.has(this.uploadId(resource))} aria-label=${`移除 ${this.resourceFilename(resource)}`}>×</button>
                  </span>`)}
                </div>` : ""}
                <div class="composer-row">
                  <textarea .value=${this.composerText} placeholder="输入消息…" ?disabled=${this.sending || this.uploading} @input=${this.updateComposer} @keydown=${this.handleComposerKeydown}></textarea>
                  <div class="composer-tools">
                    <input class="file-input" type="file" multiple @change=${this.handleFileSelection} ?disabled=${this.sending || this.uploading} />
                    <button class="attach-button" type="button" @click=${this.openFilePicker} ?disabled=${this.sending || this.uploading}>${this.uploading ? "上传中…" : "附件"}</button>
                    <button type="submit" ?disabled=${this.sending || this.uploading || this.removingUploads.size > 0 || (!this.composerText.trim() && this.uploads.length === 0)}>${this.sending ? "发送中" : "发送"}</button>
                  </div>
                </div>
              </form>
            </div>`}
          ` : html`<div class="empty">选择一个 WebUI 会话开始查看</div>`}
        </main>
      </div>
    `;
  }

  private renderSessionSection(title: string, sessions: ChatSession[], empty: string) {
    return html`
      <div class="sidebar-section">${title}</div>
      ${sessions.length === 0 && !this.error ? html`<div class="status">${empty}</div>` : ""}
      ${sessions.map((item) => html`
        <button class="session ${item.session_id === this.selectedSessionId ? "active" : ""}" ?disabled=${this.sending || this.uploading || this.removingUploads.size > 0} @click=${() => this.selectSession(item.session_id)}>
              <span class="session-title">${item.title || "未命名会话"}</span><span class="session-id" title=${item.session_id}>${item.session_id}</span><span class="session-mode">${item.read_only ? "只读 · " : ""}${item.mode}</span>
        </button>
      `)}
    `;
  }

  private async loadSessions() {
    this.loading = true;
    try {
      const requestedSessionId = this.requestedSessionId();
      const requestedSession = requestedSessionId
        ? await loadChatSession(requestedSessionId).catch(() => null)
        : null;
      const [sessionPage, externalPage, options] = await Promise.all([listChatSessions(), listExternalChatSessions(), loadChatOptions()]);
      this.sessions = mergeUniqueSessions(
        [...sessionPage.items, ...externalPage.items],
        requestedSession ? [requestedSession] : [],
      );
      this.sessionsCursor = sessionPage.next_cursor;
      this.sessionsHasMore = sessionPage.has_more;
      this.modelOptions = options.model_groups;
      this.chatControls = options.controls;
      const initialSession = requestedSession || this.sessions[0];
      if (initialSession) await this.selectSession(initialSession.session_id);
    } catch (error) { this.error = error instanceof Error ? error.message : "无法加载会话"; }
    finally { this.loading = false; }
  }

  private requestedSessionId(): string {
    if (typeof window === "undefined") return "";
    return new URLSearchParams(window.location.search).get("session_id") || "";
  }

  private handleSidebarScroll = (event: Event) => {
    const sidebar = event.currentTarget as HTMLElement;
    if (sidebar.scrollTop + sidebar.clientHeight >= sidebar.scrollHeight - 160) {
      void this.loadMoreSessions();
    }
  };

  private async loadMoreSessions() {
    if (this.sessionsLoadingMore || !this.sessionsHasMore || !this.sessionsCursor) return;
    this.sessionsLoadingMore = true;
    try {
      const page = await listChatSessions(this.sessionsCursor);
      this.sessions = mergeUniqueSessions(this.sessions, page.items);
      this.sessionsCursor = page.next_cursor;
      this.sessionsHasMore = page.has_more;
    } catch (error) {
      this.error = error instanceof Error ? error.message : "无法加载更多会话";
    } finally {
      this.sessionsLoadingMore = false;
    }
  }

  private async createSession() {
    if (this.creating) return;
    this.creating = true; this.error = "";
    try {
      const session = await createChatSession("新会话", this.newSessionMode);
      this.sessions = [session, ...this.sessions.filter((item) => item.session_id !== session.session_id)];
      await this.selectSession(session.session_id);
    } catch (error) { this.error = error instanceof Error ? error.message : "无法创建会话"; }
    finally { this.creating = false; }
  }

  private async selectSession(sessionId: string) {
    if (sessionId !== this.selectedSessionId && (this.sending || this.uploading || this.removingUploads.size > 0)) {
      this.error = "当前附件操作尚未完成，请稍后再切换会话";
      return;
    }
    this.discardDraftUploads(this.selectedSessionId, this.uploads);
    this.eventSource?.close(); this.eventSource = undefined; this.connected = false;
    this.selectedSessionId = sessionId; this.turns = []; this.cutoff = undefined; this.olderCursor = undefined; this.hasMore = false;
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      url.searchParams.set("session_id", sessionId);
      window.history.replaceState({}, "", url);
    }
    this.sidebarOpen = false;
    const session = this.sessions.find((item) => item.session_id === sessionId);
    this.composerMode = session?.mode === "chat" ? "chat" : "agent";
    this.modelGroup = "auto";
    this.reasoningEffort = "provider";
    this.editingTitle = false; this.titleDraft = "";
    this.auditOpen = false; this.auditLoading = false; this.auditEntries = []; this.auditError = ""; this.settingsOpen = false;
    this.compacting = false; this.compactionNotice = "";
    this.unread = 0; this.followingTail = true; this.lastEventId = ""; this.latestSequence = 0; this.loading = true; this.error = ""; this.uploads = []; this.removingUploads = new Set(); this.retrySubmissions = {}; this.retryingTurnId = "";
    this.hydrating = true; this.pendingEvents = [];
    if (!session?.read_only) this.openEvents(sessionId);
    try {
      const page = await loadTurns(sessionId);
      this.turns = page.items; this.cutoff = page.cutoff_sequence; this.olderCursor = page.next_before_turn_sequence ?? undefined; this.hasMore = page.has_more;
      if (!session?.read_only) await this.loadPendingApprovals(sessionId);
      this.hydrating = false;
      for (const event of this.pendingEvents.splice(0)) this.handleStreamEvent(event);
      await this.updateComplete; this.transcript()?.scrollToBottom?.();
    } catch (error) { this.error = error instanceof Error ? error.message : "无法加载会话历史"; }
    finally { this.loading = false; }
  }

  private openEvents(sessionId: string) {
    this.eventSource = openChatEvents(sessionId, this.lastEventId, (event) => this.handleStreamEvent(event));
    this.eventSource.onopen = () => { this.connected = true; };
    this.eventSource.onerror = () => { this.connected = false; };
  }

  private handleStreamEvent(event: StreamEvent) {
    if (event.session_id !== this.selectedSessionId) return;
    if (this.hydrating) {
      this.pendingEvents.push(event);
      return;
    }
    if (event.type !== "resync_required" && event.sequence <= this.latestSequence) return;
    this.lastEventId = event.event_id; this.latestSequence = Math.max(this.latestSequence, event.sequence);
    if (event.type === "resync_required") {
      void this.recoverFromResync(event);
      return;
    }
    if (event.type === "session.updated") { this.applySessionUpdate(event.payload.session); return; }
    if (event.type === "turn.accepted") this.applyAccepted(event);
    if (event.type === "message.created") this.applyMessage(event, false);
    if (event.type === "message.delta") this.applyMessage(event, true);
    if (event.type === "turn.completed") {
      this.clearRetrySubmission(event.turn_id);
      this.updateTurnStatus(event, String(event.payload.status || "completed"));
      void this.correctCompletedTurn(event.turn_id);
    }
    if (event.type === "turn.failed") {
      this.updateTurnStatus(event, "failed");
      void this.correctCompletedTurn(event.turn_id);
    }
    if (event.type === "message.reset") this.applyMessage(event, false);
    if (event.type === "delivery.backpressure") {
      this.error = typeof event.payload.message === "string"
        ? event.payload.message
        : "当前会话任务较多，请稍后重试";
    }
    if (event.type.startsWith("tool.")) this.applyToolEvent(event);
    if (event.type === "approval.requested") this.applyApproval(event);
    if (event.type === "context.compacted") {
      this.compactionNotice = this.compactionDescription(event.payload as ChatCompactionResult);
    }
    if (event.type.startsWith("message.") || event.type === "turn.accepted") {
      if (!this.followingTail) this.unread += 1; else this.scrollAfterRender();
    }
  }

  private applySessionUpdate(value: unknown) {
    if (!value || typeof value !== "object") return;
    const session = value as ChatSession;
    this.sessions = this.sessions.map((item) => item.session_id === session.session_id ? session : item);
    if (session.session_id === this.selectedSessionId) {
      this.composerMode = session.mode === "chat" ? "chat" : "agent";
    }
  }

  private applyAccepted(event: StreamEvent) {
    const payload = event.payload;
    const content = typeof payload.content === "string" ? payload.content : "";
    const senderId = typeof payload.sender_id === "string" ? payload.sender_id : "";
    const blocks = this.resourceBlocks(payload.resources, "user", senderId);
    const mode = payload.mode === "chat" ? "chat" : "agent";
    const modelGroup = typeof payload.model_group === "string" ? payload.model_group : "auto";
    const reasoningEffort = typeof payload.reasoning_effort === "string" ? payload.reasoning_effort : "provider";
    const resources = Array.isArray(payload.resources)
      ? payload.resources.filter((resource): resource is Record<string, unknown> => Boolean(resource && typeof resource === "object"))
      : [];
    this.retrySubmissions = {
      ...this.retrySubmissions,
      [event.turn_id]: { content, resources, mode, modelGroup, reasoningEffort },
    };
    if (content) blocks.unshift({ type: "text", role: "user", sender_id: senderId, text: content });
    this.upsertTurn(event, blocks, "running");
  }

  private applyApproval(event: StreamEvent) {
    this.upsertTurn(
      event,
      [{ type: "approval", role: "tool", approval: event.payload }],
      "approval",
    );
  }

  private applyToolEvent(event: StreamEvent) {
    const payload = event.payload;
    const toolName = typeof payload.tool_name === "string" ? payload.tool_name : "工具";
    const toolCallId = typeof payload.tool_call_id === "string" ? payload.tool_call_id : "";
    const status = typeof payload.status === "string"
      ? payload.status
      : event.type.replace("tool.", "");
    const current = this.turns.find((turn) => turn.turn_id === event.turn_id);
    const previous = current?.blocks.find(
      (item) => item.type === "tool" && item.tool_call_id === toolCallId,
    );
    const metadata = typeof payload.metadata === "object" && payload.metadata
      ? payload.metadata as Record<string, unknown>
      : {};
    const argumentsValue = payload.arguments ?? metadata.arguments ?? previous?.arguments;
    const result = typeof payload.result === "string"
      ? payload.result
      : previous?.result || (typeof metadata.result === "string" ? metadata.result : "");
    const resources = Array.isArray(payload.resources) && payload.resources.length
      ? payload.resources.filter((resource): resource is Record<string, unknown> => Boolean(resource && typeof resource === "object"))
      : previous?.resources;
    const text = `${toolName} · ${status}`;
    const block: ContentBlock = {
      type: "tool",
      role: "tool",
      text,
      tool_call_id: toolCallId,
      tool_name: toolName,
      arguments: argumentsValue,
      result,
      resources,
      status,
      metadata: { ...(previous?.metadata || {}), ...metadata },
    };
    if (!current) {
      this.upsertTurn(event, [block], "running");
      return;
    }
    const index = toolCallId
      ? current.blocks.findIndex(
          (item) => item.type === "tool" && item.tool_call_id === toolCallId,
        )
      : -1;
    const blocks = index >= 0
      ? current.blocks.map((item, itemIndex) => itemIndex === index ? block : item)
      : [...current.blocks, block];
    this.turns = this.turns.map((turn) => turn.turn_id === event.turn_id
      ? { ...turn, blocks }
      : turn);
  }

  private async resolveApproval(event: CustomEvent<{ sessionKey: string; decision: "allow-once" | "allow-always" | "deny" }>) {
    try {
      await resolveChatApproval(event.detail.sessionKey, event.detail.decision);
    } catch (error) {
      this.error = error instanceof Error ? error.message : "审批处理失败";
    }
  }

  private async retryTurn(event: CustomEvent<{ turnId: string }>) {
    const oldTurnId = event.detail.turnId;
    const submission = this.retrySubmissions[oldTurnId];
    if (!submission || !this.selectedSessionId || this.retryingTurnId) return;
    this.retryingTurnId = oldTurnId;
    this.error = "";
    try {
      await submitChatTurn(
        this.selectedSessionId,
        submission.content,
        createRequestId("webui-retry"),
        submission.mode,
        submission.resources,
        submission.modelGroup,
        submission.reasoningEffort,
      );
      this.clearRetrySubmission(oldTurnId);
      this.followingTail = true;
      this.scrollAfterRender();
    } catch (error) {
      this.error = error instanceof Error ? error.message : "消息重试失败";
    } finally {
      this.retryingTurnId = "";
    }
  }

  private applyMessage(event: StreamEvent, delta: boolean) {
    const payload = event.payload; const content = typeof payload.content === "string" ? payload.content : ""; const reasoning = typeof payload.reasoning_content === "string" ? payload.reasoning_content : ""; const resources = this.resourceBlocks(payload.resources, "assistant");
    const current = this.turns.find((turn) => turn.turn_id === event.turn_id);
    if (!current) { this.upsertTurn(event, [...(reasoning ? [{ type: "reasoning", role: "assistant", status: "running", text: reasoning }] : []), ...(content ? [{ type: "text", role: "assistant", text: content }] : []), ...resources], "running"); return; }
    let blocks = current.blocks;
    if (!delta) blocks = [...blocks.filter((block) => block.role !== "assistant"), ...(reasoning ? [{ type: "reasoning", role: "assistant", status: "running", text: reasoning }] : []), ...(content ? [{ type: "text", role: "assistant", text: content }] : []), ...resources];
    else if (content) {
      const index = [...blocks].reverse().findIndex((block) => block.role === "assistant" && block.type === "text"); const targetIndex = index < 0 ? -1 : blocks.length - 1 - index;
      if (targetIndex < 0) blocks = [...blocks, { type: "text", role: "assistant", text: content }];
      else blocks = blocks.map((block, blockIndex) => blockIndex === targetIndex ? { ...block, text: `${block.text || ""}${content}` } : block);
      blocks = [...blocks, ...resources];
    } else if (reasoning) {
      const targetIndex = [...blocks].reverse().findIndex((block) => block.type === "reasoning" && block.role === "assistant");
      if (targetIndex < 0) blocks = [...blocks, { type: "reasoning", role: "assistant", status: "running", text: reasoning }];
      else {
        const index = blocks.length - 1 - targetIndex;
        blocks = blocks.map((block, blockIndex) => blockIndex === index ? { ...block, text: `${block.text || ""}${reasoning}` } : block);
      }
      blocks = [...blocks, ...resources];
    } else blocks = [...blocks, ...resources];
    this.turns = this.turns.map((turn) => turn.turn_id === event.turn_id ? { ...turn, blocks } : turn);
  }

  private updateTurnStatus(event: StreamEvent, status: string) {
    this.turns = this.turns.map((turn) => turn.turn_id === event.turn_id
      ? {
        ...turn,
        status,
        blocks: turn.blocks.map((block) => block.type === "reasoning" ? { ...block, status } : block),
      }
      : turn);
  }

  private clearRetrySubmission(turnId: string) {
    if (!this.retrySubmissions[turnId]) return;
    const next = { ...this.retrySubmissions };
    delete next[turnId];
    this.retrySubmissions = next;
  }

  private upsertTurn(event: StreamEvent, blocks: ContentBlock[], status: string) {
    const existing = this.turns.find((turn) => turn.turn_id === event.turn_id);
    if (existing) { this.turns = this.turns.map((turn) => turn.turn_id === event.turn_id ? { ...turn, blocks, status } : turn); return; }
    const nextSequence = Math.max(0, ...this.turns.map((turn) => turn.turn_sequence)) + 1;
    this.turns = [...this.turns, { turn_id: event.turn_id, turn_sequence: nextSequence, created_at: event.occurred_at, status, turn_kind: "ai", blocks, metadata: { live: true } }];
  }

  private resourceBlocks(value: unknown, role: string, senderId = ""): ContentBlock[] {
    if (!Array.isArray(value)) return [];
    return value.filter((resource): resource is Record<string, unknown> => Boolean(resource && typeof resource === "object")).map((resource) => ({ type: String(resource.resource_type || "file"), role, sender_id: senderId, resource }));
  }

  private async reloadHistory(): Promise<boolean> {
    if (!this.selectedSessionId) return false;
    try {
      const page = await loadTurns(this.selectedSessionId); this.turns = page.items; this.cutoff = page.cutoff_sequence; this.olderCursor = page.next_before_turn_sequence ?? undefined; this.hasMore = page.has_more; await this.loadPendingApprovals(this.selectedSessionId); this.unread = 0; this.scrollAfterRender();
      return true;
    } catch (error) { this.error = error instanceof Error ? error.message : "历史已失同步，请刷新页面"; return false; }
  }

  private async loadPendingApprovals(sessionId: string) {
    try {
      const approvals = await loadPendingChatApprovals(sessionId);
      if (this.selectedSessionId !== sessionId) return;
      for (const approval of approvals) {
        const sessionKey = typeof approval.session_key === "string" ? approval.session_key : "";
        if (!sessionKey) continue;
        this.applyApproval({
          event_id: `approval-recovery-${sessionKey}`,
          session_id: sessionId,
          turn_id: sessionKey,
          type: "approval.requested",
          sequence: 0,
          occurred_at: typeof approval.created_at === "number" ? approval.created_at : Date.now() / 1000,
          payload: approval,
        });
      }
    } catch (error) {
      if (this.selectedSessionId === sessionId) {
        this.error = error instanceof Error ? error.message : "无法恢复待审批操作";
      }
    }
  }

  private async recoverFromResync(event: StreamEvent) {
    if (!this.selectedSessionId) return;
    const sessionId = this.selectedSessionId;
    this.hydrating = true;
    this.pendingEvents = [];
    try {
      if (!await this.reloadHistory()) throw new Error("history reload failed");
      this.hydrating = false;
      for (const pending of this.pendingEvents.splice(0)) this.handleStreamEvent(pending);
      this.eventSource?.close();
      this.eventSource = undefined;
      this.connected = false;
      this.lastEventId =
        typeof event.payload.latest_event_id === "string"
          ? event.payload.latest_event_id
          : "";
      this.openEvents(sessionId);
    } catch {
      this.hydrating = false;
      this.error = "实时事件已失同步，请刷新页面";
    }
  }

  private async correctCompletedTurn(turnId: string) {
    if (!this.selectedSessionId) return;
    try {
      const page = await loadTurns(this.selectedSessionId, { cutoff: this.cutoff });
      const freshTurn = page.items.find((turn) => turn.turn_id === turnId);
      if (!freshTurn) return;
      this.turns = this.turns.map((turn) => turn.turn_id === turnId ? freshTurn : turn);
      this.scrollAfterRender();
    } catch {
      this.error = "消息已完成，但历史校正失败；刷新后可恢复";
    }
  }

  private async loadOlder(event: CustomEvent<OlderScroll>) {
    if (!this.selectedSessionId || !this.hasMore || this.olderCursor === undefined) return;
    this.loading = true;
    try {
      const page = await loadTurns(this.selectedSessionId, { cutoff: this.cutoff, before: this.olderCursor }); this.turns = [...page.items, ...this.turns]; this.olderCursor = page.next_before_turn_sequence ?? undefined; this.hasMore = page.has_more; await this.updateComplete; this.transcript()?.restoreScroll?.(event.detail);
    } catch (error) { this.error = error instanceof Error ? error.message : "无法加载更早消息"; }
    finally { this.loading = false; }
  }

  private updateTail(event: CustomEvent<{ followingTail: boolean }>) { this.followingTail = event.detail.followingTail; if (this.followingTail) this.unread = 0; }
  private returnToTail() { this.followingTail = true; this.unread = 0; this.transcript()?.scrollToBottom?.(); }
  private scrollAfterRender() { void this.updateComplete.then(() => { if (this.followingTail) this.transcript()?.scrollToBottom?.(); }); }

  private transcript() { return this.shadowRoot?.querySelector("meow-transcript") as (HTMLElement & { scrollToBottom?: () => void; restoreScroll?: (value: OlderScroll) => void }) | null; }
  private updateComposer(event: Event) { this.composerText = (event.target as HTMLTextAreaElement).value; }
  private toggleSidebar = () => { this.sidebarOpen = !this.sidebarOpen; };
  private toggleSettings = () => { this.settingsOpen = !this.settingsOpen; this.auditOpen = false; };
  private async toggleAudit() {
    if (this.auditOpen) {
      this.auditOpen = false;
      return;
    }
    if (!this.selectedSessionId) return;
    const sessionId = this.selectedSessionId;
    this.auditOpen = true;
    this.settingsOpen = false;
    this.auditLoading = true;
    this.auditError = "";
    try {
      const entries = await loadChatAudit(sessionId);
      if (this.selectedSessionId === sessionId) this.auditEntries = entries;
    } catch (error) {
      if (this.selectedSessionId === sessionId) {
        this.auditError = error instanceof Error ? error.message : "无法加载审计记录";
      }
    } finally {
      if (this.selectedSessionId === sessionId) this.auditLoading = false;
    }
  }
  private async compactContext() {
    if (!this.selectedSessionId || this.compacting) return;
    const sessionId = this.selectedSessionId;
    this.compacting = true;
    this.compactionNotice = "正在压缩较早的模型上下文…";
    try {
      const result = await compactChatSession(sessionId);
      if (this.selectedSessionId !== sessionId) return;
      this.compactionNotice = this.compactionDescription(result);
      if (this.auditOpen) this.auditEntries = await loadChatAudit(sessionId);
    } catch (error) {
      if (this.selectedSessionId === sessionId) {
        this.compactionNotice = error instanceof Error ? error.message : "上下文压缩失败";
      }
    } finally {
      if (this.selectedSessionId === sessionId) this.compacting = false;
    }
  }
  private renderUnavailableControl(label: string, control?: ChatControlOption) {
    return html`<label class="settings-row">${label}
      <select disabled aria-label=${label} title=${control?.reason || "当前不可用"}>
        <option>${control?.default || "不可用"}</option>
      </select>
      <span class="settings-help">${control?.reason || "当前不可用"}</span>
    </label>`;
  }
  private renderThinkingControl(control?: ChatControlOption) {
    const options = control?.options || [];
    if (!control?.enabled || !options.length) return this.renderUnavailableControl("思考强度", control);
    return html`<label class="settings-row">思考强度
      <select .value=${this.reasoningEffort} @change=${this.changeReasoningEffort} aria-label="本次消息思考强度">
        <option value="provider">跟随模型配置</option>
        ${options.map((option) => html`<option value=${option}>${option === "none" ? "关闭" : option === "low" ? "低" : option === "medium" ? "中" : option === "high" ? "高" : option}</option>`)}
      </select>
      <span class="settings-help">${control.reason}</span>
    </label>`;
  }
  private compactionDescription(result: ChatCompactionResult) {
    if (!result.changed) return "当前没有可安全压缩的旧上下文。聊天记录不会被删除。";
    const saved = result.saved_tokens > 0 ? `，约节省 ${result.saved_tokens} tokens` : "";
    return `上下文已压缩${saved}；聊天记录仍可继续查看。`;
  }
  private auditActionLabel(action: string) {
    return {
      "session.created": "创建会话",
      "session.renamed": "修改标题",
      "turn.submitted": "提交消息",
      "attachment.uploaded": "上传附件",
      "attachment.discarded": "移除附件",
      "context.compacted": "压缩上下文",
    }[action] || "会话操作";
  }
  private auditDescription(entry: ChatAuditEntry) {
    const details = entry.details;
    if (entry.action === "turn.submitted") {
      const mode = details.mode === "chat" ? "Chat" : details.mode === "agent" ? "Agent" : "";
      const resourceCount = typeof details.resource_count === "number" ? details.resource_count : 0;
      const effort = typeof details.reasoning_effort === "string" && details.reasoning_effort !== "provider"
        ? `思考${details.reasoning_effort}`
        : "";
      return [mode, effort, resourceCount ? `${resourceCount} 个附件` : ""].filter(Boolean).join(" · ");
    }
    if (entry.action === "attachment.uploaded") {
      const type = details.resource_type === "image" ? "图片" : "文件";
      return typeof details.size === "number" ? `${type} · ${this.formatBytes(details.size)}` : type;
    }
    if (entry.action === "context.compacted") {
      return details.changed === true && typeof details.saved_tokens === "number"
        ? `节省约 ${details.saved_tokens} tokens`
        : "没有可压缩的旧上下文";
    }
    return entry.action === "session.created" && details.mode === "chat" ? "Chat 模式" : entry.action === "session.created" ? "Agent 模式" : "";
  }
  private auditTimestamp(timestamp: number) {
    return new Intl.DateTimeFormat("zh-CN", { dateStyle: "short", timeStyle: "short" }).format(new Date(timestamp * 1000));
  }
  private formatBytes(size: number) {
    if (size < 1024) return `${size} B`;
    if (size < 1024 * 1024) return `${(size / 1024).toFixed(1)} KB`;
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }
  private openFilePicker() {
    this.shadowRoot?.querySelector<HTMLInputElement>(".file-input")?.click();
  }
  private async handleFileSelection(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = Array.from(input.files || []);
    input.value = "";
    if (!files.length || !this.selectedSessionId || this.uploading) return;
    if (this.uploads.length + files.length > 10) {
      this.error = "一条消息最多附带 10 个附件";
      return;
    }
    this.uploading = true;
    this.error = "";
    const sessionId = this.selectedSessionId;
    try {
      for (const file of files) {
        const resource = await uploadChatResource(sessionId, file);
        if (this.selectedSessionId === sessionId) this.uploads = [...this.uploads, resource];
      }
    } catch (error) {
      this.error = error instanceof Error ? error.message : "附件上传失败";
    } finally {
      this.uploading = false;
    }
  }
  private async removeUpload(index: number) {
    const resource = this.uploads[index];
    if (!resource) return;
    const uploadId = this.uploadId(resource);
    if (!uploadId || !this.selectedSessionId) {
      this.uploads = this.uploads.filter((_resource, resourceIndex) => resourceIndex !== index);
      return;
    }
    this.removingUploads = new Set([...this.removingUploads, uploadId]);
    try {
      await discardChatResource(
        this.selectedSessionId,
        String(resource.media_uri || ""),
        uploadId,
      );
      this.uploads = this.uploads.filter((_resource, resourceIndex) => resourceIndex !== index);
    } catch (error) {
      this.error = error instanceof Error ? error.message : "附件移除失败";
    } finally {
      this.removingUploads = new Set(
        [...this.removingUploads].filter((value) => value !== uploadId),
      );
    }
  }
  private discardDraftUploads(sessionId: string, resources: Record<string, unknown>[]) {
    if (!sessionId || !resources.length) return;
    void Promise.allSettled(resources.map((resource) => {
      const uploadId = this.uploadId(resource);
      return uploadId
        ? discardChatResource(sessionId, String(resource.media_uri || ""), uploadId)
        : Promise.resolve();
    }));
  }
  private uploadId(resource: Record<string, unknown>) {
    const extra = resource.extra;
    return extra && typeof extra === "object" && typeof (extra as Record<string, unknown>).webui_upload_id === "string"
      ? (extra as Record<string, string>).webui_upload_id
      : "";
  }
  private resourceFilename(resource: Record<string, unknown>) {
    return typeof resource.filename === "string" && resource.filename ? resource.filename : "附件";
  }
  private changeNewSessionMode(event: Event) {
    const mode = (event.target as HTMLSelectElement).value;
    if (mode === "agent" || mode === "chat") this.newSessionMode = mode;
  }
  private changeComposerMode(event: Event) {
    const select = event.target as HTMLSelectElement;
    const mode = select.value;
    const session = this.sessions.find((item) => item.session_id === this.selectedSessionId);
    if (mode === "agent" || (mode === "chat" && session?.mode !== "agent")) {
      this.composerMode = mode;
    } else {
      select.value = this.composerMode;
    }
  }
  private changeModelGroup(event: Event) {
    const modelGroup = (event.target as HTMLSelectElement).value;
    if (this.modelOptions.some((option) => option.id === modelGroup)) {
      this.modelGroup = modelGroup;
      this.reasoningEffort = "provider";
    }
  }
  private changeReasoningEffort(event: Event) {
    const value = (event.target as HTMLSelectElement).value;
    const options = this.chatControls.thinking_effort?.options || [];
    if (value === "provider" || options.includes(value)) this.reasoningEffort = value;
  }
  private beginTitleEdit() {
    const session = this.sessions.find((item) => item.session_id === this.selectedSessionId);
    if (!session) return;
    this.titleDraft = session.title;
    this.editingTitle = true;
  }
  private updateTitleDraft(event: Event) {
    this.titleDraft = (event.target as HTMLInputElement).value;
  }
  private handleTitleKeydown(event: KeyboardEvent) {
    if (event.key === "Enter") {
      event.preventDefault();
      void this.saveTitle(event);
    } else if (event.key === "Escape") {
      this.cancelTitleEdit();
    }
  }
  private cancelTitleEdit = () => {
    this.editingTitle = false;
    this.titleDraft = "";
  };
  private async saveTitle(event: Event) {
    event.preventDefault();
    if (!this.selectedSessionId || !this.titleDraft.trim()) return;
    try {
      const session = await renameChatSession(this.selectedSessionId, this.titleDraft.trim());
      this.applySessionUpdate(session);
      this.cancelTitleEdit();
    } catch (error) {
      this.error = error instanceof Error ? error.message : "无法修改会话标题";
    }
  }
  private handleComposerKeydown(event: KeyboardEvent) { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void this.submitTurn(event); } }

  private async submitTurn(event: Event) {
    event.preventDefault(); const content = this.composerText.trim(); if ((!content && !this.uploads.length) || !this.selectedSessionId || this.sending || this.uploading || this.removingUploads.size > 0) return;
    this.sending = true; this.error = "";
    try { await submitChatTurn(this.selectedSessionId, content, createRequestId("webui"), this.composerMode, this.uploads, this.modelGroup, this.reasoningEffort); this.composerText = ""; this.uploads = []; this.followingTail = true; this.scrollAfterRender(); }
    catch (error) { this.error = error instanceof Error ? error.message : "消息发送失败"; }
    finally { this.sending = false; }
  }
}
