(function () {
  const menuBtn = document.getElementById("menu-btn");
  const sidebar = document.querySelector(".sidebar");
  const overlay = document.getElementById("menu-overlay");

  function openMenu() {
    sidebar.classList.add("open");
    overlay.classList.add("show");
    document.body.style.overflow = "hidden";
  }

  function closeMenu() {
    if (!sidebar || !overlay) return;
    sidebar.classList.remove("open");
    overlay.classList.remove("show");
    document.body.style.overflow = "";
  }

  if (menuBtn && sidebar && overlay) {
    menuBtn.addEventListener("click", openMenu);
    overlay.addEventListener("click", closeMenu);
  }

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeMenu();
  });

  var mediaRows = document.querySelectorAll(".media-row[data-href]");
  for (var j = 0; j < mediaRows.length; j++) {
    mediaRows[j].addEventListener("click", function (e) {
      if (e.target.closest("button, form, a")) return;
      window.location.href = this.dataset.href;
    });
    mediaRows[j].addEventListener("keydown", function (e) {
      if ((e.key === "Enter" || e.key === " ") && e.target === this) {
        e.preventDefault();
        window.location.href = this.dataset.href;
      }
    });
  }

  if (sidebar) {
    var links = sidebar.querySelectorAll("nav a");
    for (var i = 0; i < links.length; i++) {
      links[i].addEventListener("click", closeMenu);
    }
  }

  var sessionView = document.querySelector("[data-session-view]");
  if (sessionView) {
    sessionView.addEventListener("change", function () {
      if (this.value) window.location.href = this.value;
    });
  }

  var sessionStream = document.querySelector("[data-session-stream]");
  var olderButton = document.querySelector("[data-load-older]");
  var olderWrap = document.querySelector("[data-load-older-wrap]");
  if (sessionStream && olderButton) {
    var cursorCutoff = sessionStream.dataset.cursorCutoff || "";
    var cursorBefore = parseInt(sessionStream.dataset.cursorBefore || "", 10);
    var cursorMode = cursorCutoff !== "" && !Number.isNaN(cursorBefore);
    var currentPage = parseInt(sessionStream.dataset.page || "1", 10);
    var totalPages = parseInt(sessionStream.dataset.totalPages || "1", 10);
    var pageSize = parseInt(sessionStream.dataset.pageSize || "50", 10);
    var loadingOlder = false;
    var hasMoreOlder = currentPage < totalPages;
    var scrollContainer = document.querySelector(".main") || window;

    function updateOlderButton() {
      var available = cursorMode ? hasMoreOlder : currentPage < totalPages;
      olderButton.disabled = loadingOlder || !available;
      if (olderWrap) olderWrap.classList.toggle("is-hidden", !available && !loadingOlder);
      olderButton.setAttribute("aria-busy", loadingOlder ? "true" : "false");
    }

    function scrollTopOf(element) {
      return element === window ? window.scrollY : element.scrollTop;
    }

    function scrollHeightOf(element) {
      return element === window ? document.documentElement.scrollHeight : element.scrollHeight;
    }

    function scrollToTop(element, value) {
      if (element === window) window.scrollTo(0, value);
      else element.scrollTop = value;
    }

    function appendText(parent, text, className) {
      var element = document.createElement("div");
      if (className) element.className = className;
      element.textContent = String(text || "");
      parent.appendChild(element);
      return element;
    }

    function safeResourceUrl(value) {
      if (!value) return "";
      try {
        var url = new URL(String(value), window.location.href);
        if (url.origin !== window.location.origin) return "";
        return url.href;
      } catch (error) {
        return "";
      }
    }

    function formatDate(timestamp) {
      var date = new Date(Number(timestamp || 0) * 1000);
      if (Number.isNaN(date.getTime())) return "未知日期";
      return date.toLocaleDateString("zh-CN", {
        year: "numeric", month: "2-digit", day: "2-digit"
      }).replace(/\//g, "-");
    }

    function formatTimestamp(timestamp) {
      var date = new Date(Number(timestamp || 0) * 1000);
      if (Number.isNaN(date.getTime())) return "";
      return date.toLocaleString("zh-CN", {hour12: false});
    }

    function appendResource(parent, resource) {
      var previewUrl = safeResourceUrl(resource.preview_url);
      var downloadUrl = safeResourceUrl(resource.download_url);
      var type = String(resource.resource_type || "file");
      var displayType = String(resource.display_type || type);
      var filename = String(resource.filename || displayType);
      if ((resource.is_image || type === "emoji" || type === "image") && previewUrl) {
        var link = document.createElement("a");
        link.className = "conversation-resource conversation-resource-image";
        link.href = previewUrl;
        link.target = "_blank";
        link.rel = "noreferrer";
        var image = document.createElement("img");
        image.src = previewUrl;
        image.alt = filename;
        image.loading = "lazy";
        link.appendChild(image);
        parent.appendChild(link);
        return;
      }
      if (resource.is_audio && previewUrl) {
        var audioWrap = document.createElement("div");
        audioWrap.className = "conversation-resource";
        var audio = document.createElement("audio");
        audio.controls = true;
        audio.preload = "metadata";
        audio.src = previewUrl;
        audioWrap.appendChild(audio);
        parent.appendChild(audioWrap);
        return;
      }
      if (resource.is_video && previewUrl) {
        var videoWrap = document.createElement("div");
        videoWrap.className = "conversation-resource";
        var video = document.createElement("video");
        video.controls = true;
        video.preload = "metadata";
        video.src = previewUrl;
        videoWrap.appendChild(video);
        parent.appendChild(videoWrap);
        return;
      }
      if (downloadUrl) {
        var fileLink = document.createElement("a");
        fileLink.className = "conversation-resource-file";
        fileLink.href = downloadUrl;
        fileLink.target = "_blank";
        fileLink.rel = "noreferrer";
        fileLink.textContent = "📎 " + filename;
        parent.appendChild(fileLink);
        return;
      }
      var fileLabel = document.createElement("span");
      fileLabel.className = "conversation-resource-file";
      fileLabel.textContent = "📎 " + filename;
      parent.appendChild(fileLabel);
    }

    function appendBubble(parent, role, timestamp, text, resources) {
      var bubble = document.createElement("div");
      bubble.className = "conversation-bubble " + (role === "user" ? "user" : "assistant");
      var meta = document.createElement("div");
      meta.className = "conversation-bubble-meta";
      meta.textContent = (role === "user" ? "👤 用户" : "🤖 助手") +
        (timestamp ? " · " + formatTimestamp(timestamp) : "");
      bubble.appendChild(meta);
      if (text) appendText(bubble, text, "conversation-bubble-content");
      if (resources && resources.length) {
        var resourceWrap = document.createElement("div");
        resourceWrap.className = "conversation-resources";
        resources.forEach(function (resource) { appendResource(resourceWrap, resource || {}); });
        bubble.appendChild(resourceWrap);
      }
      if (!text && (!resources || !resources.length)) {
        appendText(bubble, "（无内容消息）", "conversation-bubble-content conversation-empty-message");
      }
      parent.appendChild(bubble);
    }

    function appendMeta(parent, text, className) {
      var element = document.createElement("span");
      if (className) element.className = className;
      element.textContent = String(text || "");
      parent.appendChild(element);
      return element;
    }

    function appendToolChain(parent, blocks) {
      var toolBlocks = blocks.filter(function (block) {
        return block && (block.type === "tool" || block.type === "tool_result");
      });
      if (!toolBlocks.length) return;
      var details = document.createElement("details");
      details.className = "conversation-tool-chain";
      var summary = document.createElement("summary");
      var callCount = toolBlocks.reduce(function (count, block) {
        return count + (block.type === "tool" ? (block.tool_calls || []).length : 0);
      }, 0);
      summary.textContent = "查看工具链（" + (callCount || toolBlocks.length) + " 个事件）";
      details.appendChild(summary);
      toolBlocks.forEach(function (block) {
        var event = document.createElement("div");
        event.className = "conversation-tool-event " + (block.type === "tool_result" ? "tool" : "assistant");
        appendText(event, block.type === "tool_result" ? "🔧 工具结果" : "🤖 工具调用", "msg-meta");
        if (block.type === "tool") {
          (block.tool_calls || []).forEach(function (call) {
            var fn = call && call.function ? call.function : {};
            appendText(event, String(fn.name || "未知") + "(" + String(fn.arguments || "{}") + ")", "tool-details");
          });
        } else {
          appendText(event, block.text || "（空结果）", "tool-details");
        }
        details.appendChild(event);
      });
      parent.appendChild(details);
    }

    function renderTurn(turn, previousDate) {
      var date = formatDate(turn.created_at);
      var fragment = document.createDocumentFragment();
      if (date !== previousDate) {
        var divider = document.createElement("div");
        divider.className = "conversation-date-divider";
        appendMeta(divider, date, "conversation-date-label");
        fragment.appendChild(divider);
      }
      var blocks = Array.isArray(turn.blocks) ? turn.blocks : [];
      var hasTools = blocks.some(function (block) {
        return block && (block.type === "tool" || block.type === "tool_result");
      });
      var article = document.createElement("article");
      article.className = "conversation-turn " + (hasTools ? "conversation-turn-tools" : "conversation-turn-simple");
      article.dataset.turnId = String(turn.turn_id || "");
      var meta = document.createElement("div");
      meta.className = "conversation-turn-meta";
      appendMeta(meta, "Turn " + String(turn.turn_sequence || ""));
      appendMeta(meta, String({ai: "AI 对话", ambient: "群聊闲聊", system: "系统任务", unknown: "类型未知"}[turn.turn_kind] || "类型未知"), "turn-kind turn-kind-" + String(turn.turn_kind || "unknown"));
      appendMeta(meta, date);
      if (turn.status && turn.status !== "completed" && turn.status !== "unknown") appendMeta(meta, turn.status, "turn-status");
      article.appendChild(meta);
      var messages = document.createElement("div");
      messages.className = "conversation-messages";
      blocks.forEach(function (block) {
        if (!block) return;
        if (block.type === "text" && block.role !== "tool") appendBubble(messages, block.role, turn.created_at, block.text, []);
        if (block.resource) appendBubble(messages, block.role, turn.created_at, "", [block.resource]);
      });
      if (!messages.childNodes.length && !hasTools) appendText(messages, "（无内容消息）", "conversation-bubble-content conversation-empty-message");
      article.appendChild(messages);
      appendToolChain(article, blocks);
      fragment.appendChild(article);
      return {fragment: fragment, date: date};
    }

    function renderOlderTurns(items) {
      var existingFirst = sessionStream.querySelector(".conversation-turn");
      var previousDate = existingFirst ? existingFirst.querySelector(".conversation-turn-meta span:nth-child(3)") : null;
      previousDate = previousDate ? previousDate.textContent : "";
      var fragment = document.createDocumentFragment();
      var date = previousDate;
      items.forEach(function (turn) {
        var rendered = renderTurn(turn, date);
        fragment.appendChild(rendered.fragment);
        date = rendered.date;
      });
      var empty = sessionStream.querySelector(".conversation-empty");
      if (empty) empty.remove();
      sessionStream.insertBefore(fragment, sessionStream.querySelector(".conversation-turn, .conversation-date-divider") || null);
    }

    async function loadOlderMessages() {
      if (loadingOlder || (cursorMode ? !hasMoreOlder : currentPage >= totalPages)) return;
      loadingOlder = true;
      updateOlderButton();
      var oldHeight = scrollHeightOf(scrollContainer);
      var oldScrollTop = scrollTopOf(scrollContainer);
      try {
        var chatId = sessionStream.dataset.chatId || "";
        var url = "/api/sessions/" + encodeURIComponent(chatId) + "/turns?limit=" + encodeURIComponent(pageSize);
        if (cursorMode) {
          url += "&before_turn_sequence=" + encodeURIComponent(cursorBefore);
          url += "&cutoff_sequence=" + encodeURIComponent(cursorCutoff);
        } else {
          var legacyUrl = new URL(window.location.href);
          legacyUrl.searchParams.set("page", String(currentPage + 1));
          legacyUrl.searchParams.set("page_size", String(pageSize));
          legacyUrl.searchParams.set("partial", "true");
          url = legacyUrl.toString();
        }
        var response = await fetch(url, {headers: {"X-Requested-With": "fetch", "Accept": "application/json"}});
        if (!response.ok) throw new Error("加载历史消息失败");
        if (cursorMode) {
          var payload = await response.json();
          var items = Array.isArray(payload.items) ? payload.items : [];
          renderOlderTurns(items);
          hasMoreOlder = Boolean(payload.has_more);
          if (payload.next_before_turn_sequence != null) cursorBefore = Number(payload.next_before_turn_sequence);
          sessionStream.dataset.cursorBefore = hasMoreOlder ? String(cursorBefore) : "";
        } else {
          var html = await response.text();
          var template = document.createElement("template");
          template.innerHTML = html;
          sessionStream.insertBefore(template.content, sessionStream.querySelector(".conversation-turn, .conversation-date-divider") || null);
          currentPage += 1;
          sessionStream.dataset.page = String(currentPage);
        }
        scrollToTop(scrollContainer, oldScrollTop + scrollHeightOf(scrollContainer) - oldHeight);
      } catch (error) {
        olderButton.textContent = "加载失败，重试";
      } finally {
        loadingOlder = false;
        updateOlderButton();
      }
    }

    olderButton.addEventListener("click", loadOlderMessages);
    scrollContainer.addEventListener("scroll", function () {
      if (scrollTopOf(scrollContainer) < 180) loadOlderMessages();
    }, {passive: true});
    if (cursorMode) {
      var pagination = document.querySelector("[data-cursor-pagination]");
      if (pagination) pagination.classList.add("is-hidden");
    }
    updateOlderButton();
    if (currentPage === 1 && (cursorMode ? hasMoreOlder : totalPages > 1)) {
      window.requestAnimationFrame(function () {
        scrollToTop(scrollContainer, scrollHeightOf(scrollContainer));
      });
    }
  }

  var groupOptions = document.querySelector("[data-group-options]");
  var groupSearch = document.querySelector("[data-group-search]");
  var selectedList = document.querySelector("[data-selected-list]");
  var selectedCount = document.querySelector("[data-selection-count]");
  var noResults = document.querySelector("[data-group-no-results]");

  function updateGroupPicker() {
    if (!groupOptions) return;
    var query = groupSearch ? groupSearch.value.trim().toLowerCase() : "";
    var options = groupOptions.querySelectorAll("[data-group-option]");
    var visible = 0;
    var selected = [];
    for (var optionIndex = 0; optionIndex < options.length; optionIndex++) {
      var option = options[optionIndex];
      var matches = !query || option.dataset.search.indexOf(query) !== -1;
      option.hidden = !matches;
      if (matches) visible += 1;
      var checkbox = option.querySelector("input[type=checkbox]");
      if (checkbox && checkbox.checked) {
        selected.push({id: checkbox.value, name: option.querySelector("strong").textContent});
      }
    }
    if (noResults) noResults.hidden = options.length === 0 || !query || visible !== 0;
    if (selectedCount) selectedCount.textContent = selected.length + " 个已选";
    if (selectedList) {
      selectedList.innerHTML = "";
      if (!selected.length) {
        selectedList.innerHTML = '<span class="selected-targets-empty">尚未选择目标群聊</span>';
      } else {
        for (var selectedIndex = 0; selectedIndex < selected.length; selectedIndex++) {
          var chip = document.createElement("span");
          chip.className = "target-chip";
          chip.textContent = selected[selectedIndex].name + " · " + selected[selectedIndex].id;
          selectedList.appendChild(chip);
        }
      }
    }
  }

  if (groupOptions) {
    groupOptions.addEventListener("change", updateGroupPicker);
    updateGroupPicker();
  }
  if (groupSearch) groupSearch.addEventListener("input", updateGroupPicker);

  var modeOptions = document.querySelectorAll(".mode-option");
  for (var modeIndex = 0; modeIndex < modeOptions.length; modeIndex++) {
    modeOptions[modeIndex].addEventListener("change", function () {
      for (var optionIndex = 0; optionIndex < modeOptions.length; optionIndex++) {
        var radio = modeOptions[optionIndex].querySelector("input[type=radio]");
        modeOptions[optionIndex].classList.toggle("is-selected", radio.checked);
      }
    });
  }
})();
