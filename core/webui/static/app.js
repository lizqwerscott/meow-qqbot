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
