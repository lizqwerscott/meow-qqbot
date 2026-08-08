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
})();
