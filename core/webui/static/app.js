(function () {
  const menuBtn = document.getElementById("menu-btn");
  const sidebar = document.querySelector(".sidebar");
  const overlay = document.getElementById("menu-overlay");

  if (!menuBtn || !sidebar || !overlay) return;

  function openMenu() {
    sidebar.classList.add("open");
    overlay.classList.add("show");
    document.body.style.overflow = "hidden";
  }

  function closeMenu() {
    sidebar.classList.remove("open");
    overlay.classList.remove("show");
    document.body.style.overflow = "";
  }

  menuBtn.addEventListener("click", openMenu);
  overlay.addEventListener("click", closeMenu);

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeMenu();
  });

  var links = sidebar.querySelectorAll("nav a");
  for (var i = 0; i < links.length; i++) {
    links[i].addEventListener("click", closeMenu);
  }
})();
