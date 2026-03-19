window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
  $(".navbar-burger").click(function() {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });

  const loops = Array.from(document.querySelectorAll(".sync-loop"));

  function syncLoops() {
    if (!loops.length) return;

    Promise.all(
      loops.map((video) =>
        video.readyState >= 2
          ? Promise.resolve()
          : new Promise((resolve) => {
              video.addEventListener("canplay", resolve, { once: true });
            })
      )
    ).then(() => {
      loops.forEach((video) => {
        video.pause();
        video.currentTime = 0;
      });

      requestAnimationFrame(() => {
        loops.forEach((video) => {
          const playPromise = video.play();
          if (playPromise && typeof playPromise.catch === "function") {
            playPromise.catch(() => {});
          }
        });
      });
    });
  }

  syncLoops();

  document.addEventListener("visibilitychange", function() {
    if (!document.hidden) {
      syncLoops();
    }
  });
});
