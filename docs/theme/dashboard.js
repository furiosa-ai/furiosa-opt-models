// The shared dashboard draws its own theme; follow the page's color scheme through its URL.
const followScheme = () => {
  const dark = document.body.getAttribute("data-md-color-scheme") === "slate";
  document.querySelectorAll('iframe[src*="datadoghq.com/sb/"]').forEach((frame) => {
    const url = new URL(frame.src);
    if (dark) url.searchParams.set("theme", "dark");
    else url.searchParams.delete("theme");
    if (url.href !== frame.src) frame.src = url.href;
  });
};
new MutationObserver(followScheme).observe(document.body, {
  attributes: true,
  attributeFilter: ["data-md-color-scheme"],
});
document$.subscribe(followScheme);
