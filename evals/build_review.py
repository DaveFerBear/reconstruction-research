"""Generate a static HTML review page for eyeballing the real-design corruptions.

Walks evals/data_real/, pairs each corrupted render with its original from
datasets/specs/, and emits evals/review.html — a single self-contained page
with mode filters, side-by-side image comparison, and a localStorage-backed
yes/no/skip rating UI for marking which corruptions actually look like the
target failure mode.

Run a local HTTP server from the repo root (so the page can load images via
relative paths without file:// CORS pain):

    python -m http.server
    # then open http://localhost:8000/evals/review.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "datasets" / "specs"
DATA_DIR = Path(__file__).parent / "data_real"
OUT_PATH = Path(__file__).parent / "review.html"


def _build_manifest() -> dict:
    items: list[dict] = []
    by_mode: dict[str, int] = {}
    missing_renders = 0

    for spec_dir in sorted(DATA_DIR.iterdir()):
        if not spec_dir.is_dir():
            continue
        spec_id = spec_dir.name
        original_render = SOURCE_DIR / spec_id / "render.png"
        if not original_render.exists():
            continue
        for mode_dir in sorted(spec_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            mode = mode_dir.name
            corrupted_render = mode_dir / "render.png"
            corruption_json = mode_dir / "corruption.json"
            if not corruption_json.exists():
                continue
            corruption = json.loads(corruption_json.read_text(encoding="utf-8"))
            if not corrupted_render.exists():
                missing_renders += 1
                continue
            items.append({
                "id": f"{spec_id}__{mode}",
                "spec_id": spec_id,
                "mode": mode,
                "description": corruption.get("description", ""),
                "changed": corruption.get("changed_node_indices", []),
                # Paths relative to the repo root so a server rooted there serves them.
                "original": f"datasets/specs/{spec_id}/render.png",
                "corrupted": f"evals/data_real/{spec_id}/{mode}/render.png",
            })
            by_mode[mode] = by_mode.get(mode, 0) + 1

    return {"items": items, "counts": by_mode, "missing_renders": missing_renders}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>evals/review</title>
<style>
  :root {
    --good: #2e7d32;
    --bad: #c62828;
    --skip: #757575;
    --bg: #fafafa;
    --card: #ffffff;
    --border: #e0e0e0;
    --muted: #757575;
  }
  * { box-sizing: border-box; }
  body { font: 14px -apple-system, BlinkMacSystemFont, sans-serif;
         background: var(--bg); color: #111; margin: 0; padding: 0; }
  header {
    position: sticky; top: 0; background: var(--card); border-bottom: 1px solid var(--border);
    padding: 12px 20px; z-index: 10; display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
  }
  .modes button {
    padding: 6px 12px; margin-right: 6px; background: #f0f0f0; border: 1px solid var(--border);
    border-radius: 4px; cursor: pointer; font-size: 13px;
  }
  .modes button.active { background: #1976D2; color: white; border-color: #1976D2; }
  .modes button .badge { opacity: 0.7; margin-left: 4px; font-size: 11px; }
  .tally { color: var(--muted); font-variant-numeric: tabular-nums; }
  .tally span { font-weight: 600; }
  .tally .good { color: var(--good); }
  .tally .bad { color: var(--bad); }
  .tally .skip { color: var(--skip); }
  header .actions { margin-left: auto; }
  header button.action {
    padding: 6px 10px; margin-left: 6px; background: white; border: 1px solid var(--border);
    border-radius: 4px; cursor: pointer; font-size: 12px;
  }

  main { padding: 20px; max-width: 1400px; margin: 0 auto; }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 6px;
    padding: 14px; margin-bottom: 16px;
  }
  .card.rated-good { border-left: 4px solid var(--good); }
  .card.rated-bad  { border-left: 4px solid var(--bad); }
  .card.rated-skip { border-left: 4px solid var(--skip); }
  .card-header { display: flex; gap: 12px; align-items: baseline; margin-bottom: 10px; }
  .card-header .spec-id { font-family: ui-monospace, Menlo, monospace; font-size: 12px; color: var(--muted); }
  .card-header .mode { font-weight: 600; }
  .description { font-family: ui-monospace, Menlo, monospace; font-size: 12px;
                 color: #444; padding: 6px 8px; background: #f6f6f6;
                 border-radius: 4px; margin-bottom: 12px; word-break: break-all; }

  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .pair figure { margin: 0; }
  .pair figcaption { font-size: 11px; color: var(--muted); margin-bottom: 4px; text-transform: uppercase;
                     letter-spacing: 0.5px; }
  .pair img { width: 100%; border: 1px solid var(--border); background: white; display: block; }

  .rate { display: flex; gap: 8px; margin-top: 12px; }
  .rate button {
    flex: 1; padding: 8px; border: 1px solid var(--border); background: white; border-radius: 4px;
    cursor: pointer; font-size: 13px;
  }
  .rate button.good.active { background: var(--good); color: white; border-color: var(--good); }
  .rate button.bad.active  { background: var(--bad);  color: white; border-color: var(--bad); }
  .rate button.skip.active { background: var(--skip); color: white; border-color: var(--skip); }

  .empty { padding: 60px; text-align: center; color: var(--muted); }
  kbd { font-family: ui-monospace, Menlo, monospace; background: #eee; padding: 1px 5px;
        border-radius: 3px; border: 1px solid #ccc; font-size: 11px; }
</style>
</head>
<body>
<header>
  <strong>review</strong>
  <span class="modes" id="modes"></span>
  <span class="tally" id="tally"></span>
  <span style="color: var(--muted); font-size: 12px;">
    keys: <kbd>j</kbd>/<kbd>k</kbd> nav · <kbd>1</kbd> good · <kbd>2</kbd> bad · <kbd>0</kbd> skip
  </span>
  <span class="actions">
    <button class="action" id="export">Export bad list</button>
    <button class="action" id="reset">Reset ratings</button>
  </span>
</header>
<main id="cards"></main>

<script>
const DATA = __DATA__;
const STORAGE_KEY = "evals_review_ratings_v1";

let activeMode = null;
let activeIndex = 0;

function loadRatings() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
  catch (e) { return {}; }
}
function saveRatings(r) { localStorage.setItem(STORAGE_KEY, JSON.stringify(r)); }
let ratings = loadRatings();

function modeItems(mode) {
  return DATA.items.filter(i => i.mode === mode);
}

function renderModeButtons() {
  const modes = Object.keys(DATA.counts).sort();
  const total = DATA.items.length;
  let buttons = `<button data-mode="" class="${!activeMode ? 'active':''}">all<span class="badge">${total}</span></button>`;
  for (const m of modes) {
    const cls = activeMode === m ? "active" : "";
    buttons += `<button data-mode="${m}" class="${cls}">${m}<span class="badge">${DATA.counts[m]}</span></button>`;
  }
  document.getElementById("modes").innerHTML = buttons;
  document.getElementById("modes").querySelectorAll("button").forEach(b => {
    b.addEventListener("click", () => {
      activeMode = b.dataset.mode || null;
      activeIndex = 0;
      renderAll();
    });
  });
}

function renderTally() {
  const visible = activeMode ? modeItems(activeMode) : DATA.items;
  let g=0, b=0, s=0, total = visible.length;
  for (const item of visible) {
    const v = ratings[item.id];
    if (v === "good") g++;
    else if (v === "bad") b++;
    else if (v === "skip") s++;
  }
  const rated = g + b + s;
  document.getElementById("tally").innerHTML =
    `<span>${rated}</span>/${total} rated · ` +
    `<span class="good">${g} good</span> · ` +
    `<span class="bad">${b} bad</span> · ` +
    `<span class="skip">${s} skip</span>`;
}

function renderCards() {
  const items = activeMode ? modeItems(activeMode) : DATA.items;
  const main = document.getElementById("cards");
  if (!items.length) {
    main.innerHTML = `<div class="empty">no items</div>`;
    return;
  }
  let html = "";
  items.forEach((item, idx) => {
    const r = ratings[item.id];
    const cls = r ? `rated-${r}` : "";
    html += `
      <div class="card ${cls}" id="card-${idx}" data-idx="${idx}" data-id="${item.id}">
        <div class="card-header">
          <span class="mode">${item.mode}</span>
          <span class="spec-id">${item.spec_id}</span>
        </div>
        <div class="description">${escapeHtml(item.description)}</div>
        <div class="pair">
          <figure>
            <figcaption>original (good)</figcaption>
            <img loading="lazy" src="../${item.original}" alt="original">
          </figure>
          <figure>
            <figcaption>corrupted (${item.mode})</figcaption>
            <img loading="lazy" src="../${item.corrupted}" alt="corrupted">
          </figure>
        </div>
        <div class="rate">
          <button class="good ${r==='good'?'active':''}" data-rate="good">Good corruption (1)</button>
          <button class="bad ${r==='bad'?'active':''}" data-rate="bad">Bad corruption (2)</button>
          <button class="skip ${r==='skip'?'active':''}" data-rate="skip">Skip (0)</button>
        </div>
      </div>`;
  });
  main.innerHTML = html;

  main.querySelectorAll(".rate button").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const card = btn.closest(".card");
      const id = card.dataset.id;
      const rate = btn.dataset.rate;
      if (ratings[id] === rate) delete ratings[id];
      else ratings[id] = rate;
      saveRatings(ratings);
      renderTally();
      updateCardClass(card, ratings[id]);
    });
  });
}

function updateCardClass(card, rating) {
  card.classList.remove("rated-good", "rated-bad", "rated-skip");
  if (rating) card.classList.add(`rated-${rating}`);
  card.querySelectorAll(".rate button").forEach(b => {
    b.classList.toggle("active", b.dataset.rate === rating);
  });
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderAll() {
  renderModeButtons();
  renderTally();
  renderCards();
}

function rateActive(rating) {
  const card = document.querySelector(`#card-${activeIndex}`);
  if (!card) return;
  const id = card.dataset.id;
  if (ratings[id] === rating) delete ratings[id];
  else ratings[id] = rating;
  saveRatings(ratings);
  renderTally();
  updateCardClass(card, ratings[id]);
}

function focusCard(idx) {
  const card = document.querySelector(`#card-${idx}`);
  if (!card) return;
  card.scrollIntoView({behavior: "smooth", block: "center"});
}

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const items = activeMode ? modeItems(activeMode) : DATA.items;
  if (e.key === "j") { activeIndex = Math.min(items.length - 1, activeIndex + 1); focusCard(activeIndex); }
  else if (e.key === "k") { activeIndex = Math.max(0, activeIndex - 1); focusCard(activeIndex); }
  else if (e.key === "1") rateActive("good");
  else if (e.key === "2") rateActive("bad");
  else if (e.key === "0") rateActive("skip");
});

document.getElementById("export").addEventListener("click", () => {
  const bad = DATA.items.filter(i => ratings[i.id] === "bad").map(i => ({
    spec_id: i.spec_id, mode: i.mode, description: i.description
  }));
  const blob = new Blob([JSON.stringify(bad, null, 2)], {type: "application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a"); a.href = url; a.download = "review_bad_corruptions.json";
  a.click(); URL.revokeObjectURL(url);
});
document.getElementById("reset").addEventListener("click", () => {
  if (!confirm("Clear all ratings?")) return;
  ratings = {}; saveRatings(ratings); renderAll();
});

renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the review HTML page")
    parser.add_argument("--out", default=str(OUT_PATH), help=f"Output path (default: {OUT_PATH})")
    args = parser.parse_args()

    manifest = _build_manifest()
    if manifest["missing_renders"]:
        print(
            f"warning: {manifest['missing_renders']} corruptions have no render.png "
            f"(skipped). Run `python -m evals.render_real` first."
        )
    out = Path(args.out)
    out.write_text(
        HTML_TEMPLATE.replace("__DATA__", json.dumps(manifest)),
        encoding="utf-8",
    )
    total = len(manifest["items"])
    print(f"Wrote {out} with {total} item(s) across {len(manifest['counts'])} mode(s)")
    print()
    print("To view, run a server from the repo root:")
    print("    python -m http.server")
    print("then open http://localhost:8000/evals/review.html")


if __name__ == "__main__":
    main()
