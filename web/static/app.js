// ============================================================
//  VoxSet — speech dataset builder — front-end (3-column workspace)
// ============================================================
const $  = (id) => document.getElementById(id);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------- helpers ----------
function fmtDuration(s) {
  if (!s) return "";
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const p = (n) => String(n).padStart(2, "0");
  return h ? `${h}:${p(m)}:${p(sec)}` : `${m}:${p(sec)}`;
}
function fmtSize(b) {
  if (!b) return "";
  const mb = b / 1048576;
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB`
       : mb >= 1 ? `${mb.toFixed(1)} MB` : `${(b / 1024).toFixed(0)} KB`;
}
function fmtClock(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  const ms = Math.round((sec - Math.floor(sec)) * 1000);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
}
function esc(s) { const d = document.createElement("div"); d.textContent = s ?? ""; return d.innerHTML; }

const ICO = {
  audio: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>`,
  video: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>`,
  srt: `<svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 15h5M7 11h10"/></svg>`,
  playlist: `<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h13M3 12h13M3 18h9M17 12v7l5-3.5z"/></svg>`,
  stop: `<svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor" stroke="none"><rect x="6" y="6" width="12" height="12" rx="2.5"/></svg>`,
};

// compact row action icons
const ACT = {
  srt: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 13h4M7 16h2M14 13h3M13 16h4"/></svg>`,
  save: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11m0 0 4-4m-4 4-4-4M5 20h14"/></svg>`,
  del: `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 6l12 12M18 6 6 18"/></svg>`,
};

// ============================================================
//  Modals (generic close)
// ============================================================
$$("[data-close]").forEach((b) =>
  b.addEventListener("click", () => $(b.dataset.close).classList.add("hidden")));
$$(".modal").forEach((m) =>
  m.addEventListener("click", (e) => { if (e.target === m) m.classList.add("hidden"); }));
$("refresh-btn").addEventListener("click", reloadAll);

function reloadAll() { loadLibrary(); loadSubtitles(); pollGpu(); }

// ---------- light / dark theme ----------
$("theme-btn").addEventListener("click", () => {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  const next = dark ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("wx-theme", next); } catch { /* ignore */ }
  drawChart();   // re-tint the GPU chart for the new theme
});

// ---------- collapsible mini-cards ----------
$$("[data-toggle]").forEach((head) =>
  head.addEventListener("click", () => {
    const card = head.closest(".mini-card");
    if (card) card.classList.toggle("collapsed");
    if (card && card.classList.contains("gpu-card")) drawChart();   // re-fit canvas on expand
  }));

// ============================================================
//  Add media — fetch / preview / download
// ============================================================
const urlInput = $("url"), fetchBtn = $("fetch-btn"), preview = $("preview"),
      dlControls = $("dl-controls"), qualitySel = $("quality"),
      qualityField = $("quality-field"), downloadBtn = $("download-btn");
let mode = "audio", currentInfo = null, dlTimer = null;

const QUALITY = {
  audio: [{ v: "best", t: "Best audio (mp3)" }],
  video: [
    { v: "best", t: "Best available" }, { v: "2160", t: "4K · 2160p" },
    { v: "1440", t: "1440p" }, { v: "1080", t: "1080p" },
    { v: "720", t: "720p" }, { v: "480", t: "480p" }, { v: "360", t: "360p" },
  ],
};
function fillQuality() {
  qualitySel.innerHTML = "";
  QUALITY[mode].forEach((o) => {
    const opt = document.createElement("option");
    opt.value = o.v; opt.textContent = o.t;
    qualitySel.appendChild(opt);
  });
}
$$(".seg-btn").forEach((b) => b.addEventListener("click", () => {
  $$(".seg-btn").forEach((x) => x.classList.remove("active"));
  b.classList.add("active");
  mode = b.dataset.mode; fillQuality();
}));
fillQuality();

// ---------- transcription languages (full Whisper set) ----------
const LANGS = [
  ["en","English"],["zh","Chinese"],["de","German"],["es","Spanish"],["ru","Russian"],
  ["ko","Korean"],["fr","French"],["ja","Japanese"],["pt","Portuguese"],["tr","Turkish"],
  ["pl","Polish"],["ca","Catalan"],["nl","Dutch"],["ar","Arabic"],["sv","Swedish"],
  ["it","Italian"],["id","Indonesian"],["hi","Hindi"],["fi","Finnish"],["vi","Vietnamese"],
  ["he","Hebrew"],["uk","Ukrainian"],["el","Greek"],["ms","Malay"],["cs","Czech"],
  ["ro","Romanian"],["da","Danish"],["hu","Hungarian"],["ta","Tamil"],["no","Norwegian"],
  ["th","Thai"],["ur","Urdu"],["hr","Croatian"],["bg","Bulgarian"],["lt","Lithuanian"],
  ["la","Latin"],["mi","Maori"],["ml","Malayalam"],["cy","Welsh"],["sk","Slovak"],
  ["te","Telugu"],["fa","Persian"],["lv","Latvian"],["bn","Bengali"],["sr","Serbian"],
  ["az","Azerbaijani"],["sl","Slovenian"],["kn","Kannada"],["et","Estonian"],["mk","Macedonian"],
  ["br","Breton"],["eu","Basque"],["is","Icelandic"],["hy","Armenian"],["ne","Nepali"],
  ["mn","Mongolian"],["bs","Bosnian"],["kk","Kazakh"],["sq","Albanian"],["sw","Swahili"],
  ["gl","Galician"],["mr","Marathi"],["pa","Punjabi"],["si","Sinhala"],["km","Khmer"],
  ["sn","Shona"],["yo","Yoruba"],["so","Somali"],["af","Afrikaans"],["oc","Occitan"],
  ["ka","Georgian"],["be","Belarusian"],["tg","Tajik"],["sd","Sindhi"],["gu","Gujarati"],
  ["am","Amharic"],["yi","Yiddish"],["lo","Lao"],["uz","Uzbek"],["fo","Faroese"],
  ["ht","Haitian Creole"],["ps","Pashto"],["tk","Turkmen"],["nn","Nynorsk"],["mt","Maltese"],
  ["sa","Sanskrit"],["lb","Luxembourgish"],["my","Myanmar"],["bo","Tibetan"],["tl","Tagalog"],
  ["mg","Malagasy"],["as","Assamese"],["tt","Tatar"],["haw","Hawaiian"],["ln","Lingala"],
  ["ha","Hausa"],["ba","Bashkir"],["jw","Javanese"],["su","Sundanese"],["yue","Cantonese"],
];
function fillLanguages() {
  const sel = $("language");
  const opt = (v, t, s) => { const o = document.createElement("option"); o.value = v; o.textContent = t; o.selected = !!s; return o; };
  sel.innerHTML = "";
  sel.appendChild(opt("auto", "Auto-detect"));
  sel.appendChild(opt("en", "English (en)", true));   // default
  [...LANGS].filter(([c]) => c !== "en").sort((a, b) => a[1].localeCompare(b[1]))
    .forEach(([c, n]) => sel.appendChild(opt(c, `${n} (${c})`)));
}
fillLanguages();

fetchBtn.addEventListener("click", fetchInfo);
urlInput.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); fetchInfo(); } });

function resetAddBox() {
  urlInput.value = "";
  preview.classList.add("hidden"); preview.innerHTML = "";
  dlControls.classList.add("hidden");
  currentInfo = null;
}

async function fetchInfo() {
  const url = urlInput.value.trim();
  if (!url) { urlInput.focus(); return; }
  preview.classList.remove("hidden");
  preview.innerHTML = `<div class="pv-msg">Fetching…</div>`;
  dlControls.classList.add("hidden");
  fetchBtn.disabled = true;
  try {
    const res = await fetch("/api/info?url=" + encodeURIComponent(url));
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Could not fetch info");
    currentInfo = await res.json();
    renderPreview(currentInfo);
    dlControls.classList.remove("hidden");
  } catch (err) {
    currentInfo = null;
    preview.innerHTML = `<div class="pv-msg err">${esc(err.message)}</div>`;
  } finally {
    fetchBtn.disabled = false;
  }
}

function renderPreview(info) {
  if (info.type === "playlist") {
    const rows = info.entries.map((e) => `
      <label class="pl-row">
        <input type="checkbox" class="pl-check" value="${esc(e.id)}" checked />
        ${e.thumbnail ? `<img class="pl-thumb" src="${esc(e.thumbnail)}" alt="" loading="lazy" onerror="this.classList.add('broken')" />` : `<span class="pl-thumb ph"></span>`}
        <span class="pl-title">${esc(e.title)}</span>
        <span class="pl-dur">${fmtDuration(e.duration)}</span>
      </label>`).join("");
    preview.innerHTML = `
      <div class="pv-body">
        <h3>📑 ${esc(info.title || "Playlist")}</h3>
        <p>${esc(info.uploader || "")} · ${info.count} videos</p>
        <label class="pl-row pl-all"><input type="checkbox" id="pl-all" checked /><span class="pl-title"><strong>Select all</strong></span></label>
      </div>
      <div class="pl-list">${rows}</div>`;
    const all = $("pl-all");
    all.addEventListener("change", () =>
      preview.querySelectorAll(".pl-check").forEach((c) => { c.checked = all.checked; }));
  } else {
    preview.innerHTML =
      (info.thumbnail ? `<img src="${esc(info.thumbnail)}" alt="" />` : "") +
      `<div class="pv-body"><h3>${esc(info.title || "(untitled)")}</h3>
       <p>${esc(info.uploader || "")} · ${fmtDuration(info.duration)}</p></div>`;
  }
}

downloadBtn.addEventListener("click", startDownload);
async function startDownload() {
  if (!currentInfo) return;
  const data = new FormData();
  data.set("url", urlInput.value.trim());
  data.set("mode", mode);
  data.set("quality", qualitySel.value);
  if (currentInfo.type === "playlist") {
    const ids = [...preview.querySelectorAll(".pl-check:checked")].map((c) => c.value);
    if (!ids.length) { alert("Select at least one video."); return; }
    data.set("entry_ids", ids.join(","));
  }
  downloadBtn.disabled = true;
  try {
    const res = await fetch("/api/downloads", { method: "POST", body: data });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Download failed to start");
    const { download_id } = await res.json();
    resetAddBox();
    if (dlTimer) clearInterval(dlTimer);
    dlTimer = setInterval(() => pollDownload(download_id), 1000);
    pollDownload(download_id);
  } catch (err) {
    alert(err.message);
  } finally {
    downloadBtn.disabled = false;
  }
}

async function pollDownload(id) {
  try {
    const res = await fetch("/api/downloads/" + id);
    if (!res.ok) throw new Error("lost download");
    const dl = await res.json();
    renderDownloads(dl);
    if (dl.status === "done") {
      clearInterval(dlTimer); dlTimer = null;
      loadLibrary();
      setActivityDot();
      setTimeout(() => {
        if (!dlTimer) { $("downloads").classList.add("hidden"); updateActivityEmpty(); }
      }, 5000);
    }
  } catch {
    clearInterval(dlTimer); dlTimer = null; setActivityDot();
  }
}

function renderDownloads(dl) {
  $("downloads").classList.remove("hidden");
  updateActivityEmpty(); setActivityDot();
  const done = dl.items.filter((i) => i.status === "done").length;
  const err = dl.items.filter((i) => i.status === "error").length;
  $("dl-summary").textContent = `${done}/${dl.items.length} done${err ? ` · ${err} failed` : ""}`;
  $("dl-list").innerHTML = dl.items.map((it) => {
    const p = Math.round((it.progress || 0) * 100);
    const cls = it.status === "error" ? "err" : it.status === "done" ? "done" : "";
    const right = it.status === "error" ? "failed"
      : it.status === "done" ? "✓"
      : it.status === "processing" ? "processing…" : `${p}%`;
    return `<div class="dl-row">
      <div class="dl-info"><span class="dl-title">${esc(it.title)}</span>
        <span class="dl-state ${cls}">${right}</span></div>
      <div class="bar small"><div class="bar-fill ${cls}" style="width:${it.status === "done" ? 100 : p}%"></div></div>
      ${it.error ? `<div class="dl-err">${esc(it.error)}</div>` : ""}
    </div>`;
  }).join("");
}

// ============================================================
//  Upload
// ============================================================
const fileInput = $("file"), dropzone = $("dropzone"), dropLabel = $("drop-label");
fileInput.addEventListener("change", () => { if (fileInput.files[0]) uploadFile(fileInput.files[0]); });
["dragenter", "dragover"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0]); });

async function uploadFile(file) {
  dropLabel.innerHTML = `Uploading <strong>${esc(file.name)}</strong>…`;
  const data = new FormData(); data.set("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: data });
    if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Upload failed");
    dropLabel.innerHTML = `Drop audio / video<br><em>or click to browse</em>`;
    loadLibrary();
  } catch (err) {
    dropLabel.innerHTML = `<span style="color:var(--err)">${esc(err.message)}</span>`;
  }
}

// ============================================================
//  Library column (cards)
// ============================================================
async function loadLibrary() {
  try {
    const { items } = await (await fetch("/api/library")).json();
    renderLibrary(items);
    $("lib-count").textContent = items.length || "";
  } catch {
    $("lib-list").innerHTML = ""; $("lib-empty").classList.remove("hidden");
  }
}

// leading "[Name] …" prefix → fallback playlist key for items downloaded
// before the backend stored a `playlist` field.
function bracketKey(title) {
  const m = /^\s*\[([^\]]{2,})\]/.exec(title || "");
  return m ? m[1].trim() : null;
}

function libRow(it, inGroup) {
  const meta = [it.ext.toUpperCase(), fmtSize(it.size), it.duration ? fmtDuration(it.duration) : null]
    .filter(Boolean).join(" · ");
  const srt = it.srts.length ? ` · <span class="librow-srt">${it.srts.length} dataset${it.srts.length > 1 ? "s" : ""}</span>` : "";
  // inside a playlist group, drop the redundant leading "[Playlist] " prefix
  const name = inGroup ? (it.title.replace(/^\s*\[[^\]]*\]\s*/, "").trim() || it.title) : it.title;
  return `<div class="librow">
    <span class="librow-ico ${it.kind}">${ICO[it.kind] || ICO.audio}</span>
    <div class="librow-main">
      <div class="librow-name" title="${esc(it.title)}">${esc(name)}</div>
      <div class="librow-meta">${esc(meta)}${srt}</div>
    </div>
    <div class="librow-actions">
      <button class="iconbtn primary" title="${it.srts.length ? "Re-transcribe" : "Transcribe"}" data-act="srt" data-name="${esc(it.name)}" data-title="${esc(it.title)}">${ACT.srt}</button>
      <a class="iconbtn" title="Save" href="/api/library/file/${encodeURIComponent(it.name)}" download>${ACT.save}</a>
      <button class="iconbtn danger" title="Delete" data-act="del" data-name="${esc(it.name)}">${ACT.del}</button>
    </div>
  </div>`;
}

function renderLibrary(items) {
  const empty = $("lib-empty");
  if (!items.length) { $("lib-list").innerHTML = ""; empty.classList.remove("hidden"); return; }
  empty.classList.add("hidden");

  // group consecutive items that share a playlist (or bracket-prefix fallback)
  const groups = [];
  const byKey = new Map();
  for (const it of items) {
    const key = it.playlist || bracketKey(it.title);
    if (key) {
      let g = byKey.get(key);
      if (!g) { g = { name: it.playlist || key, items: [] }; byKey.set(key, g); groups.push(g); }
      g.items.push(it);
    } else {
      groups.push({ name: null, items: [it] });
    }
  }
  // natural order within a playlist (so "2" precedes "10")
  groups.forEach((g) => g.items.sort((a, b) =>
    a.title.localeCompare(b.title, undefined, { numeric: true, sensitivity: "base" })));

  const genGroups = [];   // multi-item groups, indexed by their gen button
  $("lib-list").innerHTML = groups.map((g) => {
    if (g.items.length > 1) {
      const gi = genGroups.push(g) - 1;
      const n = g.items.length;
      return `<div class="lib-group">
        <div class="lib-group-head">
          <span class="lgh-ico">${ICO.playlist}</span>
          <span class="lgh-name" title="${esc(g.name)}">${esc(g.name)}</span>
          <span class="lgh-count">${n}</span>
          <div class="lgh-actions">
            <button class="lgh-act gen" data-gen-idx="${gi}" title="Transcribe all ${n} items">${ACT.srt}</button>
            <button class="lgh-act" data-dl-idx="${gi}" title="Download all ${n} files">${ACT.save}</button>
            <button class="lgh-act danger" data-delall-idx="${gi}" title="Delete all ${n} files">${ACT.del}</button>
          </div>
          <svg class="lgh-chev" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
        </div>
        <div class="lib-group-rows">${g.items.map((it) => libRow(it, true)).join("")}</div>
      </div>`;
    }
    return `<div class="lib-single">${libRow(g.items[0], false)}</div>`;
  }).join("");

  $("lib-list").querySelectorAll(".lib-group-head").forEach((h) =>
    h.addEventListener("click", () => h.closest(".lib-group").classList.toggle("collapsed")));
  $("lib-list").querySelectorAll("[data-gen-idx]").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();   // don't collapse the group
      const g = genGroups[+b.dataset.genIdx];
      openTxModal(g.items.map((it) => ({ name: it.name, title: it.title })));
    }));
  $("lib-list").querySelectorAll("[data-dl-idx]").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      downloadGroup(genGroups[+b.dataset.dlIdx].items);
    }));
  $("lib-list").querySelectorAll("[data-delall-idx]").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteGroup(genGroups[+b.dataset.delallIdx]);
    }));
  $("lib-list").querySelectorAll("[data-act='srt']").forEach((b) =>
    b.addEventListener("click", () => openTxModal([{ name: b.dataset.name, title: b.dataset.title }])));
  $("lib-list").querySelectorAll("[data-act='del']").forEach((b) =>
    b.addEventListener("click", () => deleteItem(b.dataset.name)));
}

async function deleteItem(name) {
  if (!confirm(`Delete "${name}" and its datasets?`)) return;
  try {
    await fetch("/api/library/" + encodeURIComponent(name), { method: "DELETE" });
    loadLibrary(); loadSubtitles();
  } catch { /* ignore */ }
}

// bulk save every file in a playlist group (staggered to dodge browser blocking)
function downloadGroup(items) {
  items.forEach((it, i) => setTimeout(() => {
    const a = document.createElement("a");
    a.href = `/api/library/file/${encodeURIComponent(it.name)}`;
    a.download = it.name;
    document.body.appendChild(a); a.click(); a.remove();
  }, i * 400));
}

// bulk delete every file in a playlist group (+ their datasets)
async function deleteGroup(g) {
  if (!confirm(`Delete all ${g.items.length} items in "${g.name}" and their datasets?`)) return;
  for (const it of g.items) {
    try { await fetch("/api/library/" + encodeURIComponent(it.name), { method: "DELETE" }); }
    catch { /* keep going */ }
  }
  loadLibrary(); loadSubtitles();
}

// ============================================================
//  Datasets column (cards)
// ============================================================
async function loadSubtitles() {
  try {
    const { items } = await (await fetch("/api/subtitles")).json();
    renderSubtitles(items);
    $("ds-count").textContent = items.length || "";
  } catch {
    $("srt-list").innerHTML = ""; $("srt-empty").classList.remove("hidden");
  }
}

function renderSubtitles(items) {
  const empty = $("srt-empty");
  if (!items.length) { $("srt-list").innerHTML = ""; empty.classList.toggle("hidden", pending.size > 0); return; }
  empty.classList.add("hidden");
  $("srt-list").innerHTML = items.map((it) => `
    <div class="rowcard clickable" data-open="${esc(it.name)}">
      <span class="cell-ico srt">${ICO.srt}</span>
      <div class="rowcard-main">
        <div class="rowcard-name" title="${esc(it.title)}">${esc(it.title)}
          ${it.lang ? `<span class="lang-badge">${esc(it.lang)}</span>` : ""}</div>
        <div class="rowcard-meta">Speech dataset · ${fmtSize(it.size)}</div>
        <div class="rowcard-actions">
          <button class="btn-xs primary" data-open="${esc(it.name)}">Open dataset</button>
          <button class="btn-xs danger" data-del="${esc(it.name)}">Delete</button>
        </div>
      </div>
    </div>`).join("");
  $("srt-list").querySelectorAll("[data-del]").forEach((b) =>
    b.addEventListener("click", (e) => { e.stopPropagation(); deleteSubtitle(b.dataset.del); }));
  $("srt-list").querySelectorAll("[data-open]").forEach((el) =>
    el.addEventListener("click", (e) => { e.stopPropagation(); openDataset(el.dataset.open); }));
}

async function deleteSubtitle(name) {
  if (!confirm(`Delete dataset "${name}"?`)) return;
  try {
    await fetch("/api/subtitles/" + encodeURIComponent(name), { method: "DELETE" });
    loadSubtitles(); loadLibrary();
  } catch { /* ignore */ }
}

// ============================================================
//  Dataset modal
// ============================================================
async function openDataset(name) {
  const modal = $("ds-modal");
  $("ds-title").textContent = "Loading…";
  $("ds-lang").classList.add("hidden");
  $("ds-meta").textContent = "";
  $("ds-nomedia").classList.add("hidden");
  $("ds-rows").innerHTML = "";
  modal.classList.remove("hidden");
  try {
    const d = await (await fetch("/api/dataset/" + encodeURIComponent(name))).json();
    $("ds-title").textContent = d.title;
    if (d.lang) { $("ds-lang").textContent = d.lang; $("ds-lang").classList.remove("hidden"); }
    $("ds-meta").textContent = `${d.count} segments · ${fmtDuration(d.duration)}`;
    $("ds-export-srt").href = "/api/srt/" + encodeURIComponent(name);
    $("ds-export-csv").onclick = () => { location.href = `/api/dataset/${encodeURIComponent(name)}/export?fmt=csv`; };

    const zipBtn = $("ds-export-zip");
    if (d.media) {
      zipBtn.disabled = false; zipBtn.style.opacity = "";
      zipBtn.onclick = () => { location.href = `/api/dataset/${encodeURIComponent(name)}/export?fmt=zip`; };
    } else {
      zipBtn.disabled = true; zipBtn.style.opacity = ".5"; zipBtn.onclick = null;
      $("ds-nomedia").classList.remove("hidden");
    }

    $("ds-rows").innerHTML = d.segments.map((s) => {
      const audio = d.media
        ? `<audio controls preload="none" src="/api/clip?file=${encodeURIComponent(d.media)}&start=${s.start}&end=${s.end}"></audio>`
        : `<span class="ds-noaudio">no source audio</span>`;
      return `<tr>
        <td class="ds-idx">${s.index}</td>
        <td>${audio}</td>
        <td class="ds-text">${esc(s.text)}</td>
        <td class="ds-time">${fmtClock(s.start)} → ${fmtClock(s.end)}<small>${(s.end - s.start).toFixed(2)}s</small></td>
      </tr>`;
    }).join("");
  } catch (err) {
    $("ds-title").textContent = "Could not load dataset";
    $("ds-rows").innerHTML = `<tr><td colspan="4" style="padding:20px;color:var(--err)">${esc(err.message)}</td></tr>`;
  }
}

// ============================================================
//  Transcription — single file or a whole playlist
// ============================================================
let modalTargets = [];                 // [{name, title}] queued by the modal
let jobTimer = null;
const pending = new Map();              // jobId -> {name, title, progress, stage, status, logs}
const STAGE_LABELS = {
  queued: "Waiting…", starting: "Spinning up…", load: "Loading model…",
  info: "Reading source…", transcribe: "Transcribing speech…",
  align: "Aligning timestamps…", diarize: "Identifying speakers…",
  write: "Writing dataset…", done: "Finished", error: "Failed",
  canceling: "Stopping…", canceled: "Stopped",
};
const STEP_ORDER = ["load", "transcribe", "align", "diarize", "write"];

// targets: array of {name, title}
function openTxModal(targets) {
  modalTargets = targets.filter(Boolean);
  if (!modalTargets.length) return;
  $("modal-file").textContent = modalTargets.length === 1
    ? modalTargets[0].name
    : `${modalTargets.length} files — generated one after another`;
  $("tx-modal").classList.remove("hidden");
}

$("tx-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!modalTargets.length) return;
  const base = {
    language: $("language").value.trim() || "en",
    model: $("model").value,
    device: $("device").value,
    compute_type: $("compute_type").value,
    batch_size: $("batch_size").value || 0,
  };
  const targets = modalTargets;
  modalTargets = [];
  $("tx-modal").classList.add("hidden");

  for (const t of targets) {
    try {
      const data = new FormData();
      data.set("filename", t.name);
      Object.entries(base).forEach(([k, v]) => data.set(k, v));
      const res = await fetch("/api/jobs", { method: "POST", body: data });
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || "Could not start job");
      const { job_id } = await res.json();
      pending.set(job_id, { name: t.name, title: t.title || t.name, progress: 0, stage: "queued", status: "queued", logs: [] });
    } catch (err) {
      alert(err.message);
    }
  }

  renderPending();
  $("job").classList.remove("hidden");
  updateActivityEmpty(); setActivityDot();
  if (!jobTimer) jobTimer = setInterval(pollAll, 1000);
  pollAll();
});

async function pollAll() {
  for (const [id, p] of [...pending]) {
    try {
      const res = await fetch("/api/jobs/" + id);
      if (!res.ok) throw new Error("lost");
      const job = await res.json();
      p.progress = job.progress || 0;
      p.stage = job.stage || job.status;
      p.status = job.status;
      p.logs = job.logs || [];
      if (job.status === "done") {
        pending.delete(id);
        loadSubtitles(); loadLibrary();
      } else if (job.status === "canceled") {
        pending.delete(id);
      } else if (job.status === "error" || job.status === "interrupted") {
        // keep the card visible so the failure (and why) isn't lost
        // ("interrupted" = the worker restarted mid-job; user retries from the library)
        p.error = job.error || (job.logs || []).slice(-1)[0] || "Transcription failed";
      }
    } catch {
      pending.delete(id);
    }
  }

  // feature the running (or next) job in the right-hand Transcription panel
  let lead = null;
  for (const p of pending.values()) { if (p.status === "running" || STEP_ORDER.includes(p.stage)) { lead = p; break; } }
  if (!lead) lead = pending.values().next().value || null;

  renderPending();

  if (lead) {
    setStage(lead.stage, lead.status);
    setProgress(lead.progress, lead.status);
    $("log").textContent = (lead.logs || []).join("\n");
    $("log").scrollTop = $("log").scrollHeight;
  }
  // stop polling once nothing is still running/queued (errored cards may linger)
  const active = [...pending.values()].some(
    (p) => p.status === "running" || p.status === "queued" || p.canceling);
  if (!active) {
    clearInterval(jobTimer); jobTimer = null;
    if (!lead) { setStage("done", "done"); setProgress(1, "done"); }
    setTimeout(() => { if (!jobTimer) { $("job").classList.add("hidden"); updateActivityEmpty(); } }, 2500);
  }
  setActivityDot();
}

// disabled "filling" placeholder cards in the centre column, one per active job
function renderPending() {
  const box = $("ds-pending");
  if (!pending.size) { box.innerHTML = ""; box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.innerHTML = [...pending.entries()].map(([id, p]) => {
    const pct = Math.round(p.progress * 100);
    const err = p.status === "error" || p.status === "interrupted";
    const stopping = !!p.canceling;
    const label = err ? `Failed — ${p.error || "see Activity log"}`
      : stopping ? "Stopping…"
      : (p.status === "queued" ? "Queued…" : (STAGE_LABELS[p.stage] || p.stage));
    const btn = err
      ? `<button class="ds-stop" data-dismiss="${id}" title="Dismiss">${ACT.del}</button>`
      : `<button class="ds-stop" data-cancel="${id}" title="Stop"${stopping ? " disabled" : ""}>${ICO.stop}</button>`;
    return `<div class="rowcard pending${err ? " err" : ""}" aria-disabled="true">
      <div class="ds-fill" style="width:${err ? 100 : pct}%"></div>
      <span class="cell-ico srt">${ICO.srt}</span>
      <div class="rowcard-main">
        <div class="rowcard-name">${esc(p.title)}</div>
        <div class="rowcard-meta" title="${esc(label)}">${esc(label)}</div>
      </div>
      <span class="ds-pct">${err ? "!" : pct + "%"}</span>
      ${btn}
    </div>`;
  }).join("");
  box.querySelectorAll("[data-cancel]").forEach((b) =>
    b.addEventListener("click", () => cancelJob(b.dataset.cancel)));
  box.querySelectorAll("[data-dismiss]").forEach((b) =>
    b.addEventListener("click", () => { pending.delete(b.dataset.dismiss); renderPending(); }));
}

async function cancelJob(id) {
  const p = pending.get(id);
  if (p) { p.canceling = true; renderPending(); }
  try { await fetch(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" }); }
  catch { /* pollAll will reconcile */ }
}
function cancelAll() { [...pending.keys()].forEach(cancelJob); }
$("job-stop").addEventListener("click", (e) => { e.stopPropagation(); cancelAll(); });

function setStage(stage, status) {
  const badge = $("stage-badge");
  badge.textContent = status === "done" ? "done"
    : (status === "error" || stage === "error") ? "error" : stage;
  badge.className = "badge " +
    (status === "done" ? "done" : (status === "error" || stage === "error") ? "error" : "running");
  $("stage-text").textContent = STAGE_LABELS[stage] || stage;
  const idx = STEP_ORDER.indexOf(stage);
  $$("#steps li").forEach((el) => {
    const pos = STEP_ORDER.indexOf(el.dataset.stage);
    el.classList.remove("active", "done");
    if (status === "done") { el.classList.add("done"); return; }
    if (idx === -1) return;
    if (pos < idx) el.classList.add("done");
    else if (pos === idx) el.classList.add("active");
  });
}
function setProgress(frac, status) {
  const p = Math.round(frac * 100);
  $("bar-fill").style.width = p + "%";
  $("pct").textContent = p + "%";
  $("bar-fill").className = "bar-fill" + (status === "done" ? " done" : status === "error" ? " error" : "");
}

// ---------- activity helpers ----------
function updateActivityEmpty() {
  const running = !$("downloads").classList.contains("hidden") || !$("job").classList.contains("hidden");
  $("activity-empty").classList.toggle("hidden", running);
}
function setActivityDot() {
  $("activity-dot").classList.toggle("on", !!dlTimer || !!jobTimer);
}

// ============================================================
//  GPU monitor
// ============================================================
const chart = $("gpu-chart");
const MAX = 60, POLL_S = 2;  // 60 samples × 2s ≈ a 2-minute window
const hist = { util: [], vram: [], memUsed: [] };  // util %, vram %, memUsed GiB
let gpuMemTotalG = 0, hoverIdx = null;
const push = (a, v) => { a.push(v); if (a.length > MAX) a.shift(); };

function drawChart() {
  const ctx = chart.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const W = chart.clientWidth, H = chart.clientHeight;
  if (!W) return;
  if (chart.width !== W * dpr || chart.height !== H * dpr) { chart.width = W * dpr; chart.height = H * dpr; }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  ctx.strokeStyle = css.getPropertyValue("--chart-grid").trim() || "rgba(28,33,48,.06)"; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = Math.round((H - 1) * i / 4) + 0.5; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  const plot = (arr, stroke, fill) => {
    if (arr.length < 2) return;
    const x = (i) => W * i / (MAX - 1), y = (v) => H - (Math.max(0, Math.min(100, v)) / 100) * (H - 6) - 3;
    const g = ctx.createLinearGradient(0, 0, 0, H); g.addColorStop(0, fill); g.addColorStop(1, "transparent");
    ctx.beginPath(); ctx.moveTo(x(0), y(arr[0])); arr.forEach((v, i) => ctx.lineTo(x(i), y(v)));
    ctx.lineTo(x(arr.length - 1), H); ctx.lineTo(x(0), H); ctx.closePath(); ctx.fillStyle = g; ctx.fill();
    ctx.beginPath(); arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
    ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
  };
  plot(hist.vram, css.getPropertyValue("--ok").trim() || "#16a34a", "rgba(22,163,74,.14)");
  plot(hist.util, css.getPropertyValue("--accent").trim() || "#6d5ef0", "rgba(109,94,240,.16)");

  // Hover crosshair: a vertical guide + a dot on each line at the hovered sample.
  if (hoverIdx != null && hist.util.length > 1) {
    const i = Math.max(0, Math.min(hist.util.length - 1, hoverIdx));
    const x = W * i / (MAX - 1), y = (v) => H - (Math.max(0, Math.min(100, v)) / 100) * (H - 6) - 3;
    ctx.strokeStyle = css.getPropertyValue("--scroll").trim() || "#c2c7d8";
    ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(x + 0.5, 0); ctx.lineTo(x + 0.5, H); ctx.stroke();
    ctx.setLineDash([]);
    const dot = (v, color) => {
      ctx.beginPath(); ctx.arc(x, y(v), 3, 0, 7);
      ctx.fillStyle = color; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = css.getPropertyValue("--panel").trim() || "#fff"; ctx.stroke();
    };
    dot(hist.vram[i], css.getPropertyValue("--ok").trim() || "#16a34a");
    dot(hist.util[i], css.getPropertyValue("--accent").trim() || "#6d5ef0");
  }
}

async function pollGpu() {
  try {
    const g = await (await fetch("/api/gpu")).json();
    if (g.available) {
      const vp = g.mem_total ? Math.round((g.mem_used / g.mem_total) * 100) : 0;
      gpuMemTotalG = g.mem_total / 1024;
      const vram = `${(g.mem_used / 1024).toFixed(1)}/${gpuMemTotalG.toFixed(0)}G`;
      $("gpu-name").textContent = g.name;
      $("gpu-util").textContent = g.util + "%";
      $("gpu-vram").textContent = vram;
      $("gpu-pulse").className = "pulse on";
      push(hist.util, g.util); push(hist.vram, vp); push(hist.memUsed, g.mem_used / 1024);
    } else {
      $("gpu-name").textContent = "No GPU detected";
      $("gpu-util").textContent = "—"; $("gpu-vram").textContent = "—";
      $("gpu-pulse").className = "pulse off";
      push(hist.util, 0); push(hist.vram, 0); push(hist.memUsed, 0);
    }
  } catch {
    $("gpu-name").textContent = "GPU unavailable"; $("gpu-pulse").className = "pulse off";
  }
  drawChart();
  if (hoverIdx != null) showGpuTip();  // keep the open tooltip in sync with new data
}

// Map a mouse X (CSS px over the canvas) to a history sample index.
function gpuIdxFromX(clientX) {
  const r = chart.getBoundingClientRect();
  const i = Math.round((clientX - r.left) / r.width * (MAX - 1));
  return Math.max(0, Math.min(hist.util.length - 1, i));
}

function showGpuTip() {
  const tip = $("gpu-tip");
  if (hoverIdx == null || hist.util.length < 2) { tip.classList.add("hidden"); return; }
  const i = Math.max(0, Math.min(hist.util.length - 1, hoverIdx));
  const ago = (hist.util.length - 1 - i) * POLL_S;
  const when = ago === 0 ? "now" : `−${ago}s`;
  const gb = hist.memUsed[i] ?? 0;
  const memLine = gpuMemTotalG
    ? `${hist.vram[i]}% · ${gb.toFixed(1)}/${gpuMemTotalG.toFixed(0)}G`
    : `${hist.vram[i]}%`;
  tip.innerHTML =
    `<div class="tip-age">${when}</div>` +
    `<div class="tip-row"><span class="lg-dot lg-util"></span>Util<b>${hist.util[i]}%</b></div>` +
    `<div class="tip-row"><span class="lg-dot lg-vram"></span>VRAM<b>${memLine}</b></div>`;
  // Position over the hovered sample, clamped inside the chart.
  const x = chart.clientWidth * i / (MAX - 1);
  tip.style.left = Math.max(34, Math.min(chart.clientWidth - 34, x)) + "px";
  tip.classList.remove("hidden");
}

chart.addEventListener("mousemove", (e) => { hoverIdx = gpuIdxFromX(e.clientX); showGpuTip(); drawChart(); });
chart.addEventListener("mouseleave", () => { hoverIdx = null; $("gpu-tip").classList.add("hidden"); drawChart(); });

// ============================================================
//  Resume in-flight work after a page reload
// ============================================================
// Jobs/downloads live in the backend, so a reload only loses the *client* view.
// Re-attach to anything still running.
async function resumeJobs() {
  try {
    const { items } = await (await fetch("/api/jobs")).json();
    if (!items || !items.length) return;
    items.forEach((j) => pending.set(j.id, {
      name: j.file, title: j.title || j.file,
      progress: j.progress || 0, stage: j.stage || j.status, status: j.status, logs: [],
    }));
    renderPending();
    $("job").classList.remove("hidden");
    updateActivityEmpty(); setActivityDot();
    if (!jobTimer) jobTimer = setInterval(pollAll, 1000);
    pollAll();
  } catch { /* ignore */ }
}
async function resumeDownloads() {
  try {
    const { items } = await (await fetch("/api/downloads")).json();
    const active = (items || [])[0];
    if (!active) return;
    if (dlTimer) clearInterval(dlTimer);
    dlTimer = setInterval(() => pollDownload(active.id), 1000);
    pollDownload(active.id);
  } catch { /* ignore */ }
}

// ============================================================
//  Init
// ============================================================
pollGpu(); setInterval(pollGpu, 2000); window.addEventListener("resize", drawChart);
loadSubtitles();
loadLibrary();
resumeJobs();
resumeDownloads();
