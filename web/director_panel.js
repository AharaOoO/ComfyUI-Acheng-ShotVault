import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { getLang, toggleLang, t, fmtBytes, pickOutdatedReasons } from "./i18n.js";
import { injectStyles, PANEL_DEFAULTS } from "./panel/styles.js";
import { openFocusPromptModal } from "./panel/prompt_modal.js";
import { renderExportModal } from "./panel/export_modal.js";

const EXT_NAME = "ShotVault.DirectorConsole.v54";
const PANEL_W = PANEL_DEFAULTS.DEF_W;
const PANEL_H = PANEL_DEFAULTS.DEF_H;
const TA_HEIGHT_STORE_KEY = "shotvault_ta_heights_v58";

function loadTaHeight(key, fallback) {
  try {
    const obj = JSON.parse(window.localStorage?.getItem(TA_HEIGHT_STORE_KEY) || "{}");
    const v = Number(obj?.[key]);
    return Number.isFinite(v) && v >= 56 && v <= 640 ? v : fallback;
  } catch {
    return fallback;
  }
}

function saveTaHeight(key, px) {
  try {
    const obj = JSON.parse(window.localStorage?.getItem(TA_HEIGHT_STORE_KEY) || "{}");
    obj[key] = Math.round(px);
    window.localStorage?.setItem(TA_HEIGHT_STORE_KEY, JSON.stringify(obj));
  } catch {}
}

function appendSafe(parent, ...items) {
  for (const item of items) {
    if (item == null || item === false) continue;
    parent.appendChild(typeof item === "string" ? document.createTextNode(item) : item);
  }
}

function extractCardSnippet(prompt, maxLen = 36) {
  const raw = String(prompt || "").trim();
  if (!raw) return t("snippet_empty");
  // Prefer actual narrative summary or [Shot 1] line instead of repetitive `subject_definitions:` boilerplate
  const mSum = raw.match(/(?:^|\n)summary\s*:\s*(?:\[[^\]]*\]\s*)?([^\n]+)/i);
  if (mSum && mSum[1]?.trim()) return mSum[1].trim().slice(0, maxLen);
  const mShot = raw.match(/\[Shot\s*1\]\s*([^\n]+)/i);
  if (mShot && mShot[1]?.trim()) return mShot[1].trim().slice(0, maxLen);
  const lines = raw.split(/\r?\n/).map((x) => x.trim()).filter(Boolean);
  for (const ln of lines) {
    if (/^(?:subject_definitions|retention_analysis|detailed_description|overall_soundscape|non_diegetic_music)\s*:$/i.test(ln)) continue;
    if (/^<Subject\s+\d+>\s+is\b/i.test(ln)) continue;
    return ln.slice(0, maxLen);
  }
  return raw.slice(0, maxLen);
}

function makeDraggableTextarea({
  value = "",
  placeholder = "",
  prefKey = "shot_prompt",
  defaultHeight = 156,
  minHeight = 68,
  maxHeight = 520,
  onInput,
  onChange,
}) {
  const curH = loadTaHeight(prefKey, defaultHeight);
  const ta = el("textarea", {
    class: "sv-ta",
    placeholder,
    style: { height: `${curH}px`, minHeight: `${minHeight}px` },
    oninput: onInput,
    onchange: onChange,
  });
  ta.value = String(value ?? "");

  const grip = el("div", {
    class: "sv-ta-grip",
    title: t("grip_title"),
  });
  grip.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    const startY = e.clientY;
    const startH = ta.getBoundingClientRect().height || curH;
    const zoom = Number(app.canvas?.ds?.scale || 1) || 1;
    grip.classList.add("dragging");
    const onMove = (ev) => {
      ev.stopPropagation();
      const nextH = Math.max(minHeight, Math.min(maxHeight, startH + (ev.clientY - startY) / zoom));
      ta.style.height = `${Math.round(nextH)}px`;
    };
    const onUp = (ev) => {
      ev.stopPropagation();
      grip.classList.remove("dragging");
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onUp, true);
      const finalH = parseFloat(ta.style.height) || startH;
      saveTaHeight(prefKey, finalH);
    };
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
  });
  return { wrap: el("div", { class: "sv-ta-wrap" }, [ta, grip]), ta };
}
const ALL_NATIVE_WIDGETS = new Set([
  "project_name", "script_format", "common_prompt", "script_text",
  "aspect_ratio", "megapixels", "fps", "duration_mode", "context_frames",
  "stage3_json_text", "audio_drive_mode", "audio_per_shot_dir", "snap_cuts_to_beat",
  "seed_video_path", "seed_trim_end_sec", "assets_root_dir",
  "auto_split_turnaround", "director_plan_json",
]);

function hideNativeWidgets(node) {
  for (const w of node.widgets || []) {
    if (w.name === "director_console") continue;
    if (!ALL_NATIVE_WIDGETS.has(w.name) && w.type === "custom") continue;
    w._origOptions = w._origOptions || w.options;
    w.type = "hidden";
    w.hidden = true;
    w.computeSize = () => [0, -4];
    w.draw = () => {};
    w.options = { ...(w._origOptions || w.options || {}), hidden: true, serialize: true };
    w.serializeValue = () => w.value;
  }
}

const GEAR_MATRIX = {
  22: { overlap: 22, trim: 22, taperTail: 22, ramp: 4, audioGuide: 20, feather: 7, pairKey: "gear_pair_22" },
  39: { overlap: 39, trim: 39, taperTail: 39, ramp: 6, audioGuide: 37, feather: 12, pairKey: "gear_pair_39" },
  56: { overlap: 56, trim: 56, taperTail: 56, ramp: 8, audioGuide: 54, feather: 16, pairKey: "gear_pair_56" },
};

function computeGear(gearVal, isCont = true) {
  const g = [22, 39, 56].includes(Number(gearVal)) ? Number(gearVal) : 22;
  const spec = GEAR_MATRIX[g];
  if (!isCont) {
    return {
      overlap: 0, trim: 0, taperTail: Math.max(22, spec.taperTail),
      ramp: spec.ramp, audioGuide: 0, feather: spec.feather, pair: t("gear_pair_first"),
    };
  }
  return { ...spec, pair: t(spec.pairKey) };
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  const tagLow = tag.toLowerCase();
  let explicitVal;
  let hasExplicitVal = false;
  if (tagLow === "button" && (!attrs || attrs.type == null)) node.type = "button";
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "checked" || k === "selected" || k === "disabled") {
      node[k] = Boolean(v);
      continue;
    }
    if (k === "value") {
      explicitVal = v == null ? "" : String(v);
      hasExplicitVal = true;
      if (tagLow !== "select") node.value = explicitVal;
      continue;
    }
    if (v == null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(node.style, v);
    else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of Array.isArray(children) ? children : [children]) {
    if (c == null || c === false) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  if (tagLow === "select") {
    if (hasExplicitVal) node.value = explicitVal;
    else {
      const selOpt = Array.from(node.options || []).find((o) => o.selected);
      if (selOpt) node.value = selOpt.value;
    }
  }
  return node;
}

function showToast(msg, msOrErr = 1800, isErr = false) {
  const errMode = typeof msOrErr === "boolean" ? msOrErr : isErr;
  const ms = typeof msOrErr === "number" ? msOrErr : (errMode ? 2800 : 1800);
  const t = el("div", {
    class: "sv-toast",
    style: errMode ? { background: "#b3263d" } : {},
    text: msg,
  });
  document.body.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

function mediaUrl(relPath, project = "") {
  if (!relPath) return "";
  if (/^https?:\/\//i.test(relPath)) return relPath;
  if (project) return `/shotvault/media?project=${encodeURIComponent(project)}&file=${encodeURIComponent(relPath)}`;
  return `/shotvault/input-media?file=${encodeURIComponent(relPath)}`;
}

async function uploadMediaFile(file, subfolder = "shotvault") {
  const fd = new FormData();
  fd.append("file", file, file.name);
  fd.append("subfolder", subfolder);
  const resp = await api.fetchApi("/shotvault/upload", { method: "POST", body: fd });
  const data = await resp.json();
  if (!data?.ok) throw new Error(data?.error || t("err_upload_failed"));
  return data;
}

function probeBrowserAudioDuration(file) {
  return new Promise((resolve) => {
    try {
      const url = URL.createObjectURL(file);
      const aud = document.createElement("audio");
      aud.preload = "metadata";
      const done = (val) => {
        URL.revokeObjectURL(url);
        resolve(Number.isFinite(val) && val > 0.1 ? Math.round(val * 1000) / 1000 : null);
      };
      aud.onloadedmetadata = () => done(aud.duration);
      aud.onerror = () => done(null);
      setTimeout(() => done(null), 2500);
      aud.src = url;
    } catch {
      resolve(null);
    }
  });
}

function pickLocalFile(accept, cb) {
  const inp = el("input", { type: "file", accept, style: { display: "none" } });
  inp.onchange = async () => {
    const f = inp.files?.[0];
    inp.remove();
    if (!f) return;
    try { await cb(f); } catch (err) { showToast(t("toast_upload_failed", { msg: err?.message || err }), 2800, true); }
  };
  document.body.appendChild(inp);
  inp.click();
}

function normSlots9(slots, imgs) {
  const out = new Array(9).fill("");
  const src = Array.isArray(slots) && slots.length ? slots : (Array.isArray(imgs) ? imgs : []);
  for (let i = 0; i < Math.min(9, src.length); i++) out[i] = String(src[i] || "").trim();
  return out;
}

function normSlots3(slots, auds) {
  const out = new Array(3).fill("");
  const src = Array.isArray(slots) && slots.length ? slots : (Array.isArray(auds) ? auds : []);
  for (let i = 0; i < Math.min(3, src.length); i++) out[i] = String(src[i] || "").trim();
  return out;
}

function resolveEffectiveAudioSlots(shot, globalAudioSlots) {
  const own = normSlots3(shot?.ref_audio_slots, shot?.ref_audios);
  const inherit = (shot && "inherit_global_audios" in shot)
    ? (shot.inherit_global_audios !== false)
    : (shot?.inherit_global_refs !== false);
  const glob = normSlots3(globalAudioSlots);
  return own.map((x, k) => x || (inherit ? glob[k] : "") || "");
}

function parseSingleShotAmpersand(text, fallbackDur = 5.0) {
  let body = String(text || "");
  let dur = Math.max(0.25, Number(fallbackDur || 5.0));
  let foundDur = false;
  const rm = body.match(/^\s*&\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*&\s*/m);
  if (rm) {
    const s0 = Number(rm[1]) * 60 + parseFloat(rm[2]);
    const s1 = Number(rm[3]) * 60 + parseFloat(rm[4]);
    dur = Math.max(0.25, s1 - s0);
    body = `${body.slice(0, rm.index)}${body.slice(rm.index + rm[0].length)}`.trim();
    foundDur = true;
  } else {
    const am = body.match(/^\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?\s*&\s*/m);
    if (am) {
      dur = Math.max(0.25, parseFloat(am[1]) || dur);
      body = `${body.slice(0, am.index)}${body.slice(am.index + am[0].length)}`.trim();
      foundDur = true;
    }
  }
  return { duration_sec: dur, found_dur: foundDur, body };
}

function mapShotTypeDirective(rawType) {
  const low = String(rawType || "").trim().toLowerCase();
  if (/^(?:0|文|文戏|drama|dialogue|emotional)$/.test(low) || low.includes("文戏") || low.includes("drama")) return 0;
  if (/^(?:2|mv|music|dance|音乐)$/.test(low)) return 2;
  return 1;
}

function splitMasterScript(text, hasSeed = false, defaultDurSec = 5.0) {
  const raw = String(text || "").trim();
  if (!raw) return [];
  const fbDur = Math.max(0.25, Number(defaultDurSec || 5.0));
  const blocks = raw.split(/\s*={3,}\s*|^\s*-{3,}\s*$/m).map((b) => b.trim()).filter(Boolean);
  return blocks.map((blk, i) => {
    const amp = parseSingleShotAmpersand(blk, fbDur);
    const lines = amp.body.split(/\r?\n/);
    let shotType = 1;
    let dur = amp.duration_sec;
    let ov = 0;
    let hardCut = null;
    const body = [];
    for (const ln of lines) {
      let cur = ln.trim();
      let consumed = false;
      const typeKv = cur.match(/^@type\s*=\s*([^\s]+)/i);
      if (typeKv) { shotType = mapShotTypeDirective(typeKv[1]); consumed = true; }
      else if (/^@(?:文戏|drama)(?:\s|$)/i.test(cur) || /^\[?(?:文戏|DRAMA)\]?$/i.test(cur)) { shotType = 0; consumed = true; }
      else if (/^@(?:武戏|action)(?:\s|$)/i.test(cur) || /^\[?(?:武戏|ACTION)\]?$/i.test(cur)) { shotType = 1; consumed = true; }
      const om = cur.match(/^@(?:overlap\s*=\s*)?(22|39|56)(?:\s|$)/i);
      if (om) { ov = Number(om[1]); consumed = true; }
      if (/^@(?:硬切|hard_?cut|cut)@?(?:\s|$)/i.test(cur)) {
        hardCut = true;
        const rest = cur.replace(/^@(?:硬切|hard_?cut|cut)@?\s*/i, "").trim();
        if (rest) body.push(rest);
        consumed = true;
      } else if (/^@(?:continue|续接|连续)@?(?:\s|$)/i.test(cur)) {
        hardCut = false;
        const rest = cur.replace(/^@(?:continue|续接|连续)@?\s*/i, "").trim();
        if (rest) body.push(rest);
        consumed = true;
      }
      if (consumed) continue;
      const dm = cur.match(/^(?:Duration|时长)\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?$/i) || cur.match(/^[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?\s*[)）]$/i);
      if (dm && !amp.found_dur) { dur = Math.max(0.25, parseFloat(dm[1]) || fbDur); continue; }
      body.push(ln);
    }
    const prompt = body.join("\n").trim();
    const inlineDur = prompt.match(/(?:^|\s)[(（]([0-9]+(?:\.[0-9]+)?)\s*s[)）]/);
    if (inlineDur && !amp.found_dur && dur === fbDur) dur = Math.max(0.25, parseFloat(inlineDur[1]) || fbDur);
    return {
      shot_index: i + 1,
      duration_sec: dur,
      shot_type: shotType,
      hard_cut: hardCut !== null ? hardCut : (i === 0 ? !hasSeed : false),
      overlap_override: ov,
      inherit_global_refs: true,
      inherit_global_audios: true,
      seed_text: "",
      prompt,
      ref_slots: new Array(9).fill(""),
      ref_images: [],
      ref_videos: [],
      ref_audio_slots: new Array(3).fill(""),
      ref_audios: [],
      guide_audio: "",
    };
  });
}

function computeContinuationStreaks(shots, hasSeed = false) {
  const streaks = [];
  let run = 0;
  for (let i = 0; i < shots.length; i++) {
    const s = shots[i];
    const isCont = i === 0 ? Boolean(hasSeed && !s.hard_cut) : !Boolean(s.hard_cut);
    run = isCont ? (run + 1) : 1;
    streaks.push(run);
  }
  return streaks;
}

function attachPanelController(node) {
  const widgetMap = {};
  for (const w of node.widgets || []) {
    widgetMap[w.name] = w;
  }
  hideNativeWidgets(node);

  const getW = (name, def = "") => (widgetMap[name] ? widgetMap[name].value : def);
  const setW = (name, val) => {
    const w = widgetMap[name];
    if (!w) return;
    w.value = val;
    if (typeof w.callback === "function") w.callback(val);
    node.setDirtyCanvas?.(true, true);
  };
  const isLinked = (name) => Boolean((node.inputs || []).find((inp) => inp.name === name && inp.link != null));
  const findGraphLink = (lid) => {
    if (lid == null || !app.graph) return null;
    if (typeof app.graph.getLink === "function") return app.graph.getLink(lid);
    if (app.graph.links instanceof Map) return app.graph.links.get(lid) || null;
    return app.graph.links?.[lid] || null;
  };
  const findGraphNode = (nid) => {
    if (nid == null || !app.graph) return null;
    if (typeof app.graph.getNodeById === "function") return app.graph.getNodeById(nid);
    return (app.graph._nodes || []).find((x) => x && x.id === nid) || null;
  };
  const isUpstreamAudioActive = () => {
    const inp = (node.inputs || []).find((x) => x.name === "master_audio_in");
    if (!inp || inp.link == null) return false;
    let cur = inp.link;
    const seen = new Set();
    for (let hop = 0; hop < 12 && cur != null; hop++) {
      if (seen.has(cur)) return false;
      seen.add(cur);
      const lk = findGraphLink(cur);
      const oid = Array.isArray(lk) ? lk[1] : lk?.origin_id;
      const src = findGraphNode(oid);
      if (!src) return false;
      if (Number(src.mode || 0) !== 0) return false;
      const cls = src.comfyClass || src.type || "";
      if (cls === "GetNode") {
        const bus = String(src.widgets?.[0]?.value ?? src.widgets_values?.[0] ?? "");
        const setter = (app.graph?._nodes || []).find(
          (n) => n && (n.comfyClass === "SetNode" || n.type === "SetNode") && String(n.widgets?.[0]?.value ?? n.widgets_values?.[0] ?? "") === bus,
        );
        if (!setter || Number(setter.mode || 0) !== 0) return false;
        cur = setter.inputs?.[0]?.link ?? null;
        continue;
      }
      if (cls === "Reroute" || cls === "RerouteNode") {
        cur = src.inputs?.[0]?.link ?? null;
        continue;
      }
      return true;
    }
    return false;
  };
  const getOpts = (name, fb) => {
    const vals = widgetMap[name]?._origOptions?.values;
    return Array.isArray(vals) && vals.length ? vals : fb;
  };

  function buildPlanFromWidgets() {
    let initPlan = {};
    try { initPlan = JSON.parse(String(getW("director_plan_json", "{}") || "{}")); } catch { initPlan = {}; }
    const defDur = Math.max(0.25, Number(initPlan.default_duration_sec || 5.0));
    const shots = Array.isArray(initPlan.shots) && initPlan.shots.length ? initPlan.shots.map((s, idx) => {
      const aSlots = normSlots3(s.ref_audio_slots, s.ref_audios);
      const inhAud = ("inherit_global_audios" in s)
        ? (s.inherit_global_audios !== false)
        : (s.inherit_global_refs !== false);
      return {
        shot_index: idx + 1,
        duration_sec: Math.max(0.25, Number(s.duration_sec || defDur)),
        shot_type: Number(s.shot_type ?? 1) === 0 ? 0 : 1,
        hard_cut: Boolean(s.hard_cut),
        overlap_override: [22, 39, 56].includes(Number(s.overlap_override)) ? Number(s.overlap_override) : 0,
        inherit_global_refs: s.inherit_global_refs !== false,
        inherit_global_audios: inhAud,
        seed_text: String(s.seed_text || ""),
        prompt: String(s.prompt || ""),
        ref_slots: normSlots9(s.ref_slots, s.ref_images),
        ref_images: (s.ref_images || []).filter(Boolean),
        ref_videos: Array.isArray(s.ref_videos) ? s.ref_videos.slice(0, 3) : [],
        ref_audio_slots: aSlots,
        ref_audios: aSlots.filter(Boolean),
        guide_audio: String(s.guide_audio || ""),
      };
    }) : splitMasterScript(String(initPlan.master_script_text || getW("script_text", "") || ""), Boolean(getW("seed_video_path", "")), defDur);
    if (!shots.length) {
      shots.push({
        shot_index: 1, duration_sec: defDur, shot_type: 1, hard_cut: !getW("seed_video_path", ""),
        overlap_override: 0, inherit_global_refs: true, inherit_global_audios: true, seed_text: "", prompt: "",
        ref_slots: new Array(9).fill(""), ref_images: [], ref_videos: [],
        ref_audio_slots: new Array(3).fill(""), ref_audios: [], guide_audio: "",
      });
    }
    const explicitMode = typeof initPlan.audio_drive_mode === "string" ? initPlan.audio_drive_mode : "";
    const rawAudioMode = String(explicitMode || getW("audio_drive_mode", "none") || "none");
    const inferredAudio = !explicitMode && rawAudioMode === "none" && shots.some((s) => Boolean(String(s.guide_audio || "").trim()));
    const finalAudioMode = ["per_shot_audio_dir", "master_song_slice"].includes(rawAudioMode)
      ? rawAudioMode
      : (inferredAudio ? "per_shot_audio_dir" : "none");
    const hasAudio = finalAudioMode !== "none";
    const rawCtx = Number(initPlan.context_frames || getW("context_frames", 22));
    const gAudSlots = normSlots3(initPlan.global_ref_audio_slots, initPlan.global_ref_audios);
    return {
      master_script_text: String(initPlan.master_script_text || getW("script_text", "") || ""),
      auto_sync_master: initPlan.auto_sync_master !== false,
      default_duration_sec: defDur,
      audio_drive_mode: finalAudioMode,
      context_frames: hasAudio ? 39 : ([22, 39, 56].includes(rawCtx) ? rawCtx : 22),
      seed_video_path: String(initPlan.seed_video_path || getW("seed_video_path", "") || ""),
      seed_trim_end_sec: Number(initPlan.seed_trim_end_sec ?? getW("seed_trim_end_sec", 0) ?? 0),
      global_ref_slots: normSlots9(initPlan.global_ref_slots, initPlan.global_ref_images),
      global_ref_audio_slots: gAudSlots,
      global_ref_audios: gAudSlots.filter(Boolean),
      retry_lock: {
        enabled: Boolean(initPlan.retry_lock?.enabled),
        shot_index: Math.max(1, Number(initPlan.retry_lock?.shot_index || 1)),
        seed_mode: String(initPlan.retry_lock?.seed_mode || "randomize"),
        seed_text: String(initPlan.retry_lock?.seed_text || ""),
        prompt_override: String(initPlan.retry_lock?.prompt_override || ""),
        tail_bridge_frames: [0, 1, 5, 22].includes(Number(initPlan.retry_lock?.tail_bridge_frames))
          ? Number(initPlan.retry_lock.tail_bridge_frames)
          : 0,
        tail_bridge_noise: initPlan.retry_lock?.tail_bridge_noise === "reverse_taper" ? "reverse_taper" : "clean",
        auto_assemble_after_reroll: initPlan.retry_lock?.auto_assemble_after_reroll !== false,
      },
      anti_degradation: {
        mode: ["standard", "digital_human_low_noise", "digital_human_static_bg"].includes(String(initPlan.anti_degradation?.mode || ""))
          ? String(initPlan.anti_degradation.mode)
          : "standard",
        noise_keep_ratio: Number.isFinite(Number(initPlan.anti_degradation?.noise_keep_ratio))
          ? Math.max(0.05, Math.min(1.0, Number(initPlan.anti_degradation.noise_keep_ratio)))
          : 0.40,
        static_bg_pull: Number.isFinite(Number(initPlan.anti_degradation?.static_bg_pull))
          ? Math.max(0.0, Math.min(0.35, Number(initPlan.anti_degradation.static_bg_pull)))
          : 0.20,
      },
      shots,
    };
  }

  const initBuiltPlan = buildPlanFromWidgets();
  const state = {
    tab: "edit",
    sel: 0,
    playerMode: "seam",
    serverState: { shots: [], master_exports: [], master_sync_status: null },
    inventory: null,
    selectedTakes: new Set(),
    cleanerOpen: false,
    cleanerTab: "takes",
    exportOpen: false,
    exportBusy: false,
    exportDraft: {
      stale_policy: "allow_with_warning",
      output_mode: "overwrite_master",
      export_fcpxml: true,
      prepend_seed_video: true,
      audio_crossfade_ms: 15,
    },
    lastExport: null,
    periodicCutN: 4,
    masterExpanded: initBuiltPlan.shots.length <= 1,
    repairBusy: false,
    mediaBust: 0,
    lastVideoTime: 0,
    repairDrafts: {},
    syncTimer: null,
    plan: initBuiltPlan,
  };

  function isAudioMode() {
    const m = String(state.plan.audio_drive_mode || "none");
    return m === "per_shot_audio_dir" || m === "master_song_slice";
  }

  function serializePlan() {
    const p = state.plan;
    if (!["none", "per_shot_audio_dir", "master_song_slice"].includes(String(p.audio_drive_mode || "none"))) {
      p.audio_drive_mode = "none";
    }
    const audioOn = isAudioMode();
    if (audioOn) p.context_frames = 39;
    else if (![22, 39, 56].includes(Number(p.context_frames))) p.context_frames = 22;
    const shotsOut = p.shots.map((s, i) => {
      const slots = normSlots9(s.ref_slots, s.ref_images);
      const aSlots = normSlots3(s.ref_audio_slots, s.ref_audios);
      const ov = audioOn ? 0 : Number(s.overlap_override || 0);
      return {
        shot_index: i + 1, duration_sec: Number(s.duration_sec || p.default_duration_sec || 5.0), shot_type: Number(s.shot_type ?? 1),
        hard_cut: Boolean(s.hard_cut), overlap_override: ov,
        inherit_global_refs: s.inherit_global_refs !== false,
        inherit_global_audios: s.inherit_global_audios !== false,
        seed_text: String(s.seed_text || ""),
        prompt: String(s.prompt || ""), ref_slots: slots, ref_images: slots.filter(Boolean),
        ref_videos: s.ref_videos || [], ref_audio_slots: aSlots, ref_audios: aSlots.filter(Boolean), guide_audio: String(s.guide_audio || ""),
      };
    });
    const gSlots = normSlots9(p.global_ref_slots);
    const gAudSlots = normSlots3(p.global_ref_audio_slots, p.global_ref_audios);
    const jsonObj = {
      master_script_text: p.master_script_text, auto_sync_master: p.auto_sync_master,
      default_duration_sec: Number(p.default_duration_sec || 5.0),
      audio_drive_mode: p.audio_drive_mode,
      context_frames: audioOn ? 39 : Number(p.context_frames || 22), seed_video_path: p.seed_video_path, seed_trim_end_sec: p.seed_trim_end_sec,
      global_ref_slots: gSlots, global_ref_images: gSlots.filter(Boolean),
      global_ref_audio_slots: gAudSlots, global_ref_audios: gAudSlots.filter(Boolean), retry_lock: p.retry_lock,
      anti_degradation: p.anti_degradation || { mode: "standard", noise_keep_ratio: 0.40, static_bg_pull: 0.20 },
      shots: shotsOut,
    };
    const serialized = JSON.stringify(jsonObj, null, 2);
    setW("director_plan_json", serialized);
    setW("context_frames", String(audioOn ? 39 : Number(p.context_frames || 22)));
    setW("audio_drive_mode", p.audio_drive_mode);
    setW("seed_video_path", p.seed_video_path);
    setW("seed_trim_end_sec", p.seed_trim_end_sec);
    return serialized;
  }

  async function syncServer() {
    const serialized = serializePlan();
    try {
      const r = await api.fetchApi("/shotvault/sync-plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: String(getW("project_name", "MyH3Film") || "MyH3Film"),
          fps: Number(getW("fps", 24) || 24),
          common_prompt: String(getW("common_prompt", "") || ""),
          director_plan_json: serialized,
        }),
      });
      if (r.ok) { state.serverState = await r.json(); patchFooter(); }
    } catch {}
  }

  function scheduleSync(immediate = false) {
    serializePlan();
    clearTimeout(state.syncTimer);
    state.syncTimer = setTimeout(syncServer, immediate ? 40 : 380);
  }

  async function refreshInventory(silent = false) {
    const proj = String(getW("project_name", "MyH3Film") || "MyH3Film");
    try {
      const r = await api.fetchApi(`/shotvault/history-inventory?project=${encodeURIComponent(proj)}`);
      if (r.ok) {
        state.inventory = await r.json();
        if (!silent) renderBody();
      }
    } catch {}
  }

  async function refreshState() {
    const proj = String(getW("project_name", "MyH3Film") || "MyH3Film");
    try {
      const [r1, r2] = await Promise.all([
        api.fetchApi(`/shotvault/state?project=${encodeURIComponent(proj)}`),
        api.fetchApi(`/shotvault/history-inventory?project=${encodeURIComponent(proj)}`),
      ]);
      if (r1.ok) state.serverState = await r1.json();
      if (r2.ok) state.inventory = await r2.json();
      renderAll();
    } catch {}
  }

  function findDownstreamVaultNodes() {
    const nodes = app.graph?._nodes || [];
    const loads = nodes.filter((n) => n?.comfyClass === "ShotVaultLoad");
    const dispatches = nodes.filter((n) => n?.comfyClass === "ShotVaultLoopDispatch");
    return { loads, dispatches };
  }

  function runChecks() {
    // Calm UI principle: ONLY report genuine hard/blocking errors — zero heuristic nag warnings!
    const issues = [];
    const { loads, dispatches } = findDownstreamVaultNodes();
    if (!loads.length && !dispatches.length) {
      issues.push({ lv: "warn", shot: -1, msg: t("chk_no_downstream") });
    }
    if (state.plan.audio_drive_mode === "master_song_slice" && !isUpstreamAudioActive()) {
      issues.push({ lv: "err", shot: -1, msg: t("chk_audio_bypass") });
    }
    state.plan.shots.forEach((s, idx) => {
      if (!String(s.prompt || "").trim()) issues.push({ lv: "err", shot: idx, msg: t("chk_empty_prompt", { i: idx + 1 }) });
      if (Number(s.duration_sec || 0) <= 0.25) issues.push({ lv: "warn", shot: idx, msg: t("chk_short_dur", { i: idx + 1, dur: s.duration_sec }) });
    });
    return issues;
  }

  function applyMasterSplit(preserveMedia = true) {
    const parsed = splitMasterScript(
      state.plan.master_script_text,
      Boolean(state.plan.seed_video_path),
      state.plan.default_duration_sec || 5.0,
    );
    if (!parsed.length) return false;
    state.plan.shots = parsed.map((ns, i) => {
      const old = preserveMedia ? state.plan.shots[i] : null;
      if (!old) return ns;
      const oldAudSlots = normSlots3(old.ref_audio_slots, old.ref_audios);
      return {
        ...ns,
        overlap_override: isAudioMode() ? 0 : (ns.overlap_override || old.overlap_override || 0),
        inherit_global_refs: old.inherit_global_refs !== false,
        inherit_global_audios: ("inherit_global_audios" in old) ? (old.inherit_global_audios !== false) : (old.inherit_global_refs !== false),
        seed_text: old.seed_text || "",
        ref_slots: normSlots9(old.ref_slots, old.ref_images),
        ref_images: old.ref_images || [],
        ref_videos: old.ref_videos || [],
        ref_audio_slots: oldAudSlots,
        ref_audios: oldAudSlots.filter(Boolean),
        guide_audio: old.guide_audio || "",
      };
    });
    if (state.sel >= state.plan.shots.length) state.sel = Math.max(0, state.plan.shots.length - 1);
    return true;
  }

  function composeMasterFromShots() {
    const txt = state.plan.shots.map((s, idx) => {
      const lines = [`& ${Number(s.duration_sec || 5.0)} &`];
      if (Number(s.shot_type ?? 1) === 0) lines.push(t("tag_type_drama"));
      if (idx > 0 && s.hard_cut) lines.push("@cut@");
      else if (idx > 0 && !s.hard_cut) lines.push("@continue@");
      else if (idx === 0 && !s.hard_cut && state.plan.seed_video_path) lines.push("@continue@");
      if ([22, 39, 56].includes(Number(s.overlap_override))) lines.push(`@overlap=${Number(s.overlap_override)}`);
      lines.push(String(s.prompt || "").trim());
      return lines.join("\n");
    }).join("\n===\n");
    state.plan.master_script_text = txt;
    setW("script_text", txt);
    return txt;
  }

  let stripWrap = null;
  let editorWrap = null;

  function applyPanelSize(targetW, targetH) {
    const w = Math.max(PANEL_DEFAULTS.MIN_W, Math.min(PANEL_DEFAULTS.MAX_W, Math.round(Number(targetW) || PANEL_W)));
    const h = Math.max(PANEL_DEFAULTS.MIN_H, Math.min(PANEL_DEFAULTS.MAX_H, Math.round(Number(targetH) || PANEL_H)));
    host.style.height = `${h}px`;
    root.classList.toggle("sv-w-wide", w >= PANEL_DEFAULTS.WIDE_BP);
    node.size = [w, h + PANEL_DEFAULTS.CHROME_H];
    node.setDirtyCanvas?.(true, true);
  }

  function syncResponsiveFromNode() {
    const nw = Math.max(PANEL_DEFAULTS.MIN_W, Math.min(PANEL_DEFAULTS.MAX_W, Math.round(Number(node.size?.[0]) || PANEL_W)));
    const nh = Math.max(PANEL_DEFAULTS.MIN_H, Math.min(PANEL_DEFAULTS.MAX_H, Math.round((Number(node.size?.[1]) || (PANEL_H + PANEL_DEFAULTS.CHROME_H)) - PANEL_DEFAULTS.CHROME_H)));
    host.style.height = `${nh}px`;
    root.classList.toggle("sv-w-wide", nw >= PANEL_DEFAULTS.WIDE_BP);
    return { w: nw, h: nh };
  }

  function patchFooter() {
    const p = state.plan;
    const totalSec = p.shots.reduce((a, s) => a + Number(s.duration_sec || 0), 0).toFixed(1);
    const actCnt = p.shots.filter((s) => Number(s.shot_type ?? 1) === 1).length;
    const drmCnt = p.shots.length - actCnt;
    const svShots = state.serverState?.shots || [];
    const okCnt = svShots.filter((x) => x.status === "VALID" || x.status === "LOCKED_KEEP" || Boolean(x.locked_keep)).length;
    const lockTag = p.retry_lock?.enabled ? t("lock_tag", { i: p.retry_lock.shot_index }) : "";
    const modeTag = isAudioMode() ? t("mode_audio") : t("mode_regular", { g: p.context_frames });
    const summaryText = t("footer_line", { n: p.shots.length, dur: totalSec, mode: modeTag, act: actCnt, drm: drmCnt, ok: okCnt, lock: lockTag });
    ftrEl.innerHTML = "";
    const sizeBtns = el("div", { class: "sv-size-bar" }, [
      el("span", { style: { fontSize: "10.5px", color: "#7f889e" }, text: "📐" }),
      el("button", { class: "sv-btn xs", text: t("sz_compact"), title: "1020×720", onclick: () => applyPanelSize(PANEL_DEFAULTS.PRESETS.compact.w, PANEL_DEFAULTS.PRESETS.compact.h) }),
      el("button", { class: "sv-btn xs", text: t("sz_std"), title: "1180×840", onclick: () => applyPanelSize(PANEL_DEFAULTS.PRESETS.standard.w, PANEL_DEFAULTS.PRESETS.standard.h) }),
      el("button", { class: "sv-btn xs", text: t("sz_wide"), title: t("sz_wide_title"), onclick: () => applyPanelSize(PANEL_DEFAULTS.PRESETS.wide.w, PANEL_DEFAULTS.PRESETS.wide.h) }),
    ]);
    appendSafe(ftrEl, el("span", { style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, text: summaryText }), sizeBtns);
  }

  function patchCardPreview(idx) {
    if (!stripWrap) return;
    const card = stripWrap.querySelector(`[data-sv-card="${idx}"]`);
    if (!card) return;
    const s = state.plan.shots[idx];
    const sn = card.querySelector(".sv-card-sn");
    const dur = card.querySelector(".sv-card-dur");
    if (sn) sn.textContent = extractCardSnippet(s?.prompt || "", 36);
    if (dur) dur.textContent = `${Number(s?.duration_sec || 5).toFixed(1)}s`;
    patchFooter();
  }

  function renderHeader() {
    hdrEl.innerHTML = "";
    const issues = runChecks();
    const audioOn = isAudioMode();
    if (audioOn) state.plan.context_frames = 39;
    const projNode = isLinked("project_name")
      ? el("span", { class: "sv-link-lbl", text: t("lbl_project_linked") })
      : el("input", {
          class: "sv-inp",
          value: String(getW("project_name", "MyH3Film")),
          style: { width: "120px" },
          title: t("proj_input_title"),
          oninput: (e) => { setW("project_name", e.target.value); scheduleSync(); },
        });
    const modeBtns = [
      el("button", {
        class: `sv-btn ${!audioOn ? "pri" : ""}`,
        text: t("btn_regular"),
        title: t("title_btn_regular"),
        onclick: () => {
          state.plan.audio_drive_mode = "none";
          scheduleSync(true);
          renderAll();
          showToast(t("toast_mode_regular"));
        },
      }),
      el("button", {
        class: `sv-btn ${audioOn ? "ok" : ""}`,
        text: t("btn_audio"),
        title: t("title_btn_audio"),
        onclick: () => {
          if (state.plan.audio_drive_mode === "none") {
            state.plan.audio_drive_mode = isUpstreamAudioActive() ? "master_song_slice" : "per_shot_audio_dir";
          }
          state.plan.context_frames = 39;
          scheduleSync(true);
          renderAll();
          showToast(t("toast_mode_audio"));
        },
      }),
      audioOn
        ? el("select", {
            class: "sv-sel",
            onchange: (e) => { state.plan.audio_drive_mode = e.target.value; scheduleSync(true); renderAll(); },
          }, [
            el("option", { value: "per_shot_audio_dir", selected: state.plan.audio_drive_mode !== "master_song_slice", text: t("opt_per_shot_audio") }),
            el("option", { value: "master_song_slice", selected: state.plan.audio_drive_mode === "master_song_slice", text: t("opt_master_slice") }),
          ])
        : null,
    ];
    const rlPill = state.plan.retry_lock?.enabled
      ? el("button", {
          class: "sv-btn warn",
          text: t("rl_pill_locked", { i: state.plan.retry_lock.shot_index }),
          onclick: () => {
            state.plan.retry_lock.enabled = false;
            scheduleSync(true);
            renderAll();
            showToast(t("toast_rl_unlocked"));
          },
        })
      : null;
    const gearCtrl = audioOn
      ? [el("span", { class: "sv-link-lbl", text: t("lbl_gear_locked") })]
      : [22, 39].map((g) => el("button", {
          class: `sv-btn ${Number(state.plan.context_frames) === g ? "pri" : ""}`,
          text: t("gear_btn_lbl", { g }),
          onclick: () => {
            state.plan.context_frames = g;
            scheduleSync(true);
            renderAll();
          },
        }));
    const loopRounds = state.plan.retry_lock?.enabled ? 1 : state.plan.shots.length;
    const totalDur = state.plan.shots.reduce((a, s) => a + Number(s.duration_sec || 0), 0).toFixed(1);
    const activeRepairCnt = (state.serverState?.shots || []).filter((x) => x.seam_repair?.enabled && (x.seam_repair?.choice || "repaired") === "repaired").length;
    const syncStatus = state.serverState?.master_sync_status || null;
    const masterOutdated = Boolean(syncStatus?.is_outdated && syncStatus?.has_master);
    const adMode = String(state.plan.anti_degradation?.mode || "standard");
    const adBadge = adMode !== "standard"
      ? el("button", {
          class: "sv-btn act",
          text: adMode === "digital_human_static_bg" ? t("ad_badge_bg") : t("ad_badge_low"),
          title: t("ad_hdr"),
          onclick: () => { state.tab = "gear"; renderTabs(); renderBody(); },
        })
      : null;
    const exportBtnText = masterOutdated
      ? t("btn_export_master_outdated")
      : (activeRepairCnt > 0 ? t("btn_export_master_rep", { n: activeRepairCnt }) : t("btn_export_master"));
    const reasonSep = getLang() === "en" ? ", " : "、";
    appendSafe(
      hdrEl,
      el("div", { class: "sv-title" }, [t("app_title"), el("span", { class: "sv-badge", text: t("badge_summary", { n: state.plan.shots.length, dur: totalDur, rounds: loopRounds }) })]),
      el("span", { text: t("lbl_project") }),
      projNode,
      ...modeBtns,
      adBadge,
      rlPill,
      el("span", { class: "sv-hdr-sp" }),
      el("span", { text: t("lbl_overlap") }),
      ...gearCtrl,
      el("button", {
        class: `sv-btn ${masterOutdated ? "warn" : "ok"}`,
        text: exportBtnText,
        title: masterOutdated
          ? t("outdated_tip", { reasons: pickOutdatedReasons(syncStatus).join(reasonSep) })
          : t("exp_title"),
        onclick: async () => {
          state.exportOpen = true;
          state.lastExport = null;
          await refreshState();
        },
      }),
      el("button", {
        class: "sv-btn act",
        text: t("btn_lang"),
        title: t("title_btn_lang"),
        onclick: () => {
          toggleLang();
          showToast(t("toast_lang_switched"));
        },
      }),
      el("button", { class: "sv-btn", text: t("btn_refresh"), onclick: () => refreshState() }),
      el("button", {
        class: `sv-btn ${issues.length ? "warn" : "ok"}`,
        text: issues.length ? t("btn_checks", { n: issues.length }) : t("btn_ready"),
        onclick: () => { state.tab = "gear"; renderTabs(); renderBody(); },
      }),
    );
  }

  function renderTabs() {
    tabsEl.innerHTML = "";
    const defs = [
      ["edit", t("tab_edit")],
      ["reroll", t("tab_reroll")],
      ["shot0", t("tab_shot0")],
      ["gear", t("tab_gear")],
    ];
    for (const [id, label] of defs) {
      tabsEl.appendChild(el("div", {
        class: `sv-tab ${state.tab === id ? "on" : ""}`,
        text: label,
        onclick: () => { state.tab = id; renderTabs(); renderBody(); },
      }));
    }
  }

  function render9SlotGrid(slots, globalSlots, inheritGlobal, onUpdate, subfolder) {
    const grid = el("div", { class: "sv-grid9" });
    for (let k = 0; k < 9; k++) {
      const own = String(slots[k] || "").trim();
      const gVal = inheritGlobal && globalSlots ? String(globalSlots[k] || "").trim() : "";
      const eff = own || gVal;
      const box = el("div", { class: `sv-slot ${eff ? "has" : ""}` });
      box.ondragover = (e) => { e.preventDefault(); box.style.borderColor = "#2f6df6"; };
      box.ondragleave = () => { box.style.borderColor = ""; };
      box.ondrop = async (e) => {
        e.preventDefault();
        box.style.borderColor = "";
        const f = e.dataTransfer?.files?.[0];
        if (!f) return;
        try {
          const res = await uploadMediaFile(f, subfolder);
          onUpdate(k, res.rel_path);
        } catch (err) { showToast(t("toast_upload_failed", { msg: err.message })); }
      };
      const imgBox = el("div", {
        class: "sv-slot-img",
        title: eff ? eff : t("slot_upload_title", { k: k + 1 }),
        onclick: () => pickLocalFile("image/*", async (f) => {
          try {
            const res = await uploadMediaFile(f, subfolder);
            onUpdate(k, res.rel_path);
          } catch (err) { showToast(t("toast_upload_failed", { msg: err.message })); }
        }),
      });
      if (eff) imgBox.appendChild(el("img", { src: mediaUrl(eff) }));
      else imgBox.textContent = `+ Picture ${k + 1}`;
      const ft = el("div", { class: "sv-slot-ft" }, [
        el("span", { text: own ? t("slot_own", { k: k + 1 }) : (gVal ? t("slot_inherited", { k: k + 1 }) : t("slot_empty", { k: k + 1 })) }),
        own ? el("button", { class: "sv-btn dng", style: { padding: "1px 5px", fontSize: "10px" }, text: "×", onclick: (e) => { e.stopPropagation(); onUpdate(k, ""); } }) : null,
      ]);
      box.append(imgBox, ft);
      grid.appendChild(box);
    }
    return grid;
  }

  let previewAud = null;
  let previewPath = "";
  function stopAudioPreview() {
    if (previewAud) {
      try { previewAud.pause(); previewAud.src = ""; } catch {}
      previewAud = null;
    }
    previewPath = "";
  }
  function toggleAudioPreview(relPath, onStateChange) {
    const p = String(relPath || "").trim();
    if (!p) return;
    if (previewAud && previewPath === p && !previewAud.paused) {
      stopAudioPreview();
      onStateChange?.();
      return;
    }
    stopAudioPreview();
    try {
      const aud = new Audio(mediaUrl(p));
      previewAud = aud;
      previewPath = p;
      aud.onended = () => { if (previewAud === aud) { previewAud = null; previewPath = ""; onStateChange?.(); } };
      aud.onerror = () => { if (previewAud === aud) { previewAud = null; previewPath = ""; onStateChange?.(); } };
      aud.play().catch(() => { if (previewAud === aud) { previewAud = null; previewPath = ""; onStateChange?.(); } });
      onStateChange?.();
    } catch {
      stopAudioPreview();
      onStateChange?.();
    }
  }

  function render3AudioSlotGrid(slots, globalSlots, inheritGlobal, muted, onUpdate, subfolder, onRerender) {
    const grid = el("div", { class: "sv-aud-grid3" });
    const ownSlots = normSlots3(slots);
    const globSlots = normSlots3(globalSlots);
    for (let k = 0; k < 3; k++) {
      const own = ownSlots[k];
      const gVal = inheritGlobal && globalSlots ? globSlots[k] : "";
      const eff = own || gVal;
      const fn = eff ? eff.split("/").pop() : "";
      const isPlaying = Boolean(eff && previewAud && !previewAud.paused && previewPath === eff);
      const cls = `sv-aud-slot${muted ? " muted" : ""}${own ? " own" : (gVal ? " inh" : "")}`;
      const box = el("div", { class: cls });
      if (!muted) {
        box.ondragover = (e) => { e.preventDefault(); box.style.borderColor = "#2f6df6"; };
        box.ondragleave = () => { box.style.borderColor = ""; };
        box.ondrop = async (e) => {
          e.preventDefault();
          box.style.borderColor = "";
          const f = e.dataTransfer?.files?.[0];
          if (!f) return;
          try {
            const res = await uploadMediaFile(f, subfolder);
            onUpdate(k, res.rel_path);
            showToast(t("toast_ref_audio_added", { name: res.rel_path.split("/").pop() }));
          } catch (err) { showToast(t("toast_upload_failed", { msg: err?.message || err }), true); }
        };
      }
      const badgeTxt = own
        ? t("aud_slot_own", { k: k + 1 })
        : (gVal ? t("aud_slot_inherited", { k: k + 1 }) : `<Audio ${k + 1}>`);
      const top = el("div", { class: "sv-aud-top" }, [
        el("span", { style: { fontWeight: "600", color: own ? "#8df2c2" : (gVal ? "#9cc2ff" : "#8c95ad") }, text: badgeTxt }),
        el("div", { class: "sv-row", style: { gap: "3px" } }, [
          eff && !muted ? el("button", {
            class: `sv-btn xs ${isPlaying ? "warn" : ""}`,
            title: t("title_play_audio"),
            text: isPlaying ? t("btn_stop_audio") : t("btn_play_audio"),
            onclick: (e) => { e.stopPropagation(); toggleAudioPreview(eff, onRerender); },
          }) : null,
          own && !muted ? el("button", {
            class: "sv-btn dng xs",
            text: "×",
            onclick: (e) => {
              e.stopPropagation();
              if (previewPath === own) stopAudioPreview();
              onUpdate(k, "");
            },
          }) : null,
        ]),
      ]);
      const nameRow = el("div", {
        class: `sv-aud-name${eff ? "" : " empty"}`,
        title: eff ? eff : t("aud_slot_upload_title", { k: k + 1 }),
        text: eff ? fn : t("aud_slot_empty", { k: k + 1 }),
        onclick: () => {
          if (muted) return;
          pickLocalFile("audio/*", async (f) => {
            try {
              const res = await uploadMediaFile(f, subfolder);
              onUpdate(k, res.rel_path);
              showToast(t("toast_ref_audio_added", { name: res.rel_path.split("/").pop() }));
            } catch (err) { showToast(t("toast_upload_failed", { msg: err?.message || err }), true); }
          });
        },
      });
      box.append(top, nameRow);
      grid.appendChild(box);
    }
    return grid;
  }

  function renderFilmstrip() {
    if (!stripWrap) return;
    stripWrap.innerHTML = "";
    const svMap = new Map((state.serverState?.shots || []).map((x) => [Number(x.shot_index), x]));
    state.plan.shots.forEach((s, idx) => {
      const sv = svMap.get(idx + 1);
      const actTake = (sv?.takes || []).find((tk) => tk.take_id === sv?.active_take_id) || (sv?.takes || [])[0];
      const ownCnt = (s.ref_slots || []).filter(Boolean).length;
      const effImg = s.ref_slots.find(Boolean) || (s.inherit_global_refs ? state.plan.global_ref_slots.find(Boolean) : "");
      const thumbSrc = actTake?.thumb_rel ? mediaUrl(actTake.thumb_rel, String(getW("project_name", "MyH3Film"))) : (effImg ? mediaUrl(effImg) : "");
      const isAct = Number(s.shot_type ?? 1) === 1;
      const audioOn = isAudioMode();
      const effAudCnt = audioOn ? 0 : resolveEffectiveAudioSlots(s, state.plan.global_ref_audio_slots).filter(Boolean).length;
      const effGear = audioOn ? 39 : (s.overlap_override || state.plan.context_frames);
      const card = el("div", {
        class: `sv-card ${state.sel === idx ? "on" : ""}`,
        "data-sv-card": String(idx),
        onclick: () => { state.sel = idx; renderFilmstrip(); renderShotEditor(); },
      });
      const th = el("div", { class: "sv-thumb" });
      if (thumbSrc) th.appendChild(el("img", { src: thumbSrc }));
      else th.textContent = t("card_unrendered", { i: idx + 1 });
      const badges = el("div", { class: "sv-row", style: { gap: "3px", justifyContent: "space-between" } }, [
        el("span", { style: { fontWeight: "700" }, text: `#${idx + 1}` }),
        el("span", { class: isAct ? "sv-tag-a" : "sv-tag-d", text: isAct ? t("tag_action_short") : t("tag_drama_short") }),
        el("span", {
          class: "sv-badge",
          style: (idx > 0 && s.hard_cut)
            ? { background: "#173c47", color: "#9ee8ff", borderColor: "#2d7c94" }
            : {},
          title: (idx > 0 && s.hard_cut) ? t("card_cut_tip") : t("card_cont_tip", { g: effGear }),
          text: s.hard_cut ? (idx > 0 ? "✂@cut@" : t("card_first")) : `${audioOn ? "🔒" : "🔗"}${effGear}f`,
        }),
        el("span", { class: "sv-card-dur", style: { color: "#9da5ba", fontSize: "10.5px" }, text: `${Number(s.duration_sec || 5).toFixed(1)}s` }),
      ]);
      const sn = el("div", {
        class: "sv-card-sn",
        style: { fontSize: "10.5px", color: "#aeb6cb", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" },
        text: extractCardSnippet(s.prompt, 36),
      });
      const meta = el("div", { style: { fontSize: "10px", color: "#7f889e", display: "flex", justifyContent: "space-between" } }, [
        el("span", { text: `🖼 ${ownCnt}/9 · ${audioOn ? "🔇" : `🔊 ${effAudCnt}/3`}` }),
        el("span", { text: sv?.status || "UNRENDERED" }),
      ]);
      card.append(th, badges, sn, meta);
      stripWrap.appendChild(card);
    });
  }

  function renderShotEditor() {
    if (!editorWrap) return;
    editorWrap.innerHTML = "";
    const idx = Math.max(0, Math.min(state.sel, state.plan.shots.length - 1));
    state.sel = idx;
    const s = state.plan.shots[idx];
    const audioOn = isAudioMode();
    const effGear = audioOn ? 39 : (s.overlap_override || state.plan.context_frames);
    const gInfo = computeGear(effGear, idx > 0 || Boolean(state.plan.seed_video_path) ? !s.hard_cut : false);

    const durInp = el("input", {
      class: "sv-inp", type: "number", step: "0.5", min: "0.5", max: "20", value: String(s.duration_sec),
      style: { width: "62px" },
      oninput: (e) => { s.duration_sec = Math.max(0.25, parseFloat(e.target.value) || 5.0); patchCardPreview(idx); scheduleSync(); },
    });

    const gearNode = audioOn
      ? el("span", { class: "sv-link-lbl", text: t("ed_gear_audio_lock") })
      : el("select", {
          class: "sv-sel",
          onchange: (e) => { s.overlap_override = Number(e.target.value || 0); scheduleSync(true); renderFilmstrip(); renderShotEditor(); },
        }, [
          el("option", { value: "0", selected: s.overlap_override === 0, text: t("ed_gear_follow", { g: state.plan.context_frames }) }),
          el("option", { value: "22", selected: s.overlap_override === 22, text: t("ed_gear_22") }),
          el("option", { value: "39", selected: s.overlap_override === 39, text: t("ed_gear_39") }),
          el("option", { value: "56", selected: s.overlap_override === 56, text: t("ed_gear_56") }),
        ]);

    const topCtrl = el("div", { class: "sv-row" }, [
      el("strong", { text: t("ed_shot_hdr", { i: idx + 1 }) }),
      el("button", {
        class: `sv-btn ${Number(s.shot_type ?? 1) === 1 ? "warn" : ""}`,
        text: t("btn_action_flow"),
        onclick: () => { s.shot_type = 1; scheduleSync(); renderFilmstrip(); renderShotEditor(); },
      }),
      el("button", {
        class: `sv-btn ${Number(s.shot_type ?? 1) === 0 ? "pri" : ""}`,
        text: t("btn_drama_flow"),
        onclick: () => { s.shot_type = 0; scheduleSync(); renderFilmstrip(); renderShotEditor(); },
      }),
      el("span", { text: t("lbl_dur_amp") }),
      durInp,
      el("span", { text: t("lbl_shot_overlap") }),
      gearNode,
      el("label", { class: "sv-row", style: { gap: "4px", cursor: "pointer" } }, [
        el("input", {
          type: "checkbox", checked: s.hard_cut,
          onchange: (e) => { s.hard_cut = e.target.checked; scheduleSync(); renderFilmstrip(); renderShotEditor(); },
        }),
        el("span", { text: t("lbl_hard_cut") }),
      ]),
      el("span", { text: "🎲Seed:" }),
      el("input", {
        class: "sv-inp", value: s.seed_text || "", placeholder: t("ph_seed_empty"), style: { width: "92px" },
        oninput: (e) => { s.seed_text = e.target.value; scheduleSync(); },
      }),
      el("span", { class: "sv-badge", text: t("badge_eff_gear", { trim: gInfo.trim, taper: gInfo.taperTail }) }),
    ]);

    const audioOnChips = isAudioMode();
    const chipTags = [
      ...Array.from({ length: 9 }, (_, k) => `<Picture ${k + 1}>`),
      ...(audioOnChips ? [] : ["<Audio 1>", "<Audio 2>", "<Audio 3>"]),
      "<Video 1>", "<Video 2>",
    ];

    const { wrap: promptWrap, ta: promptTa } = makeDraggableTextarea({
      value: s.prompt,
      prefKey: "shot_prompt",
      defaultHeight: 156,
      minHeight: 84,
      maxHeight: 520,
      placeholder: t("ph_shot_prompt"),
      onInput: (e) => { s.prompt = e.target.value; patchCardPreview(idx); scheduleSync(); },
      onChange: (e) => {
        const amp = parseSingleShotAmpersand(e.target.value, s.duration_sec);
        if (amp.found_dur) {
          s.duration_sec = amp.duration_sec;
          s.prompt = amp.body;
          e.target.value = amp.body;
          durInp.value = String(amp.duration_sec);
          patchCardPreview(idx);
          scheduleSync(true);
          showToast(t("toast_dur_extracted", { dur: amp.duration_sec }));
        }
      },
    });

    const promptHdr = el("div", { class: "sv-box-hdr" }, [
      el("span", { style: { fontSize: "11.5px", color: "#b6bfd6" }, text: t("lbl_shot_prompt_hdr", { i: idx + 1, n: (s.prompt || "").length }) }),
      el("button", {
        class: "sv-btn act xs",
        text: t("btn_focus_edit"),
        onclick: () => openFocusPromptModal({
          title: t("focus_title_shot", { i: idx + 1 }),
          value: s.prompt,
          placeholder: promptTa.placeholder,
          chips: [...chipTags, "& 8 &"],
          el,
          t,
          onSave: (val) => {
            s.prompt = val;
            promptTa.value = val;
            patchCardPreview(idx);
            scheduleSync(true);
            renderShotEditor();
          },
        }),
      }),
    ]);

    const chips = el("div", { class: "sv-chips" }, chipTags.map((tag) => el("button", {
      class: "sv-chip",
      text: tag,
      onclick: () => {
        const st = promptTa.selectionStart ?? s.prompt.length;
        const en = promptTa.selectionEnd ?? s.prompt.length;
        const next = `${s.prompt.slice(0, st)}${tag}${s.prompt.slice(en)}`;
        s.prompt = next;
        promptTa.value = next;
        promptTa.focus();
        promptTa.setSelectionRange(st + tag.length, st + tag.length);
        patchCardPreview(idx);
        scheduleSync();
      },
    })));

    const refBar = el("div", { class: "sv-box-hdr" }, [
      el("span", { text: t("lbl_shot_slots", { i: idx + 1 }) }),
      el("div", { class: "sv-row" }, [
        el("label", { class: "sv-row", style: { gap: "4px", fontWeight: "normal", cursor: "pointer" } }, [
          el("input", {
            type: "checkbox", checked: s.inherit_global_refs !== false,
            onchange: (e) => { s.inherit_global_refs = e.target.checked; scheduleSync(); renderShotEditor(); },
          }),
          el("span", { text: t("lbl_inherit_global") }),
        ]),
        el("button", {
          class: "sv-btn act",
          text: t("btn_copy_refs_forward"),
          onclick: () => {
            if (idx >= state.plan.shots.length - 1) {
              return showToast(t("toast_already_last"));
            }
            const resolved = s.ref_slots.map((x, k) => x || (s.inherit_global_refs ? state.plan.global_ref_slots[k] : "") || "");
            for (let j = idx + 1; j < state.plan.shots.length; j++) {
              state.plan.shots[j].ref_slots = [...resolved];
              state.plan.shots[j].ref_images = resolved.filter(Boolean);
            }
            scheduleSync(true);
            renderFilmstrip();
            showToast(t("toast_refs_copied_forward", { i: idx + 1, n: state.plan.shots.length - idx - 1 }));
          },
        }),
        el("button", {
          class: "sv-btn dng", text: t("btn_clear_shot_refs"),
          onclick: () => { s.ref_slots = new Array(9).fill(""); s.ref_images = []; scheduleSync(); renderFilmstrip(); renderShotEditor(); },
        }),
      ]),
    ]);

    const grid9 = render9SlotGrid(s.ref_slots, state.plan.global_ref_slots, s.inherit_global_refs !== false, (k, val) => {
      s.ref_slots[k] = val;
      s.ref_images = s.ref_slots.filter(Boolean);
      scheduleSync();
      renderFilmstrip();
      renderShotEditor();
    }, `shotvault/shot_${idx + 1}`);

    const mmRow = el("div", { class: "sv-row", style: { marginTop: "4px" } }, [
      el("button", {
        class: `sv-btn ${s.guide_audio ? "ok" : ""}`,
        text: s.guide_audio ? t("btn_guide_audio_set", { name: s.guide_audio.split("/").pop() }) : t("btn_guide_audio_up"),
        onclick: () => pickLocalFile("audio/*", async (f) => {
          const [r, probedDur] = await Promise.all([
            uploadMediaFile(f, `shotvault/shot_${idx + 1}`),
            probeBrowserAudioDuration(f),
          ]);
          s.guide_audio = r.rel_path;
          if (probedDur) s.duration_sec = Math.max(0.25, probedDur);
          if (state.plan.audio_drive_mode === "none") state.plan.audio_drive_mode = "per_shot_audio_dir";
          state.plan.context_frames = 39;
          scheduleSync(true);
          renderAll();
          showToast(probedDur ? t("toast_audio_probed", { dur: probedDur.toFixed(2) }) : t("toast_audio_attached"));
        }),
      }),
      s.guide_audio ? el("button", { class: "sv-btn dng", text: t("btn_clear_audio"), onclick: () => { s.guide_audio = ""; scheduleSync(true); renderAll(); } }) : null,
      el("button", {
        class: `sv-btn ${(s.ref_videos || []).length ? "ok" : ""}`,
        text: t("btn_ref_video", { n: (s.ref_videos || []).length }),
        onclick: () => pickLocalFile("video/*", async (f) => {
          const r = await uploadMediaFile(f, `shotvault/shot_${idx + 1}`);
          s.ref_videos = [...(s.ref_videos || []), { path: r.rel_path, with_audio: true }].slice(-3);
          scheduleSync();
          renderShotEditor();
          showToast(t("toast_ref_video_added", { name: r.rel_path.split("/").pop() }));
        }),
      }),
      (s.ref_videos || []).length ? el("button", { class: "sv-btn dng", text: t("btn_clear_video"), onclick: () => { s.ref_videos = []; scheduleSync(); renderShotEditor(); } }) : null,
    ]);

    const mmChips = [];
    (s.ref_videos || []).forEach((vp, vi) => {
      const vObj = typeof vp === "string" ? { path: vp, with_audio: true } : (vp || {});
      const fn = String(vObj.path || "").split("/").pop();
      mmChips.push(el("span", { class: "sv-chip" }, [
        `<Video ${vi + 1}> ${fn} `,
        audioOnChips
          ? el("span", { style: { fontSize: "10px", color: "#9da5ba" }, text: t("lbl_video_only") })
          : el("label", { style: { display: "inline-flex", alignItems: "center", gap: "2px", fontSize: "10px", cursor: "pointer" } }, [
              el("input", {
                type: "checkbox", checked: vObj.with_audio !== false,
                onchange: (e) => { s.ref_videos[vi] = { path: vObj.path, with_audio: e.target.checked }; scheduleSync(); renderShotEditor(); },
              }),
              t("lbl_with_audio"),
            ]),
        el("button", {
          class: "sv-btn dng", style: { padding: "0 4px", fontSize: "10px" }, text: "×",
          onclick: (e) => { e.stopPropagation(); s.ref_videos.splice(vi, 1); scheduleSync(); renderShotEditor(); },
        }),
      ]));
    });
    const mmChipBar = mmChips.length ? el("div", { class: "sv-chips" }, mmChips) : null;

    s.ref_audio_slots = normSlots3(s.ref_audio_slots, s.ref_audios);
    const ownAudAny = s.ref_audio_slots.some(Boolean);
    const vidWithAudioCnt = (s.ref_videos || []).filter((vp) => {
      const vObj = typeof vp === "string" ? { path: vp, with_audio: true } : (vp || {});
      return Boolean(String(vObj.path || "").trim()) && vObj.with_audio !== false;
    }).length;
    const voiceHdr = el("div", { class: "sv-box-hdr" }, [
      el("span", { text: t("lbl_shot_audio_slots", { i: idx + 1 }) }),
      el("div", { class: "sv-row", style: { gap: "4px" } }, [
        el("label", { class: "sv-row", style: { gap: "4px", fontWeight: "normal", cursor: "pointer" } }, [
          el("input", {
            type: "checkbox",
            checked: s.inherit_global_audios !== false,
            disabled: audioOnChips,
            onchange: (e) => { s.inherit_global_audios = e.target.checked; scheduleSync(); renderFilmstrip(); renderShotEditor(); },
          }),
          el("span", { text: t("lbl_inherit_global_audios") }),
        ]),
        el("button", {
          class: "sv-btn act xs",
          disabled: audioOnChips,
          style: audioOnChips ? { opacity: "0.5", cursor: "not-allowed" } : {},
          text: t("btn_copy_voices_forward"),
          onclick: () => {
            if (audioOnChips) return;
            if (idx >= state.plan.shots.length - 1) return showToast(t("toast_already_last"));
            const resolved = resolveEffectiveAudioSlots(s, state.plan.global_ref_audio_slots);
            if (!resolved.some(Boolean)) return showToast(t("toast_no_voices_to_copy"), true);
            for (let j = idx + 1; j < state.plan.shots.length; j++) {
              state.plan.shots[j].ref_audio_slots = [...resolved];
              state.plan.shots[j].ref_audios = resolved.filter(Boolean);
            }
            scheduleSync(true);
            renderFilmstrip();
            showToast(t("toast_voices_copied_forward", { i: idx + 1, n: state.plan.shots.length - idx - 1 }));
          },
        }),
        el("button", {
          class: "sv-btn pri xs",
          disabled: audioOnChips,
          style: audioOnChips ? { opacity: "0.5", cursor: "not-allowed" } : {},
          text: t("btn_apply_voices_all"),
          onclick: () => {
            if (audioOnChips) return;
            const resolved = resolveEffectiveAudioSlots(s, state.plan.global_ref_audio_slots);
            if (!resolved.some(Boolean)) return showToast(t("toast_no_voices_to_copy"), true);
            state.plan.global_ref_audio_slots = [...resolved];
            state.plan.global_ref_audios = resolved.filter(Boolean);
            state.plan.shots.forEach((sh) => {
              sh.inherit_global_audios = true;
              sh.ref_audio_slots = new Array(3).fill("");
              sh.ref_audios = [];
            });
            scheduleSync(true);
            renderFilmstrip();
            renderShotEditor();
            showToast(t("toast_voices_applied_all", { i: idx + 1, n: state.plan.shots.length }));
          },
        }),
        ownAudAny ? el("button", {
          class: "sv-btn dng xs",
          text: t("btn_clear_voice"),
          onclick: () => {
            s.ref_audio_slots = new Array(3).fill("");
            s.ref_audios = [];
            scheduleSync();
            renderFilmstrip();
            renderShotEditor();
          },
        }) : null,
      ]),
    ]);
    const voiceGrid = render3AudioSlotGrid(
      s.ref_audio_slots,
      state.plan.global_ref_audio_slots,
      s.inherit_global_audios !== false,
      audioOnChips,
      (k, val) => {
        s.ref_audio_slots[k] = val;
        s.ref_audios = s.ref_audio_slots.filter(Boolean);
        scheduleSync();
        renderFilmstrip();
        renderShotEditor();
      },
      `shotvault/shot_${idx + 1}`,
      () => renderShotEditor(),
    );
    const voiceBox = el("div", { class: "sv-aud-box" }, [
      voiceHdr,
      audioOnChips ? el("span", { class: "sv-link-lbl", text: t("lbl_audio_drive_mute") }) : null,
      voiceGrid,
      (!audioOnChips && vidWithAudioCnt > 0)
        ? el("div", { class: "sv-aud-warn", text: t("warn_video_audio_shift", { n: vidWithAudioCnt }) })
        : null,
    ]);

    const leftPane = el("div", { class: "sv-shot-pane" }, [promptHdr, promptWrap, chips, mmRow, mmChipBar]);
    const rightPane = el("div", { class: "sv-shot-pane" }, [refBar, grid9, voiceBox]);
    appendSafe(editorWrap, topCtrl, el("div", { class: "sv-shot-split" }, [leftPane, rightPane]));
  }

  function renderTab1() {
    const masterBox = el("div", { class: "sv-box" });
    let mTimer = null;
    const { wrap: masterWrap, ta: masterTa } = makeDraggableTextarea({
      value: state.plan.master_script_text,
      prefKey: "master_script",
      defaultHeight: 150,
      minHeight: 80,
      maxHeight: 460,
      placeholder: t("ph_master_script"),
      onInput: (e) => {
        state.plan.master_script_text = e.target.value;
        setW("script_text", e.target.value);
        clearTimeout(mTimer);
        if (state.plan.auto_sync_master) {
          mTimer = setTimeout(() => {
            if (applyMasterSplit(true)) { renderFilmstrip(); renderShotEditor(); scheduleSync(); }
          }, 320);
        } else { scheduleSync(); }
      },
    });
    const defDurInp = el("input", {
      class: "sv-inp", type: "number", step: "0.5", min: "0.5", max: "20",
      value: String(state.plan.default_duration_sec || 5.0), style: { width: "54px" },
      title: t("title_default_dur"),
      oninput: (e) => {
        state.plan.default_duration_sec = Math.max(0.25, parseFloat(e.target.value) || 5.0);
        if (state.plan.auto_sync_master && applyMasterSplit(true)) {
          renderFilmstrip(); renderShotEditor();
        }
        scheduleSync();
      },
    });
    const commonInp = el("input", {
      class: "sv-inp", value: String(getW("common_prompt", "")), style: { flex: "1" },
      placeholder: t("ph_common_prompt"),
      oninput: (e) => { setW("common_prompt", e.target.value); scheduleSync(); },
    });
    const masterLen = (state.plan.master_script_text || "").length;
    appendSafe(
      masterBox,
      el("div", { class: "sv-box-hdr" }, [
        el("div", { class: "sv-row" }, [
          el("button", {
            class: "sv-btn xs",
            text: state.masterExpanded ? t("btn_collapse") : t("btn_expand"),
            onclick: () => { state.masterExpanded = !state.masterExpanded; renderBody(); },
          }),
          el("span", { text: t("lbl_master_hdr", { n: state.plan.shots.length, chars: masterLen }) }),
        ]),
        el("div", { class: "sv-row" }, [
          el("span", { style: { fontWeight: "normal", fontSize: "11px" }, text: t("lbl_default_dur") }),
          defDurInp,
          el("label", { class: "sv-row", style: { gap: "4px", fontWeight: "normal", cursor: "pointer" } }, [
            el("input", { type: "checkbox", checked: state.plan.auto_sync_master, onchange: (e) => { state.plan.auto_sync_master = e.target.checked; scheduleSync(); } }),
            el("span", { text: t("lbl_live_split") }),
          ]),
          el("button", {
            class: "sv-btn act",
            text: t("btn_focus_edit"),
            onclick: () => openFocusPromptModal({
              title: t("focus_title_master"),
              value: state.plan.master_script_text,
              placeholder: masterTa.placeholder,
              chips: ["===\n& 8 &\n@continue@\n", "===\n& 8 &\n@cut@\n", `${t("tag_type_drama")}\n`, "<Picture 1>", "<Picture 2>", "<Picture 3>"],
              el,
              t,
              onSave: (val) => {
                state.plan.master_script_text = val;
                setW("script_text", val);
                if (state.plan.auto_sync_master) applyMasterSplit(true);
                scheduleSync(true);
                renderBody();
              },
            }),
          }),
          el("button", {
            class: "sv-btn pri", text: t("btn_split_now"),
            onclick: () => {
              state.plan.master_script_text = masterTa.value;
              setW("script_text", masterTa.value);
              if (applyMasterSplit(true)) { scheduleSync(true); renderFilmstrip(); renderShotEditor(); showToast(t("toast_split_ok", { n: state.plan.shots.length })); }
            },
          }),
          el("button", {
            class: "sv-btn", text: t("btn_merge_back"),
            title: t("title_merge_back"),
            onclick: () => {
              masterTa.value = composeMasterFromShots();
              state.masterExpanded = true;
              scheduleSync(true);
              renderBody();
              showToast(t("toast_merged_back"));
            },
          }),
          el("button", {
            class: "sv-btn", text: t("btn_backend_parse"),
            onclick: async () => {
              try {
                const r = await api.fetchApi("/shotvault/parse-preview", {
                  method: "POST", headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({
                    script_text: masterTa.value, common_prompt: String(getW("common_prompt", "")),
                    stage3_json_text: String(getW("stage3_json_text", "")), script_format: String(getW("script_format", "auto")),
                    fps: Number(getW("fps", 24) || 24), seed_video_path: state.plan.seed_video_path,
                    default_duration_sec: Number(state.plan.default_duration_sec || 5.0),
                  }),
                });
                const d = await r.json();
                if (d?.ok && Array.isArray(d.shots) && d.shots.length) {
                  state.plan.shots = d.shots.map((x, i) => {
                    const oldS = state.plan.shots[i] || {};
                    const xAud = normSlots3(x.ref_audio_slots, x.ref_audios);
                    const aSlots = xAud.some(Boolean) ? xAud : normSlots3(oldS.ref_audio_slots, oldS.ref_audios);
                    const inhAud = ("inherit_global_audios" in oldS)
                      ? (oldS.inherit_global_audios !== false)
                      : (x.inherit_global_audios !== false);
                    return {
                      ...oldS, ...x,
                      inherit_global_audios: inhAud,
                      ref_slots: normSlots9(x.ref_slots, oldS.ref_slots || x.ref_images),
                      ref_audio_slots: aSlots,
                      ref_audios: aSlots.filter(Boolean),
                    };
                  });
                  scheduleSync(true); renderFilmstrip(); renderShotEditor(); showToast(t("toast_backend_parsed", { n: d.shots.length }));
                }
              } catch (err) { showToast(t("toast_parse_failed", { msg: err.message })); }
            },
          }),
        ]),
      ]),
      state.masterExpanded ? masterWrap : null,
      state.masterExpanded ? el("div", { class: "sv-row" }, [el("span", { text: t("lbl_common_prompt") }), commonInp]) : null,
    );

    const stripBox = el("div", { class: "sv-box" });
    stripBox.append(
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("lbl_filmstrip_hdr") }),
        el("div", { class: "sv-row" }, [
          el("button", {
            class: "sv-btn pri",
            text: t("btn_all_continue"),
            title: t("title_all_continue"),
            onclick: () => {
              state.plan.shots.forEach((sh, k) => {
                sh.hard_cut = k === 0 ? !Boolean(state.plan.seed_video_path) : false;
              });
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); renderHeader();
              showToast(t("toast_all_continue", { n: state.plan.shots.length }));
            },
          }),
          el("select", {
            class: "sv-sel",
            style: { width: "86px" },
            title: t("title_periodic_n"),
            onchange: (e) => { state.periodicCutN = Math.max(2, Number(e.target.value) || 4); },
          }, [3, 4, 5].map((n) => el("option", { value: String(n), selected: Number(state.periodicCutN) === n, text: t("opt_every_n", { n }) }))),
          el("button", {
            class: "sv-btn act",
            text: t("btn_periodic_cut"),
            title: t("title_periodic_cut"),
            onclick: () => {
              const step = Math.max(2, Number(state.periodicCutN) || 4);
              const cutNums = [];
              state.plan.shots.forEach((sh, k) => {
                if (k === 0) {
                  sh.hard_cut = !Boolean(state.plan.seed_video_path);
                } else if (k % step === 0) {
                  sh.hard_cut = true;
                  cutNums.push(`#${k + 1}`);
                } else {
                  sh.hard_cut = false;
                }
              });
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); renderHeader();
              showToast(cutNums.length ? t("toast_periodic_inserted", { list: cutNums.join(", ") }) : t("toast_periodic_none", { n: step }));
            },
          }),
          el("button", {
            class: "sv-btn ok", text: t("btn_add_shot"),
            onclick: () => {
              state.plan.shots.push({
                shot_index: state.plan.shots.length + 1, duration_sec: Number(state.plan.default_duration_sec || 5.0), shot_type: 1, hard_cut: false,
                overlap_override: 0, inherit_global_refs: true, inherit_global_audios: true, seed_text: "", prompt: "",
                ref_slots: new Array(9).fill(""), ref_images: [], ref_videos: [],
                ref_audio_slots: new Array(3).fill(""), ref_audios: [], guide_audio: "",
              });
              state.sel = state.plan.shots.length - 1;
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); patchFooter();
            },
          }),
          el("button", {
            class: "sv-btn act", text: t("btn_dup_shot"),
            onclick: () => {
              const cur = state.plan.shots[state.sel];
              if (!cur) return;
              const copy = JSON.parse(JSON.stringify(cur));
              copy.hard_cut = false;
              state.plan.shots.splice(state.sel + 1, 0, copy);
              state.sel = state.sel + 1;
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); patchFooter();
            },
          }),
          el("button", {
            class: "sv-btn", text: t("btn_move_prev"),
            onclick: () => {
              if (state.sel <= 0) return;
              const tmp = state.plan.shots[state.sel - 1];
              state.plan.shots[state.sel - 1] = state.plan.shots[state.sel];
              state.plan.shots[state.sel] = tmp;
              state.sel -= 1;
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); patchFooter();
            },
          }),
          el("button", {
            class: "sv-btn", text: t("btn_move_next"),
            onclick: () => {
              if (state.sel >= state.plan.shots.length - 1) return;
              const tmp = state.plan.shots[state.sel + 1];
              state.plan.shots[state.sel + 1] = state.plan.shots[state.sel];
              state.plan.shots[state.sel] = tmp;
              state.sel += 1;
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); patchFooter();
            },
          }),
          el("button", {
            class: "sv-btn dng", text: t("btn_del_shot"),
            onclick: () => {
              if (state.plan.shots.length <= 1) return showToast(t("toast_keep_one"));
              state.plan.shots.splice(state.sel, 1);
              state.sel = Math.max(0, Math.min(state.sel, state.plan.shots.length - 1));
              scheduleSync(true); renderFilmstrip(); renderShotEditor(); patchFooter();
            },
          }),
        ]),
      ]),
      (stripWrap = el("div", { class: "sv-strip" })),
    );
    editorWrap = el("div", { class: "sv-box" });
    bodyEl.append(masterBox, stripBox, editorWrap);
    renderFilmstrip();
    renderShotEditor();
  }

  function renderCleanupModal(proj) {
    if (!state.cleanerOpen) return null;
    const inv = state.inventory || { shots: [], unused_count: 0, unused_bytes: 0, repair_cache: { count: 0, bytes: 0, files: [] }, recycle_bin: { batches: [], total_bytes: 0, count: 0 }, folders: {} };
    const allDeletable = [];
    const bytesByTake = {};
    for (const sh of inv.shots || []) {
      for (const tk of sh.takes || []) {
        bytesByTake[tk.take_id] = Number(tk.bytes || 0);
        if (tk.deletable) allDeletable.push(tk.take_id);
      }
    }
    const selIds = [...state.selectedTakes].filter((id) => allDeletable.includes(id));
    const selBytes = selIds.reduce((a, id) => a + (bytesByTake[id] || 0), 0);

    const doCleanup = async (payload) => {
      try {
        const r = await api.fetchApi("/shotvault/cleanup-items", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: proj, ...payload }),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          if (j.state) state.serverState = j.state;
          if (j.inventory) state.inventory = j.inventory;
          for (const d of j.deleted_takes || []) state.selectedTakes.delete(d);
          showToast(t("toast_cleaned", { n: (j.deleted_takes || []).length || j.file_count || 0, sz: fmtBytes(j.freed_bytes || 0) }));
          renderBody();
        } else {
          showToast(j.error || "Cleanup failed", true);
        }
      } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
    };

    const doRecycle = async (action, batch_ids = null) => {
      try {
        const r = await api.fetchApi("/shotvault/recycle-bin", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: proj, action, batch_ids }),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          if (j.state) state.serverState = j.state;
          if (j.inventory) state.inventory = j.inventory;
          if (action === "restore") showToast(t("toast_restored", { n: (j.restored_takes || []).length, f: j.restored_files || 0 }));
          else showToast(t("toast_emptied", { sz: fmtBytes(j.freed_bytes || 0) }));
          renderBody();
        }
      } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
    };

    const tabBar = el("div", { class: "sv-row", style: { padding: "6px 12px", background: "#161820", borderBottom: "1px solid #2b3042" } }, [
      el("button", { class: `sv-btn ${state.cleanerTab === "takes" ? "pri" : ""}`, text: t("cl_tab_takes", { n: inv.unused_count || 0 }), onclick: () => { state.cleanerTab = "takes"; renderBody(); } }),
      el("button", { class: `sv-btn ${state.cleanerTab === "cache" ? "pri" : ""}`, text: t("cl_tab_cache", { n: inv.repair_cache?.count || 0 }), onclick: () => { state.cleanerTab = "cache"; renderBody(); } }),
      el("button", { class: `sv-btn ${state.cleanerTab === "recycle" ? "pri" : ""}`, text: t("cl_tab_recycle", { n: inv.recycle_bin?.count || 0, sz: fmtBytes(inv.recycle_bin?.total_bytes || 0) }), onclick: () => { state.cleanerTab = "recycle"; renderBody(); } }),
    ]);

    const mBody = el("div", { class: "sv-modal-body" });
    if (state.cleanerTab === "takes") {
      mBody.appendChild(el("div", { class: "sv-row" }, [
        el("button", { class: "sv-btn act", text: t("cl_sel_all_unused"), onclick: () => { allDeletable.forEach((id) => state.selectedTakes.add(id)); renderBody(); } }),
        el("button", { class: "sv-btn", text: t("cl_sel_none"), onclick: () => { state.selectedTakes.clear(); renderBody(); } }),
        el("span", { style: { color: "#9da5ba", fontSize: "11px" }, text: `shots: ${fmtBytes(inv.folders?.shots)} · vault: ${fmtBytes(inv.folders?.vault)} · master: ${fmtBytes(inv.folders?.master)}` }),
      ]));
      let shownAny = false;
      for (const sh of inv.shots || []) {
        if (!(sh.takes || []).length) continue;
        shownAny = true;
        const rows = (sh.takes || []).map((tk) => {
          const checked = state.selectedTakes.has(tk.take_id);
          const protLabels = (tk.protected_reasons || []).map((r) => t(r === "active" ? "prot_active" : (r === "locked" ? "prot_locked" : (r === "parent_ref" ? "prot_parent" : "prot_seed"))));
          return el("div", { class: `sv-chk-row ${tk.is_active ? "active" : ""}` }, [
            el("label", { class: "sv-row", style: { cursor: tk.deletable ? "pointer" : "not-allowed", opacity: tk.deletable ? "1" : "0.65" } }, [
              el("input", {
                type: "checkbox",
                checked,
                disabled: !tk.deletable,
                onchange: (e) => {
                  if (e.target.checked) state.selectedTakes.add(tk.take_id);
                  else state.selectedTakes.delete(tk.take_id);
                  renderBody();
                },
              }),
              el("span", { text: `${tk.take_id} · Seed:${tk.seed_text || "-"} · ${tk.width}×${tk.height} · ${fmtBytes(tk.bytes)}` }),
              ...protLabels.map((pl) => el("span", { class: "sv-pill-prot", text: `🔒 ${pl}` })),
            ]),
            tk.deletable
              ? el("button", { class: "sv-btn dng", style: { padding: "2px 6px" }, text: "🗑", title: "Delete this take", onclick: () => doCleanup({ scope: "selected", take_ids: [tk.take_id], mode: "recycle" }) })
              : null,
          ]);
        });
        mBody.appendChild(el("div", { class: "sv-box", style: { padding: "6px 8px" } }, [
          el("div", { class: "sv-box-hdr" }, [el("span", { text: `Shot #${sh.shot_index}` })]),
          ...rows,
        ]));
      }
      if (!shownAny) mBody.appendChild(el("span", { style: { color: "#8890a6" }, text: t("cl_empty_hint") }));
    } else if (state.cleanerTab === "cache") {
      const files = inv.repair_cache?.files || [];
      if (!files.length) {
        mBody.appendChild(el("span", { style: { color: "#8890a6" }, text: t("cl_empty_hint") }));
      } else {
        mBody.appendChild(el("button", {
          class: "sv-btn warn",
          text: t("cl_clear_cache", { sz: fmtBytes(inv.repair_cache?.bytes || 0) }),
          onclick: () => doCleanup({ scope: "repair_cache_only", mode: "recycle" }),
        }));
        for (const f of files) {
          mBody.appendChild(el("div", { class: "sv-chk-row" }, [
            el("span", { text: f.name }),
            el("span", { style: { color: "#9da5ba" }, text: fmtBytes(f.bytes) }),
          ]));
        }
      }
    } else {
      const batches = inv.recycle_bin?.batches || [];
      if (!batches.length) {
        mBody.appendChild(el("span", { style: { color: "#8890a6" }, text: t("cl_empty_hint") }));
      } else {
        mBody.appendChild(el("button", {
          class: "sv-btn dng",
          text: t("cl_empty_bin", { sz: fmtBytes(inv.recycle_bin?.total_bytes || 0) }),
          onclick: () => {
            if (!window.confirm(t("cl_confirm_empty", { sz: fmtBytes(inv.recycle_bin?.total_bytes || 0) }))) return;
            doRecycle("empty");
          },
        }));
        for (const b of batches) {
          mBody.appendChild(el("div", { class: "sv-chk-row" }, [
            el("span", { text: `${b.batch_id} · ${(b.take_ids || []).length ? b.take_ids.join(", ") : b.scope} · ${b.file_count} files · ${fmtBytes(b.bytes)}` }),
            el("div", { class: "sv-row" }, [
              b.restorable ? el("button", { class: "sv-btn ok", text: t("cl_restore_batch"), onclick: () => doRecycle("restore", [b.batch_id]) }) : null,
              el("button", { class: "sv-btn dng", text: "🗑", onclick: () => doRecycle("empty", [b.batch_id]) }),
            ]),
          ]));
        }
      }
    }

    const mFtr = el("div", { class: "sv-modal-ftr" }, [
      state.cleanerTab === "takes"
        ? el("div", { class: "sv-row" }, [
            el("button", {
              class: "sv-btn warn",
              disabled: !selIds.length,
              text: t("cl_recycle_sel", { n: selIds.length, sz: fmtBytes(selBytes) }),
              onclick: () => {
                if (!selIds.length) return showToast(t("toast_no_selection"), true);
                doCleanup({ scope: "selected", take_ids: selIds, mode: "recycle" });
              },
            }),
            el("button", {
              class: "sv-btn dng",
              disabled: !selIds.length,
              text: t("cl_perm_sel"),
              onclick: () => {
                if (!selIds.length) return showToast(t("toast_no_selection"), true);
                if (!window.confirm(t("cl_confirm_perm", { n: selIds.length, sz: fmtBytes(selBytes) }))) return;
                doCleanup({ scope: "selected", take_ids: selIds, mode: "permanent" });
              },
            }),
          ])
        : el("span", {}),
      el("button", { class: "sv-btn", text: t("cl_close"), onclick: () => { state.cleanerOpen = false; renderBody(); } }),
    ]);

    return el("div", { class: "sv-modal-bg", onclick: (e) => { if (e.target === e.currentTarget) { state.cleanerOpen = false; renderBody(); } } }, [
      el("div", { class: "sv-modal" }, [
        el("div", { class: "sv-modal-hdr" }, [
          el("span", { text: t("cl_title") }),
          el("button", { class: "sv-btn", text: "×", onclick: () => { state.cleanerOpen = false; renderBody(); } }),
        ]),
        tabBar,
        mBody,
        mFtr,
      ]),
    ]);
  }

  function renderTab2() {
    const svShots = state.serverState?.shots || [];
    const proj = String(getW("project_name", "MyH3Film") || "MyH3Film");
    const rl = state.plan.retry_lock;
    const pills = el("div", { class: "sv-row" }, state.plan.shots.map((_, i) => {
      const shSv = svShots.find((x) => Number(x.shot_index) === i + 1) || {};
      const srOn = Boolean(shSv.seam_repair?.enabled && (shSv.seam_repair?.choice || "repaired") === "repaired");
      return el("button", {
        class: `sv-btn ${state.sel === i ? "pri" : (srOn ? "ok" : "")}`,
        text: `${srOn ? "✨" : ""}#${i + 1}`,
        onclick: () => {
          state.sel = i;
          renderBody();
        },
      });
    }));

    const idx = Math.max(0, Math.min(state.sel, state.plan.shots.length - 1));
    const shotNum = idx + 1;
    const sv = svShots.find((x) => Number(x.shot_index) === shotNum) || {};
    const invShot = (state.inventory?.shots || []).find((x) => Number(x.shot_index) === shotNum) || { takes: [] };
    const invTakeMap = Object.fromEntries((invShot.takes || []).map((tk) => [tk.take_id, tk]));
    const isLocked = Boolean(sv.locked_keep);
    const takes = sv.takes || [];
    const actTake = takes.find((tk) => tk.take_id === sv.active_take_id) || takes[0];
    const diag = actTake?.seam_diagnostics || {};
    const trimRep = diag?.seam_trim_v53 || null;
    const cRep = trimRep?.color || trimRep || null;
    const rRep = trimRep?.rife || null;
    const seamRep = sv.seam_repair || {};

    // Per-shot repair draft settings (default retry_policy = "off" as requested)
    if (!state.repairDrafts[shotNum]) {
      state.repairDrafts[shotNum] = {
        mode: String(seamRep.mode || "rife"),
        rife_bridge_frames: Number(seamRep.rife_bridge_frames || 5),
        preserve_duration: Boolean(seamRep.preserve_duration || false),
        structure_threshold: Number(seamRep.structure_threshold ?? 0.38),
        coverage_threshold: Number(seamRep.coverage_threshold ?? 0.08),
        retry_policy: String(seamRep.retry_policy || "off"),
      };
    }
    const draft = state.repairDrafts[shotNum];

    // Honest seam-window diagnostics (distinguish real seam jump vs interior VFX flash)
    const lang = getLang();
    const seamEvents = Array.isArray(diag.events) ? diag.events : [];
    const interiorEvents = Array.isArray(diag.interior_events) ? diag.interior_events : [];
    const realSeamEvents = seamEvents.filter((ev) => !diag.join_frame || (Number(ev.frame || 0) - Number(diag.join_frame || 0)) < 6);
    const lateEvents = interiorEvents.length ? interiorEvents : seamEvents.filter((ev) => diag.join_frame && (Number(ev.frame || 0) - Number(diag.join_frame || 0)) >= 6);
    const hasWarn = realSeamEvents.length > 0;

    const seamRepMatchesTake = Boolean(actTake && seamRep.take_id && seamRep.take_id === actTake.take_id);
    let diagText = t("no_take_yet");
    if (actTake) {
      if (seamRepMatchesTake && seamRep.report) {
        const rr = seamRep.report;
        const sumTxt = lang === "en" ? (rr.summary_en || rr.summary_zh || "") : (rr.summary_zh || rr.summary_en || "");
        const rawPk = rr.raw_seam_qc?.peak_jump_ratio;
        const newPk = rr.seam_qc?.peak_jump_ratio;
        const deltaStr = (rawPk != null && newPk != null) ? ` · Peak ${rawPk}→${newPk}` : "";
        diagText = `${sumTxt}${deltaStr}`;
      } else if (diag.join_frame) {
        if (hasWarn) {
          const evSummary = realSeamEvents.slice(0, 2).map((ev) => `${ev.kind}@f${ev.frame}(Δ=${Number(ev.difference || 0).toFixed(3)})`).join(" · ");
          diagText = `@#${diag.join_frame} · ${evSummary} ${t("seam_warn")}`;
        } else if (lateEvents.length) {
          const sec = ((Number(lateEvents[0].frame || 0) - Number(diag.join_frame || 0)) / Math.max(1, Number(diag.fps || 24))).toFixed(2);
          diagText = t("seam_interior_only", { sec });
        } else {
          diagText = `@#${diag.join_frame} · ${t("seam_smooth")}`;
        }
      } else {
        diagText = `Take ${actTake.take_id} · ${actTake.width || "?"}×${actTake.height || "?"}`;
      }
    }

    const deBefore = Number(cRep?.delta_e_eff ?? trimRep?.de_before ?? 0);
    const deAfter = Number(cRep?.delta_e_after ?? trimRep?.de_after ?? deBefore);
    const fpRep = seamRepMatchesTake ? (seamRep.report || null) : null;
    const telemNode = (trimRep || fpRep) ? el("div", { class: "sv-telem" }, [
      el("span", { text: `🎨 Lab: ΔE ${deBefore.toFixed(2)} → ${deAfter.toFixed(2)}` }),
      el("span", { text: fpRep ? `📐 eI: err=${fpRep.structureError ?? "-"} / cov=${fpRep.flowCoverage ?? "-"}` : `🌊 RIFE: ${rRep?.applied_rife ? "ON" : "OFF"}` }),
      el("span", { text: fpRep ? `⚡ ${fpRep.backend || fpRep.applied || "-"} (Δf=${fpRep.frame_delta ?? 0})` : `🎯 ${cRep?.mode || "-"}` }),
    ]) : null;

    const repairedRel = seamRepMatchesTake ? (seamRep.repaired_seam_rel || "") : "";
    const hasRepaired = Boolean(repairedRel);
    const activeSrcRel = state.playerMode === "repaired"
      ? repairedRel
      : (state.playerMode === "seam" ? actTake?.seam_rel : actTake?.video_rel);

    const switchPlayerMode = (nextMode) => {
      const curVid = bodyEl.querySelector("video.sv-seam-player");
      if (curVid && Number.isFinite(curVid.currentTime)) state.lastVideoTime = curVid.currentTime;
      state.playerMode = nextMode;
      renderBody();
    };

    const resolvedSrc = activeSrcRel ? mediaUrl(activeSrcRel, proj) : "";
    const videoEl = resolvedSrc
      ? el("video", {
          class: "sv-seam-player",
          controls: true,
          loop: true,
          src: state.mediaBust ? `${resolvedSrc}${resolvedSrc.includes("?") ? "&" : "?"}cb=${state.mediaBust}` : resolvedSrc,
          style: { width: "100%", maxHeight: "235px", borderRadius: "6px", background: "#000" },
          onloadedmetadata: (e) => {
            if (state.lastVideoTime > 0 && state.playerMode !== "full" && e.target.duration > state.lastVideoTime) {
              try { e.target.currentTime = state.lastVideoTime; } catch {}
            }
          },
        })
      : el("div", {
          class: "sv-thumb",
          style: { height: "185px", padding: "12px", textAlign: "center" },
          text: state.playerMode === "repaired" ? t("no_repaired_yet") : t("waiting_media"),
        });

    const saveSeamChoice = async (patch) => {
      try {
        const r = await api.fetchApi("/shotvault/set-seam-choice", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: proj, shot_index: shotNum, ...patch }),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          if (j.state) state.serverState = j.state;
          showToast(t("toast_choice_saved", { i: shotNum }));
          renderBody();
        }
      } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
    };

    const runSingleSeamRepair = async () => {
      if (state.repairBusy) return;
      state.repairBusy = true;
      renderBody();
      try {
        const r = await api.fetchApi("/shotvault/repair-seam", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            project: proj,
            shot_index: shotNum,
            take_id: actTake?.take_id,
            ...draft,
          }),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          if (j.state) state.serverState = j.state;
          state.mediaBust = Date.now();
          state.playerMode = "repaired";
          const rep = j.seam_repair?.report || {};
          const msg = getLang() === "en" ? (rep.summary_en || rep.applied) : (rep.summary_zh || rep.applied);
          showToast(t("toast_repaired_ok", { msg }));
          await refreshInventory(true);
        } else {
          showToast(t("toast_repair_fail", { msg: j.error || r.statusText }), true);
        }
      } catch (err) {
        showToast(t("toast_repair_fail", { msg: err?.message || err }), true);
      } finally {
        state.repairBusy = false;
        renderBody();
      }
    };

    const repairEnabled = Boolean(seamRep.enabled);
    const repairChoice = String(seamRep.choice || (hasRepaired ? "repaired" : "original"));

    const repairBar = el("div", { class: "sv-repair-bar" }, [
      el("div", {
        style: {
          fontSize: "10.5px",
          lineHeight: "1.4",
          color: "#b7c0d6",
          background: "rgba(180, 140, 48, 0.10)",
          border: "1px solid rgba(198, 150, 56, 0.28)",
          borderRadius: "5px",
          padding: "4px 7px",
        },
        text: t("repair_exp_note"),
      }),
      el("div", { class: "sv-row", style: { justifyContent: "space-between" } }, [
        el("button", {
          class: `sv-btn ${repairEnabled ? "ok" : ""}`,
          text: repairEnabled ? t("repair_on") : t("repair_off"),
          onclick: () => saveSeamChoice({ enabled: !repairEnabled, choice: !repairEnabled ? (hasRepaired ? "repaired" : "repaired") : "original", ...draft }),
        }),
        el("div", { class: "sv-row" }, [
          hasRepaired
            ? el("button", {
                class: `sv-btn ${repairEnabled && repairChoice === "repaired" ? "ok" : ""}`,
                text: t("use_repaired"),
                onclick: () => saveSeamChoice({ enabled: true, choice: "repaired" }),
              })
            : null,
          hasRepaired
            ? el("button", {
                class: `sv-btn ${!repairEnabled || repairChoice === "original" ? "warn" : ""}`,
                text: t("use_original"),
                onclick: () => saveSeamChoice({ enabled: false, choice: "original" }),
              })
            : null,
        ]),
      ]),
      el("div", { class: "sv-row" }, [
        el("span", { style: { fontSize: "11px", color: "#9da5ba" }, text: t("lbl_repair_mode") }),
        el("select", {
          class: "sv-sel",
          onchange: (e) => { draft.mode = e.target.value; renderBody(); },
        }, [
          ["rife", t("opt_mode_rife")],
          ["auto_align", t("opt_mode_align")],
          ["cut", t("opt_mode_cut")],
        ].map(([v, lbl]) => el("option", { value: v, selected: draft.mode === v, text: lbl }))),
        draft.mode === "rife"
          ? el("span", { style: { fontSize: "11px", color: "#9da5ba" }, text: t("lbl_bridge_frames") })
          : null,
        draft.mode === "rife"
          ? el("select", {
              class: "sv-sel",
              onchange: (e) => { draft.rife_bridge_frames = Number(e.target.value); },
            }, [3, 5, 7].map((n) => el("option", { value: String(n), selected: Number(draft.rife_bridge_frames) === n, text: `${n}f` })))
          : null,
        el("span", { style: { fontSize: "11px", color: "#9da5ba" }, text: t("lbl_retry") }),
        el("select", {
          class: "sv-sel",
          onchange: (e) => { draft.retry_policy = e.target.value; },
        }, [
          ["off", t("opt_retry_off")],
          ["once_ladder", t("opt_retry_ladder")],
        ].map(([v, lbl]) => el("option", { value: v, selected: draft.retry_policy === v, text: lbl }))),
        el("label", { class: "sv-row", style: { gap: "3px", fontSize: "11px", cursor: "pointer" } }, [
          el("input", { type: "checkbox", checked: Boolean(draft.preserve_duration), onchange: (e) => { draft.preserve_duration = e.target.checked; } }),
          t("lbl_preserve_dur"),
        ]),
      ]),
      el("div", { class: "sv-row", style: { justifyContent: "space-between" } }, [
        el("details", { class: "sv-adv" }, [
          el("summary", { style: { fontSize: "11px" }, text: t("lbl_adv_thresh") }),
          el("div", { class: "sv-row", style: { marginTop: "4px" } }, [
            el("span", { style: { fontSize: "11px" }, text: "struct<" }),
            el("input", { class: "sv-inp", type: "number", step: "0.02", min: "0.10", max: "0.80", value: String(draft.structure_threshold), style: { width: "62px" }, oninput: (e) => { draft.structure_threshold = parseFloat(e.target.value) || 0.38; } }),
            el("span", { style: { fontSize: "11px" }, text: "cov>" }),
            el("input", { class: "sv-inp", type: "number", step: "0.01", min: "0.01", max: "0.50", value: String(draft.coverage_threshold), style: { width: "62px" }, oninput: (e) => { draft.coverage_threshold = parseFloat(e.target.value) || 0.08; } }),
          ]),
        ]),
        el("button", {
          class: "sv-btn pri",
          disabled: state.repairBusy || !actTake || (shotNum <= 1 && !state.plan.seed_video_path && !state.serverState?.seed_shot?.take_id),
          text: state.repairBusy ? t("btn_repairing") : t("btn_repair_now"),
          onclick: runSingleSeamRepair,
        }),
      ]),
    ]);

    const left = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t2_left_hdr", { i: shotNum }) }),
        el("div", { class: "sv-row" }, [
          el("button", { class: `sv-btn ${state.playerMode === "seam" ? "pri" : ""}`, text: t("btn_orig_3s"), onclick: () => switchPlayerMode("seam") }),
          el("button", { class: `sv-btn ${state.playerMode === "repaired" ? "ok" : ""}`, text: t("btn_repaired_3s"), onclick: () => switchPlayerMode("repaired") }),
          el("button", { class: `sv-btn ${state.playerMode === "full" ? "pri" : ""}`, text: t("btn_full_take"), onclick: () => switchPlayerMode("full") }),
        ]),
      ]),
      el("div", { class: `sv-badge ${hasWarn ? "sv-tag-a" : "sv-tag-l"}`, text: diagText }),
      telemNode,
      videoEl,
      repairBar,
    ]);

    // Right column: Take management + Single / Multi / Select-All Cleanup
    const shotDeletableIds = takes.map((tk) => tk.take_id).filter((id) => {
      const info = invTakeMap[id];
      return info ? info.deletable : (id !== sv.active_take_id && !isLocked);
    });
    const shotSelectedIds = shotDeletableIds.filter((id) => state.selectedTakes.has(id));
    const allShotSelected = shotDeletableIds.length > 0 && shotSelectedIds.length === shotDeletableIds.length;

    const cleanTakesQuick = async (ids) => {
      if (!ids.length) return showToast(t("toast_no_selection"), true);
      try {
        const r = await api.fetchApi("/shotvault/cleanup-items", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: proj, scope: "selected", take_ids: ids, mode: "recycle" }),
        });
        const j = await r.json();
        if (r.ok && j.ok) {
          if (j.state) state.serverState = j.state;
          if (j.inventory) state.inventory = j.inventory;
          for (const d of j.deleted_takes || []) state.selectedTakes.delete(d);
          showToast(t("toast_cleaned", { n: (j.deleted_takes || []).length, sz: fmtBytes(j.freed_bytes || 0) }));
          renderBody();
        }
      } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
    };

    const outdatedSet = new Set(state.serverState?.master_sync_status?.outdated_shots || []);
    const takeCards = takes.length ? takes.map((tk) => {
      const info = invTakeMap[tk.take_id];
      const isAct = tk.take_id === sv.active_take_id;
      const deletable = info ? Boolean(info.deletable) : (!isAct && !isLocked);
      const checked = deletable && state.selectedTakes.has(tk.take_id);
      const szText = info ? ` · ${fmtBytes(info.bytes)}` : "";
      const protReasons = info?.protected_reasons || (isAct ? ["active"] : (isLocked ? ["locked"] : []));
      const bridgeBadge = Number(tk.tail_bridge_frames || 0) > 0
        ? el("span", { class: "sv-badge", style: { background: "#143045", color: "#9ee8ff", borderColor: "#2d7c94" }, text: t("badge_bidir", { f: tk.tail_bridge_frames }) })
        : null;
      const pendingBadge = (isAct && outdatedSet.has(shotNum))
        ? el("span", { class: "sv-badge", style: { background: "#472b11", color: "#ffd7a1", borderColor: "#c67a22" }, text: t("badge_pending_merge") })
        : null;
      return el("div", { class: `sv-chk-row ${isAct ? "active" : ""}` }, [
        el("label", { class: "sv-row", style: { gap: "5px", cursor: deletable ? "pointer" : "default", flex: "1", minWidth: "0" } }, [
          el("input", {
            type: "checkbox",
            checked,
            disabled: !deletable,
            title: deletable ? "Select for cleanup" : `Protected: ${protReasons.join(", ")}`,
            onchange: (e) => {
              if (e.target.checked) state.selectedTakes.add(tk.take_id);
              else state.selectedTakes.delete(tk.take_id);
              renderBody();
            },
          }),
          el("span", { style: { overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }, text: `${tk.take_id} · Seed:${tk.seed_text || "-"} · ${tk.width || 0}×${tk.height || 0}${szText}` }),
          bridgeBadge,
          pendingBadge,
        ]),
        el("div", { class: "sv-row", style: { flexWrap: "nowrap" } }, [
          el("button", {
            class: `sv-btn ${isAct ? "ok" : ""}`,
            text: isAct ? t("btn_active_take") : t("btn_use_take"),
            onclick: async () => {
              try {
                const r = await api.fetchApi("/shotvault/select-take", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project: proj, shot_index: shotNum, take_id: tk.take_id }) });
                if (r.ok) { state.serverState = await r.json(); await refreshInventory(true); renderAll(); }
              } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
            },
          }),
          deletable
            ? el("button", {
                class: "sv-btn dng",
                style: { padding: "3px 6px" },
                text: "🗑",
                title: "Move this take to Recycle Bin",
                onclick: () => cleanTakesQuick([tk.take_id]),
              })
            : el("span", { class: "sv-pill-prot", text: "🔒" }),
        ]),
      ]);
    }) : [el("span", { style: { color: "#8890a6" }, text: t("no_takes") })];

    const { wrap: rlWrap, ta: rlTa } = makeDraggableTextarea({
      value: rl.prompt_override || "",
      prefKey: "reroll_prompt",
      defaultHeight: 82,
      minHeight: 58,
      maxHeight: 320,
      placeholder: t("rl_prompt_ph"),
      onInput: (e) => { rl.prompt_override = e.target.value; scheduleSync(); },
    });

    const nextSv = svShots.find((x) => Number(x.shot_index) === shotNum + 1) || null;
    const hasNextRendered = Boolean(shotNum < state.plan.shots.length && nextSv?.active_take_id);
    if (![0, 1, 5, 22].includes(Number(rl.tail_bridge_frames))) rl.tail_bridge_frames = 0;

    const takesCard = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t2_right_hdr") }),
        el("div", { class: "sv-row" }, [
          el("button", {
            class: `sv-btn ${isLocked ? "ok" : ""}`,
            title: t("title_protect_shot"),
            text: isLocked ? t("btn_locked_shot") : t("btn_lock_shot"),
            onclick: async () => {
              try {
                const r = await api.fetchApi("/shotvault/lock-shot", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ project: proj, shot_index: shotNum, locked: !isLocked }) });
                if (r.ok) { state.serverState = await r.json(); await refreshInventory(true); renderBody(); }
              } catch (err) { showToast(`Error: ${err?.message || err}`, true); }
            },
          }),
          el("button", {
            class: "sv-btn act",
            text: t("btn_open_cleaner"),
            onclick: async () => {
              state.cleanerOpen = true;
              await refreshInventory(true);
              renderBody();
            },
          }),
        ]),
      ]),
      el("div", { class: "sv-row" }, [
        el("button", {
          class: "sv-btn",
          disabled: !shotDeletableIds.length,
          text: allShotSelected ? t("btn_unselect_all_shot") : t("btn_select_all_shot"),
          onclick: () => {
            if (allShotSelected) shotDeletableIds.forEach((id) => state.selectedTakes.delete(id));
            else shotDeletableIds.forEach((id) => state.selectedTakes.add(id));
            renderBody();
          },
        }),
        el("button", {
          class: "sv-btn dng",
          disabled: !shotSelectedIds.length,
          text: t("btn_del_selected", { n: shotSelectedIds.length }),
          onclick: () => cleanTakesQuick(shotSelectedIds),
        }),
      ]),
      ...takeCards,
    ]);

    const isThisShotLocked = Boolean(rl.enabled && Number(rl.shot_index) === shotNum);
    const isOtherShotLocked = Boolean(rl.enabled && Number(rl.shot_index) !== shotNum);
    const otherLockBanner = isOtherShotLocked
      ? el("div", { class: "sv-outdated-banner", style: { display: "flex", alignItems: "center", justifyContent: "space-between", gap: "6px", flexWrap: "wrap" } }, [
          el("span", { text: t("rl_other_locked_warn", { locked: rl.shot_index, cur: shotNum }) }),
          el("div", { class: "sv-row" }, [
            el("button", {
              class: "sv-btn warn xs",
              text: t("rl_switch_lock_btn", { cur: shotNum }),
              onclick: () => { rl.enabled = true; rl.shot_index = shotNum; scheduleSync(true); renderAll(); },
            }),
            el("button", {
              class: "sv-btn xs",
              text: t("rl_unlock_btn"),
              onclick: () => { rl.enabled = false; scheduleSync(true); renderAll(); },
            }),
          ]),
        ])
      : null;

    const tailOpts = hasNextRendered
      ? [
          [0, t("rl_tail_off")],
          [1, t("rl_tail_1")],
          [5, t("rl_tail_5")],
          [22, t("rl_tail_22")],
        ]
      : [[0, t("rl_tail_disabled_opt")]];
    const displayedTailVal = hasNextRendered ? Number(rl.tail_bridge_frames || 0) : 0;

    const rerollCard = el("div", { class: `sv-box ${isThisShotLocked ? "reroll-on" : ""}` }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t2_reroll_hdr") }),
        el("button", {
          class: "sv-btn act xs",
          text: t("btn_focus_edit"),
          onclick: () => openFocusPromptModal({
            title: t("focus_title_reroll", { i: shotNum }),
            value: rl.prompt_override || "",
            placeholder: t("rl_prompt_ph"),
            chips: Array.from({ length: 9 }, (_, k) => `<Picture ${k + 1}>`),
            el,
            t,
            onSave: (val) => {
              rl.prompt_override = val;
              rlTa.value = val;
              scheduleSync(true);
              renderBody();
            },
          }),
        }),
      ]),
      otherLockBanner,
      el("label", { class: "sv-row", style: { gap: "6px", cursor: "pointer" } }, [
        el("input", {
          type: "checkbox", checked: isThisShotLocked,
          onchange: (e) => {
            if (e.target.checked) {
              rl.enabled = true;
              rl.shot_index = shotNum;
            } else {
              rl.enabled = false;
            }
            scheduleSync(true);
            renderAll();
          },
        }),
        el("strong", { text: t("rl_enable", { i: shotNum }) }),
      ]),
      el("div", { class: "sv-row" }, [
        el("select", { class: "sv-sel", onchange: (e) => { rl.seed_mode = e.target.value; scheduleSync(); } }, [
          ["randomize", t("rl_seed_rand")], ["fixed", t("rl_seed_fixed")], ["delta_+1", t("rl_seed_delta")],
        ].map(([v, lbl]) => el("option", { value: v, selected: rl.seed_mode === v, text: lbl }))),
        el("input", { class: "sv-inp", value: rl.seed_text || "", placeholder: t("rl_seed_ph"), style: { width: "104px" }, oninput: (e) => { rl.seed_text = e.target.value; scheduleSync(); } }),
      ]),
      el("div", { class: "sv-row" }, [
        el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: hasNextRendered ? t("rl_tail_lbl", { next: shotNum + 1 }) : t("rl_tail_unavail") }),
        el("select", {
          class: "sv-sel",
          disabled: !hasNextRendered,
          onchange: (e) => { rl.tail_bridge_frames = Number(e.target.value || 0); scheduleSync(true); renderBody(); },
        }, tailOpts.map(([v, lbl]) => el("option", { value: String(v), selected: displayedTailVal === v, text: lbl }))),
        (hasNextRendered && Number(rl.tail_bridge_frames) === 22)
          ? el("select", {
              class: "sv-sel",
              title: t("title_tail_noise"),
              onchange: (e) => { rl.tail_bridge_noise = e.target.value; scheduleSync(true); },
            }, [
              ["clean", t("rl_tail_clean")],
              ["reverse_taper", t("rl_tail_rev")],
            ].map(([v, lbl]) => el("option", { value: v, selected: (rl.tail_bridge_noise || "clean") === v, text: lbl })))
          : null,
      ]),
      rlWrap,
      el("label", { class: "sv-row", style: { gap: "5px", fontSize: "11.5px", cursor: "pointer" } }, [
        el("input", {
          type: "checkbox",
          checked: rl.auto_assemble_after_reroll !== false,
          onchange: (e) => { rl.auto_assemble_after_reroll = e.target.checked; scheduleSync(true); renderBody(); },
        }),
        el("span", { text: t("rl_auto_asm") }),
      ]),
      isThisShotLocked
        ? el("div", { class: "sv-guide-strip", text: (rl.auto_assemble_after_reroll !== false) ? t("rl_guide_auto", { i: shotNum }) : t("rl_guide_manual", { i: shotNum }) })
        : null,
    ]);

    const rightCol = el("div", { style: { display: "flex", flexDirection: "column", gap: "8px", minWidth: "0" } }, [takesCard, rerollCard]);

    bodyEl.append(pills, el("div", { class: "sv-grid2" }, [left, rightCol]));
    const modal = renderCleanupModal(proj);
    if (modal) bodyEl.appendChild(modal);
  }

  function renderTab3() {
    const p = state.plan;
    const left = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t3_left_hdr") }),
        p.seed_video_path ? el("button", { class: "sv-btn dng", text: t("btn_clear_shot0"), onclick: () => { p.seed_video_path = ""; scheduleSync(true); renderBody(); } }) : null,
      ]),
      el("div", { class: "sv-row" }, [
        el("button", { class: "sv-btn pri", text: t("btn_upload_shot0"), onclick: () => pickLocalFile("video/*", async (f) => { const r = await uploadMediaFile(f, "shotvault/seed"); p.seed_video_path = r.rel_path; scheduleSync(true); renderBody(); showToast(t("toast_shot0_imported")); }) }),
        el("span", { text: t("lbl_seed_trim") }),
        el("input", { class: "sv-inp", type: "number", step: "0.1", min: "0", value: String(p.seed_trim_end_sec || 0), style: { width: "70px" }, oninput: (e) => { p.seed_trim_end_sec = Math.max(0, parseFloat(e.target.value) || 0); scheduleSync(); } }),
      ]),
      p.seed_video_path
        ? el("video", { controls: true, src: mediaUrl(p.seed_video_path), style: { width: "100%", maxHeight: "230px", borderRadius: "6px", background: "#000" } })
        : el("div", { class: "sv-thumb", style: { height: "160px" }, text: t("ph_shot0_empty") }),
    ]);

    p.global_ref_audio_slots = normSlots3(p.global_ref_audio_slots, p.global_ref_audios);
    const gAudMuted = isAudioMode();
    const gAudHasAny = p.global_ref_audio_slots.some(Boolean);
    const right = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t3_right_hdr") }),
        el("button", {
          class: "sv-btn act", text: t("btn_copy_global_refs"),
          onclick: () => {
            p.shots.forEach((s) => { s.ref_slots = [...p.global_ref_slots]; s.ref_images = s.ref_slots.filter(Boolean); });
            scheduleSync(true); showToast(t("toast_global_refs_copied"));
          },
        }),
      ]),
      render9SlotGrid(p.global_ref_slots, null, false, (k, v) => { p.global_ref_slots[k] = v; scheduleSync(); renderBody(); }, "shotvault/global"),
      el("div", { class: "sv-aud-box", style: { marginTop: "6px" } }, [
        el("div", { class: "sv-box-hdr" }, [
          el("span", { text: t("lbl_global_voice_hdr") }),
          el("div", { class: "sv-row", style: { gap: "4px" } }, [
            el("button", {
              class: "sv-btn act xs",
              disabled: gAudMuted,
              style: gAudMuted ? { opacity: "0.5", cursor: "not-allowed" } : {},
              text: t("btn_copy_global_voices"),
              onclick: () => {
                if (gAudMuted) return;
                const gSlots = normSlots3(p.global_ref_audio_slots, p.global_ref_audios);
                if (!gSlots.some(Boolean)) return showToast(t("toast_no_voices_to_copy"), true);
                p.shots.forEach((sh) => {
                  sh.ref_audio_slots = [...gSlots];
                  sh.ref_audios = gSlots.filter(Boolean);
                });
                scheduleSync(true);
                showToast(t("toast_global_voices_copied", { n: p.shots.length }));
              },
            }),
            gAudHasAny ? el("button", {
              class: "sv-btn dng xs",
              text: t("btn_clear_global_voice"),
              onclick: () => {
                p.global_ref_audio_slots = new Array(3).fill("");
                p.global_ref_audios = [];
                scheduleSync();
                renderBody();
              },
            }) : null,
          ]),
        ]),
        gAudMuted ? el("span", { class: "sv-link-lbl", text: t("lbl_mv_muted") }) : null,
        render3AudioSlotGrid(
          p.global_ref_audio_slots,
          null,
          false,
          gAudMuted,
          (k, v) => {
            p.global_ref_audio_slots[k] = v;
            p.global_ref_audios = p.global_ref_audio_slots.filter(Boolean);
            scheduleSync();
            renderBody();
          },
          "shotvault/global",
          () => renderBody(),
        ),
      ]),
    ]);
    bodyEl.append(el("div", { class: "sv-grid2" }, [left, right]));
  }

  function renderTab4() {
    const g = state.plan.context_frames;
    const rows = [22, 39, 56].map((val) => {
      const m = computeGear(val, true);
      return el("tr", { style: val === g ? { background: "#1b2742", fontWeight: "700" } : {} }, [
        el("td", { text: t("tbl_gear_cell", { v: val, cur: val === g ? t("tbl_gear_cur") : "" }) }),
        el("td", { text: `${m.overlap}f` }), el("td", { text: `${m.trim}f` }), el("td", { text: `${m.taperTail}f` }),
        el("td", { text: `${m.ramp}f` }), el("td", { text: `${m.audioGuide}f` }), el("td", { text: `${m.feather} tick` }),
        el("td", { text: m.pair }), el("td", { text: `forLoop -> clean_tail[-${val}f]` }),
      ]);
    });
    const gearBox = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("t4_gear_hdr") }),
        isLinked("context_frames") ? el("span", { class: "sv-link-lbl", text: t("lbl_ctx_linked") }) : null,
      ]),
      el("table", { class: "sv-tbl" }, [
        el("thead", {}, [el("tr", {}, [t("tbl_th_gear"), "overlap", "trim_head", "taper_tail(#749)", "taper_ramp", "audio_guide", "feather_ticks", t("tbl_th_grid"), t("tbl_th_forloop")].map((h) => el("th", { text: h })))]),
        el("tbody", {}, rows),
      ]),
    ]);

    const cfgFields = [
      ["aspect_ratio", t("cfg_aspect_ratio"), "select", ["16:9 (1344x768)", "9:16 (768x1344)", "4:3 (1152x864)", "3:4 (864x1152)", "1:1 (1024x1024)", "2.39:1 (1536x640)"]],
      ["megapixels", t("cfg_megapixels"), "float", 1.0],
      ["fps", t("cfg_fps"), "int", 24],
      ["duration_mode", t("cfg_duration_mode"), "select", ["zero_tail_trim", "cumulative_cursor", "net_preserve"]],
      ["script_format", t("cfg_script_format"), "select", ["auto", "1114_markdown", "kkdemabu_dual_jsonc", "cinema_pack_json"]],
      ["audio_drive_mode", t("cfg_audio_drive_mode"), "select", ["none", "per_shot_audio_dir", "master_song_slice"]],
      ["snap_cuts_to_beat", t("cfg_snap_cuts"), "bool", false],
      ["auto_split_turnaround", t("cfg_auto_split"), "bool", false],
    ];
    const grid4 = el("div", { class: "sv-grid4" }, cfgFields.map(([k, lbl, kind, fb]) => {
      const wrap = el("div", { class: "sv-box", style: { padding: "6px" } }, [el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: lbl })]);
      if (isLinked(k)) { wrap.appendChild(el("span", { class: "sv-link-lbl", text: t("lbl_linked_ctrl") })); return wrap; }
      if (kind === "select") {
        const opts = getOpts(k, fb);
        const cur = k === "audio_drive_mode" ? String(state.plan.audio_drive_mode || getW(k, opts[0])) : String(getW(k, opts[0]));
        wrap.appendChild(el("select", {
          class: "sv-sel",
          onchange: (e) => {
            setW(k, e.target.value);
            if (k === "audio_drive_mode") {
              state.plan.audio_drive_mode = e.target.value;
              if (e.target.value !== "none") state.plan.context_frames = 39;
              scheduleSync(true);
              renderAll();
              return;
            }
            scheduleSync();
          },
        }, opts.map((o) => el("option", { value: String(o), selected: String(o) === cur, text: String(o) }))));
      } else if (kind === "bool") {
        wrap.appendChild(el("input", { type: "checkbox", checked: Boolean(getW(k, fb)), onchange: (e) => { setW(k, e.target.checked); scheduleSync(); } }));
      } else {
        wrap.appendChild(el("input", { class: "sv-inp", type: (kind === "float" || kind === "int") ? "number" : "text", step: kind === "float" ? "0.05" : "1", value: String(getW(k, fb)), oninput: (e) => { setW(k, kind === "float" ? (parseFloat(e.target.value) || fb) : (kind === "int" ? (parseInt(e.target.value, 10) || fb) : e.target.value)); scheduleSync(); } }));
      }
      return wrap;
    }));

    const advFields = [
      ["audio_per_shot_dir", t("cfg_audio_dir"), "text", ""],
      ["assets_root_dir", t("cfg_assets_dir"), "text", ""],
    ];
    const advTa = el("textarea", {
      class: "sv-ta", rows: 3, style: { width: "100%", resize: "vertical" },
      placeholder: t("ph_stage3_json"),
      oninput: (e) => { setW("stage3_json_text", e.target.value); scheduleSync(); },
    });
    advTa.value = String(getW("stage3_json_text", "") || "");
    const advDetails = el("details", { class: "sv-adv" }, [
      el("summary", { text: t("adv_v17_summary") }),
      el("div", { class: "sv-grid2", style: { marginTop: "6px" } }, advFields.map(([k, lbl, _, fb]) => el("div", { class: "sv-box", style: { padding: "6px" } }, [
        el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: lbl }),
        el("input", { class: "sv-inp", value: String(getW(k, fb) || ""), oninput: (e) => { setW(k, e.target.value); scheduleSync(); } }),
      ]))),
      el("div", { style: { marginTop: "6px" } }, [advTa]),
    ]);

    const issues = runChecks();
    const { loads, dispatches } = findDownstreamVaultNodes();
    const chkBox = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr" }, [
        el("span", { text: t("chk_hdr", { d: dispatches.length, l: loads.length }) }),
      ]),
      ...(issues.length ? issues.map((it) => el("div", { class: "sv-row", style: { justifyContent: "space-between", background: "#14151b", padding: "4px 8px", borderRadius: "6px" } }, [
        el("span", { style: { color: it.lv === "err" ? "#ff9ca8" : "#ffd08c" }, text: `${it.lv === "err" ? "❌" : "⚠"} ${it.msg}` }),
        it.shot >= 0 ? el("button", { class: "sv-btn", text: t("btn_jump_shot", { i: it.shot + 1 }), onclick: () => { state.sel = it.shot; state.tab = "edit"; renderTabs(); renderBody(); } }) : null,
      ])) : [el("span", { style: { color: "#7fe0b6" }, text: t("t4_chk_ok") })]),
    ]);
    const adCfg = state.plan.anti_degradation || (state.plan.anti_degradation = { mode: "standard", noise_keep_ratio: 0.40, static_bg_pull: 0.20 });
    const adDefs = [
      ["standard", t("ad_std"), t("ad_std_desc")],
      ["digital_human_low_noise", t("ad_low"), t("ad_low_desc")],
      ["digital_human_static_bg", t("ad_bg"), t("ad_bg_desc")],
    ];
    const adCards = el("div", { style: { display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "8px" } }, adDefs.map(([mVal, titleTxt, descTxt]) => {
      const on = String(adCfg.mode || "standard") === mVal;
      return el("div", {
        style: {
          padding: "8px",
          borderRadius: "6px",
          border: `1px solid ${on ? "#2f6df6" : "#2c3040"}`,
          background: on ? "#18243d" : "#14161e",
          cursor: "pointer",
          display: "flex",
          flexDirection: "column",
          gap: "4px",
        },
        onclick: () => {
          adCfg.mode = mVal;
          if (mVal === "digital_human_low_noise" && adCfg.noise_keep_ratio === 0.36) adCfg.noise_keep_ratio = 0.40;
          if (mVal === "digital_human_static_bg" && adCfg.noise_keep_ratio === 0.40) adCfg.noise_keep_ratio = 0.36;
          scheduleSync(true);
          renderAll();
        },
      }, [
        el("div", { class: "sv-row", style: { justifyContent: "space-between", fontWeight: "700", fontSize: "12px", color: on ? "#8ec5ff" : "#dde3f5" } }, [
          el("span", { text: `${on ? "● " : "○ "}${titleTxt}` }),
        ]),
        el("div", { style: { fontSize: "11px", color: "#9da6bc", lineHeight: "1.4" }, text: descTxt }),
      ]);
    }));
    const adAdv = el("details", { class: "sv-adv", style: { marginTop: "6px" } }, [
      el("summary", { style: { fontSize: "11px" }, text: t("ad_adv") }),
      el("div", { class: "sv-row", style: { marginTop: "6px", gap: "12px" } }, [
        el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: t("ad_lbl_noise") }),
        el("input", {
          class: "sv-inp",
          type: "number",
          step: "0.02",
          min: "0.05",
          max: "1.0",
          style: { width: "70px" },
          value: String(adCfg.noise_keep_ratio ?? 0.40),
          oninput: (e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) { adCfg.noise_keep_ratio = Math.max(0.05, Math.min(1.0, v)); scheduleSync(); }
          },
        }),
        el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: t("ad_lbl_bg_pull") }),
        el("input", {
          class: "sv-inp",
          type: "number",
          step: "0.02",
          min: "0",
          max: "0.35",
          style: { width: "70px" },
          value: String(adCfg.static_bg_pull ?? 0.20),
          oninput: (e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v)) { adCfg.static_bg_pull = Math.max(0.0, Math.min(0.35, v)); scheduleSync(); }
          },
        }),
      ]),
    ]);
    const adBox = el("div", { class: "sv-box" }, [
      el("div", { class: "sv-box-hdr", text: t("ad_hdr") }),
      adCards,
      adAdv,
    ]);

    bodyEl.append(gearBox, adBox, el("div", { class: "sv-box" }, [el("div", { class: "sv-box-hdr", text: t("t4_cfg_hdr") }), grid4, advDetails]), chkBox);
  }

  function renderBody() {
    bodyEl.innerHTML = "";
    if (state.tab === "edit") renderTab1();
    else if (state.tab === "reroll") renderTab2();
    else if (state.tab === "shot0") renderTab3();
    else renderTab4();
    const proj = String(getW("project_name", "MyH3Film") || "MyH3Film");
    const expModal = renderExportModal({
      state,
      proj,
      api,
      el,
      t,
      fmtBytes,
      mediaUrl,
      syncServer,
      renderBody,
      renderAll,
      showToast,
    });
    if (expModal) bodyEl.appendChild(expModal);
    patchFooter();
  }

  function renderAll() {
    hideNativeWidgets(node);
    renderHeader();
    renderTabs();
    renderBody();
  }

  const host = el("div", { class: "sv-host" });
  const root = el("div", { class: "sv-root" });
  const hdrEl = el("div", { class: "sv-hdr" });
  const tabsEl = el("div", { class: "sv-tabs" });
  const bodyEl = el("div", { class: "sv-body" });
  const ftrEl = el("div", { class: "sv-ftr" });
  root.append(hdrEl, tabsEl, bodyEl, ftrEl);
  host.appendChild(root);
  root.addEventListener("wheel", (e) => {
    const canScroll = bodyEl.scrollHeight > bodyEl.clientHeight && !e.ctrlKey && !e.metaKey;
    if (!canScroll) app.canvas?.processMouseWheel?.(e);
  }, { passive: true });
  root.addEventListener("pointerdown", (e) => {
    if (e.target?.closest?.("button, input, textarea, select, label, summary, video, a, .sv-ta-grip, .sv-modal-bg, .sv-modal, .sv-slot, .sv-aud-slot, .sv-card, .sv-tab, .sv-chip")) {
      e.stopPropagation();
    }
  });
  root.addEventListener("mousedown", (e) => {
    if (e.target?.closest?.("button, input, textarea, select, label, summary, video, a, .sv-ta-grip, .sv-modal-bg, .sv-modal, .sv-slot, .sv-aud-slot, .sv-card, .sv-tab, .sv-chip")) {
      e.stopPropagation();
    }
  });
  root.addEventListener("dblclick", (e) => e.stopPropagation());
  const stopCanvasKey = (e) => {
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || e.target?.isContentEditable) {
      e.stopPropagation();
    }
  };
  root.addEventListener("keydown", stopCanvasKey);
  root.addEventListener("keyup", stopCanvasKey);
  function reloadFromWidgets() {
    hideNativeWidgets(node);
    state.plan = buildPlanFromWidgets();
    setW("audio_drive_mode", state.plan.audio_drive_mode);
    setW("context_frames", String(state.plan.context_frames));
    state.masterExpanded = state.plan.shots.length <= 1;
    state.sel = Math.max(0, Math.min(state.sel, state.plan.shots.length - 1));
    syncResponsiveFromNode();
    renderAll();
    return refreshState();
  }

  const onLangChange = () => renderAll();
  window.addEventListener("acheng:lang-change", onLangChange);
  const origRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    stopAudioPreview();
    window.removeEventListener("acheng:lang-change", onLangChange);
    return origRemoved?.apply(this, args);
  };

  renderAll();
  setTimeout(refreshState, 120);
  return { node, host, root, state, serializePlan, reloadFromWidgets, renderAll, isUpstreamAudioActive, applyPanelSize, syncResponsiveFromNode };
}

app.registerExtension({
  name: EXT_NAME,
  async nodeCreated(node) {
    if (node.comfyClass !== "ShotVaultPlan") return;
    injectStyles();
    hideNativeWidgets(node);
    const ctrl = attachPanelController(node);
    const domWidget = node.addDOMWidget("director_console", "div", ctrl.host, {
      serialize: false,
      hideOnZoom: false,
      getMinHeight: () => PANEL_DEFAULTS.MIN_H,
      getMaxHeight: () => PANEL_DEFAULTS.MAX_H,
    });
    domWidget.computeSize = (width) => {
      const curH = Math.max(PANEL_DEFAULTS.MIN_H, Math.min(PANEL_DEFAULTS.MAX_H, Math.round((Number(node.size?.[1]) || (PANEL_H + PANEL_DEFAULTS.CHROME_H)) - PANEL_DEFAULTS.CHROME_H)));
      return [Math.max(PANEL_DEFAULTS.MIN_W - 24, Number(width || PANEL_W)), curH];
    };
    const initW = Math.max(PANEL_DEFAULTS.MIN_W, Math.min(PANEL_DEFAULTS.MAX_W, Number(node.size?.[0] || PANEL_W)));
    const initH = Math.max(PANEL_DEFAULTS.MIN_H + PANEL_DEFAULTS.CHROME_H, Math.min(PANEL_DEFAULTS.MAX_H + PANEL_DEFAULTS.CHROME_H, Number(node.size?.[1] || (PANEL_H + PANEL_DEFAULTS.CHROME_H))));
    node.size = [initW, initH];
    ctrl.syncResponsiveFromNode();

    const origResize = node.onResize;
    node.onResize = function (size) {
      if (Array.isArray(size) && size.length >= 2) {
        size[0] = Math.max(PANEL_DEFAULTS.MIN_W, Math.min(PANEL_DEFAULTS.MAX_W, Math.round(size[0])));
        size[1] = Math.max(PANEL_DEFAULTS.MIN_H + PANEL_DEFAULTS.CHROME_H, Math.min(PANEL_DEFAULTS.MAX_H + PANEL_DEFAULTS.CHROME_H, Math.round(size[1])));
      }
      const r = origResize?.apply(this, arguments);
      this._svCtrl?.syncResponsiveFromNode?.();
      return r;
    };
    const origCfg = node.onConfigure;
    node.onConfigure = function (...args) {
      this._svHydrating = true;
      const r = origCfg?.apply(this, args);
      hideNativeWidgets(this);
      setTimeout(() => {
        hideNativeWidgets(this);
        this._svCtrl?.reloadFromWidgets?.();
        this._svHydrating = false;
      }, 0);
      return r;
    };
    const origConn = node.onConnectionsChange;
    node.onConnectionsChange = function (type, index, connected, ...rest) {
      const r = origConn?.apply(this, [type, index, connected, ...rest]);
      if (!this._svHydrating && type === 1 && index === 0 && connected && this._svCtrl?.state?.plan?.audio_drive_mode === "none" && this._svCtrl?.isUpstreamAudioActive?.()) {
        this._svCtrl.state.plan.audio_drive_mode = "master_song_slice";
        this._svCtrl.state.plan.context_frames = 39;
        this._svCtrl.serializePlan();
      }
      this._svCtrl?.renderAll?.();
      return r;
    };
    node._svCtrl = ctrl;
  },
});
