import { app } from "../../scripts/app.js";

const STORAGE_KEY = "acheng_ui_lang";

const NODE_LABELS = {
  AchengFuzzPuppyClipMerge: {
    title: {
      zh: "Acheng FuzzPuppy Clip Merge · 视频无缝合并",
      en: "Acheng FuzzPuppy Clip Merge · Video Seam Merge",
    },
    widgets: {
      mode: { zh: "合并模式 (mode)", en: "Mode" },
      rife_bridge_frames: { zh: "RIFE 跨缝帧数", en: "RIFE Bridge Frames" },
      rife_backend: { zh: "RIFE 推理后端", en: "RIFE Backend" },
      preserve_duration: { zh: "保持原时长 (MV零帧差)", en: "Preserve Duration (Zero Delta)" },
      max_search_sec: { zh: "切点搜寻窗口 (秒)", en: "Seam Search Window (s)" },
      structure_threshold: { zh: "硬切结构差阈值", en: "Structure Threshold" },
      coverage_threshold: { zh: "硬切块覆盖阈值", en: "Coverage Threshold" },
      release_seconds: { zh: "对齐缓释时长 (秒)", en: "Release Duration (s)" },
      is_continuation: { zh: "是否续段", en: "Is Continuation" },
      codec: { zh: "编码格式", en: "Codec" },
      enable_repair: { zh: "开启修复 (关=原版直拼)", en: "Enable Repair (Off=Original)" },
      retry_policy: { zh: "不通过自动兜底", en: "Retry Fallback Policy" },
      ui_lang: { zh: "诊断摘要语言", en: "Diagnostics Language" },
    },
  },
  AchengSeamBridge: {
    title: {
      zh: "Acheng Seam Bridge · 图像接缝修复",
      en: "Acheng Seam Bridge · Image Seam Repair",
    },
    widgets: {
      mode: { zh: "修复模式 (mode)", en: "Mode" },
      join_frame: { zh: "接缝帧序号", en: "Join Frame" },
      max_bridge_frames: { zh: "最大桥接帧数", en: "Max Bridge Frames" },
      release_frames: { zh: "缓释帧数", en: "Release Frames" },
      sensitivity_ratio: { zh: "跳变灵敏度", en: "Sensitivity Ratio" },
      protect_hard_cuts: { zh: "硬切保护", en: "Protect Hard Cuts" },
      repair_white_head: { zh: "修复白头/黑头", en: "Repair White/Black Head" },
      is_continuation: { zh: "是否续段", en: "Is Continuation" },
      auto_bridge_backend: { zh: "自动桥接后端", en: "Auto Bridge Backend" },
      rife_checkpoint: { zh: "RIFE 模型权重", en: "RIFE Checkpoint" },
      auto_align_color: { zh: "自动局部对齐校色", en: "Auto Align & Color" },
    },
  },
};

function getLang() {
  try {
    return window.localStorage?.getItem(STORAGE_KEY) === "en" ? "en" : "zh";
  } catch {
    return "zh";
  }
}

function setLang(lang) {
  const norm = lang === "en" ? "en" : "zh";
  try {
    window.localStorage?.setItem(STORAGE_KEY, norm);
  } catch {
    // ignore
  }
  window.dispatchEvent(new CustomEvent("acheng:lang-change", { detail: { lang: norm } }));
}

function applyLangToNode(node, lang) {
  const spec = NODE_LABELS[node?.comfyClass];
  if (!spec) return;
  const l = lang === "en" ? "en" : "zh";
  node.title = spec.title[l] || node.title;
  for (const w of node.widgets || []) {
    const lbl = spec.widgets[w.name]?.[l];
    if (lbl) w.label = lbl;
    if (w.name === "ui_lang" && (w.value === "zh" || w.value === "en")) {
      w.value = l;
    }
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

function syncAllSeamBridgeNodes(lang) {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if (NODE_LABELS[n?.comfyClass]) applyLangToNode(n, lang);
  }
}

app.registerExtension({
  name: "Acheng.SeamBridge.BilingualUI",
  async setup() {
    window.addEventListener("acheng:lang-change", (e) => {
      const lang = e?.detail?.lang || getLang();
      syncAllSeamBridgeNodes(lang);
    });
  },
  async nodeCreated(node) {
    if (!NODE_LABELS[node?.comfyClass]) return;
    applyLangToNode(node, getLang());
    const origMenu = node.getExtraMenuOptions;
    node.getExtraMenuOptions = function (canvas, options) {
      origMenu?.apply(this, arguments);
      const cur = getLang();
      options.push(null, {
        content: cur === "zh" ? "🌐 Switch SeamBridge UI to English" : "🌐 切换 SeamBridge 界面为中文",
        callback: () => setLang(cur === "zh" ? "en" : "zh"),
      });
    };
    const origCfg = node.onConfigure;
    node.onConfigure = function (...args) {
      const r = origCfg?.apply(this, args);
      applyLangToNode(this, getLang());
      return r;
    };
  },
});
