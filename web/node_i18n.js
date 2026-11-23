/**
 * Unified canvas node-title & widget-label bilingual (ZH / EN) localization
 * for all 10 nodes in ComfyUI-Universal-ShotVault (8 ShotVault* + 2 AchengSeamBridge*).
 *
 * Strictly display-layer only: never mutates Python INPUT_TYPES keys, widget names,
 * or serialized workflow widget values (except optional `ui_lang` diagnostics enum).
 */

import { app } from "../../scripts/app.js";
import { getLang, setLang } from "./i18n.js";

export const NODE_I18N_MAP = {
  ShotVaultPlan: {
    titles: {
      zh: "ShotVault Plan (可视化分镜规划/音轨编排)",
      en: "ShotVault Plan (Visual Storyboard & Audio Planner)",
    },
    widgets: {
      project_name: { zh: "项目名称", en: "Project Name" },
      script_format: { zh: "脚本格式", en: "Script Format" },
      common_prompt: { zh: "公共画风词", en: "Common Prompt" },
      script_text: { zh: "总脚本文本", en: "Master Script" },
      aspect_ratio: { zh: "画幅比例", en: "Aspect Ratio" },
      megapixels: { zh: "目标像素 (MP)", en: "Megapixels" },
      fps: { zh: "帧率 (FPS)", en: "FPS" },
      duration_mode: { zh: "时长计算模式", en: "Duration Mode" },
      context_frames: { zh: "衔接重叠帧", en: "Context Frames" },
      audio_drive_mode: { zh: "音频驱动模式", en: "Audio Drive Mode" },
      snap_cuts_to_beat: { zh: "切点吸附节拍", en: "Snap Cuts to Beat" },
      auto_split_turnaround: { zh: "四宫格自动裁图", en: "Auto-Split Turnaround" },
    },
  },
  ShotVaultLoad: {
    titles: {
      zh: "ShotVault Load (选段/双尾与多模加载)",
      en: "ShotVault Load (Shot Selector & Context Loader)",
    },
    widgets: {
      start_mode: { zh: "启动模式", en: "Start Mode" },
      shot_numbers: { zh: "镜号范围", en: "Shot Numbers" },
      continue_from: { zh: "续接来源", en: "Continue From" },
      seed_mode: { zh: "Seed 策略", en: "Seed Mode" },
      shot_seed: { zh: "默认 Seed", en: "Shot Seed" },
      seed_text: { zh: "指定 Seed 文本", en: "Seed Text" },
      av_sync_lead_sec: { zh: "音画提前量 (秒)", en: "AV Sync Lead (s)" },
      parent_take_override: { zh: "指定上段 Take", en: "Parent Take Override" },
      keep_downstream_on_reroll: { zh: "重抽保留下游", en: "Keep Downstream on Reroll" },
    },
  },
  ShotVaultLoopDispatch: {
    titles: {
      zh: "ShotVault Loop Dispatch (外部循环分发/22·39档/9图)",
      en: "ShotVault Loop Dispatch (Loop & Bidir Anchor Dispatcher)",
    },
    widgets: {
      segment_index: { zh: "循环序号", en: "Segment Index" },
      index_base: { zh: "索引基准", en: "Index Base" },
      honor_director_retry_lock: { zh: "响应单段重抽锁", en: "Honor Single-Shot Reroll Lock" },
      seed_mode: { zh: "Seed 策略", en: "Seed Mode" },
      base_seed: { zh: "基础 Seed", en: "Base Seed" },
      seed_text: { zh: "指定 Seed 文本", en: "Seed Text" },
    },
  },
  ShotVaultH3Conditioning: {
    titles: {
      zh: "ShotVault H3 Conditioning (分镜多模参照编码)",
      en: "ShotVault H3 Conditioning (Per-Shot Multimodal Encoder)",
    },
    widgets: {
      ref_image_size: { zh: "参考图尺寸策略", en: "Ref Image Size" },
    },
  },
  ShotVaultWindowGuider: {
    titles: {
      zh: "ShotVault Window Guider (零劣化窗口引导)",
      en: "ShotVault Window Guider (Context Window & Bidir Guider)",
    },
    widgets: {
      pass_stage: { zh: "采样阶段", en: "Pass Stage" },
      window_mode: { zh: "窗口模式", en: "Window Mode" },
      window_frames: { zh: "窗口帧数", en: "Window Frames" },
      window_overlap: { zh: "窗口重叠帧", en: "Window Overlap" },
      continuation_frames: { zh: "续接帧数", en: "Continuation Frames" },
      keyframe_until_sigma: { zh: "关键帧终止 Sigma", en: "Keyframe Until Sigma" },
      audio_feather_ticks: { zh: "音频羽化步数", en: "Audio Feather Ticks" },
      rebase_2nd_pass_audio: { zh: "二采音频对齐", en: "Rebase 2nd-Pass Audio" },
      lock_target_audio: { zh: "锁定目标音轨", en: "Lock Target Audio" },
      cfg: { zh: "CFG 引导强度", en: "CFG" },
    },
  },
  ShotVaultSeamTrim: {
    titles: {
      zh: "ShotVault Seam Trim (智能ΔE锁色/RIFE微桥)",
      en: "ShotVault Seam Trim (Head Trim & CIE Lab Color Lock)",
    },
    widgets: {
      trim_head_frames: { zh: "裁头帧数 (-1自动)", en: "Trim Head Frames (-1 Auto)" },
      keep_frames: { zh: "净留帧数 (-1自动)", en: "Keep Frames (-1 Auto)" },
      color_match_mode: { zh: "锁色算法", en: "Color Match Mode" },
      seam_frames: { zh: "校色过渡帧", en: "Seam Frames" },
      strength_curve: { zh: "校色强度曲线", en: "Strength Curve" },
      rife_bridge_mode: { zh: "RIFE 微桥模式", en: "RIFE Bridge Mode" },
      bridge_max_frames: { zh: "微桥最大帧数", en: "Bridge Max Frames" },
      color_deadzone_de: { zh: "ΔE 死区阈值", en: "Color Deadzone ΔE" },
      global_lock_strength: { zh: "全局锁色强度", en: "Global Lock Strength" },
      jump_spike_threshold: { zh: "跳变检测阈值", en: "Jump Spike Threshold" },
    },
  },
  ShotVaultSave: {
    titles: {
      zh: "ShotVault Save (双轨隔离归档)",
      en: "ShotVault Save (Isolated Take Archive & Auto-Assemble)",
    },
    widgets: {
      auto_queue_next: { zh: "自动排队下一镜", en: "Auto Queue Next" },
      crf: { zh: "MP4 画质 (CRF)", en: "MP4 Quality (CRF)" },
      codec: { zh: "视频编码", en: "Codec" },
    },
  },
  ShotVaultAssemble: {
    titles: {
      zh: "ShotVault Assemble (单遍PCM总装)",
      en: "ShotVault Assemble (Single-Pass PCM Master)",
    },
    widgets: {
      project_name: { zh: "项目名称", en: "Project Name" },
      stale_policy: { zh: "标红镜头策略", en: "Stale Policy" },
      audio_source_policy: { zh: "音轨来源策略", en: "Audio Source Policy" },
      audio_crossfade_ms: { zh: "音频淡化 (ms)", en: "Audio Crossfade (ms)" },
      post_mux_offset_ms: { zh: "音画微调偏移 (ms)", en: "Post-Mux Offset (ms)" },
      prepend_seed_video: { zh: "拼接 Shot 0 底板", en: "Prepend Shot 0 Seed" },
      export_fcpxml: { zh: "同步导出 FCPXML", en: "Export FCPXML" },
    },
  },
  AchengFuzzPuppyClipMerge: {
    titles: {
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
    titles: {
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

function isKnownDefaultTitle(node, spec) {
  const cur = String(node?.title || "").trim();
  if (!cur) return true;
  if (cur === node?.comfyClass) return true;
  if (cur === spec.titles.zh || cur === spec.titles.en) return true;
  // Also accept legacy SeamBridge display title
  if (node?.comfyClass === "AchengSeamBridge" && cur === "Acheng Seam Bridge · 接缝修复") return true;
  return false;
}

export function applyLangToCanvasNode(node, lang) {
  const spec = NODE_I18N_MAP[node?.comfyClass];
  if (!spec) return;
  const l = lang === "en" ? "en" : "zh";
  if (isKnownDefaultTitle(node, spec)) {
    node.title = spec.titles[l] || node.title;
  }
  for (const w of node.widgets || []) {
    if (w.hidden || w.type === "hidden" || w.name === "director_console") continue;
    const lbl = spec.widgets?.[w.name]?.[l];
    if (lbl) w.label = lbl;
    if (w.name === "ui_lang" && (w.value === "zh" || w.value === "en")) {
      w.value = l;
    }
  }
  app.graph?.setDirtyCanvas?.(true, true);
}

function syncAllCanvasNodes(lang) {
  const nodes = app.graph?._nodes || [];
  for (const n of nodes) {
    if (NODE_I18N_MAP[n?.comfyClass]) {
      applyLangToCanvasNode(n, lang);
    }
  }
}

if (!window.__achengNodeI18nRegistered) {
  window.__achengNodeI18nRegistered = true;

  app.registerExtension({
    name: "ShotVault.CanvasNodes.BilingualUI",
    async setup() {
      window.addEventListener("acheng:lang-change", (e) => {
        const lang = e?.detail?.lang || getLang();
        syncAllCanvasNodes(lang);
      });
    },
    async nodeCreated(node) {
      if (!NODE_I18N_MAP[node?.comfyClass]) return;
      applyLangToCanvasNode(node, getLang());

      if (!node._svI18nMenuPatched) {
        node._svI18nMenuPatched = true;
        const origMenu = node.getExtraMenuOptions;
        node.getExtraMenuOptions = function (canvas, options) {
          origMenu?.apply(this, arguments);
          const cur = getLang();
          options.push(null, {
            content: cur === "zh" ? "🌐 Switch ShotVault UI to English" : "🌐 切换 ShotVault 界面为中文",
            callback: () => setLang(cur === "zh" ? "en" : "zh"),
          });
        };
      }

      const origCfg = node.onConfigure;
      node.onConfigure = function (...args) {
        const r = origCfg?.apply(this, args);
        applyLangToCanvasNode(this, getLang());
        return r;
      };
    },
  });
}
