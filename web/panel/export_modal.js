/**
 * In-panel Master Export Modal with exact manifest-based outdated detection,
 * stale-shot policy control, live HTML5 playback, and safe MP4/FCPXML download.
 */

export function renderExportModal({
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
}) {
  if (!state.exportOpen) return null;
  const svShots = state.serverState?.shots || [];
  const totalPlanned = state.plan.shots.length;
  const renderedShots = svShots.filter((x) => Boolean(x.active_take_id));
  const staleShots = svShots.filter((x) => x.status === "STALE");
  const unrenderedShots = svShots.filter((x) => x.status === "UNRENDERED" || !x.active_take_id);
  const activeRepairs = svShots.filter(
    (x) => x.seam_repair?.enabled && (x.seam_repair?.choice || "repaired") === "repaired" && (x.seam_repair?.mode || "rife") !== "cut",
  );
  const hasSeed = Boolean(state.serverState?.seed_shot?.take_id || state.plan.seed_video_path);
  const existingMasters = (state.serverState?.master_exports || []).filter((x) => x.ext === ".mp4" || x.ext === ".mov");
  const syncStatus = state.serverState?.master_sync_status || { has_master: existingMasters.length > 0, is_outdated: false, outdated_reasons: [] };
  const draft = state.exportDraft;

  const runExportNow = async () => {
    if (state.exportBusy) return;
    state.exportBusy = true;
    renderBody();
    try {
      await syncServer();
      const r = await api.fetchApi("/shotvault/assemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project: proj,
          stale_policy: draft.stale_policy,
          output_mode: draft.output_mode,
          export_fcpxml: Boolean(draft.export_fcpxml),
          prepend_seed_video: Boolean(draft.prepend_seed_video),
          audio_crossfade_ms: Number(draft.audio_crossfade_ms ?? 15),
        }),
      });
      const j = await r.json();
      if (r.ok && j.ok) {
        if (j.state) state.serverState = j.state;
        state.mediaBust = Date.now();
        state.lastExport = j;
        const rep = j.report || {};
        showToast(t("exp_ok", {
          n: rep.total_clips ?? renderedShots.length,
          dur: Number(rep.expected_duration_sec || 0).toFixed(1),
          sz: fmtBytes(j.file_bytes || 0),
        }));
      } else {
        showToast(t("exp_fail", { msg: j.error || r.statusText }), 3200, true);
      }
    } catch (err) {
      showToast(t("exp_fail", { msg: err?.message || err }), 3200, true);
    } finally {
      state.exportBusy = false;
      renderAll();
    }
  };

  const resData = state.lastExport && state.lastExport.ok ? state.lastExport : null;
  const stillOutdated = Boolean(syncStatus.is_outdated) && !resData;
  const outdatedBanner = stillOutdated
    ? el("div", { class: "sv-outdated-banner" }, [
        syncStatus.has_master
          ? t("exp_outdated_banner", { reasons: (syncStatus.outdated_reasons || []).slice(0, 6).join("、") })
          : t("exp_no_master_banner"),
      ])
    : (resData
        ? el("div", { class: "sv-fresh-banner", text: t("exp_fresh_banner", { n: resData.report?.total_clips ?? renderedShots.length, dur: Number(resData.report?.expected_duration_sec || 0).toFixed(1) }) })
        : null);

  const repairListStr = activeRepairs
    .map((x) => `#${x.shot_index}(${x.seam_repair?.mode || "rife"}${x.seam_repair?.mode === "rife" ? ` ${x.seam_repair?.rife_bridge_frames || 5}f` : ""})`)
    .join(", ");

  const summaryBox = el("div", { class: "sv-box", style: { background: "#14161e", gap: "5px" } }, [
    el("div", { class: "sv-row", style: { justifyContent: "space-between" } }, [
      el("span", { style: { fontWeight: "700", color: "#dde4f7" }, text: `📦 ${proj} · 已渲染 ${renderedShots.length}/${totalPlanned} 镜` }),
      staleShots.length
        ? el("span", { class: "sv-badge", style: { background: "#4a3312", color: "#ffd27d", borderColor: "#a67521" }, text: `⚠ 标红 STALE: ${staleShots.map((x) => `#${x.shot_index}`).join(", ")}` })
        : el("span", { class: "sv-badge", style: { background: "#143d2c", color: "#8ff0c2", borderColor: "#25875f" }, text: "✅ 全片绿灯" }),
    ]),
    el("div", { style: { fontSize: "11px", color: activeRepairs.length ? "#8ff0c2" : "#9da6bc" }, text: activeRepairs.length ? `✨ ${t("exp_with_repairs", { n: activeRepairs.length, list: repairListStr })}` : `🎞 ${t("exp_no_repairs")}` }),
    unrenderedShots.length
      ? el("div", { style: { fontSize: "11px", color: "#ff9ca8" }, text: `未渲染镜头：${unrenderedShots.map((x) => `#${x.shot_index}`).join(", ")}（合并时将自动跳过未出片镜头）` })
      : null,
    existingMasters.length
      ? el("div", { style: { fontSize: "11px", color: "#8c95ab" }, text: t("exp_existing", { name: existingMasters[0].name, sz: fmtBytes(existingMasters[0].bytes) }) })
      : null,
  ]);

  const optBox = el("div", { class: "sv-box", style: { gap: "8px" } }, [
    el("div", { class: "sv-row" }, [
      el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: t("exp_policy_lbl") }),
      el("select", {
        class: "sv-sel",
        onchange: (e) => { draft.stale_policy = e.target.value; },
      }, [
        ["allow_with_warning", t("exp_policy_allow")],
        ["block_and_report", t("exp_policy_strict")],
      ].map(([v, lbl]) => el("option", { value: v, selected: draft.stale_policy === v, text: lbl }))),
    ]),
    el("div", { class: "sv-row" }, [
      el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: t("exp_out_lbl") }),
      el("select", {
        class: "sv-sel",
        onchange: (e) => { draft.output_mode = e.target.value; },
      }, [
        ["overwrite_master", t("exp_out_overwrite")],
        ["timestamped", t("exp_out_ts")],
      ].map(([v, lbl]) => el("option", { value: v, selected: draft.output_mode === v, text: lbl }))),
      el("span", { style: { fontSize: "11px", color: "#9da6bc" }, text: t("exp_xfade") }),
      el("input", {
        class: "sv-inp",
        type: "number",
        min: "0",
        max: "200",
        step: "1",
        style: { width: "58px" },
        value: String(draft.audio_crossfade_ms ?? 15),
        oninput: (e) => { draft.audio_crossfade_ms = Math.max(0, Math.min(200, Number(e.target.value) || 0)); },
      }),
    ]),
    el("div", { class: "sv-row", style: { gap: "14px" } }, [
      el("label", { class: "sv-row", style: { gap: "4px", fontSize: "11.5px", cursor: "pointer" } }, [
        el("input", { type: "checkbox", checked: Boolean(draft.export_fcpxml), onchange: (e) => { draft.export_fcpxml = e.target.checked; } }),
        t("exp_fcpxml"),
      ]),
      hasSeed
        ? el("label", { class: "sv-row", style: { gap: "4px", fontSize: "11.5px", cursor: "pointer" } }, [
            el("input", { type: "checkbox", checked: Boolean(draft.prepend_seed_video), onchange: (e) => { draft.prepend_seed_video = e.target.checked; } }),
            t("exp_seed"),
          ])
        : null,
    ]),
  ]);

  const previewRel = resData?.master_rel || (existingMasters[0]?.rel || "");
  const previewFcpRel = resData?.fcpxml_rel || ((state.serverState?.master_exports || []).find((x) => x.ext === ".fcpxml")?.rel || "");
  const previewAbs = resData?.master_mp4 || "";
  const bust = state.mediaBust || Date.now();
  const isStaleDiskPreview = Boolean(previewRel && !resData && syncStatus.is_outdated);
  const resBox = previewRel
    ? el("div", { class: "sv-box", style: { gap: "6px" } }, [
        el("div", { class: "sv-row", style: { justifyContent: "space-between" } }, [
          el("span", {
            style: { fontWeight: "700", fontSize: "11.5px", color: isStaleDiskPreview ? "#ffd08c" : "#8ff0c2" },
            text: resData
              ? `✅ 刚刚导出：${previewRel} (${fmtBytes(resData.file_bytes || 0)} · ${Number(resData.report?.expected_duration_sec || 0).toFixed(1)}s)`
              : (isStaleDiskPreview ? `⚠ 磁盘旧成片（未含最新重抽/修复）：${previewRel}` : `🎞 当前磁盘成片：${previewRel}`),
          }),
          el("div", { class: "sv-row" }, [
            previewAbs
              ? el("button", {
                  class: "sv-btn",
                  text: t("exp_copy_path"),
                  onclick: async () => {
                    try { await navigator.clipboard.writeText(previewAbs); showToast(t("exp_copied")); } catch { showToast(previewAbs, 3000); }
                  },
                })
              : null,
            el("a", {
              class: `sv-btn ${isStaleDiskPreview ? "warn" : "pri"}`,
              style: { textDecoration: "none" },
              href: `${mediaUrl(previewRel, proj)}&download=1`,
              download: previewRel.split("/").pop(),
              text: isStaleDiskPreview ? t("exp_dl_mp4_old") : t("exp_dl_mp4"),
            }),
            previewFcpRel
              ? el("a", {
                  class: "sv-btn act",
                  style: { textDecoration: "none" },
                  href: `${mediaUrl(previewFcpRel, proj)}&download=1`,
                  download: previewFcpRel.split("/").pop(),
                  text: t("exp_dl_fcpxml"),
                })
              : null,
          ]),
        ]),
        previewAbs ? el("div", { style: { fontSize: "10.5px", color: "#8f98ae", wordBreak: "break-all" }, text: previewAbs }) : null,
        el("video", {
          src: `${mediaUrl(previewRel, proj)}&t=${bust}`,
          controls: true,
          preload: "metadata",
          style: { width: "100%", maxHeight: "260px", background: "#090a0d", borderRadius: "6px" },
        }),
      ])
    : null;

  const mBody = el("div", { class: "sv-modal-body" }, [outdatedBanner, summaryBox, optBox, resBox]);
  const mFtr = el("div", { class: "sv-modal-ftr" }, [
    el("button", {
      class: `sv-btn ${stillOutdated ? "pri" : "ok"}`,
      disabled: state.exportBusy || renderedShots.length === 0,
      text: state.exportBusy ? t("exp_btn_running") : t("exp_btn_run"),
      onclick: runExportNow,
    }),
    el("button", { class: "sv-btn", text: t("cl_close"), onclick: () => { state.exportOpen = false; renderBody(); } }),
  ]);

  return el("div", { class: "sv-modal-bg", onclick: (e) => { if (e.target === e.currentTarget && !state.exportBusy) { state.exportOpen = false; renderBody(); } } }, [
    el("div", { class: "sv-modal" }, [
      el("div", { class: "sv-modal-hdr" }, [
        el("span", { text: t("exp_title") }),
        el("button", { class: "sv-btn", text: "×", onclick: () => { if (!state.exportBusy) { state.exportOpen = false; renderBody(); } } }),
      ]),
      mBody,
      mFtr,
    ]),
  ]);
}
