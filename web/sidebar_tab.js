import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { toggleLang, t } from "./i18n.js";

const TAB_ID = "shotvault.director";
let activeRefreshFn = null;
let activeContainer = null;

function createEl(tag, attrs = {}, children = []) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style" && typeof v === "object") {
      Object.assign(el.style, v);
    } else if (k.startsWith("on") && typeof v === "function") {
      el.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== undefined && v !== null) {
      el.setAttribute(k, String(v));
    }
  }
  for (const ch of Array.isArray(children) ? children : [children]) {
    if (typeof ch === "string") el.appendChild(document.createTextNode(ch));
    else if (ch) el.appendChild(ch);
  }
  return el;
}

function statusBadge(status, promptEdited) {
  const colors = {
    VALID: "#2e7d32",
    STALE: "#c62828",
    LOCKED_KEEP: "#ef6c00",
    UNRENDERED: "#546e7a",
  };
  const wrap = createEl("span", { style: { display: "inline-flex", gap: "4px", alignItems: "center" } });
  wrap.appendChild(createEl("span", {
    style: {
      fontSize: "10px", fontWeight: "700", padding: "2px 6px", borderRadius: "4px",
      background: colors[status] || "#455a64", color: "#fff",
    },
  }, status || "UNKNOWN"));
  if (promptEdited) {
    wrap.appendChild(createEl("span", {
      style: {
        fontSize: "10px", padding: "2px 5px", borderRadius: "4px",
        background: "#3949ab", color: "#e8eaf6",
      },
    }, "PROMPT_EDITED"));
  }
  return wrap;
}

async function queueShotsNonDestructive(shotSpec, startMode = "manual_picks", customSeedText = "") {
  const p = await app.graphToPrompt();
  const promptObj = p?.output || {};
  let patched = false;
  for (const node of Object.values(promptObj)) {
    if (!node || node.class_type !== "ShotVaultLoad") continue;
    node.inputs = node.inputs || {};
    node.inputs.start_mode = startMode;
    node.inputs.shot_numbers = String(shotSpec);
    if (customSeedText) {
      node.inputs.seed_mode = "custom_string";
      node.inputs.seed_text = String(customSeedText);
    }
    patched = true;
  }
  if (!patched) {
    const isSingle = startMode === "manual_picks" && /^\d+$/.test(String(shotSpec).trim());
    const targetIdx = isSingle ? Math.max(1, parseInt(String(shotSpec).trim(), 10)) : 1;
    for (const node of Object.values(promptObj)) {
      if (!node || node.class_type !== "ShotVaultPlan") continue;
      node.inputs = node.inputs || {};
      let dplan = {};
      try { dplan = JSON.parse(String(node.inputs.director_plan_json || "{}") || "{}"); } catch { dplan = {}; }
      dplan.retry_lock = {
        ...(dplan.retry_lock || {}),
        enabled: isSingle,
        shot_index: targetIdx,
        seed_mode: customSeedText ? "fixed" : (dplan.retry_lock?.seed_mode || "randomize"),
        seed_text: customSeedText ? String(customSeedText) : (dplan.retry_lock?.seed_text || ""),
      };
      node.inputs.director_plan_json = JSON.stringify(dplan);
      patched = true;
    }
    if (patched && customSeedText) {
      for (const node of Object.values(promptObj)) {
        if (!node || node.class_type !== "ShotVaultLoopDispatch") continue;
        node.inputs = node.inputs || {};
        node.inputs.seed_mode = "custom_string";
        node.inputs.seed_text = String(customSeedText);
      }
    }
  }
  if (!patched) return false;
  await api.fetchApi("/shotvault/resume-auto", { method: "POST" }).catch(() => {});
  await api.queuePrompt(0, p);
  return true;
}

function buildSidebarDom(container) {
  container.innerHTML = "";
  Object.assign(container.style, {
    height: "100%", overflowY: "auto", padding: "10px", boxSizing: "border-box",
    fontFamily: "var(--comfy-font, sans-serif)", fontSize: "12px", color: "var(--input-text, #ddd)",
  });

  let currentProject = "MyH3Film";
  const titleEl = createEl("div", { style: { fontWeight: "700", fontSize: "13px", marginBottom: "6px" } }, t("sb_title"));
  const projSelect = createEl("select", { style: { flex: "1", padding: "4px 6px", borderRadius: "4px" } });
  const statusLine = createEl("div", { style: { fontSize: "11px", opacity: "0.85", margin: "6px 0" } }, t("sb_ready"));

  const playerBox = createEl("div", { style: { display: "none", marginBottom: "8px", border: "1px solid #444", borderRadius: "6px", padding: "6px", background: "#111" } });
  const videoEl = createEl("video", { controls: "true", preload: "none", style: { width: "100%", maxHeight: "220px", borderRadius: "4px" } });
  const playerTitle = createEl("div", { style: { fontSize: "11px", marginBottom: "4px" } });
  const closePlayer = () => {
    videoEl.pause();
    videoEl.removeAttribute("src");
    videoEl.load();
    playerBox.style.display = "none";
  };
  const closePlayerBtn = createEl("button", { onClick: closePlayer, style: { cursor: "pointer", padding: "2px 6px" } }, t("sb_close"));
  playerBox.appendChild(createEl("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center" } }, [
    playerTitle,
    closePlayerBtn,
  ]));
  playerBox.appendChild(videoEl);

  const listBox = createEl("div", { style: { display: "flex", flexDirection: "column", gap: "8px" } });

  async function refreshState() {
    try {
      const pRes = await api.fetchApi("/shotvault/projects");
      const pJson = await pRes.json();
      const projects = pJson.projects || [];
      if (projects.length && !projects.includes(currentProject)) {
        currentProject = projects[0];
      }
      projSelect.innerHTML = "";
      for (const name of projects.length ? projects : [currentProject]) {
        const opt = createEl("option", { value: name }, name);
        if (name === currentProject) opt.selected = true;
        projSelect.appendChild(opt);
      }

      const sRes = await api.fetchApi(`/shotvault/state?project=${encodeURIComponent(currentProject)}`);
      const state = await sRes.json();
      const shots = state.shots || [];
      const staleList = shots.filter((s) => s.status === "STALE" || s.status === "UNRENDERED").map((s) => s.shot_index);
      statusLine.textContent = t("sb_status_line", { proj: state.project_name, n: shots.length, stale: staleList.length });
      renderShots(state);
    } catch (err) {
      statusLine.textContent = t("sb_refresh_fail", { msg: err?.message || err });
    }
  }

  function renderShots(state) {
    listBox.innerHTML = "";
    const shots = state.shots || [];
    for (const sh of shots) {
      const card = createEl("div", {
        style: { border: "1px solid var(--border-color, #444)", borderRadius: "6px", padding: "8px", background: "rgba(255,255,255,0.03)" },
      });
      const typeName = sh.shot_type === 0 ? t("sb_type_drama") : (sh.shot_type === 2 ? t("sb_type_mv") : t("sb_type_action"));
      card.appendChild(createEl("div", { style: { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "4px" } }, [
        createEl("strong", {}, `#${sh.shot_index} · ${typeName} (${Number(sh.duration_sec || 0).toFixed(2)}s)`),
        statusBadge(sh.status, sh.prompt_edited),
      ]));

      if (sh.local_prompt) {
        card.appendChild(createEl("div", { style: { fontSize: "11px", opacity: "0.85", marginBottom: "6px", wordBreak: "break-word" } }, sh.local_prompt.slice(0, 110)));
      }

      const activeTake = (sh.takes || []).find((tk) => tk.take_id === sh.active_take_id);
      if (activeTake?.thumb_rel) {
        card.appendChild(createEl("img", {
          src: `/shotvault/media?project=${encodeURIComponent(currentProject)}&file=${encodeURIComponent(activeTake.thumb_rel)}`,
          style: { width: "100%", maxHeight: "120px", objectFit: "cover", borderRadius: "4px", marginBottom: "6px" },
        }));
      }

      if ((sh.takes || []).length > 0) {
        const takeSel = createEl("select", { style: { width: "100%", marginBottom: "6px", padding: "3px" } });
        for (const tk of sh.takes) {
          const opt = createEl("option", { value: tk.take_id }, `${tk.take_id} (seed:${tk.seed_text || "-"})`);
          if (tk.take_id === sh.active_take_id) opt.selected = true;
          takeSel.appendChild(opt);
        }
        takeSel.addEventListener("change", async () => {
          await api.fetchApi("/shotvault/select-take", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ project: currentProject, shot_index: sh.shot_index, take_id: takeSel.value }),
          });
          refreshState();
        });
        card.appendChild(takeSel);
      }

      const actions = createEl("div", { style: { display: "flex", gap: "4px", flexWrap: "wrap", alignItems: "center" } });
      actions.appendChild(createEl("button", {
        onClick: async () => {
          try {
            const ok = await queueShotsNonDestructive(String(sh.shot_index), "manual_picks");
            if (!ok) statusLine.textContent = t("sb_no_node");
          } catch (err) {
            statusLine.textContent = t("sb_queue_fail", { msg: err?.message || err });
          }
        },
        style: { cursor: "pointer", padding: "3px 7px", fontSize: "11px" },
      }, t("sb_btn_reroll")));

      if (activeTake?.seam_rel) {
        actions.appendChild(createEl("button", {
          onClick: () => {
            playerTitle.textContent = t("sb_orig_seam_title", { id: activeTake.take_id });
            videoEl.src = `/shotvault/media?project=${encodeURIComponent(currentProject)}&file=${encodeURIComponent(activeTake.seam_rel)}`;
            playerBox.style.display = "block";
            videoEl.play().catch(() => {});
          },
          style: { cursor: "pointer", padding: "3px 7px", fontSize: "11px" },
        }, t("sb_btn_orig_3s")));
      }
      if (sh.seam_repair?.repaired_seam_rel) {
        actions.appendChild(createEl("button", {
          onClick: () => {
            playerTitle.textContent = t("sb_rep_seam_title", { id: activeTake?.take_id || sh.shot_index });
            videoEl.src = `/shotvault/media?project=${encodeURIComponent(currentProject)}&file=${encodeURIComponent(sh.seam_repair.repaired_seam_rel)}&cb=${Date.now()}`;
            playerBox.style.display = "block";
            videoEl.play().catch(() => {});
          },
          style: { cursor: "pointer", padding: "3px 7px", fontSize: "11px" },
        }, t("sb_btn_rep_3s")));
      }
      if (activeTake?.video_rel) {
        actions.appendChild(createEl("button", {
          onClick: () => {
            playerTitle.textContent = t("sb_full_title", { id: activeTake.take_id });
            videoEl.src = `/shotvault/media?project=${encodeURIComponent(currentProject)}&file=${encodeURIComponent(activeTake.video_rel)}`;
            playerBox.style.display = "block";
            videoEl.play().catch(() => {});
          },
          style: { cursor: "pointer", padding: "3px 7px", fontSize: "11px" },
        }, t("sb_btn_full")));
      }

      const lockCb = createEl("input", { type: "checkbox" });
      lockCb.checked = !!sh.locked_keep;
      lockCb.addEventListener("change", async () => {
        await api.fetchApi("/shotvault/lock-shot", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ project: currentProject, shot_index: sh.shot_index, locked: lockCb.checked }),
        });
        refreshState();
      });
      actions.appendChild(createEl("label", { style: { fontSize: "11px", display: "inline-flex", alignItems: "center", gap: "3px", marginLeft: "auto" } }, [lockCb, t("sb_lbl_lock")]));

      card.appendChild(actions);
      listBox.appendChild(card);
    }
  }

  projSelect.addEventListener("change", () => {
    currentProject = projSelect.value;
    refreshState();
  });

  const langBtn = createEl("button", {
    onClick: () => { toggleLang(); },
    style: { cursor: "pointer", padding: "4px 7px" },
  }, t("btn_lang"));
  const refreshBtn = createEl("button", { onClick: refreshState, style: { cursor: "pointer", padding: "4px 8px" } }, t("sb_btn_refresh"));
  const cleanBtn = createEl("button", {
    onClick: async () => {
      await api.fetchApi("/shotvault/cleanup-recycle", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: currentProject }),
      });
      refreshState();
    },
    style: { cursor: "pointer", padding: "4px 8px" },
  }, t("sb_btn_clean"));

  const resumeBtn = createEl("button", {
    onClick: async () => {
      try {
        const ok = await queueShotsNonDestructive("1", "sequence_auto");
        if (!ok) statusLine.textContent = t("sb_no_node");
      } catch (err) {
        statusLine.textContent = t("sb_queue_fail", { msg: err?.message || err });
      }
    },
    style: { flex: "1", cursor: "pointer", padding: "5px", fontWeight: "600" },
  }, t("sb_btn_resume"));

  const rerunAllBtn = createEl("button", {
    onClick: async () => {
      try {
        const ok = await queueShotsNonDestructive("0", "sequence_auto");
        if (!ok) statusLine.textContent = t("sb_no_node");
      } catch (err) {
        statusLine.textContent = t("sb_queue_fail", { msg: err?.message || err });
      }
    },
    style: { cursor: "pointer", padding: "5px" },
  }, t("sb_btn_rerun_all"));

  const assembleBtn = createEl("button", {
    onClick: async () => {
      statusLine.textContent = t("sb_assembling");
      const res = await api.fetchApi("/shotvault/assemble", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project: currentProject, stale_policy: "block_and_report" }),
      });
      const data = await res.json();
      statusLine.textContent = data.ok ? t("sb_assembled_ok", { path: data.master_mp4 }) : t("sb_assemble_blocked", { msg: data.error || t("sb_unknown_err") });
    },
    style: { cursor: "pointer", padding: "5px", fontWeight: "600" },
  }, t("sb_btn_assemble"));

  const abortBtn = createEl("button", {
    onClick: async () => {
      await api.fetchApi("/shotvault/cancel-auto", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true }),
      });
      statusLine.textContent = t("sb_aborted");
    },
    style: { cursor: "pointer", padding: "5px", color: "#ff8a80" },
  }, t("sb_btn_abort"));

  if (container._svLangHandler) {
    window.removeEventListener("acheng:lang-change", container._svLangHandler);
  }
  container._svLangHandler = () => {
    titleEl.textContent = t("sb_title");
    closePlayerBtn.textContent = t("sb_close");
    langBtn.textContent = t("btn_lang");
    refreshBtn.textContent = t("sb_btn_refresh");
    cleanBtn.textContent = t("sb_btn_clean");
    resumeBtn.textContent = t("sb_btn_resume");
    rerunAllBtn.textContent = t("sb_btn_rerun_all");
    assembleBtn.textContent = t("sb_btn_assemble");
    abortBtn.textContent = t("sb_btn_abort");
    refreshState();
  };
  window.addEventListener("acheng:lang-change", container._svLangHandler);

  const toolbar = createEl("div", { style: { display: "flex", gap: "4px", flexWrap: "wrap", marginBottom: "6px" } }, [
    projSelect,
    langBtn,
    refreshBtn,
    cleanBtn,
  ]);

  const batchBar = createEl("div", { style: { display: "flex", gap: "4px", flexWrap: "wrap", marginBottom: "6px" } }, [
    resumeBtn,
    rerunAllBtn,
    assembleBtn,
    abortBtn,
  ]);

  container.appendChild(titleEl);
  container.appendChild(toolbar);
  container.appendChild(batchBar);
  container.appendChild(statusLine);
  container.appendChild(playerBox);
  container.appendChild(listBox);

  activeContainer = container;
  activeRefreshFn = refreshState;
  refreshState();
}

app.registerExtension({
  name: "ComfyUI.Universal.ShotVault",
  async setup() {
    if (app.extensionManager?.registerSidebarTab) {
      app.extensionManager.registerSidebarTab({
        id: TAB_ID,
        icon: "pi pi-film",
        title: "ShotVault",
        tooltip: t("sb_tooltip"),
        type: "custom",
        render(el) {
          buildSidebarDom(el);
        },
      });
    }

    const abortScoped = (ev) => {
      const pid = ev?.detail?.prompt_id || "";
      api.fetchApi("/shotvault/cancel-auto", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt_id: pid, force: false }),
      }).catch(() => {});
    };
    api.addEventListener("execution_interrupted", abortScoped);
    api.addEventListener("execution_error", abortScoped);
    api.addEventListener("executed", (ev) => {
      const out = ev?.detail?.output;
      if (out?.shotvault_saved || out?.shotvault_assembled || out?.shotvault_deferred) {
        if (activeContainer?.isConnected && typeof activeRefreshFn === "function") {
          activeRefreshFn();
        }
      }
    });
  },
});
