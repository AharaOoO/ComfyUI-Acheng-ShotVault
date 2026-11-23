/**
 * Viewport-level Focus Prompt Editor modal (`position: fixed`, immune to ComfyUI canvas zoom).
 * Supports quick tag insertion (<Picture 1..9>, @cut@, @continue@, & 8 &), live stats, and Ctrl+Enter save.
 */

export function openFocusPromptModal({
  title,
  value,
  placeholder = "",
  chips = [],
  onSave,
  el,
  t,
}) {
  const existing = document.getElementById("sv-focus-prompt-overlay");
  if (existing) existing.remove();

  let currentText = String(value ?? "");
  const ta = el("textarea", {
    class: "sv-focus-ta",
    placeholder,
  });
  ta.value = currentText;

  const statEl = el("span", { style: { fontSize: "11.5px", color: "#9da6bc" } });
  const updateStat = () => {
    const lines = currentText ? currentText.split(/\r?\n/).length : 0;
    statEl.textContent = t("focus_stats", { chars: currentText.length, lines });
  };
  updateStat();

  ta.addEventListener("input", () => {
    currentText = ta.value;
    updateStat();
  });

  const close = () => {
    window.removeEventListener("keydown", onKey, true);
    overlay.remove();
  };
  const saveAndClose = () => {
    try {
      onSave?.(ta.value);
    } finally {
      close();
    }
  };
  const onKey = (e) => {
    if (e.key === "Escape") {
      e.stopPropagation();
      e.preventDefault();
      close();
    } else if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.stopPropagation();
      e.preventDefault();
      saveAndClose();
    }
  };

  const chipRow = chips.length
    ? el("div", { class: "sv-chips" }, chips.map((tag) => el("button", {
        class: "sv-chip",
        text: tag,
        onclick: () => {
          const st = ta.selectionStart ?? ta.value.length;
          const en = ta.selectionEnd ?? ta.value.length;
          const next = `${ta.value.slice(0, st)}${tag}${ta.value.slice(en)}`;
          ta.value = next;
          currentText = next;
          ta.focus();
          ta.setSelectionRange(st + tag.length, st + tag.length);
          updateStat();
        },
      })))
    : null;

  const modal = el("div", { class: "sv-focus-modal" }, [
    el("div", { class: "sv-modal-hdr" }, [
      el("span", { text: title }),
      el("div", { class: "sv-row" }, [
        statEl,
        el("button", { class: "sv-btn", text: "×", onclick: close }),
      ]),
    ]),
    el("div", { class: "sv-modal-body", style: { flex: "1", display: "flex", flexDirection: "column", gap: "8px", overflow: "hidden" } }, [
      chipRow,
      ta,
    ]),
    el("div", { class: "sv-modal-ftr" }, [
      el("span", { style: { fontSize: "11px", color: "#8e97ad" }, text: t("focus_hint") }),
      el("div", { class: "sv-row" }, [
        el("button", { class: "sv-btn", text: t("focus_cancel"), onclick: close }),
        el("button", { class: "sv-btn pri", text: t("focus_save"), onclick: saveAndClose }),
      ]),
    ]),
  ]);

  const overlay = el("div", {
    id: "sv-focus-prompt-overlay",
    class: "sv-focus-bg",
    onclick: (e) => { if (e.target === overlay) close(); },
  }, [modal]);

  // Stop canvas hotkeys/panning from ever interfering inside viewport modal
  for (const evName of ["pointerdown", "mousedown", "wheel", "keydown", "keyup"]) {
    overlay.addEventListener(evName, (e) => e.stopPropagation());
  }
  window.addEventListener("keydown", onKey, true);
  document.body.appendChild(overlay);
  setTimeout(() => { ta.focus(); }, 20);
}
