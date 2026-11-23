/**
 * ShotVault v5.8 responsive stylesheet & layout sizing constants.
 * Lives under `web/panel/` so ComfyUI does not auto-execute it as a standalone extension.
 */

export const STYLE_ID = "shotvault-planner-v58-css";

export const PANEL_DEFAULTS = {
  MIN_W: 980,
  MAX_W: 1860,
  DEF_W: 1180,
  MIN_H: 660,
  MAX_H: 1400,
  DEF_H: 840,
  CHROME_H: 88,
  WIDE_BP: 1380,
  PRESETS: {
    compact: { w: 1020, h: 720 },
    standard: { w: 1180, h: 840 },
    wide: { w: 1460, h: 960 },
  },
};

export function injectStyles() {
  for (const oldId of ["shotvault-director-v51-css", "shotvault-director-v54-css"]) {
    const old = document.getElementById(oldId);
    if (old) old.remove();
  }
  if (document.getElementById(STYLE_ID)) return;
  const st = document.createElement("style");
  st.id = STYLE_ID;
  st.textContent = `
.sv-host{position:relative;width:100%;height:${PANEL_DEFAULTS.DEF_H}px;min-width:${PANEL_DEFAULTS.MIN_W - 24}px;min-height:${PANEL_DEFAULTS.MIN_H}px;overflow:hidden}
.sv-root{position:absolute;inset:0;box-sizing:border-box;width:100%;height:100%;display:flex;flex-direction:column;background:#15161b;color:#e6e8ef;font:12px/1.45 "Inter","Microsoft YaHei",system-ui,sans-serif;border:1px solid #2b2f3b;border-radius:10px;overflow:hidden;user-select:none}
.sv-root *{box-sizing:border-box}
.sv-root [hidden]{display:none !important}
.sv-hdr{flex:none;display:flex;flex-wrap:wrap;align-items:center;gap:5px;padding:7px 10px;background:#1b1d24;border-bottom:1px solid #2a2e3a}
.sv-title{font-weight:700;font-size:13px;color:#fff;display:flex;align-items:center;gap:6px;white-space:nowrap}
.sv-badge{font-size:10px;padding:2px 6px;border-radius:4px;background:#252936;color:#9cb3ff;border:1px solid #353b4e;white-space:nowrap}
.sv-hdr-sp{flex:1;min-width:6px}
.sv-inp,.sv-sel,.sv-ta{background:#101116;color:#e6e8ef;border:1px solid #2e3340;border-radius:6px;padding:4px 7px;font:inherit;outline:none;user-select:text}
.sv-inp:focus,.sv-sel:focus,.sv-ta:focus{border-color:#3a76ff}
.sv-ta{font-size:12.5px;line-height:1.55;padding:7px 9px;width:100%;resize:none}
.sv-ta-wrap{display:flex;flex-direction:column;width:100%}
.sv-ta-wrap .sv-ta{border-radius:6px 6px 0 0;border-bottom-color:#262a36}
.sv-ta-grip{height:11px;width:100%;background:#1a1c25;border:1px solid #2e3340;border-top:none;border-radius:0 0 6px 6px;cursor:ns-resize;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .12s}
.sv-ta-grip:hover,.sv-ta-grip.dragging{background:#242c42;border-color:#3a76ff}
.sv-ta-grip::after{content:"";width:38px;height:3px;border-radius:2px;background:#48516b}
.sv-ta-grip:hover::after,.sv-ta-grip.dragging::after{background:#7aa7ff}
.sv-btn{background:#232631;color:#dce1ec;border:1px solid #343847;border-radius:6px;padding:4px 9px;font:11.5px/1.3 inherit;cursor:pointer;display:inline-flex;align-items:center;gap:4px;white-space:nowrap;transition:background .1s,border-color .1s}
.sv-btn:hover{background:#2d3242;border-color:#4b536b}
.sv-btn:disabled{opacity:.48;cursor:not-allowed}
.sv-btn.pri{background:#2f6df6;border-color:#4580ff;color:#fff;font-weight:600}
.sv-btn.pri:hover{background:#3b78ff}
.sv-btn.act{background:#24355a;border-color:#356ae0;color:#c3d6ff}
.sv-btn.warn{background:#472b11;border-color:#c67a22;color:#ffd7a1;font-weight:600}
.sv-btn.ok{background:#143627;border-color:#218559;color:#8ff0c3;font-weight:600}
.sv-btn.dng{background:#3b1b22;border-color:#8a2f3d;color:#ffb4bf}
.sv-btn.xs{padding:2px 6px;font-size:10.5px}
.sv-tabs{flex:none;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:5px;padding:6px 10px;background:#181920;border-bottom:1px solid #2a2d36}
.sv-tab{padding:6px 8px;text-align:center;border-radius:6px;cursor:pointer;background:#1d1f28;color:#9aa2b6;border:1px solid #292c38;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sv-tab:hover{color:#dce3f5;border-color:#3a4054}
.sv-tab.on{background:#2f6df6;color:#fff;border-color:#4e86ff}
.sv-body{flex:1;min-height:0;overflow-y:auto;overflow-x:hidden;overscroll-behavior:contain;padding:8px 10px;display:flex;flex-direction:column;gap:8px}
.sv-box{background:#1d1e25;border:1px solid #2a2e3a;border-radius:8px;padding:8px 10px;display:flex;flex-direction:column;gap:6px}
.sv-box.reroll-on{border-color:#3a76ff;box-shadow:0 0 0 1px rgba(58,118,255,.35) inset;background:#191f2e}
.sv-box-hdr{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;font-weight:600;color:#d6dbee}
.sv-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.sv-strip{display:flex;gap:7px;overflow-x:auto;padding:4px 2px;scrollbar-width:thin}
.sv-card{min-width:158px;max-width:158px;background:#181920;border:1px solid #2c303c;border-radius:7px;padding:6px;cursor:pointer;display:flex;flex-direction:column;gap:4px;flex-shrink:0}
.sv-card.on{border-color:#2f6df6;box-shadow:0 0 0 1px #2f6df6 inset;background:#1b2030}
.sv-thumb{width:100%;height:74px;background:#0d0e13;border-radius:5px;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#666d80;font-size:11px;position:relative}
.sv-thumb img{width:100%;height:100%;object-fit:contain;display:block}
.sv-tag-a{background:#472909;color:#ffbe6b;border:1px solid #8c5012;border-radius:4px;padding:1px 5px;font-size:10px;font-weight:700}
.sv-tag-d{background:#1f2152;color:#b2b9ff;border:1px solid #484cc4;border-radius:4px;padding:1px 5px;font-size:10px;font-weight:700}
.sv-shot-split{display:flex;flex-direction:column;gap:8px}
.sv-root.sv-w-wide .sv-shot-split{display:grid;grid-template-columns:minmax(0,1.2fr) minmax(360px,1fr);gap:10px;align-items:start}
.sv-shot-pane{display:flex;flex-direction:column;gap:6px;min-width:0}
.sv-grid9{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px}
.sv-slot{background:#15161c;border:1px dashed #363b4a;border-radius:6px;padding:5px;display:flex;flex-direction:column;gap:4px;position:relative}
.sv-slot.has{border-style:solid;border-color:#444b5e}
.sv-slot-img{width:100%;height:76px;background:#0d0e13;border-radius:4px;overflow:hidden;display:flex;align-items:center;justify-content:center;color:#6c748a;font-size:10px;cursor:pointer}
.sv-slot-img img{width:100%;height:100%;object-fit:contain;display:block}
.sv-slot-ft{display:flex;align-items:center;justify-content:space-between;font-size:10px;color:#9da5ba;gap:3px}
.sv-chips{display:flex;flex-wrap:wrap;gap:4px}
.sv-chip{background:#232735;color:#b9c7ef;border:1px solid #353b50;border-radius:4px;padding:2px 6px;font-size:10.5px;cursor:pointer}
.sv-chip:hover{background:#2f6df6;color:#fff}
.sv-grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}
.sv-grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.sv-telem{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:5px;background:#13141a;border:1px solid #272b37;border-radius:6px;padding:6px;font-size:11px}
.sv-adv summary{cursor:pointer;color:#9cb3ff;font-weight:600;padding:4px 0}
.sv-tbl{width:100%;border-collapse:collapse;font-size:11.5px}
.sv-tbl th,.sv-tbl td{border:1px solid #2a2d36;padding:4px 7px;text-align:left}
.sv-tbl th{background:#181920;color:#aeb7cf}
.sv-ftr{flex:none;padding:5px 10px;background:#181920;border-top:1px solid #2a2d36;display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:11px;color:#9ba3b8}
.sv-link-lbl{font-size:10.5px;color:#7fe0b6;background:#143025;border:1px solid #21664c;padding:2px 6px;border-radius:4px}
.sv-toast{position:fixed;right:18px;bottom:18px;background:#2f6df6;color:#fff;padding:7px 12px;border-radius:7px;font-size:12px;z-index:100020;box-shadow:0 4px 16px rgba(0,0,0,.45)}
.sv-repair-bar{background:#14161e;border:1px solid #2b3145;border-radius:7px;padding:7px 8px;display:flex;flex-direction:column;gap:6px}
.sv-chk-row{display:flex;align-items:center;justify-content:space-between;gap:6px;padding:4px 6px;background:#14151b;border-radius:6px;border:1px solid #272a34}
.sv-chk-row.active{border-color:#2f6df6}
.sv-guide-strip{padding:6px 8px;border-radius:6px;background:#13213d;border:1px solid #2f5cc0;color:#c3daff;font-size:11px;line-height:1.45}
.sv-outdated-banner{padding:8px 10px;border-radius:7px;background:#3c2310;border:1px solid #cc7b22;color:#ffe0b2;font-size:11.5px;line-height:1.5;font-weight:600}
.sv-fresh-banner{padding:7px 10px;border-radius:7px;background:#123425;border:1px solid #22875b;color:#9df5cd;font-size:11.5px;font-weight:600}
.sv-modal-bg{position:absolute;inset:0;background:rgba(10,11,16,.84);z-index:200;display:flex;align-items:center;justify-content:center;padding:14px}
.sv-modal{width:100%;max-width:880px;max-height:94%;background:#1a1c24;border:1px solid #3a4158;border-radius:10px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 12px 36px rgba(0,0,0,.6)}
.sv-modal-hdr{display:flex;align-items:center;justify-content:space-between;padding:9px 12px;background:#20232e;border-bottom:1px solid #2f3547;font-weight:700;color:#fff}
.sv-modal-body{flex:1;min-height:0;overflow-y:auto;padding:10px 12px;display:flex;flex-direction:column;gap:8px}
.sv-modal-ftr{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:8px 12px;background:#181a22;border-top:1px solid #2f3547}
.sv-pill-prot{font-size:10px;padding:1px 5px;border-radius:4px;background:#282c3d;color:#9ea9c6;border:1px solid #3a4058}
.sv-focus-bg{position:fixed;inset:0;background:rgba(7,8,12,.82);z-index:100000;display:flex;align-items:center;justify-content:center;padding:24px;backdrop-filter:blur(2px)}
.sv-focus-modal{width:min(1120px,84vw);height:min(780px,82vh);background:#181a22;border:1px solid #3f4b6a;border-radius:12px;display:flex;flex-direction:column;overflow:hidden;box-shadow:0 20px 60px rgba(0,0,0,.75);color:#e6e8ef;font:13px/1.5 "Inter","Microsoft YaHei",system-ui,sans-serif}
.sv-focus-ta{flex:1;min-height:0;width:100%;background:#0f1015;color:#eef1f8;border:1px solid #2e3445;border-radius:8px;padding:12px 14px;font:13.5px/1.65 "Consolas","JetBrains Mono","Microsoft YaHei",monospace;outline:none;resize:none;user-select:text}
.sv-focus-ta:focus{border-color:#3a76ff}
.sv-size-bar{display:inline-flex;align-items:center;gap:3px}
`;
  document.head.appendChild(st);
}
