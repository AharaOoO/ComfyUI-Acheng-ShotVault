"""Phase 13: Exhaustive i18n zero-leak guard, 10-node canvas i18n, backend bilingual contracts, and B1-B4 UX audit tests."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import types
import unittest

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_NAME = "sv_p13_pkg"
if PKG_NAME not in sys.modules:
    _pkg = types.ModuleType(PKG_NAME)
    _pkg.__path__ = [REPO_ROOT]
    _pkg.__package__ = PKG_NAME
    sys.modules[PKG_NAME] = _pkg

from sv_p13_pkg.core.seam_qc import analyze_seam_diagnostics  # type: ignore
from sv_p13_pkg.core.vault_store import VaultStore, compute_master_sync_status  # type: ignore
from sv_p13_pkg.nodes.plan_node import ShotVaultPlan  # type: ignore

CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def _read_web(rel: str) -> str:
    with open(os.path.join(REPO_ROOT, "web", rel), "r", encoding="utf-8") as f:
        return f.read()


class TestPhase13I18nAndUxAudit(unittest.TestCase):
    def test_i18n_dict_symmetry_and_english_purity(self):
        txt = _read_web("i18n.js")
        m = re.search(r"zh\s*:\s*\{(.*?)\n\s*\}\s*,\s*\n\s*en\s*:\s*\{(.*?)\n\s*\}", txt, re.S)
        self.assertIsNotNone(m, "Failed to locate zh/en blocks in web/i18n.js")
        zh_block, en_block = m.group(1), m.group(2)
        zh_keys = set(re.findall(r"^\s*([a-zA-Z0-9_]+)\s*:", zh_block, re.M))
        en_keys = set(re.findall(r"^\s*([a-zA-Z0-9_]+)\s*:", en_block, re.M))

        self.assertGreaterEqual(len(zh_keys), 250)
        self.assertEqual(zh_keys, en_keys, f"Key mismatch: zh-en={zh_keys - en_keys}, en-zh={en_keys - zh_keys}")

        # Zero Chinese characters inside English dictionary values (except btn_lang: "🌐 中文")
        bad_en = []
        for line in en_block.splitlines():
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("btn_lang:"):
                continue
            if CJK_RE.search(s):
                bad_en.append(s)
        self.assertEqual(bad_en, [], f"Found Chinese in EN dictionary entries: {bad_en}")

        # Verify smart initial locale detection & pickOutdatedReasons helper exist
        self.assertIn("Comfy.Settings.Comfy.Locale", txt)
        self.assertIn("navigator?.language", txt)
        self.assertIn("export function pickOutdatedReasons", txt)

        # Verify every t("key") call across all frontend JS files exists in both zh and en
        missing = []
        for rel in ("director_panel.js", "sidebar_tab.js", "panel/export_modal.js", "panel/prompt_modal.js"):
            js = _read_web(rel)
            for km in re.finditer(r'\bt\(\s*["\']([a-zA-Z0-9_]+)["\']', js):
                k = km.group(1)
                if k not in zh_keys or k not in en_keys:
                    missing.append((rel, k))
        self.assertEqual(missing, [], f"Missing i18n keys referenced in UI: {missing}")

    def test_zero_hardcoded_chinese_ui_strings_across_frontend(self):
        """Static guard: zero non-localized Chinese UI strings in any frontend JS outside i18n.js/node_i18n.js."""
        targets = (
            "director_panel.js",
            "sidebar_tab.js",
            "panel/export_modal.js",
            "panel/prompt_modal.js",
            "panel/styles.js",
        )
        # Only bilingual script-syntax regex matchers (parsing user-written @文戏/@硬切/&秒&) are permitted to contain CJK
        allowed_regex_hints = (
            ".match(",
            ".test(",
            ".replace(",
            'low.includes("文戏")',
        )
        violations = []
        for rel in targets:
            lines = _read_web(rel).splitlines()
            for idx, line in enumerate(lines, 1):
                code_part = re.sub(r"//.*$", "", line)
                if not CJK_RE.search(code_part):
                    continue
                if any(tok in code_part for tok in allowed_regex_hints):
                    continue
                violations.append(f"{rel}:L{idx}: {code_part.strip()[:120]}")
        self.assertEqual(violations, [], "Hardcoded Chinese UI strings found:\n" + "\n".join(violations))

        # B2 check: zero stale '导演' across all frontend JS (including i18n.js and node_i18n.js)
        for rel in (*targets, "i18n.js", "node_i18n.js"):
            txt = _read_web(rel)
            self.assertNotIn("导演", txt, f"Stale '导演' string found in web/{rel}")

    def test_canvas_node_i18n_covers_all_10_nodes_with_idempotent_guard(self):
        txt = _read_web("node_i18n.js")
        expected_classes = [
            "ShotVaultPlan",
            "ShotVaultLoad",
            "ShotVaultLoopDispatch",
            "ShotVaultH3Conditioning",
            "ShotVaultWindowGuider",
            "ShotVaultSeamTrim",
            "ShotVaultSave",
            "ShotVaultAssemble",
            "AchengFuzzPuppyClipMerge",
            "AchengSeamBridge",
        ]
        for cls_name in expected_classes:
            self.assertIn(f"{cls_name}:", txt, f"Missing canvas i18n definition for {cls_name}")
        self.assertIn("__achengNodeI18nRegistered", txt)
        self.assertIn("acheng:lang-change", txt)
        self.assertIn("isKnownDefaultTitle", txt)

    def test_b1_reroll_lock_mismatch_guard_and_b3_disabled_tail_bridge(self):
        dp = _read_web("director_panel.js")
        self.assertIn("isThisShotLocked", dp)
        self.assertIn("isOtherShotLocked", dp)
        self.assertIn("rl_other_locked_warn", dp)
        self.assertIn("rl_switch_lock_btn", dp)
        self.assertIn("rl_tail_disabled_opt", dp)

    def test_backend_bilingual_contracts_and_clean_defaults(self):
        # 1. ShotVaultPlan clean neutral defaults (no hardcoded Chinese prompt for fresh nodes)
        it = ShotVaultPlan.INPUT_TYPES()
        self.assertEqual(it["required"]["common_prompt"][1]["default"], "")
        self.assertEqual(it["required"]["script_text"][1]["default"], "")

        # 2. seam_qc first-shot returns both messages_zh and messages_en
        dummy = torch.zeros((4, 32, 32, 3), dtype=torch.float32)
        qc = analyze_seam_diagnostics(None, dummy, fps=24)
        self.assertTrue(qc.get("messages_zh"))
        self.assertTrue(qc.get("messages_en"))
        self.assertFalse(CJK_RE.search(" ".join(qc["messages_en"])))

        # 3. compute_master_sync_status returns both outdated_reasons and outdated_reasons_en
        with tempfile.TemporaryDirectory() as td:
            store = VaultStore(td)
            proj = store.load_project()
            proj["shots"]["1"] = {
                "active_take_id": "shot_0001_take_001",
                "takes": {"shot_0001_take_001": {"take_id": "shot_0001_take_001", "video_rel_path": "shots/shot_0001_take_001.mp4"}},
            }
            store.save_project(proj)
            st = compute_master_sync_status(store, [{"shot_index": 1}])
            self.assertTrue(st["is_outdated"])
            self.assertTrue(st["outdated_reasons"])
            self.assertTrue(st["outdated_reasons_en"])
            self.assertFalse(CJK_RE.search(" ".join(st["outdated_reasons_en"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
