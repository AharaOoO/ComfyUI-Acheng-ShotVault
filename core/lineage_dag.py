"""Lineage DAG evaluator: STALE propagation, @cut isolation, LOCKED_KEEP, and non-destructive PROMPT_EDITED."""

from __future__ import annotations

from typing import Any


class LineageDAG:
    """Evaluate deterministic parent-child consistency across a multi-shot plan."""

    @staticmethod
    def evaluate(
        shots_plan: list[dict[str, Any]],
        project_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Evaluate statuses from shot 1..N.

        Statuses:
          - `UNRENDERED`: no active take saved yet.
          - `VALID`: rendered and parent lineage matches (or hard cut `@cut` resets lineage).
          - `STALE`: rendered as continuation from an inactive/outdated parent take.
          - `LOCKED_KEEP`: lineage mismatches, but `locked_keep` allows assembly with warning.
        Secondary flag:
          - `prompt_edited`: prompt SHA-256 changed after take render (never invalidates to STALE).
        """
        shots_state = project_state.get("shots") or {}
        seed_shot = project_state.get("seed_shot") or {}
        seed_take_id = str(seed_shot.get("take_id") or "")

        results: list[dict[str, Any]] = []
        prev_active_take = seed_take_id
        prev_bridged_next = ""
        for shot in shots_plan:
            idx = int(shot["shot_index"])
            rec = shots_state.get(str(idx)) or {}
            active_id = str(rec.get("active_take_id") or "")
            locked_keep = bool(rec.get("locked_keep", False))
            takes = rec.get("takes") or {}
            active_meta = takes.get(active_id) or {}
            is_cont = bool(shot.get("is_continuation", idx > 1))

            if not active_id or not active_meta:
                status = "UNRENDERED"
                prompt_edited = False
            else:
                saved_hash = str(active_meta.get("prompt_sha256") or "")
                prompt_edited = bool(saved_hash and saved_hash != str(shot.get("prompt_sha256") or ""))
                if not is_cont:
                    status = "VALID"
                else:
                    actual_parent = str(active_meta.get("parent_take_id") or "")
                    accepted_parent = str(rec.get("accepted_parent_take_id") or "")
                    expected_parent = prev_active_take
                    if expected_parent and (
                        actual_parent == expected_parent
                        or (prev_bridged_next and prev_bridged_next == active_id)
                    ):
                        status = "VALID"
                    elif locked_keep or (accepted_parent and accepted_parent == expected_parent):
                        status = "LOCKED_KEEP"
                    else:
                        status = "STALE"

            results.append({
                "shot_index": idx,
                "shot_id": shot.get("shot_id", f"shot_{idx:04d}"),
                "is_continuation": is_cont,
                "active_take_id": active_id,
                "parent_take_id": str(active_meta.get("parent_take_id") or ""),
                "expected_parent_take_id": prev_active_take if is_cont else "",
                "status": status,
                "prompt_edited": prompt_edited,
                "locked_keep": locked_keep,
                "take_count": len(takes),
            })
            prev_active_take = active_id
            prev_bridged_next = str(active_meta.get("bridged_next_take_id") or "")

        return results
