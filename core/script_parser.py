"""Multi-format script parser: 1114 markdown, MV timestamps, kkdemabu dual JSONC, cinema pack JSON."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from ..compat import sha256_text

_SEGMENT_SPLIT_RE = re.compile(r"\s*={3,}\s*|^\s*-{3,}\s*$", re.MULTILINE)
_TIME_RANGE_RE = re.compile(
    r"^\s*&\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\s*&\s*",
    re.MULTILINE,
)
_SECONDS_HEADER_RE = re.compile(r"^\s*&\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?\s*&\s*", re.MULTILINE)
_PARENS_DUR_RE = re.compile(
    r"^\s*(?:(?:Duration|时长)\s*[:：]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?|[（(]\s*([0-9]+(?:\.[0-9]+)?)\s*(?:s|秒)?\s*[)）])\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_SPEECH_SPAN_RE = re.compile(r"\[(\d{1,2}):(\d{2}(?:\.\d+)?)\s*-\s*(\d{1,2}):(\d{2}(?:\.\d+)?)\]\s*<d>.*?</d>", re.DOTALL)
_TYPE_DIRECTIVE_RE = re.compile(r"@type\s*=\s*([^\s\r\n]+)", re.IGNORECASE)
_SHORT_TYPE_DIRECTIVE_RE = re.compile(r"@(文戏|武戏)\b")
_OVERLAP_DIRECTIVE_RE = re.compile(r"(?:@overlap\s*=\s*(22|39|56)|@(22|39|56)\b)", re.IGNORECASE)
_CUT_DIRECTIVE_RE = re.compile(r"(?:^|(?<=\s))@(?:cut|硬切|hard_?cut)@?(?=\s|$)", re.IGNORECASE)
_CONT_DIRECTIVE_RE = re.compile(r"(?:^|(?<=\s))@(?:continue|续接|连续)@?(?=\s|$)", re.IGNORECASE)
_SEG_DIRECTIVE_RE = re.compile(r"(?:^|(?<=\s))@segment@?(?=\s|$)", re.IGNORECASE)
_AUDIO_DIRECTIVE_RE = re.compile(r"@audio_file\s+([^\r\n]+)", re.IGNORECASE)
_REF_IMAGE_DIRECTIVE_RE = re.compile(r"@ref_image\s+([^\r\n]+)", re.IGNORECASE)
_REF_VIDEO_DIRECTIVE_RE = re.compile(r"@ref_video\s+([^\r\n]+)", re.IGNORECASE)
_REF_AUDIO_DIRECTIVE_RE = re.compile(r"@ref_audio\s+([^\r\n]+)", re.IGNORECASE)


def _mmss_to_sec(minutes_str: str, sec_str: str) -> float:
    return int(minutes_str) * 60.0 + float(sec_str)


def strip_jsonc_comments(text: str) -> str:
    """Strip `//...` and `/*...*/` comments plus trailing commas outside JSON string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_str = False
    escaped = False
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and text[i] not in ("\n", "\r"):
                i += 1
            continue
        if ch == "/" and nxt == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i = min(n, i + 2)
            continue
        out.append(ch)
        i += 1
    cleaned = "".join(out)
    # Remove trailing commas before } or ] outside strings cleanly via token scan
    res: list[str] = []
    in_str = False
    escaped = False
    length = len(cleaned)
    for idx, ch in enumerate(cleaned):
        if in_str:
            res.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            res.append(ch)
            continue
        if ch == ",":
            j = idx + 1
            while j < length and cleaned[j].isspace():
                j += 1
            if j < length and cleaned[j] in ("}", "]"):
                continue
        res.append(ch)
    return "".join(res)


def _map_shot_type(raw_type: Any) -> int:
    text = ("" if raw_type is None else str(raw_type)).strip().lower()
    if any(k in text for k in ("文戏", "drama", "dialogue", "文", "emotional")):
        return 0
    if text in ("2", "mv", "music", "dance", "音乐"):
        return 2
    if text in ("0",):
        return 0
    return 1


def _is_hard_cut_text(raw_conn: Any) -> Optional[bool]:
    text = str(raw_conn or "").strip().lower()
    if not text:
        return None
    if any(k in text for k in ("硬切", "切镜", "cut", "hard", "开场", "首镜", "flashback")):
        return True
    if any(k in text for k in ("接续", "连续", "continue", "extend", "跟随", "连镜")):
        return False
    return None


def _combine_prompt(common_prompt: str, shot_prompt: str) -> str:
    common = (common_prompt or "").strip()
    local = (shot_prompt or "").strip()
    if common and local:
        return f"{common}\n{local}"
    return local or common


def _extract_speech_spans(body: str, fps: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for m in _SPEECH_SPAN_RE.finditer(body):
        s0 = _mmss_to_sec(m.group(1), m.group(2))
        s1 = _mmss_to_sec(m.group(3), m.group(4))
        if s1 > s0:
            spans.append((max(0, round(s0 * fps)), max(1, round(s1 * fps))))
    return spans


def _parse_1114_markdown(
    script_text: str,
    common_prompt: str,
    fps: int,
    has_seed_video: bool,
    default_duration_sec: float = 5.0,
) -> list[dict]:
    fallback_dur = max(0.25, float(default_duration_sec or 5.0))
    raw_blocks = [b.strip() for b in _SEGMENT_SPLIT_RE.split(script_text or "") if b.strip()]
    if not raw_blocks:
        raw_blocks = [(script_text or "").strip()] if (script_text or "").strip() else [f"& {fallback_dur:g} & 默认分镜"]

    shots: list[dict] = []
    for idx, block in enumerate(raw_blocks):
        shot_idx = idx + 1
        dur = fallback_dur
        t_start = None
        t_end = None
        body = block

        m_range = _TIME_RANGE_RE.search(body)
        if m_range:
            s0 = _mmss_to_sec(m_range.group(1), m_range.group(2))
            s1 = _mmss_to_sec(m_range.group(3), m_range.group(4))
            t_start = s0
            t_end = max(s0 + 0.25, s1)
            dur = max(0.25, t_end - t_start)
            body = (body[:m_range.start()] + body[m_range.end():]).strip()
        else:
            m_sec = _SECONDS_HEADER_RE.search(body)
            if m_sec:
                dur = max(0.25, float(m_sec.group(1)))
                body = (body[:m_sec.start()] + body[m_sec.end():]).strip()
            else:
                m_pdur = _PARENS_DUR_RE.search(body)
                if m_pdur:
                    dur = max(0.25, float(m_pdur.group(1) or m_pdur.group(2)))
                    body = (body[:m_pdur.start()] + body[m_pdur.end():]).strip()

        has_cut = bool(_CUT_DIRECTIVE_RE.search(body))
        has_cont = bool(_CONT_DIRECTIVE_RE.search(body))
        has_segment = bool(_SEG_DIRECTIVE_RE.search(body))
        m_type = _TYPE_DIRECTIVE_RE.search(body)
        if m_type:
            shot_type = _map_shot_type(m_type.group(1))
        else:
            m_stype = _SHORT_TYPE_DIRECTIVE_RE.search(body)
            shot_type = _map_shot_type(m_stype.group(1)) if m_stype else 1
        m_ov = _OVERLAP_DIRECTIVE_RE.search(body)
        ov_raw = int(m_ov.group(1) or m_ov.group(2)) if m_ov else 0
        overlap_override = ov_raw if ov_raw in (22, 39, 56) else 0
        m_audio = _AUDIO_DIRECTIVE_RE.search(body)
        audio_file = m_audio.group(1).strip().strip('"').strip("'") if m_audio else ""
        ref_images = [x.strip().strip('"').strip("'") for x in _REF_IMAGE_DIRECTIVE_RE.findall(body) if x.strip()]
        ref_videos = [x.strip().strip('"').strip("'") for x in _REF_VIDEO_DIRECTIVE_RE.findall(body) if x.strip()]
        ref_audios = [x.strip().strip('"').strip("'") for x in _REF_AUDIO_DIRECTIVE_RE.findall(body) if x.strip()]

        clean_body = body
        for rx in (_CUT_DIRECTIVE_RE, _CONT_DIRECTIVE_RE, _SEG_DIRECTIVE_RE):
            clean_body = rx.sub("", clean_body)
        clean_body = _TYPE_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _SHORT_TYPE_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _OVERLAP_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _AUDIO_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _REF_IMAGE_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _REF_VIDEO_DIRECTIVE_RE.sub("", clean_body)
        clean_body = _REF_AUDIO_DIRECTIVE_RE.sub("", clean_body)
        clean_body = re.sub(r"\n{3,}", "\n\n", clean_body).strip()

        if has_cut:
            is_cont = False
        elif has_cont:
            is_cont = True
        elif shot_idx == 1:
            is_cont = bool(has_seed_video)
        else:
            is_cont = True

        full_prompt = _combine_prompt(common_prompt, clean_body)
        shots.append({
            "shot_index": shot_idx,
            "shot_id": f"shot_{shot_idx:04d}",
            "duration_sec": float(dur),
            "t_start_sec": t_start,
            "t_end_sec": t_end,
            "shot_type": shot_type,
            "is_drama_mode": (shot_type == 0),
            "is_continuation": is_cont,
            "overlap_override": overlap_override,
            "segment_only": has_segment,
            "audio_file": audio_file,
            "ref_images": ref_images[:9],
            "ref_slots": list(ref_images[:9]),
            "ref_videos": ref_videos,
            "ref_audios": ref_audios,
            "inherit_global_refs": True,
            "speech_spans": _extract_speech_spans(clean_body, fps),
            "local_prompt": clean_body,
            "prompt": full_prompt,
            "prompt_sha256": sha256_text(full_prompt),
        })
    return shots


def _extract_json_records(obj: Any) -> list[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("shots", "events", "segments", "分镜", "list", "data"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [obj]
    return []


def _parse_jsonc_payload(
    script_text: str,
    common_prompt: str,
    stage3_json_text: str,
    fps: int,
    has_seed_video: bool,
) -> list[dict]:
    parsed_s2 = json.loads(strip_jsonc_comments(script_text))
    records_s2 = _extract_json_records(parsed_s2)

    records_s3: list[dict] = []
    if (stage3_json_text or "").strip():
        parsed_s3 = json.loads(strip_jsonc_comments(stage3_json_text))
        records_s3 = _extract_json_records(parsed_s3)

    s3_by_id: dict[str, dict] = {}
    for i, item in enumerate(records_s3):
        key = str(
            item.get("id")
            or item.get("shot_id")
            or item.get("镜号")
            or item.get("序号")
            or (i + 1)
        )
        s3_by_id[key] = item

    shots: list[dict] = []
    for idx, item in enumerate(records_s2):
        shot_idx = idx + 1
        key = str(
            item.get("id")
            or item.get("shot_id")
            or item.get("镜号")
            or item.get("序号")
            or shot_idx
        )
        extra = s3_by_id.get(key) or (records_s3[idx] if idx < len(records_s3) else {})
        merged = {**item, **extra}

        dur_raw = (
            merged.get("duration_sec")
            or merged.get("duration")
            or merged.get("时长")
            or merged.get("seconds")
            or 5.0
        )
        try:
            dur = max(0.25, float(str(dur_raw).rstrip("s秒 ")))
        except ValueError:
            dur = 5.0

        shot_type = _map_shot_type(merged.get("shot_type") if "shot_type" in merged else merged.get("类型"))
        conn_flag = _is_hard_cut_text(merged.get("场景衔接") or merged.get("transition"))
        if "is_continuation" in merged:
            is_cont = bool(merged["is_continuation"])
        elif "hard_cut" in merged:
            is_cont = not bool(merged["hard_cut"])
        elif conn_flag is not None:
            is_cont = not conn_flag
        elif shot_idx == 1:
            is_cont = bool(has_seed_video)
        else:
            is_cont = True

        local_prompt = str(
            merged.get("prompt")
            or merged.get("video_prompt")
            or merged.get("画面提示词")
            or merged.get("视频文本")
            or merged.get("描述")
            or ""
        ).strip()
        audio_file = str(merged.get("audio_file") or merged.get("音频") or "").strip()
        full_prompt = _combine_prompt(common_prompt, local_prompt)
        shots.append({
            "shot_index": shot_idx,
            "shot_id": f"shot_{shot_idx:04d}",
            "duration_sec": float(dur),
            "t_start_sec": merged.get("t_start_sec"),
            "t_end_sec": merged.get("t_end_sec"),
            "shot_type": shot_type,
            "is_drama_mode": (shot_type == 0),
            "is_continuation": is_cont,
            "segment_only": bool(merged.get("segment_only", False)),
            "audio_file": audio_file,
            "ref_images": list(merged.get("ref_images") or ()),
            "ref_videos": list(merged.get("ref_videos") or ()),
            "ref_audios": list(merged.get("ref_audios") or ()),
            "inherit_global_refs": bool(merged.get("inherit_global_refs", True)),
            "speech_spans": _extract_speech_spans(local_prompt, fps),
            "local_prompt": local_prompt,
            "prompt": full_prompt,
            "prompt_sha256": sha256_text(full_prompt),
        })
    return shots


def _norm_slots(raw_slots: Any, fallback: Any = ()) -> list[str]:
    src = raw_slots if isinstance(raw_slots, (list, tuple)) else fallback
    out = [str(x or "").strip() for x in list(src or ())[:9]]
    while out and not out[-1]:
        out.pop()
    return out


def parse_director_plan_json(
    director_plan_json: str,
    common_prompt: str = "",
    fps: int = 24,
    has_seed_video: bool = False,
) -> tuple[dict[str, Any], list[dict]]:
    """Parse structured v5.x Director Console JSON into `(global_cfg, normalized_shots)`."""
    raw = (director_plan_json or "").strip()
    if not raw:
        return {}, []
    try:
        obj = json.loads(strip_jsonc_comments(raw))
    except Exception:
        return {}, []
    if not isinstance(obj, dict):
        return {}, []

    seed_path = str(obj.get("seed_video_path") or "").strip()
    effective_has_seed = bool(has_seed_video or seed_path)
    has_glob_slots = "global_ref_slots" in obj
    glob_slots = _norm_slots(obj.get("global_ref_slots"), obj.get("global_ref_images"))
    glob_imgs = [x for x in glob_slots if x] if has_glob_slots else [
        str(x).strip() for x in (obj.get("global_ref_images") or ()) if str(x).strip()
    ][:9]
    ctx_raw = obj.get("context_frames")
    ctx_val = int(ctx_raw) if str(ctx_raw or "").isdigit() and int(ctx_raw) in (22, 39, 56) else 0
    audio_mode_raw = str(obj.get("audio_drive_mode") or "").strip()
    try:
        default_dur = max(0.25, float(obj.get("default_duration_sec") or 5.0))
    except (TypeError, ValueError):
        default_dur = 5.0
    retry_raw = obj.get("retry_lock") if isinstance(obj.get("retry_lock"), dict) else {}
    try:
        tb_raw = int(retry_raw.get("tail_bridge_frames") or 0)
    except (TypeError, ValueError):
        tb_raw = 0
    tb_frames = tb_raw if tb_raw in (0, 1, 5, 22) else 0
    tb_noise = str(retry_raw.get("tail_bridge_noise") or "clean").strip()
    if tb_noise not in ("clean", "reverse_taper"):
        tb_noise = "clean"
    retry_lock = {
        "enabled": bool(retry_raw.get("enabled", False)),
        "shot_index": max(1, int(retry_raw.get("shot_index") or 1)),
        "seed_mode": str(retry_raw.get("seed_mode") or "randomize"),
        "seed_text": str(retry_raw.get("seed_text") or "").strip(),
        "prompt_override": str(retry_raw.get("prompt_override") or ""),
        "tail_bridge_frames": tb_frames,
        "tail_bridge_noise": tb_noise,
        "auto_assemble_after_reroll": bool(retry_raw.get("auto_assemble_after_reroll", True)),
    }

    ad_raw = obj.get("anti_degradation") if isinstance(obj.get("anti_degradation"), dict) else {}
    ad_mode = str(ad_raw.get("mode") or "standard").strip()
    if ad_mode not in ("standard", "digital_human_low_noise", "digital_human_static_bg"):
        ad_mode = "standard"
    default_noise_keep = 0.36 if ad_mode == "digital_human_static_bg" else 0.40
    try:
        ad_noise_keep = float(ad_raw.get("noise_keep_ratio", default_noise_keep))
    except (TypeError, ValueError):
        ad_noise_keep = default_noise_keep
    try:
        ad_bg_pull = float(ad_raw.get("static_bg_pull", 0.20))
    except (TypeError, ValueError):
        ad_bg_pull = 0.20
    anti_degradation = {
        "mode": ad_mode,
        "noise_keep_ratio": max(0.05, min(1.0, ad_noise_keep)),
        "static_bg_pull": max(0.0, min(0.35, ad_bg_pull)),
    }

    global_cfg: dict[str, Any] = {
        "seed_video_path": seed_path,
        "seed_trim_end_sec": float(obj.get("seed_trim_end_sec") or 0.0),
        "global_ref_images": glob_imgs,
        "global_ref_audios": [str(x).strip() for x in (obj.get("global_ref_audios") or ()) if str(x).strip()][:3],
        "context_frames": ctx_val,
        "audio_drive_mode": audio_mode_raw,
        "default_duration_sec": default_dur,
        "master_script_text": str(obj.get("master_script_text") or ""),
        "auto_sync_master": bool(obj.get("auto_sync_master", True)),
        "retry_lock": retry_lock,
        "anti_degradation": anti_degradation,
    }
    if has_glob_slots:
        global_cfg["global_ref_slots"] = glob_slots

    raw_shots = obj.get("shots")
    if (not isinstance(raw_shots, list) or not raw_shots) and global_cfg["master_script_text"].strip():
        parsed_master = _parse_1114_markdown(
            global_cfg["master_script_text"],
            common_prompt=common_prompt,
            fps=fps,
            has_seed_video=effective_has_seed,
            default_duration_sec=default_dur,
        )
        return global_cfg, parsed_master
    if not isinstance(raw_shots, list) or not raw_shots:
        return global_cfg, []

    shots: list[dict] = []
    for item in raw_shots:
        if not isinstance(item, dict):
            continue
        shot_idx = len(shots) + 1
        try:
            dur = max(0.25, float(item.get("duration_sec", default_dur)))
        except (TypeError, ValueError):
            dur = default_dur
        shot_type = _map_shot_type(item.get("shot_type", 1))
        if "hard_cut" in item:
            is_cont = not bool(item["hard_cut"])
            if shot_idx == 1 and not bool(item["hard_cut"]):
                is_cont = bool(effective_has_seed)
        elif shot_idx == 1:
            is_cont = bool(effective_has_seed)
        else:
            is_cont = True

        ov_raw = int(item.get("overlap_override") or 0)
        ov_ovr = ov_raw if ov_raw in (22, 39, 56) else 0
        has_shot_slots = "ref_slots" in item
        shot_slots = _norm_slots(item.get("ref_slots"), item.get("ref_images")) if (has_shot_slots or has_glob_slots) else None
        shot_imgs = (
            [x for x in (shot_slots or ()) if x][:9]
            if has_shot_slots
            else [str(x).strip() for x in (item.get("ref_images") or ()) if str(x).strip()][:9]
        )

        local_prompt = str(item.get("prompt") or "").strip()
        full_prompt = _combine_prompt(common_prompt, local_prompt)
        guide_audio = str(item.get("guide_audio") or item.get("audio_file") or "").strip()
        sh_dict: dict[str, Any] = {
            "shot_index": shot_idx,
            "shot_id": f"shot_{shot_idx:04d}",
            "duration_sec": float(dur),
            "t_start_sec": item.get("t_start_sec"),
            "t_end_sec": item.get("t_end_sec"),
            "shot_type": shot_type,
            "is_drama_mode": (shot_type == 0),
            "is_continuation": is_cont,
            "explicit_hard_cut": bool(item.get("hard_cut", False)),
            "overlap_override": ov_ovr,
            "segment_only": bool(item.get("segment_only", False)),
            "audio_file": guide_audio,
            "ref_images": shot_imgs,
            "ref_videos": list(item.get("ref_videos") or ()),
            "ref_audios": [str(x).strip() for x in (item.get("ref_audios") or ()) if str(x).strip()],
            "inherit_global_refs": bool(item.get("inherit_global_refs", True)),
            "seed_text": str(item.get("seed_text") or "").strip(),
            "speech_spans": _extract_speech_spans(local_prompt, fps),
            "local_prompt": local_prompt,
            "prompt": full_prompt,
            "prompt_sha256": sha256_text(full_prompt),
        }
        if shot_slots is not None:
            sh_dict["ref_slots"] = shot_slots
        shots.append(sh_dict)
    return global_cfg, shots


def parse_script(
    script_text: str,
    common_prompt: str = "",
    stage3_json_text: str = "",
    script_format: str = "auto",
    fps: int = 24,
    has_seed_video: bool = False,
    default_duration_sec: float = 5.0,
) -> list[dict]:
    """Parse 1114 markdown, MV timestamped ranges, kkdemabu dual JSONC, or cinema pack JSON."""
    fmt = (script_format or "auto").strip()
    if fmt == "auto":
        stripped = strip_jsonc_comments(script_text or "").lstrip()
        fmt = "kkdemabu_dual_jsonc" if stripped.startswith(("[", "{")) else "1114_markdown"
    if fmt in ("kkdemabu_dual_jsonc", "cinema_pack_json"):
        return _parse_jsonc_payload(script_text, common_prompt, stage3_json_text, fps, has_seed_video)
    return _parse_1114_markdown(script_text, common_prompt, fps, has_seed_video, default_duration_sec=default_duration_sec)


def parse_shot_numbers(spec: str, total_shots: int) -> tuple[bool, list[int]]:
    """Parse `'1'`, `'0'` (all reset), `'3-5'`, `'6,9'` into `(is_full_reset, sorted_unique_indices)`."""
    total_shots = max(1, int(total_shots))
    text = str(spec or "1").strip()
    if text == "0":
        return True, list(range(1, total_shots + 1))
    picks: list[int] = []
    seen = set()
    for part in re.split(r"[,;\s]+", text):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bounds = part.split("-", 1)
            try:
                lo = max(1, int(bounds[0]))
                hi = min(total_shots, int(bounds[1]))
            except ValueError:
                continue
            for k in range(lo, hi + 1):
                if k not in seen:
                    seen.add(k)
                    picks.append(k)
        else:
            try:
                val = int(part)
            except ValueError:
                continue
            if 1 <= val <= total_shots and val not in seen:
                seen.add(val)
                picks.append(val)
    if not picks:
        picks = [1]
    return False, picks
