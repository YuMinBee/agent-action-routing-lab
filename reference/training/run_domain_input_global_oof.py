import argparse
import csv
import json
import math
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


FULL_LABELS = np.array(
    [
        "apply_patch",
        "ask_user",
        "edit_file",
        "glob_pattern",
        "grep_search",
        "lint_or_typecheck",
        "list_directory",
        "plan_task",
        "read_file",
        "respond_only",
        "run_bash",
        "run_tests",
        "web_search",
        "write_file",
    ],
    dtype=object,
)

USER_TIERS = ["free", "pro", "enterprise", "unknown"]
LANG_PREFS = ["en", "ko", "mixed", "unknown"]
CI_STATUSES = ["none", "passed", "failed", "unknown", "other"]
PRIMARY_LANGS = [
    "py",
    "js",
    "jsx",
    "ts",
    "tsx",
    "css",
    "html",
    "json",
    "yaml",
    "yml",
    "toml",
    "md",
    "go",
    "rs",
    "java",
    "kt",
    "cpp",
    "c",
    "sh",
    "sql",
    "vue",
    "dockerfile",
    "none",
    "other",
]

PATH_RE = re.compile(
    r"[\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|go|rs|java|kt|cpp|c|h|css|html|sql|sh|ps1)",
    re.I,
)
GLOB_RE = re.compile(r"(?:\*\*?[/.\w-]*|[/.\w-]*\*\*?|[/.\w-]*\*[/.\w-]*|\{[^}]+\}|\[[^\]]+\])")


def compact(value, limit=700):
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def session_id(sample_id):
    return sample_id.split("-step_")[0]


def assistant_actions(row):
    return [item for item in row.get("history") or [] if item.get("name")]


def previous_actions(row):
    return [str(item.get("name")) for item in assistant_actions(row)]


def last_result(row):
    for item in reversed(row.get("history") or []):
        if item.get("result_summary"):
            return compact(item.get("result_summary"), 500)
    return ""


def last_user(row):
    for item in reversed(row.get("history") or []):
        if item.get("role") == "user":
            return compact(item.get("content"), 500)
    return ""


def word_count(text):
    return len(re.findall(r"\S+", str(text or "")))


def bucket_num(value, cuts, prefix="le"):
    try:
        value = int(float(value))
    except Exception:
        return "na"
    for cut in cuts:
        if value <= cut:
            return f"{prefix}_{cut}"
    return f"gt_{cuts[-1]}"


def onehot(value, choices):
    value = str(value or "unknown").lower()
    if value not in choices:
        value = "other" if "other" in choices else "unknown"
    return [1.0 if value == choice else 0.0 for choice in choices]


def primary_lang(workspace):
    mix = workspace.get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    lang = str(max(mix.items(), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0)[0]).lower()
    return lang if lang in PRIMARY_LANGS else "other"


def count_user_turns(row):
    return sum(1 for item in row.get("history") or [] if item.get("role") == "user")


def count_result_failures(row):
    total = 0
    for item in row.get("history") or []:
        text = str(item.get("result_summary") or "").lower()
        if "failed" in text or "failure" in text or "error" in text or re.search(r"exit(?: code)?[=: ]+[1-9]", text):
            total += 1
    return total


def extract_arg_values(row, keys, limit=8):
    values = []
    for item in reversed(row.get("history") or []):
        args = item.get("args")
        if not isinstance(args, dict):
            continue
        for key in keys:
            value = args.get(key)
            if not value:
                continue
            if isinstance(value, list):
                values.extend(compact(v, 100) for v in value)
            else:
                values.append(compact(value, 120))
        if len(values) >= limit:
            break
    return list(reversed(values[-limit:]))


def dedupe_tail(values, limit=8):
    out = []
    seen = set()
    for value in reversed(values):
        value = compact(value, 140)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return list(reversed(out[-limit:]))


def recent_paths(row, limit=8):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    values = [str(path) for path in ws.get("open_files") or []]
    values.extend(extract_arg_values(row, ["path", "paths", "file", "files"], limit=limit))
    for item in reversed(row.get("history") or []):
        chunks = [item.get("result_summary")]
        args = item.get("args")
        if isinstance(args, dict):
            chunks.extend(args.values())
        elif args:
            chunks.append(args)
        for chunk in chunks:
            if chunk:
                values.extend(PATH_RE.findall(str(chunk)))
        if len(values) >= limit * 2:
            break
    return dedupe_tail(values, limit=limit)


def recent_patterns(row, limit=6):
    values = extract_arg_values(row, ["pattern", "query", "glob"], limit=limit)
    values.extend(GLOB_RE.findall(compact(row.get("current_prompt"), 600)))
    return dedupe_tail(values, limit=limit)


def recent_cmds(row, limit=6):
    return extract_arg_values(row, ["cmd", "command", "commands", "script"], limit=limit)


def recent_targets(row, limit=6):
    return extract_arg_values(row, ["target", "targets", "name", "symbol"], limit=limit)


def extract_count(text, patterns):
    low = str(text or "").lower()
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def last_result_status(result):
    low = str(result or "").lower()
    if not low:
        return "none"
    if any(x in low for x in ["failed", "failure", "error"]) or re.search(r"exit(?: code)?[=: ]+[1-9]", low):
        return "fail"
    if any(x in low for x in ["ok", "passed", "success"]) or "exit=0" in low or "exit code 0" in low:
        return "ok"
    return "other"


def prompt_length_bucket(prompt):
    n_words = word_count(prompt)
    if n_words <= 5:
        return "le5"
    if n_words <= 10:
        return "le10"
    if n_words <= 18:
        return "le18"
    if n_words <= 30:
        return "le30"
    return "gt30"


def budget_bucket(meta):
    return bucket_num((meta or {}).get("budget_tokens_remaining"), [5000, 12000, 20000], prefix="le")


def last_result_bucket(result):
    low = str(result or "").lower()
    if not low:
        return "none"
    if (
        "no match" in low
        or "0 match" in low
        or "no result" in low
        or "0 result" in low
        or "empty" in low
    ):
        return "empty"
    return last_result_status(low)


def sim_query_header(row, variant=None):
    meta = row.get("session_meta") or {}
    parts = [
        "query:",
        "[SRC=sim]",
    ]
    if variant is not None:
        parts.append(f"[AUG=v{variant}]")
    parts.extend(
        [
            f"[STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}]",
            f"[PLEN={prompt_length_bucket(row.get('current_prompt'))}]",
            f"[BUDGET={budget_bucket(meta)}]",
            f"[LASTRES={last_result_bucket(last_result(row))}]",
        ]
    )
    return " ".join(parts)


def au_query_header(row, variant=None):
    meta = row.get("session_meta") or {}
    parts = [
        "query:",
        "[SRC=au]",
    ]
    if variant is not None:
        parts.append(f"[AUG=v{variant}]")
    parts.extend(
        [
            f"[STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}]",
            f"[PLEN={prompt_length_bucket(row.get('current_prompt'))}]",
            f"[BUDGET={budget_bucket(meta)}]",
            f"[LASTRES={last_result_bucket(last_result(row))}]",
        ]
    )
    return " ".join(parts)


def result_type(row):
    actions = previous_actions(row)
    last_action = actions[-1] if actions else "NONE"
    result = last_result(row).lower()
    if last_action == "read_file" or " read " in f" {result} ":
        return "read"
    if last_action == "grep_search" or "matches" in result or "occurrences" in result:
        return "grep"
    if last_action == "glob_pattern":
        return "glob"
    if last_action == "list_directory" or "entries" in result:
        return "list"
    if last_action == "run_tests" or "tests" in result or "pytest" in result:
        return "test"
    if last_action == "lint_or_typecheck" or any(t in result for t in ["lint", "type", "mypy", "tsc", "clippy"]):
        return "lint"
    if last_action == "run_bash" or result.startswith("exit="):
        return "bash"
    if last_action in {"apply_patch", "edit_file", "write_file"} or any(t in result for t in ["patched", "modified", "wrote"]):
        return "patch"
    if not result:
        return "none"
    return "other"


def norm_path(value):
    return str(value or "").replace("\\", "/").strip().lower()


def read_paths_from_history(row):
    out = []
    for item in row.get("history") or []:
        if item.get("name") != "read_file":
            continue
        args = item.get("args")
        if not isinstance(args, dict):
            continue
        for key in ("path", "file", "filepath"):
            if args.get(key):
                out.append(norm_path(args.get(key)))
    return {path for path in out if path}


def path_flags(paths, open_files, read_paths):
    path_set = {norm_path(path) for path in paths if norm_path(path)}
    open_set = {norm_path(path) for path in open_files if norm_path(path)}
    return int(bool(path_set & read_paths)), int(bool(path_set & open_set))


def args_light_text(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = [str(x) for x in (ws.get("open_files") or [])]
    paths = recent_paths(row, limit=6)
    patterns = recent_patterns(row, limit=5)
    cmds = recent_cmds(row, limit=4)
    targets = recent_targets(row, limit=4)
    path_read, path_open = path_flags(paths, open_files, read_paths_from_history(row))
    path_names = " ".join(f"path={Path(path).name}" for path in paths[-4:])
    pattern_text = " ".join(f"pattern={compact(pattern, 60)}" for pattern in patterns[-3:])
    cmd_text = " ".join(f"cmd={compact(cmd, 80)}" for cmd in cmds[-3:])
    target_text = " ".join(f"target={compact(target, 60)}" for target in targets[-3:])
    return " ".join(
        [
            f"paths_count={bucket_num(len(paths), [0, 1, 2, 4, 8], prefix='le')}",
            f"patterns_count={bucket_num(len(patterns), [0, 1, 2, 4], prefix='le')}",
            f"cmds_count={bucket_num(len(cmds), [0, 1, 2, 4], prefix='le')}",
            f"targets_count={bucket_num(len(targets), [0, 1, 2, 4], prefix='le')}",
            f"path_already_read={path_read}",
            f"path_in_open_files={path_open}",
            path_names,
            pattern_text,
            cmd_text,
            target_text,
        ]
    )


def result_light_text(row):
    result = last_result(row)
    low = result.lower()
    files_count = extract_count(result, [r"(\d+)\s+files?", r"patched\s+(\d+)\s+files?", r"modified\s+(\d+)\s+files?"])
    matches_count = extract_count(result, [r"(\d+)\s+matches?", r"found\s+(\d+)\s+occurrences?", r"(\d+)\s+occurrences?"])
    lines_count = extract_count(result, [r"\((\d+)l\)", r"(\d+)\s+lines?", r"read\s+[^ ]+\s+\((\d+)l\)"])
    tests_count = extract_count(result, [r"(\d+)\s+tests?", r"(\d+)\s+passed", r"collected\s+(\d+)"])
    failed_count = extract_count(result, [r"(\d+)\s+failed", r"failures?\s*[:=]\s*(\d+)"])
    no_matches = int(bool(re.search(r"\b(no|0)\s+(matches|results|occurrences)\b|not\s+found|found\s+0", low)))
    no_output = int("no output" in low or "empty output" in low)
    return " ".join(
        [
            f"last_result_type={result_type(row)}",
            f"last_result_status={last_result_status(result)}",
            f"last_result_empty={int(not bool(low.strip()))}",
            f"last_result_no_matches={no_matches}",
            f"last_result_no_output={no_output}",
            f"count_files={bucket_num(files_count, [0, 1, 2, 5, 10, 25], prefix='le')}",
            f"count_matches={bucket_num(matches_count, [0, 1, 2, 5, 10, 25], prefix='le')}",
            f"count_lines={bucket_num(lines_count, [0, 20, 80, 200, 500], prefix='le')}",
            f"count_tests={bucket_num(tests_count, [0, 1, 2, 5, 10, 25], prefix='le')}",
            f"count_failed={bucket_num(failed_count, [0, 1, 2, 5, 10], prefix='le')}",
            f"summary={compact(result, 220)}",
        ]
    )


def last_args_compact(row, limit=180):
    for item in reversed(row.get("history") or []):
        if item.get("name") and item.get("args"):
            return compact(json.dumps(item.get("args"), ensure_ascii=False, sort_keys=True), limit)
    return "none"


def recent_action_tail(row, n=6):
    actions = previous_actions(row)
    return ">".join(actions[-n:]) if actions else "START"


def compact_events(row, limit=6):
    events = []
    for item in reversed(row.get("history") or []):
        name = item.get("name")
        if not name:
            continue
        args = compact(json.dumps(item.get("args") or {}, ensure_ascii=False, sort_keys=True), 110)
        result = compact(item.get("result_summary"), 120)
        events.append(f"{name} args={args} result={result}")
        if len(events) >= limit:
            break
    return list(reversed(events))


def extract_symbols_from_text(text, limit=12):
    text = str(text or "")
    symbols = []
    patterns = [
        r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b",
        r"`([^`]{2,80})`",
        r"['\"]([^'\"]{2,80})['\"]",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                match = next((m for m in match if m), "")
            token = compact(match, 80)
            if token and token.lower() not in {"the", "and", "for", "with", "this", "that", "please"}:
                symbols.append(token)
    return dedupe_tail(symbols, limit=limit)


def explicit_hint_text(row):
    prompt = compact(row.get("current_prompt"), 900)
    last = last_user(row)
    combined = f"{prompt} {last}"
    paths = dedupe_tail(PATH_RE.findall(combined), limit=8)
    paths.extend(recent_paths(row, 6))
    paths = dedupe_tail(paths, limit=10)
    filenames = [Path(path).name for path in paths]
    exts = []
    for path in paths:
        m = re.search(r"\.([A-Za-z0-9]{1,12})$", str(path))
        if m:
            exts.append(m.group(1).lower())
    quoted = []
    for pattern in [r"`([^`]{1,80})`", r"['\"]([^'\"]{1,80})['\"]"]:
        quoted.extend(compact(x, 80) for x in re.findall(pattern, combined))
    return " ".join(
        [
            "paths=" + " ".join(paths[-8:]),
            "files=" + " ".join(dedupe_tail(filenames, 8)),
            "ext=" + " ".join(sorted(set(exts))) if exts else "ext=none",
            "symbols=" + " ".join(extract_symbols_from_text(combined, 10)),
            "quoted=" + " ".join(dedupe_tail(quoted, 8)),
            "cmds=" + " ".join(recent_cmds(row, 5)),
            "patterns=" + " ".join(recent_patterns(row, 5)),
        ]
    )


def state_short_text(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = [str(x) for x in (ws.get("open_files") or [])]
    exts = []
    for path in open_files:
        m = re.search(r"\.([A-Za-z0-9]{1,12})$", path)
        if m:
            exts.append(m.group(1).lower())
    return " ".join(
        [
            f"open_count={bucket_num(len(open_files), [0, 1, 2, 5, 10], prefix='le')}",
            f"open_exts={' '.join(sorted(set(exts))) or 'none'}",
            f"git_dirty={str(ws.get('git_dirty', 'unknown')).lower()}",
            f"ci={ws.get('last_ci_status', 'unknown')}",
            f"lang={primary_lang(ws)}",
            f"loc={bucket_num(ws.get('loc'), [1000, 5000, 20000, 50000], prefix='le')}",
            f"turn={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}",
        ]
    )


def last_line_text(row):
    actions = previous_actions(row)
    last = actions[-1] if actions else "NONE"
    return " ".join(
        [
            f"action={last}",
            f"args={last_args_compact(row, 160)}",
            f"result_type={result_type(row)}",
            f"status={last_result_status(last_result(row))}",
            f"hint={compact(last_result(row), 120)}",
        ]
    )


def action_trace_text(row):
    actions = previous_actions(row)
    counts = Counter(actions)
    last = actions[-1] if actions else "NONE"
    prev = actions[-2] if len(actions) >= 2 else "NONE"
    tail = " ".join(actions[-8:]) if actions else "NONE"
    prompt = compact(row.get("current_prompt"), 1000)
    return " ".join(
        [
            f"last_action={last}",
            f"last2_actions={prev}_{last}",
            f"tail={tail}",
            f"na={len(actions)}",
            f"nu={count_user_turns(row)}",
            f"hlen={len(row.get('history') or [])}",
            f"fail={count_result_failures(row)}",
            f"patch={counts.get('apply_patch', 0) + counts.get('edit_file', 0) + counts.get('write_file', 0)}",
            f"tests={counts.get('run_tests', 0)}",
            f"read={counts.get('read_file', 0)}",
            f"grep={counts.get('grep_search', 0)}",
            f"bash={counts.get('run_bash', 0)}",
            f"plen={bucket_num(word_count(prompt), [4, 10, 30, 60], prefix='le')}",
            f"c_read={int(counts.get('read_file', 0) > 0)}",
            f"c_edit={int(counts.get('edit_file', 0) + counts.get('apply_patch', 0) + counts.get('write_file', 0) > 0)}",
        ]
    )


def session_meta_text(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = [str(x) for x in (ws.get("open_files") or [])]
    exts = []
    for path in open_files:
        m = re.search(r"\.([A-Za-z0-9]{1,12})$", path)
        if m:
            exts.append(m.group(1).lower())
    file_tokens = " ".join(f"file={Path(path).name}" for path in open_files[-4:])
    ext_tokens = " ".join(f"ext_{ext}" for ext in sorted(set(exts)))
    return " ".join(
        [
            f"user_tier={meta.get('user_tier', 'unknown')}",
            f"language_pref={meta.get('language_pref', 'unknown')}",
            f"budget={bucket_num(meta.get('budget_tokens_remaining'), [5000, 10000, 30000, 70000, 150000], prefix='le')}",
            f"turn={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}",
            f"elapsed={bucket_num(meta.get('elapsed_session_sec'), [120, 300, 900, 1800], prefix='le')}",
            f"loc={bucket_num(ws.get('loc'), [1000, 5000, 20000, 50000], prefix='le')}",
            f"git_dirty={str(ws.get('git_dirty', 'unknown')).lower()}",
            f"last_ci_status={ws.get('last_ci_status', 'unknown')}",
            f"lang={primary_lang(ws)}",
            f"open_files_count={bucket_num(len(open_files), [0, 1, 2, 5, 10], prefix='le')}",
            file_tokens,
            ext_tokens,
        ]
    )


def build_sim_text(row):
    meta = row.get("session_meta") or {}
    events = compact_events(row, 6)
    paths = recent_paths(row, 10)
    exts = []
    for path in paths:
        m = re.search(r"\.([A-Za-z0-9]{1,12})$", str(path))
        if m:
            exts.append(m.group(1).lower())
    symbols = extract_symbols_from_text(" ".join([compact(row.get("current_prompt"), 1000), last_user(row), " ".join(paths)]), 14)
    return "\n".join(
        [
            sim_query_header(row),
            "[CURRENT_PROMPT]",
            compact(row.get("current_prompt"), 1000),
            "",
            "[ACTION_TAIL]",
            recent_action_tail(row, 6),
            "",
            "[LAST]",
            last_line_text(row),
            "",
            "[FLOW]",
            "\n".join(events) if events else "START",
            "",
            "[STATE]",
            state_short_text(row),
            "",
            "[PATHS]",
            " ".join(paths) + " ext=" + (" ".join(sorted(set(exts))) if exts else "none"),
            "",
            "[SYMBOLS]",
            " ".join(symbols),
        ]
    )


def build_au_text(row):
    meta = row.get("session_meta") or {}
    open_files = [str(x) for x in ((meta.get("workspace") or {}).get("open_files") or [])]
    edited = []
    for item in reversed(row.get("history") or []):
        if item.get("name") in {"edit_file", "apply_patch", "write_file"}:
            args = item.get("args")
            if isinstance(args, dict):
                for key in ["path", "file", "paths", "files"]:
                    val = args.get(key)
                    if isinstance(val, list):
                        edited.extend(str(v) for v in val)
                    elif val:
                        edited.append(str(val))
        if len(edited) >= 6:
            break
    return "\n".join(
        [
            au_query_header(row),
            "[CURRENT_PROMPT]",
            compact(row.get("current_prompt"), 700),
            "",
            "[DIRECT_HINTS]",
            explicit_hint_text(row),
            "",
            "[ACTION_TAIL]",
            recent_action_tail(row, 3),
            "",
            "[LAST]",
            last_line_text(row),
            "",
            "[STATE]",
            " ".join(
                [
                    state_short_text(row),
                    "edited=" + " ".join(dedupe_tail(edited, 6)),
                    "visible=" + " ".join(Path(path).name for path in open_files[-6:]),
                ]
            ),
        ]
    )


def build_text(row, input_src="sim"):
    if input_src == "au":
        return build_au_text(row)
    return build_sim_text(row)


def structured_features(row):
    actions = previous_actions(row)
    counts = Counter(actions)
    last = actions[-1] if actions else "NONE"
    prev = actions[-2] if len(actions) >= 2 else "NONE"
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    prompt = compact(row.get("current_prompt"), 1200)
    open_files = ws.get("open_files") or []

    feats = []
    feats.extend([math.log1p(counts.get(label, 0)) / 3.0 for label in FULL_LABELS])
    feats.extend([1.0 if last == label else 0.0 for label in FULL_LABELS])
    feats.extend([1.0 if prev == label else 0.0 for label in FULL_LABELS])
    feats.extend(onehot(meta.get("user_tier", "unknown"), USER_TIERS))
    feats.extend(onehot(meta.get("language_pref", "unknown"), LANG_PREFS))
    feats.extend(onehot(ws.get("last_ci_status", "unknown"), CI_STATUSES))
    feats.extend(onehot(primary_lang(ws), PRIMARY_LANGS))

    failed_count = count_result_failures(row)
    patch_count = counts.get("apply_patch", 0) + counts.get("edit_file", 0) + counts.get("write_file", 0)
    test_count = counts.get("run_tests", 0)
    numeric = [
        math.log1p(len(actions)) / 4.0,
        math.log1p(count_user_turns(row)) / 4.0,
        math.log1p(float(meta.get("budget_tokens_remaining") or 0)) / 13.0,
        math.log1p(float(meta.get("turn_index") or 0)) / 4.0,
        math.log1p(float(meta.get("elapsed_session_sec") or 0)) / 8.0,
        math.log1p(float(ws.get("loc") or 0)) / 12.0,
        float(bool(ws.get("git_dirty"))),
        math.log1p(len(open_files)) / 3.0,
        math.log1p(word_count(prompt)) / 5.0,
        math.log1p(len(prompt)) / 8.0,
        math.log1p(len(row.get("history") or [])) / 4.0,
        math.log1p(failed_count) / 3.0,
        math.log1p(patch_count) / 3.0,
        math.log1p(test_count) / 3.0,
    ]
    feats.extend(numeric)
    if len(feats) != 93:
        raise ValueError(f"structured feature dimension mismatch: {len(feats)}")
    return np.asarray(feats, dtype=np.float32)


def load_all_rows(data_dir, input_src="sim"):
    data_dir = Path(data_dir)
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    rows = load_jsonl(data_dir / "train.jsonl")
    ids = np.asarray([row["id"] for row in rows], dtype=object)
    sessions = np.asarray([session_id(row["id"]) for row in rows], dtype=object)
    y_action = np.asarray([labels[row["id"]] for row in rows], dtype=object)
    texts = [build_text(row, input_src=input_src) for row in rows]
    feats = np.vstack([structured_features(row) for row in rows])
    return rows, ids, sessions, y_action, texts, feats


class TextStructuredDataset(Dataset):
    def __init__(self, texts, features, y=None):
        self.texts = list(texts)
        self.features = np.asarray(features, dtype=np.float32)
        self.y = y

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx], "features": self.features[idx]}
        if self.y is not None:
            item["label"] = int(self.y[idx])
        return item


def make_collate(tokenizer, max_length):
    def collate(batch):
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["features"] = torch.tensor(np.vstack([item["features"] for item in batch]), dtype=torch.float32)
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return encoded

    return collate


class TextStructuredGlobalModel(nn.Module):
    def __init__(self, model_dir, feature_dim=93, dropout=0.2, gradient_checkpointing=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        hidden = int(getattr(self.config, "hidden_size"))
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2 + feature_dim),
            nn.Linear(hidden * 2 + feature_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, len(FULL_LABELS)),
        )

    def pool_mean(self, output, attention_mask):
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, input_ids, attention_mask, features, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        cls = output.last_hidden_state[:, 0]
        mean = self.pool_mean(output, attention_mask)
        features = self.feature_norm(features.to(mean.dtype))
        return self.classifier(torch.cat([cls, mean, features], dim=-1))


def model_inputs(batch):
    keys = {"input_ids", "attention_mask", "token_type_ids", "features"}
    return {k: v for k, v in batch.items() if k in keys}


def predict(model, loader, device):
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(**model_inputs(batch))
            chunks.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.vstack(chunks)


def train_fold(model, train_loader, valid_loader, y_valid, args, device, save_dir=None, tokenizer=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    ce = nn.CrossEntropyLoss()
    best = {"macro_f1": -1.0, "accuracy": -1.0, "epoch": -1, "proba": None}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.fp16)):
                logits = model(**model_inputs(batch))
                loss = ce(logits, batch["labels"]) / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        proba = predict(model, valid_loader, device)
        pred = FULL_LABELS[proba.argmax(axis=1)]
        macro_f1 = float(f1_score(y_valid, pred, labels=FULL_LABELS, average="macro", zero_division=0))
        acc = float(accuracy_score(y_valid, pred))
        print(
            f"  epoch={epoch} loss={np.mean(losses):.5f} valid_macro_f1={macro_f1:.6f} "
            f"acc={acc:.6f} sec={time.time() - t0:.1f}",
            flush=True,
        )
        if macro_f1 > best["macro_f1"]:
            best = {"macro_f1": macro_f1, "accuracy": acc, "epoch": epoch, "proba": proba}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / "model.pt")
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_dir)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def write_metrics(out_dir, name, y_true, pred):
    report = classification_report(y_true, pred, labels=FULL_LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / f"{name}_class_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y_true, pred, labels=FULL_LABELS)
    pd.DataFrame(cm, index=FULL_LABELS, columns=FULL_LABELS).to_csv(
        out_dir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig"
    )
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, pred, labels=FULL_LABELS, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--fold-dir", default="experiments/oof/hierarchical_story_state_transition_sgd_targetctx_20260702_rerun")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-name", default="text_structured_global")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--folds", default="2", help="Comma-separated 0-based fold ids")
    parser.add_argument("--input-src", choices=["sim", "au"], default="sim")
    parser.add_argument("--id-prefix", default="", help="Optional sample id/session prefix filter, e.g. sess_sim")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/oof/{args.artifact_name}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    rows, ids, sessions, y_action, texts, features = load_all_rows(args.data_dir, input_src=args.input_src)
    fold_dir = Path(args.fold_dir)
    fold_ids = np.load(fold_dir / "fold_ids.npy")
    fold_ids_ref = np.load(fold_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    if not np.array_equal(ids.astype(str), fold_ids_ref):
        raise ValueError("sample_ids do not align with fold_dir")
    if args.id_prefix:
        selected_mask = np.asarray([str(sample_id).startswith(args.id_prefix) for sample_id in ids], dtype=bool)
    else:
        selected_mask = np.ones(len(ids), dtype=bool)
    if not selected_mask.any():
        raise ValueError(f"id_prefix selected no rows: {args.id_prefix!r}")
    action_to_idx = {label: idx for idx, label in enumerate(FULL_LABELS)}
    y_idx = np.asarray([action_to_idx[label] for label in y_action], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"rows={len(texts)} selected={int(selected_mask.sum())} input_src={args.input_src} "
        f"id_prefix={args.id_prefix or 'ALL'} folds={dict(zip(*np.unique(fold_ids[selected_mask], return_counts=True)))} "
        f"max_length={args.max_length} feature_dim={features.shape[1]}",
        flush=True,
    )

    oof_proba = np.zeros((len(texts), len(FULL_LABELS)), dtype=np.float32)
    filled = np.zeros(len(texts), dtype=bool)
    fold_rows = []
    requested = {int(item.strip()) for item in args.folds.split(",") if item.strip()}
    folds = [fold for fold in sorted(np.unique(fold_ids)) if int(fold) in requested]
    collate = make_collate(tokenizer, args.max_length)

    for fold in folds:
        t0 = time.time()
        valid_idx = np.where((fold_ids == fold) & selected_mask)[0]
        train_idx = np.where((fold_ids != fold) & selected_mask)[0]
        if len(valid_idx) == 0 or len(train_idx) == 0:
            print(f"fold={int(fold)+1} skipped train={len(train_idx)} valid={len(valid_idx)}", flush=True)
            continue
        print(f"fold={int(fold)+1} train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        model = TextStructuredGlobalModel(
            args.model_dir,
            feature_dim=features.shape[1],
            gradient_checkpointing=args.gradient_checkpointing,
        )
        model.to(device)
        train_loader = DataLoader(
            TextStructuredDataset([texts[i] for i in train_idx], features[train_idx], y_idx[train_idx]),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        valid_loader = DataLoader(
            TextStructuredDataset([texts[i] for i in valid_idx], features[valid_idx], y_idx[valid_idx]),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        save_dir = out_dir / f"fold_{int(fold)+1}_model" if args.save_fold_models else None
        best = train_fold(model, train_loader, valid_loader, y_action[valid_idx], args, device, save_dir, tokenizer)
        oof_proba[valid_idx] = best["proba"]
        filled[valid_idx] = True
        fold_rows.append(
            {
                "fold": int(fold) + 1,
                "best_epoch": int(best["epoch"]),
                "macro_f1": float(best["macro_f1"]),
                "accuracy": float(best["accuracy"]),
                "seconds": round(time.time() - t0, 2),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    covered = filled
    pred = FULL_LABELS[oof_proba[covered].argmax(axis=1)]
    metrics = write_metrics(out_dir, args.artifact_name, y_action[covered], pred)
    np.save(out_dir / "classes.npy", FULL_LABELS.astype(str))
    np.save(out_dir / "sample_ids.npy", ids.astype(str))
    np.save(out_dir / "session_ids.npy", sessions.astype(str))
    np.save(out_dir / "y_true.npy", y_action.astype(str))
    np.save(out_dir / "fold_ids.npy", fold_ids)
    np.save(out_dir / "filled.npy", filled)
    np.save(out_dir / f"oof_{args.artifact_name}.npy", oof_proba)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{args.artifact_name}_folds.csv", index=False)
    summary = {"global": metrics, "folds": fold_rows, "covered_rows": int(covered.sum())}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
