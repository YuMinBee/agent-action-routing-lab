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

from klue_infer import predict_klue_proba_subset


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


def sim_query_header(row):
    meta = row.get("session_meta") or {}
    return " ".join(
        [
            "query:",
            "[SRC=sim]",
            f"[STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}]",
            f"[PLEN={prompt_length_bucket(row.get('current_prompt'))}]",
            f"[BUDGET={budget_bucket(meta)}]",
            f"[LASTRES={last_result_bucket(last_result(row))}]",
        ]
    )


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
            f"query: [SRC=au] [STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}]",
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
        self.encoder = AutoModel.from_config(self.config)
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


def load_quantized_state(path, device):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state" in payload:
        restored = {}
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        for key, value in payload["state"].items():
            if isinstance(value, dict) and "q" in value and "scale" in value:
                restored[key] = value["q"].to(dtype=dtype) * float(value["scale"])
            elif isinstance(value, dict) and "value" in value:
                restored[key] = value["value"]
            else:
                restored[key] = value
        return restored
    return payload


def load_delta_quantized_state(model_dir, device):
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "delta_meta.json").read_text(encoding="utf-8"))
    base_path = (model_dir / meta["base_model_relpath"]).resolve()
    base_payload = torch.load(base_path, map_location="cpu")
    delta_payload = torch.load(model_dir / "model_delta.pt", map_location="cpu")
    base_state = base_payload["state"] if isinstance(base_payload, dict) and "state" in base_payload else base_payload
    delta_state = delta_payload["state"] if isinstance(delta_payload, dict) and "state" in delta_payload else delta_payload
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    restored = {}
    for key, value in delta_state.items():
        if isinstance(value, dict) and "delta" in value and "scale" in value:
            base_value = base_state[key]
            if not (isinstance(base_value, dict) and "q" in base_value):
                raise ValueError(f"Delta base tensor is not quantized: {key}")
            q = (
                base_value["q"].to(dtype=torch.int16)
                + value["delta"].to(dtype=torch.int16)
            ).to(dtype=torch.int8)
            restored[key] = q.to(dtype=dtype) * float(value["scale"])
        elif isinstance(value, dict) and "value" in value:
            restored[key] = value["value"]
        else:
            restored[key] = value
    return restored


def describe_dir(path, limit=40):
    try:
        items = []
        for idx, item in enumerate(sorted(Path(path).iterdir(), key=lambda x: x.name)):
            if idx >= limit:
                items.append("...")
                break
            suffix = "/" if item.is_dir() else ""
            items.append(item.name + suffix)
        return f"{path}: " + ", ".join(items)
    except Exception as exc:
        return f"{path}: <unavailable {exc}>"


def resolve_model_dir(name):
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    roots = [script_dir, cwd, Path("/app"), Path("/work")]
    roots.extend(script_dir.parents)
    roots.extend(cwd.parents)

    candidates = []
    seen = set()
    for root in roots:
        for candidate in [
            root / "model" / name,
            root / name,
            root / "submit" / "model" / name,
            root / "app" / "model" / name,
        ]:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    def valid(path):
        return (
            (path / "config.json").exists()
            and (
                (path / "model_int8.pt").exists()
                or (path / "model.pt").exists()
                or ((path / "model_delta.pt").exists() and (path / "delta_meta.json").exists())
            )
            and (path / "tokenizer.json").exists()
        )

    for candidate in candidates:
        if valid(candidate):
            return candidate

    search_roots = []
    for root in [script_dir, cwd, Path("/app")]:
        try:
            if root.exists() and root.is_dir() and str(root) not in {str(x) for x in search_roots}:
                search_roots.append(root)
        except Exception:
            pass
    for root in search_roots:
        try:
            for config_path in root.rglob("config.json"):
                parent = config_path.parent
                if parent.name == name and valid(parent):
                    return parent
        except Exception:
            pass

    diagnostics = [
        f"script_dir={script_dir}",
        f"cwd={cwd}",
        describe_dir(script_dir),
        describe_dir(cwd),
        describe_dir(Path("/app")),
        "checked=" + " | ".join(str(path) for path in candidates[:20]),
    ]
    raise FileNotFoundError(f"Could not locate model directory for {name!r}. " + " ; ".join(diagnostics))


def resolve_klue_model_dir(name):
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    roots = [script_dir, cwd, Path("/app"), Path("/work")]
    roots.extend(script_dir.parents)
    roots.extend(cwd.parents)

    candidates = []
    seen = set()
    for root in roots:
        for candidate in [
            root / "model" / name,
            root / name,
            root / "submit" / "model" / name,
            root / "app" / "model" / name,
        ]:
            key = str(candidate)
            if key not in seen:
                seen.add(key)
                candidates.append(candidate)

    def valid(path):
        return (
            (path / "structured_config.json").exists()
            and ((path / "model_int8.pt").exists() or (path / "model.pt").exists())
            and (path / "tokenizer.json").exists()
        )

    for candidate in candidates:
        if valid(candidate):
            return candidate

    diagnostics = [
        f"script_dir={script_dir}",
        f"cwd={cwd}",
        describe_dir(script_dir),
        describe_dir(cwd),
        describe_dir(Path("/app")),
        "checked=" + " | ".join(str(path) for path in candidates[:20]),
    ]
    raise FileNotFoundError(
        f"Could not locate KLUE model directory for {name!r}. " + " ; ".join(diagnostics)
    )


def load_inference_model(model_dir, device):
    model = TextStructuredGlobalModel(model_dir, feature_dim=93)
    if device.type == "cuda":
        model.half()
    model_dir = Path(model_dir)
    if (model_dir / "model_delta.pt").exists():
        state = load_delta_quantized_state(model_dir, device)
    else:
        state_path = model_dir / "model_int8.pt"
        if not state_path.exists():
            state_path = model_dir / "model.pt"
        state = load_quantized_state(state_path, device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def predict_subset(rows, model_dir, input_src, max_length, device, batch_size=32):
    if not rows:
        return np.zeros((0, len(FULL_LABELS)), dtype=np.float32)
    texts = [build_text(row, input_src=input_src) for row in rows]
    features = np.vstack([structured_features(row) for row in rows]).astype(np.float32)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    loader = DataLoader(
        TextStructuredDataset(texts, features),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, max_length),
        num_workers=0,
    )
    model = load_inference_model(model_dir, device)
    proba = predict(model, loader, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return proba


def apply_class_bias(proba, bias, domain):
    before = proba.argmax(axis=1)
    score = np.log(np.clip(proba.astype(np.float64), 1e-9, 1.0))
    score += np.asarray(bias, dtype=np.float64).reshape(1, -1)
    score -= score.max(axis=1, keepdims=True)
    adjusted = np.exp(score)
    adjusted /= np.clip(adjusted.sum(axis=1, keepdims=True), 1e-12, None)
    changed = int((before != adjusted.argmax(axis=1)).sum())
    print(f"class_bias_{domain}_changed={changed}/{len(proba)}", flush=True)
    return adjusted.astype(np.float32)


def main():
    script_dir = Path(__file__).resolve().parent
    with open(script_dir / "model" / "class_bias.json", encoding="utf-8") as handle:
        class_bias = json.load(handle)
    if class_bias.get("labels") != FULL_LABELS.tolist():
        raise ValueError("class bias label order mismatch")
    candidates = [Path("data"), script_dir / "data", Path("/data"), Path("../data"), Path("open/data"), script_dir / "open" / "data"]
    data_dir = next((path for path in candidates if (path / "test.jsonl").exists()), Path("data"))
    rows = load_jsonl(data_dir / "test.jsonl")
    ids = np.asarray([str(row["id"]) for row in rows], dtype=object)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_proba = np.zeros((len(rows), len(FULL_LABELS)), dtype=np.float32)
    au_mask = np.asarray([sample_id.startswith("sess_au") for sample_id in ids], dtype=bool)
    sim_mask = ~au_mask
    override_global_indices = np.zeros(0, dtype=np.int64)
    override_label_ids = np.zeros(0, dtype=np.int64)

    if sim_mask.any():
        sim_indices = np.where(sim_mask)[0]
        sim_rows = [rows[i] for i in sim_indices]
        sim_proba_78425 = predict_subset(sim_rows, resolve_model_dir("sim"), "sim", 512, device, batch_size=8)
        sim_proba_78374 = predict_subset(sim_rows, resolve_model_dir("sim_alt"), "sim", 512, device, batch_size=8)
        sim_proba = (0.60 * sim_proba_78425 + 0.40 * sim_proba_78374).astype(np.float32)
        sim_proba = sim_proba / np.clip(sim_proba.sum(axis=1, keepdims=True), 1e-12, None)

        pred_a = sim_proba_78425.argmax(axis=1)
        pred_b = sim_proba_78374.argmax(axis=1)
        top2 = np.sort(np.partition(sim_proba, -2, axis=1)[:, -2:], axis=1)
        margin = top2[:, 1] - top2[:, 0]
        klue_candidate_indices = np.flatnonzero((pred_a != pred_b) & (margin < 0.08))
        if len(klue_candidate_indices):
            klue_rows = [sim_rows[i] for i in klue_candidate_indices]
            klue_proba = predict_klue_proba_subset(
                klue_rows,
                resolve_klue_model_dir("sim_klue"),
                device,
                batch_size=64,
                label_order=FULL_LABELS.tolist(),
            )
            klue_pred = klue_proba.argmax(axis=1)
            klue_conf = klue_proba.max(axis=1)
            agrees_with_either = (
                (klue_pred == pred_a[klue_candidate_indices])
                | (klue_pred == pred_b[klue_candidate_indices])
            )
            accepted = agrees_with_either & (klue_conf >= 0.55)
            accepted_local = klue_candidate_indices[accepted]
            override_global_indices = sim_indices[accepted_local]
            override_label_ids = klue_pred[accepted]
            print(
                f"klue_tiebreak_candidates={len(klue_candidate_indices)}/{len(sim_rows)} "
                f"accepted={len(override_global_indices)} margin_lt=0.08 conf_ge=0.55",
                flush=True,
            )
        else:
            print(
                f"klue_tiebreak_candidates=0/{len(sim_rows)} accepted=0 "
                "margin_lt=0.08 conf_ge=0.55",
                flush=True,
            )

        sim_proba = apply_class_bias(sim_proba, class_bias["bias_sim"], "sim")
        final_proba[sim_indices] = sim_proba
    if au_mask.any():
        au_rows = [rows[i] for i in np.where(au_mask)[0]]
        au_proba_plus_mined = predict_subset(
            au_rows, resolve_model_dir("au"), "au", 448, device, batch_size=16
        )
        au_proba_para_only = predict_subset(
            au_rows, resolve_model_dir("au_alt"), "au", 448, device, batch_size=16
        )
        au_proba = (0.50 * au_proba_plus_mined + 0.50 * au_proba_para_only).astype(np.float32)
        au_proba = au_proba / np.clip(au_proba.sum(axis=1, keepdims=True), 1e-12, None)
        au_proba = apply_class_bias(au_proba, class_bias["bias_au"], "au")
        final_proba[np.where(au_mask)[0]] = au_proba

    pred = FULL_LABELS[final_proba.argmax(axis=1)]
    if len(override_global_indices):
        pred[override_global_indices] = FULL_LABELS[override_label_ids]
        print(f"klue_tiebreak_overrides={len(override_global_indices)}", flush=True)
    pred_map = dict(zip([str(x) for x in ids], pred))
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    sample["id"] = sample["id"].astype(str)
    sample["action"] = sample["id"].map(pred_map)
    if sample["action"].isna().any():
        missing = sample.loc[sample["action"].isna(), "id"].head(10).tolist()
        raise ValueError(f"id mismatch between test.jsonl and sample_submission.csv: {missing}")
    out_dir = script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_dir / "submission.csv", index=False)
    cwd_out = Path("output")
    try:
        if cwd_out.resolve() != out_dir.resolve():
            cwd_out.mkdir(parents=True, exist_ok=True)
            sample.to_csv(cwd_out / "submission.csv", index=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
