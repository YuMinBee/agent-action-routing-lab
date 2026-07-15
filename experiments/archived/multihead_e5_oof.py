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

GROUP_TO_LABELS = {
    "inspect": ["read_file", "grep_search", "glob_pattern", "list_directory"],
    "modify": ["edit_file", "apply_patch", "write_file"],
    "validate": ["run_tests", "lint_or_typecheck", "run_bash"],
    "reason": ["plan_task", "ask_user", "web_search"],
}
GROUPS = np.array(list(GROUP_TO_LABELS), dtype=object)
LABEL_TO_GROUP = {label: group for group, labels in GROUP_TO_LABELS.items() for label in labels}
LABEL_TO_GROUP["respond_only"] = "reason"
RESPOND_LABELS = np.array(["not_respond_only", "respond_only"], dtype=object)

PATH_RE = re.compile(r"[\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|go|rs|java|kt|cpp|c|h|css|html|sql|sh|ps1)", re.I)
GLOB_RE = re.compile(r"(?:\*\*?[/.\w-]*|[/.\w-]*\*\*?|[/.\w-]*\*[/.\w-]*|\{[^}]+\}|\[[^\]]+\])")


def compact(value, limit=700):
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def bucket_count(value, cuts=(0, 1, 2, 5, 10, 25, 50, 100)):
    try:
        value = int(float(value))
    except Exception:
        return "na"
    for cut in cuts:
        if value <= cut:
            return f"le_{cut}"
    return f"gt_{cuts[-1]}"


def bucket_turn(value):
    try:
        value = int(float(value))
    except Exception:
        return "na"
    if value <= 2:
        return "early"
    if value <= 6:
        return "mid"
    if value <= 12:
        return "late"
    return "deep"


def session_id(sample_id):
    return sample_id.split("-step_")[0]


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def previous_actions(row):
    return [str(item.get("name")) for item in row.get("history") or [] if item.get("name")]


def assistant_actions(row):
    return [item for item in row.get("history") or [] if item.get("name")]


def last_result(row):
    for item in reversed(row.get("history") or []):
        if item.get("result_summary"):
            return compact(item.get("result_summary"), 500)
    return ""


def last_user_text(row):
    for item in reversed(row.get("history") or []):
        if item.get("role") == "user" and item.get("content"):
            return compact(item.get("content"), 500)
    return ""


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
                values.extend(compact(v, 120) for v in value)
            else:
                values.append(compact(value, 160))
        if len(values) >= limit:
            break
    return list(reversed(values[-limit:]))


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
    deduped = []
    seen = set()
    for value in reversed(values):
        value = compact(value, 160)
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return list(reversed(deduped[-limit:]))


def recent_patterns(row, limit=6):
    values = extract_arg_values(row, ["pattern", "query", "glob"], limit=limit)
    prompt = compact(row.get("current_prompt"), 600)
    values.extend(GLOB_RE.findall(prompt))
    deduped = []
    seen = set()
    for value in reversed(values):
        value = compact(value, 120)
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return list(reversed(deduped[-limit:]))


def recent_targets(row, limit=6):
    return extract_arg_values(row, ["target", "scope", "command", "cmd", "name"], limit=limit)


def recent_cmds(row, limit=6):
    return extract_arg_values(row, ["cmd", "command", "script"], limit=limit)


def result_status(text):
    low = text.lower()
    if not low:
        return "none"
    if any(t in low for t in ["error", "failed", "failure", "exception", "traceback", "panic", "hunk", "conflict", "에러", "실패"]):
        return "error"
    if any(t in low for t in ["pass", "passed", "success", "green", "ok", "exit=0", "no issues", "통과", "성공"]):
        return "ok"
    return "other"


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
    if last_action in {"run_tests"} or "tests" in result or "pytest" in result:
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


def extract_count(text, patterns):
    low = text.lower()
    for pattern in patterns:
        match = re.search(pattern, low)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def flag(text, terms):
    low = text.lower()
    return int(any(term in low for term in terms))


def recent_action_count(actions, action):
    return sum(1 for item in actions[-16:] if item == action)


def count_tests_buckets(text):
    total = extract_count(text, [r"(\d+)\s+tests?", r"(\d+)\s+passed", r"collected\s+(\d+)"])
    failed = extract_count(text, [r"(\d+)\s+failed", r"failures?\s*[:=]\s*(\d+)"])
    return bucket_count(total, cuts=(0, 1, 2, 5, 10, 25, 50, 100)), bucket_count(
        failed, cuts=(0, 1, 2, 5, 10)
    )


def diff_count_buckets(text):
    low = text.lower()
    match = re.search(r"([0-9]+)\s*(?:insertions?|added).{0,30}?([0-9]+)\s*(?:deletions?|removed)", low)
    if match:
        return bucket_count(match.group(1)), bucket_count(match.group(2))
    plus = extract_count(text, [r"(\d+)\+"])
    minus = extract_count(text, [r"(\d+)-"])
    return bucket_count(plus), bucket_count(minus)


def exit_zero_flag(text):
    low = text.lower()
    if "exit=0" in low or "exit code 0" in low:
        return "1"
    if re.search(r"exit(?: code)?[=: ]+[1-9]", low):
        return "0"
    return "na"


def recent_n_files_bucket(row):
    return bucket_count(len(recent_paths(row, limit=20)), cuts=(0, 1, 2, 4, 8, 16))


def budget_critical(value):
    try:
        return int(float(value) < 5000)
    except Exception:
        return 0


def common_text(row):
    actions = previous_actions(row)
    last = actions[-1] if actions else "NONE"
    prev = actions[-2] if len(actions) >= 2 else "NONE"
    recent = actions[-16:]
    bigrams = " ".join(f"{a}>{b}" for a, b in zip(recent[:-1], recent[1:]))
    result = last_result(row)
    prompt = compact(row.get("current_prompt"), 1200)
    last_user = last_user_text(row)
    paths = recent_paths(row, limit=8)
    patterns = recent_patterns(row, limit=6)
    cmds = recent_cmds(row, limit=6)
    targets = recent_targets(row, limit=6)
    prompt_signal = " ".join([prompt, last_user])
    context_signal = " ".join([" ".join(paths), " ".join(patterns), " ".join(cmds), " ".join(targets), result, " ".join(recent)])
    signal_text = " ".join([prompt_signal, context_signal])
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = [str(x) for x in (ws.get("open_files") or [])]
    open_exts = []
    for path in open_files:
        match = re.search(r"\.([A-Za-z0-9]{1,8})$", path)
        if match:
            open_exts.append(match.group(1).lower())
    mix = ws.get("language_mix") or {}
    top_lang = "unknown"
    if isinstance(mix, dict) and mix:
        top_lang = str(max(mix.items(), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0)[0])
    files_count = extract_count(result, [r"(\d+)\s+files?", r"patched\s+(\d+)\s+files?", r"modified\s+(\d+)\s+files?"])
    matches_count = extract_count(result, [r"(\d+)\s+matches?", r"found\s+(\d+)\s+occurrences?", r"(\d+)\s+occurrences?"])
    lines_count = extract_count(result, [r"\((\d+)l\)", r"(\d+)\s+lines?", r"read\s+[^ ]+\s+\((\d+)l\)"])
    count_tests, count_failed = count_tests_buckets(result)
    count_added, count_removed = diff_count_buckets(result)
    rtype = result_type(row)
    hint_items = [
        "ih=" + str(flag(signal_text, ['read', 'open', 'show', 'grep', 'search', 'find', 'glob', 'pattern', 'list', 'directory', 'folder', 'file', '\uc5f4\uc5b4', '\uc77d\uc5b4', '\ubcf4\uc5ec', '\ucc3e\uc544', '\uac80\uc0c9', '\ud638\ucd9c', '\ucc38\uc870', '\ubaa9\ub85d', '\ub514\ub809\ud1a0\ub9ac', '\ud3f4\ub354', '\ud30c\uc77c'])),
        "ip=" + str(flag(prompt_signal, ['read', 'open', 'show', 'grep', 'search', 'find', 'glob', 'pattern', 'list', 'directory', 'folder', 'file', '\uc5f4\uc5b4', '\uc77d\uc5b4', '\ubcf4\uc5ec', '\ucc3e\uc544', '\uac80\uc0c9', '\ud638\ucd9c', '\ucc38\uc870', '\ubaa9\ub85d', '\ub514\ub809\ud1a0\ub9ac', '\ud3f4\ub354', '\ud30c\uc77c'])),
        "ic=" + str(flag(context_signal, ['read', 'open', 'grep', 'search', 'find', 'glob', 'list', 'directory', 'folder', 'file', 'matches', 'occurrences', '\ucc3e\uc544', '\uac80\uc0c9', '\ud638\ucd9c', '\ucc38\uc870', '\ubaa9\ub85d', '\ub514\ub809\ud1a0\ub9ac', '\ud3f4\ub354', '\ud30c\uc77c'])),
        f"path={int(bool(paths))}",
        f"patt={int(bool(patterns))}",
        f"cmd={int(bool(cmds))}",
        f"nof={int(len(open_files) == 0)}",
        f"hof={int(len(open_files) > 0)}",
        f"rr={int(rtype == 'read')}",
        f"rg={int(rtype == 'grep')}",
        f"rb={int(rtype == 'glob')}",
        f"rl={int(rtype == 'list')}",
        "mh=" + str(flag(signal_text, ['edit', 'modify', 'change', 'update', 'tweak', 'patch', 'write', 'fix', 'refactor', 'replace', '\uc218\uc815', '\uace0\uccd0', '\ubc14\uafd4', '\ubcc0\uacbd'])),
        "wh=" + str(flag(signal_text, ['write', 'create file', 'new file', 'from scratch', 'save', '\uc0c8 \ud30c\uc77c', '\uc791\uc131', '\ub9cc\ub4e4'])),
        "nf=" + str(flag(signal_text, ['new file', 'create', 'add a file', 'create file', '\uc0c8 \ud30c\uc77c', '\uc0c8\ub85c', '\ub9cc\ub4e4'])),
        "fix=" + str(flag(signal_text, ['fix', 'bug', 'broken', 'error', 'failing', 'failed', 'failure', 'issue', '\ubc84\uadf8', '\uae68', '\uc5d0\ub7ec', '\uc2e4\ud328'])),
        "ref=" + str(flag(signal_text, ['refactor', 'cleanup', 'clean up', 'simplify', 'restructure', '\ub9ac\ud329\ud130', '\uc815\ub9ac', '\uad6c\uc870'])),
        "rep=" + str(flag(signal_text, ['replace', 'rewrite', 'swap', 'rename', 'substitute', '\ub300\uccb4', '\ub2e4\uc2dc \uc368', '\ubc14\uafd4\uc368'])),
        "mf=" + str(int(len(paths) >= 2 or flag(signal_text, ['multiple files', 'both', 'all of', 'across', 'together', '\ub450 \ud30c\uc77c', '\uc804\uccb4', '\uac19\uc774', '\ud55c \ubc88\uc5d0']))),
        "vh=" + str(flag(signal_text, ['test', 'tests', 'pytest', 'jest', 'vitest', 'spec', 'go test', 'cargo test', 'lint', 'eslint', 'ruff', 'pylint', 'clippy', 'typecheck', 'type check', 'tsc', 'mypy', 'pyright', 'build', 'compile', 'docker', 'image', '\ud14c\uc2a4\ud2b8', '\ub9b0\ud2b8', '\ud0c0\uc785', '\uc815\uc801\ubd84\uc11d', '\ube4c\ub4dc', '\ucef4\ud30c\uc77c'])),
        "tc=" + str(flag(signal_text, ['typecheck', 'type check', 'tsc', 'mypy', 'pyright', 'types', '\ud0c0\uc785', '\uc815\uc801\ubd84\uc11d'])),
        "bh=" + str(flag(signal_text, ['build', 'compile', 'docker', 'image', 'webpack', 'vite build', 'cargo build', '\ube4c\ub4dc', '\ucef4\ud30c\uc77c'])),
        "run=" + str(flag(signal_text, ['run', 'start', 'serve', 'execute', 'script', '\ub3cc\ub824', '\uc2e4\ud589', '\ub744\uc6cc'])),
        "sh=" + str(flag(signal_text, ['bash', 'shell', 'terminal', 'command', 'cmd', 'script', '\uba85\ub839', '\uc2a4\ud06c\ub9bd\ud2b8', '\uc258'])),
        "ci=" + str(flag(signal_text, ['ci', 'workflow', 'github actions', 'action', 'pipeline', 'red', 'green', '\uc6cc\ud06c\ud50c\ub85c', '\ud30c\uc774\ud504\ub77c\uc778'])),
        "rh=" + str(flag(signal_text, ['plan', 'steps', 'approach', 'strategy', 'outline', 'break down', 'explain', 'ask', 'question', 'latest', 'docs', 'should', '\uacc4\ud68d', '\ub2e8\uacc4', '\uc811\uadfc', '\uc804\ub7b5', '\uc124\uba85', '\ubb3c\uc5b4', '\uc9c8\ubb38'])),
        "resp=" + str(flag(signal_text, ['answer', 'respond', 'tell me', 'what is', 'why', 'explain', 'summarize', '\ub2f5\ubcc0', '\uc54c\ub824', '\ubb50\uc57c', '\uc65c', '\uc124\uba85'])),
        "ans=" + str(flag(signal_text, ['answer', 'tell me', 'what is', 'why', 'how does', '\uc54c\ub824', '\ubb50\uc57c', '\uc65c', '\uc124\uba85'])),
        "sum=" + str(flag(signal_text, ['summary', 'summarize', 'recap', 'explain', '\uc694\uc57d', '\uc815\ub9ac', '\uc124\uba85'])),
        "web=" + str(flag(signal_text, ['web', 'internet', 'online', 'google', 'browse', 'latest', 'recent', 'current', 'today', 'docs', 'official', 'version', '\uc6f9', '\uc778\ud130\ub137', '\uac80\uc0c9', '\ucd5c\uc2e0', '\uc694\uc998', '\uc624\ub298', '\ubb38\uc11c', '\uacf5\uc2dd'])),
        "doc=" + str(flag(signal_text, ['docs', 'documentation', 'official', 'reference', 'api reference', '\ubb38\uc11c', '\uacf5\uc2dd', '\ub808\ud37c\ub7f0\uc2a4'])),
        "lat=" + str(flag(signal_text, ['latest', 'recent', 'today', 'current', 'newest', 'version', '\ucd5c\uc2e0', '\uc694\uc998', '\uc624\ub298'])),
        "unc=" + str(flag(signal_text, ['not sure', 'unclear', 'ambiguous', 'maybe', 'confirm', '\uc560\ub9e4', '\ubaa8\ub974', '\ubd88\ud655\uc2e4', '\ud655\uc778'])),
        "cho=" + str(flag(signal_text, ['which', 'choose', 'option', 'prefer', 'or', '\uc5b4\ub290', '\uc120\ud0dd', '\uc635\uc158', '\ubb50\ub85c'])),
        "ask=" + str(flag(signal_text, ['ask', 'which', 'choose', 'confirm', 'clarify', 'ambiguous', 'option', 'should i', 'do you want', '\ubb3c\uc5b4', '\uc9c8\ubb38', '\ud655\uc778', '\uc5b4\ub290', '\uc120\ud0dd'])),
        "plan=" + str(flag(signal_text, ['plan', 'steps', 'approach', 'strategy', 'outline', 'break down', 'before', 'first', '\uacc4\ud68d', '\ub2e8\uacc4', '\uc811\uadfc', '\uc804\ub7b5'])),
        "test=" + str(flag(signal_text, ['test', 'tests', 'pytest', 'vitest', 'jest', 'spec', 'cargo test', 'go test', '\ud14c\uc2a4\ud2b8'])),
        "lint=" + str(flag(signal_text, ['lint', 'eslint', 'ruff', 'pylint', 'clippy', 'typecheck', 'type check', 'tsc', 'mypy', '\ub9b0\ud2b8', '\ud0c0\uc785'])),
        "edit=" + str(flag(signal_text, ['edit', 'modify', 'change', 'update', 'tweak', 'fix', '\uc218\uc815', '\uace0\uccd0', '\ubc14\uafd4', '\ubcc0\uacbd'])),
        "patch=" + str(flag(signal_text, ['patch', 'diff', 'apply', 'apply patch', 'multiple files', 'both files', '\ud328\uce58', '\ud55c \ubc88\uc5d0', '\uac19\uc774'])),
        f"lp={int(len(open_files) == 0)}",
    ]
    return "\n".join(
        [
            "[COMMON]",
            "",
            "[PROMPT]",
            prompt,
            "",
            "[LAST_USER]",
            last_user,
            "",
            "[FLOW]",
            f"last={last}",
            f"prev={prev}",
            f"last2={prev}>{last}",
            f"recent={' '.join(recent)}",
            f"bigrams={bigrams}",
            f"test_count={min(recent_action_count(actions, 'run_tests'), 5)}",
            f"lint_count={min(recent_action_count(actions, 'lint_or_typecheck'), 5)}",
            f"bash_count={min(recent_action_count(actions, 'run_bash'), 5)}",
            f"edit_count={min(recent_action_count(actions, 'edit_file'), 5)}",
            f"patch_count={min(recent_action_count(actions, 'apply_patch'), 5)}",
            f"write_count={min(recent_action_count(actions, 'write_file'), 5)}",
            "",
            "[HINTS]",
            " ".join(hint_items),
            "",
            "[ARGS]",
            f"paths={' '.join(paths)}",
            f"patterns={' '.join(patterns)}",
            f"cmds={' '.join(cmds)}",
            f"targets={' '.join(targets)}",
            f"n_files={recent_n_files_bucket(row)}",
            "",
            "[RESULT]",
            f"status={result_status(result)}",
            f"type={rtype}",
            f"count_files_bucket={bucket_count(files_count)}",
            f"count_matches_bucket={bucket_count(matches_count)}",
            f"count_lines_bucket={bucket_count(lines_count, cuts=(0, 20, 80, 200, 500, 1000))}",
            f"count_tests={count_tests}",
            f"count_failed_tests={count_failed}",
            f"count_added={count_added}",
            f"count_removed={count_removed}",
            f"exit_zero={exit_zero_flag(result)}",
            "",
            "[STATE]",
            f"git_dirty={ws.get('git_dirty', '')}",
            f"open_files_count={bucket_count(len(open_files), cuts=(0, 1, 2, 4, 8, 16))}",
            f"open_exts={' '.join(sorted(set(open_exts))) if open_exts else 'none'}",
            f"last_ci_status={compact(ws.get('last_ci_status'), 80) or 'unknown'}",
            f"top_lang={top_lang}",
            f"turn_bucket={bucket_turn(meta.get('turn_index'))}",
            f"budget_bucket={bucket_count(meta.get('budget_tokens_remaining'), cuts=(0, 5000, 10000, 20000, 50000))}",
            f"budget_critical={budget_critical(meta.get('budget_tokens_remaining'))}",
        ]
    )


def load_all_rows(data_dir):
    data_dir = Path(data_dir)
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    rows = load_jsonl(data_dir / "train.jsonl")
    ids = np.asarray([row["id"] for row in rows], dtype=object)
    sessions = np.asarray([session_id(row["id"]) for row in rows], dtype=object)
    y_action = np.asarray([labels[row["id"]] for row in rows], dtype=object)
    y_group = np.asarray([LABEL_TO_GROUP[label] for label in y_action], dtype=object)
    texts = [common_text(row) for row in rows]
    return rows, ids, sessions, y_action, y_group, texts


class MultiHeadDataset(Dataset):
    def __init__(self, texts, y_action=None, y_group=None, y_local=None, y_respond=None):
        self.texts = list(texts)
        self.y_action = y_action
        self.y_group = y_group
        self.y_local = y_local
        self.y_respond = y_respond

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx]}
        if self.y_action is not None:
            item["action_label"] = int(self.y_action[idx])
            item["group_label"] = int(self.y_group[idx])
            item["local_label"] = int(self.y_local[idx])
            item["respond_label"] = int(self.y_respond[idx])
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
        if "action_label" in batch[0]:
            encoded["action_labels"] = torch.tensor([item["action_label"] for item in batch], dtype=torch.long)
            encoded["group_labels"] = torch.tensor([item["group_label"] for item in batch], dtype=torch.long)
            encoded["local_labels"] = torch.tensor([item["local_label"] for item in batch], dtype=torch.long)
            encoded["respond_labels"] = torch.tensor([item["respond_label"] for item in batch], dtype=torch.long)
        return encoded

    return collate


class MultiHeadE5(nn.Module):
    def __init__(self, model_dir, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        hidden = int(getattr(self.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.group_head = nn.Linear(hidden, len(GROUPS))
        self.global_head = nn.Linear(hidden, len(FULL_LABELS))
        self.respond_head = nn.Linear(hidden, len(RESPOND_LABELS))
        self.specialist_heads = nn.ModuleDict(
            {group: nn.Linear(hidden, len(labels)) for group, labels in GROUP_TO_LABELS.items()}
        )

    def pool(self, output, attention_mask):
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        pooled = self.dropout(self.pool(output, attention_mask))
        return {
            "group": self.group_head(pooled),
            "global": self.global_head(pooled),
            "respond": self.respond_head(pooled),
            "specialist": {group: head(pooled) for group, head in self.specialist_heads.items()},
        }


def make_label_arrays(y_action, y_group):
    action_to_idx = {label: idx for idx, label in enumerate(FULL_LABELS)}
    group_to_idx = {group: idx for idx, group in enumerate(GROUPS)}
    local_maps = {
        group: {label: idx for idx, label in enumerate(labels)}
        for group, labels in GROUP_TO_LABELS.items()
    }
    y_action_idx = np.asarray([action_to_idx[label] for label in y_action], dtype=np.int64)
    y_group_idx = np.asarray([group_to_idx[group] for group in y_group], dtype=np.int64)
    y_local_idx = np.asarray(
        [local_maps[group].get(label, -1) for label, group in zip(y_action, y_group)],
        dtype=np.int64,
    )
    y_respond_idx = np.asarray([int(label == "respond_only") for label in y_action], dtype=np.int64)
    return y_action_idx, y_group_idx, y_local_idx, y_respond_idx


def compute_loss(outputs, batch, args):
    ce = nn.CrossEntropyLoss()
    group_labels = batch["group_labels"]
    action_labels = batch["action_labels"]
    local_labels = batch["local_labels"]
    respond_labels = batch["respond_labels"]
    loss_group = ce(outputs["group"], group_labels)
    loss_global = ce(outputs["global"], action_labels)
    reason_idx = int(np.where(GROUPS == "reason")[0][0])
    reason_mask = group_labels == reason_idx
    if reason_mask.any():
        loss_respond = ce(outputs["respond"][reason_mask], respond_labels[reason_mask])
    else:
        loss_respond = torch.tensor(0.0, device=group_labels.device)
    loss_spec = torch.tensor(0.0, device=group_labels.device)
    active_groups = 0
    for group_idx, group in enumerate(GROUPS):
        mask = (group_labels == group_idx) & (local_labels >= 0)
        if mask.any():
            loss_spec = loss_spec + ce(outputs["specialist"][str(group)][mask], local_labels[mask])
            active_groups += 1
    if active_groups:
        loss_spec = loss_spec / active_groups
    return (
        args.group_loss_weight * loss_group
        + args.spec_loss_weight * loss_spec
        + args.respond_loss_weight * loss_respond
        + args.global_loss_weight * loss_global
    )


def to_action_proba(outputs, global_blend):
    group_prob = torch.softmax(outputs["group"], dim=-1)
    global_prob = torch.softmax(outputs["global"], dim=-1)
    respond_prob = torch.softmax(outputs["respond"], dim=-1)
    hier_prob = torch.zeros_like(global_prob)
    action_to_idx = {label: idx for idx, label in enumerate(FULL_LABELS)}
    for group_idx, group in enumerate(GROUPS):
        spec_prob = torch.softmax(outputs["specialist"][str(group)], dim=-1)
        if str(group) == "reason":
            reason_mass = group_prob[:, group_idx]
            not_respond = respond_prob[:, 0]
            is_respond = respond_prob[:, 1]
            hier_prob[:, action_to_idx["respond_only"]] = reason_mass * is_respond
            for local_idx, label in enumerate(GROUP_TO_LABELS[str(group)]):
                hier_prob[:, action_to_idx[label]] = reason_mass * not_respond * spec_prob[:, local_idx]
            continue
        for local_idx, label in enumerate(GROUP_TO_LABELS[str(group)]):
            hier_prob[:, action_to_idx[label]] = group_prob[:, group_idx] * spec_prob[:, local_idx]
    if global_blend <= 0:
        return hier_prob
    return (1.0 - global_blend) * hier_prob + global_blend * global_prob


def predict(model, loader, device, global_blend):
    model.eval()
    action_chunks = []
    group_chunks = []
    global_chunks = []
    respond_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            model_inputs = {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(**model_inputs)
            action_chunks.append(to_action_proba(outputs, global_blend).detach().cpu().numpy())
            group_chunks.append(torch.softmax(outputs["group"], dim=-1).detach().cpu().numpy())
            global_chunks.append(torch.softmax(outputs["global"], dim=-1).detach().cpu().numpy())
            respond_chunks.append(torch.softmax(outputs["respond"], dim=-1).detach().cpu().numpy())
    return np.vstack(action_chunks), np.vstack(group_chunks), np.vstack(global_chunks), np.vstack(respond_chunks)


def train_fold(model, train_loader, valid_loader, y_valid, args, device, save_dir=None, tokenizer=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    best = {
        "macro_f1": -1.0,
        "epoch": -1,
        "proba": None,
        "group_proba": None,
        "global_proba": None,
        "respond_proba": None,
    }
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            model_inputs = {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.fp16)):
                outputs = model(**model_inputs)
                loss = compute_loss(outputs, batch, args) / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        valid_proba, valid_group_proba, valid_global_proba, valid_respond_proba = predict(
            model,
            valid_loader,
            device,
            args.valid_global_blend,
        )
        valid_pred = FULL_LABELS[valid_proba.argmax(axis=1)]
        macro_f1 = float(f1_score(y_valid, valid_pred, labels=FULL_LABELS, average="macro", zero_division=0))
        acc = float(accuracy_score(y_valid, valid_pred))
        print(
            f"  epoch={epoch} loss={np.mean(losses):.5f} valid_macro_f1={macro_f1:.6f} "
            f"acc={acc:.6f} sec={time.time() - t0:.1f}",
            flush=True,
        )
        if macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": macro_f1,
                "accuracy": acc,
                "epoch": epoch,
                "proba": valid_proba,
                "group_proba": valid_group_proba,
                "global_proba": valid_global_proba,
                "respond_proba": valid_respond_proba,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / "model.pt")
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_dir)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def write_metrics(out_dir, name, y_true, pred, labels):
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / f"{name}_class_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--fold-dir", default="experiments/oof/hierarchical_story_state_transition_sgd_targetctx_20260702_rerun")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-name", default="multihead_e5_common")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--group-loss-weight", type=float, default=0.5)
    parser.add_argument("--spec-loss-weight", type=float, default=1.0)
    parser.add_argument("--respond-loss-weight", type=float, default=0.5)
    parser.add_argument("--global-loss-weight", type=float, default=0.3)
    parser.add_argument("--valid-global-blend", type=float, default=0.0)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--folds", default=None, help="Comma-separated 0-based fold ids to run, e.g. 0,1,2")
    parser.add_argument("--debug-train-samples", type=int, default=None)
    parser.add_argument("--debug-valid-samples", type=int, default=None)
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

    rows, ids, sessions, y_action, y_group, texts = load_all_rows(args.data_dir)
    fold_dir = Path(args.fold_dir)
    fold_ids = np.load(fold_dir / "fold_ids.npy")
    fold_ids_ref = np.load(fold_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    if not np.array_equal(ids.astype(str), fold_ids_ref):
        raise ValueError("sample_ids do not align with fold_dir")
    y_action_idx, y_group_idx, y_local_idx, y_respond_idx = make_label_arrays(y_action, y_group)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"rows={len(texts)} folds={dict(zip(*np.unique(fold_ids, return_counts=True)))} "
        f"max_length={args.max_length}",
        flush=True,
    )

    oof_proba = np.zeros((len(texts), len(FULL_LABELS)), dtype=np.float32)
    oof_group = np.zeros((len(texts), len(GROUPS)), dtype=np.float32)
    oof_global = np.zeros((len(texts), len(FULL_LABELS)), dtype=np.float32)
    oof_respond = np.zeros((len(texts), len(RESPOND_LABELS)), dtype=np.float32)
    filled = np.zeros(len(texts), dtype=bool)
    fold_rows = []
    folds = sorted(np.unique(fold_ids))
    if args.folds:
        requested_folds = {int(item.strip()) for item in args.folds.split(",") if item.strip()}
        folds = [fold for fold in folds if int(fold) in requested_folds]
    if args.max_folds:
        folds = folds[: args.max_folds]
    if not folds:
        raise ValueError("No folds selected")

    for fold in folds:
        t0 = time.time()
        valid_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        if args.debug_train_samples:
            train_idx = train_idx[: args.debug_train_samples]
        if args.debug_valid_samples:
            valid_idx = valid_idx[: args.debug_valid_samples]
        print(f"fold={int(fold)+1} train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        model = MultiHeadE5(args.model_dir, gradient_checkpointing=args.gradient_checkpointing)
        model.to(device)
        collate = make_collate(tokenizer, args.max_length)
        train_loader = DataLoader(
            MultiHeadDataset(
                [texts[i] for i in train_idx],
                y_action_idx[train_idx],
                y_group_idx[train_idx],
                y_local_idx[train_idx],
                y_respond_idx[train_idx],
            ),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        valid_loader = DataLoader(
            MultiHeadDataset(
                [texts[i] for i in valid_idx],
                y_action_idx[valid_idx],
                y_group_idx[valid_idx],
                y_local_idx[valid_idx],
                y_respond_idx[valid_idx],
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        save_dir = out_dir / f"fold_{int(fold)+1}_model" if args.save_fold_models else None
        best = train_fold(model, train_loader, valid_loader, y_action[valid_idx], args, device, save_dir, tokenizer)
        oof_proba[valid_idx] = best["proba"]
        oof_group[valid_idx] = best["group_proba"]
        oof_global[valid_idx] = best["global_proba"]
        oof_respond[valid_idx] = best["respond_proba"]
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
    np.save(out_dir / "classes.npy", FULL_LABELS.astype(str))
    np.save(out_dir / "group_classes.npy", GROUPS.astype(str))
    np.save(out_dir / "sample_ids.npy", ids.astype(str))
    np.save(out_dir / "session_ids.npy", sessions.astype(str))
    np.save(out_dir / "y_true.npy", y_action.astype(str))
    np.save(out_dir / "y_group.npy", y_group.astype(str))
    np.save(out_dir / "fold_ids.npy", fold_ids)
    np.save(out_dir / "filled.npy", filled)
    np.save(out_dir / f"oof_{args.artifact_name}.npy", oof_proba)
    np.save(out_dir / f"oof_{args.artifact_name}_group.npy", oof_group)
    np.save(out_dir / f"oof_{args.artifact_name}_global.npy", oof_global)
    np.save(out_dir / f"oof_{args.artifact_name}_respond.npy", oof_respond)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{args.artifact_name}_folds.csv", index=False)

    y_eval = y_action[covered]
    pred = FULL_LABELS[oof_proba[covered].argmax(axis=1)]
    metrics = write_metrics(out_dir, args.artifact_name, y_eval, pred, FULL_LABELS)
    group_pred = GROUPS[oof_group[covered].argmax(axis=1)]
    group_metrics = write_metrics(out_dir, f"{args.artifact_name}_group", y_group[covered], group_pred, GROUPS)
    global_pred = FULL_LABELS[oof_global[covered].argmax(axis=1)]
    global_metrics = write_metrics(out_dir, f"{args.artifact_name}_global", y_eval, global_pred, FULL_LABELS)
    reason_mask = covered & (y_group == "reason")
    respond_true = np.where(y_action[reason_mask] == "respond_only", "respond_only", "not_respond_only")
    respond_pred = RESPOND_LABELS[oof_respond[reason_mask].argmax(axis=1)]
    respond_metrics = write_metrics(
        out_dir,
        f"{args.artifact_name}_respond_reason_only",
        respond_true,
        respond_pred,
        RESPOND_LABELS,
    )

    blend_rows = []
    for blend in np.linspace(0.0, 0.6, 13):
        blended = (1.0 - blend) * oof_proba[covered] + blend * oof_global[covered]
        blend_pred = FULL_LABELS[blended.argmax(axis=1)]
        blend_rows.append(
            {
                "global_blend": float(blend),
                "macro_f1": float(f1_score(y_eval, blend_pred, labels=FULL_LABELS, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_eval, blend_pred)),
            }
        )
    blend_df = pd.DataFrame(blend_rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    blend_df.to_csv(out_dir / f"{args.artifact_name}_blend_search.csv", index=False)
    summary = {
        "multihead": metrics,
        "group": group_metrics,
        "global": global_metrics,
        "respond_reason_only": respond_metrics,
        "best_blend": blend_df.iloc[0].to_dict(),
        "folds": fold_rows,
        "covered_rows": int(covered.sum()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
