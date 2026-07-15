import argparse
import csv
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion


MODEL_PRESETS = {
    "prompt_word": {
        "text": "prompt",
        "vectorizer": "word",
        "state": False,
        "prior": False,
        "artifact": "prompt_word",
        "note": "current_prompt TF-IDF baseline",
    },
    "prompt_word_char": {
        "text": "prompt",
        "vectorizer": "word_char",
        "state": False,
        "prior": False,
        "artifact": "prompt_word_char",
        "note": "current_prompt word+char TF-IDF",
    },
    "story_word": {
        "text": "story",
        "vectorizer": "word",
        "state": False,
        "prior": False,
        "artifact": "story_word",
        "note": "story_text TF-IDF baseline",
    },
    "story_word_char": {
        "text": "story",
        "vectorizer": "word_char",
        "state": False,
        "prior": False,
        "artifact": "story_word_char",
        "note": "story_text word+char TF-IDF",
    },
    "story_state": {
        "text": "story_state",
        "vectorizer": "word_char",
        "state": False,
        "prior": False,
        "artifact": "story_state",
        "note": "story_text with explicit state tokens",
    },
    "state": {
        "text": None,
        "vectorizer": None,
        "state": True,
        "prior": False,
        "artifact": "state",
        "note": "session state and previous action feature model",
    },
    "transition": {
        "text": None,
        "vectorizer": None,
        "state": False,
        "prior": True,
        "artifact": "transition",
        "note": "fold-safe transition prior feature model",
    },
    "story_state_transition": {
        "text": "story_state",
        "vectorizer": "word_char",
        "state": False,
        "prior": True,
        "artifact": "story_state_transition",
        "note": "story_state plus fold-safe transition prior features",
    },
    "jm_text": {
        "text": "jm_text",
        "vectorizer": "jm_word_char",
        "state": False,
        "prior": False,
        "artifact": "jm_text",
        "note": "JM-style engineered text with prompt repeats, action n-grams, keywords, and meta tokens",
    },
    "jm_text_transition": {
        "text": "jm_text",
        "vectorizer": "jm_word_char",
        "state": False,
        "prior": True,
        "artifact": "jm_text_transition",
        "note": "JM-style engineered text plus fold-safe transition prior features",
    },
    "five_view": {
        "text": None,
        "views": True,
        "vectorizer": "five_view",
        "state": True,
        "prior": False,
        "artifact": "five_view",
        "note": "5 separated TF-IDF views plus structured session/action features",
    },
    "five_view_transition": {
        "text": None,
        "views": True,
        "vectorizer": "five_view",
        "state": True,
        "prior": True,
        "artifact": "five_view_transition",
        "note": "5 separated TF-IDF views plus structured features and fold-safe transition prior",
    },
}

DEFAULT_MODELS = [
    "prompt_word",
    "prompt_word_char",
    "story_word",
    "story_word_char",
    "story_state",
    "state",
    "transition",
    "story_state_transition",
    "jm_text",
    "jm_text_transition",
    "five_view",
    "five_view_transition",
]


@dataclass
class Dataset:
    ids: list[str]
    rows: list[dict]
    y: np.ndarray
    groups: np.ndarray
    classes: np.ndarray


def compact(value, limit=500):
    if value is None:
        return ""
    text = str(value).replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def fast_compact(value, limit=500):
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit]


def session_id(sample_id):
    return sample_id.split("-step_")[0]


def bucket(value, size):
    try:
        return str(int(float(value) // size))
    except Exception:
        return "na"


def bucket_le(value, bins):
    try:
        x = float(value)
    except Exception:
        return "unknown"
    for bound in bins:
        if x <= bound:
            return f"le_{bound}"
    return f"gt_{bins[-1]}"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_dataset(data_dir, max_sessions=None):
    data_dir = Path(data_dir)
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}

    rows = load_jsonl(data_dir / "train.jsonl")
    if max_sessions:
        seen = set()
        keep_sessions = set()
        for row in rows:
            sid = session_id(row["id"])
            if sid not in seen:
                seen.add(sid)
                if len(keep_sessions) < max_sessions:
                    keep_sessions.add(sid)
            if len(keep_sessions) >= max_sessions and len(seen) > max_sessions:
                break
        rows = [row for row in rows if session_id(row["id"]) in keep_sessions]

    ids = [row["id"] for row in rows]
    y = np.array([labels[row["id"]] for row in rows])
    groups = np.array([session_id(row["id"]) for row in rows])
    classes = np.array(sorted(set(y)))
    return Dataset(ids=ids, rows=rows, y=y, groups=groups, classes=classes)


def previous_actions(row):
    names = []
    for item in row.get("history") or []:
        name = item.get("name")
        if name:
            names.append(str(name))
    return names


PATH_RE = re.compile(
    r"([A-Za-z]:[\\/]|[./\\][\w./\\-]+|\b[\w./\\-]+\.(py|js|ts|tsx|json|csv|md|txt|yaml|yml|ipynb|go|rs|java|kt|css|scss|html|toml|ini|cfg|lock)\b)",
    flags=re.IGNORECASE,
)


def has_any(text, needles):
    lower = str(text).lower()
    return any(needle in lower for needle in needles)


def inspect_context_tokens(row):
    current = prompt_text(row)
    current_l = current.lower()
    history = row.get("history") or []
    actions = previous_actions(row)
    tokens = []

    prompt_paths = PATH_RE.findall(current)
    has_prompt_path = bool(prompt_paths)
    has_wildcard = bool(re.search(r"(\*\.[a-z0-9]+|\*\*/|\*)", current_l))
    if has_prompt_path:
        tokens.append("inspect_prompt_has_path")
        tokens.append("inspect_prompt_path_count_" + bucket_le(len(prompt_paths), [1, 2, 4]))
    if has_wildcard:
        tokens.append("inspect_prompt_has_wildcard")

    read_intent = has_any(
        current_l,
        [
            "open",
            "read",
            "show me",
            "pull up",
            "cat ",
            "view",
            "look at",
            "\uc5f4\uc5b4",
            "\uc77d",
            "\ubcf4\uc5ec",
            "\ubd10",
        ],
    )
    search_intent = has_any(
        current_l,
        [
            "where",
            "where else",
            "find",
            "search",
            "grep",
            "reference",
            "references",
            "usage",
            "usages",
            "occurrence",
            "occurrences",
            "who calls",
            "called from",
            "defined",
            "\uc5b4\ub514",
            "\uc5b4\ub514\uc11c",
            "\ucc3e",
            "\uac80\uc0c9",
            "\ud638\ucd9c",
            "\ucc38\uc870",
            "\uc815\uc758",
            "\uc4f0",
        ],
    )
    list_intent = has_any(
        current_l,
        [
            "list",
            "ls ",
            "tree",
            "directory",
            "folder",
            "what files",
            "which files",
            "contents of",
            "\ubaa9\ub85d",
            "\ud3f4\ub354",
            "\ub514\ub809\ud1a0\ub9ac",
            "\ubb50\ubb50",
            "\uad6c\uc870",
        ],
    )
    glob_intent = has_wildcard or has_any(
        current_l,
        [
            "glob",
            "pattern",
            "matching files",
            "all files",
            "every file",
            "recursive",
            "recursively",
            "extension",
            "\ud328\ud134",
            "\ud655\uc7a5\uc790",
            "\uc804\ubd80",
            "\uc7ac\uadc0",
        ],
    )
    deixis = has_any(
        current_l,
        [
            "it",
            "that",
            "that file",
            "the file",
            "the one",
            "this one",
            "one of",
            "\uadf8\uac70",
            "\uc774\uac70",
            "\uadf8 \ud30c\uc77c",
            "\uc774 \ud30c\uc77c",
        ],
    )

    for name, enabled in [
        ("inspect_intent_read", read_intent),
        ("inspect_intent_search", search_intent),
        ("inspect_intent_list", list_intent),
        ("inspect_intent_glob", glob_intent),
        ("inspect_deictic_target", deixis),
    ]:
        if enabled:
            tokens.append(name)

    last_action = actions[-1] if actions else "NONE"
    if actions:
        tokens.append("inspect_last_action_" + last_action)

    last_result = ""
    last_args = ""
    for item in reversed(history):
        if not last_result and item.get("result_summary"):
            last_result = compact(item.get("result_summary"), 500).lower()
        if not last_args:
            args = item.get("args")
            if isinstance(args, dict):
                last_args = " ".join(f"{k}={compact(v, 250)}" for k, v in args.items()).lower()
            elif args:
                last_args = compact(args, 350).lower()
        if last_result and last_args:
            break

    last_context = " ".join([last_action.lower(), last_result, last_args])
    result_paths = PATH_RE.findall(last_context)
    if result_paths:
        tokens.append("inspect_recent_result_has_path")
        tokens.append("inspect_recent_result_path_count_" + bucket_le(len(result_paths), [1, 2, 4, 8]))
    if re.search(r"\b\d+\s+(entries|items)\b", last_result) or "listed" in last_result:
        tokens.append("inspect_recent_result_listed")
    if re.search(r"\b\d+\s+(files?\s+)?matched\b", last_result) or "files matched" in last_result:
        tokens.append("inspect_recent_result_globbed")
    if re.search(r"\b\d+\s+(matches|occurrences)\b", last_result) or "found " in last_result:
        tokens.append("inspect_recent_result_grepped")
    if "read " in last_result or "read_file" in last_context:
        tokens.append("inspect_recent_result_read")

    known_target = has_prompt_path or (
        deixis
        and (
            last_action in {"read_file", "grep_search", "glob_pattern", "list_directory"}
            or bool(result_paths)
            or any(tok.startswith("inspect_recent_result_") for tok in tokens)
        )
    )
    needs_discovery = search_intent and not known_target
    if known_target:
        tokens.append("inspect_target_known")
    if needs_discovery:
        tokens.append("inspect_target_unknown_search")

    for intent_name, enabled in [
        ("read", read_intent),
        ("search", search_intent),
        ("list", list_intent),
        ("glob", glob_intent),
    ]:
        if enabled:
            tokens.append(f"inspect_combo_last_{last_action}_intent_{intent_name}")
            if known_target:
                tokens.append(f"inspect_combo_known_intent_{intent_name}")

    return tokens


def state_tokens(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = ws.get("open_files") or []
    lang_mix = ws.get("language_mix") or {}
    tokens = [
        "tier_" + compact(meta.get("user_tier"), 80),
        "lang_" + compact(meta.get("language_pref"), 80),
        "ci_" + compact(ws.get("last_ci_status"), 80),
        "dirty_" + str(ws.get("git_dirty", "")),
        "turn_" + bucket(meta.get("turn_index"), 1),
        "budget_bin_" + bucket(meta.get("budget_tokens_remaining"), 10000),
        "elapsed_bin_" + bucket(meta.get("elapsed_session_sec"), 60),
        "loc_bin_" + bucket(ws.get("loc"), 5000),
    ]
    tokens.extend("open_file_" + compact(x, 120) for x in open_files)
    tokens.extend("langmix_" + compact(k, 40) for k in lang_mix.keys())
    tokens.extend(inspect_context_tokens(row))
    return [t for t in tokens if t and not t.endswith("_")]


def prompt_text(row):
    return compact(row.get("current_prompt"), 2200)


def keyword_tokens(text):
    t = str(text).lower()
    groups = {
        "KW_READ_FILE": [
            "read file",
            "open file",
            "view file",
            "show file",
            "inspect file",
            "cat ",
            "print file",
            "check file contents",
            "파일 읽",
            "파일 열",
            "파일 확인",
            "내용 확인",
            "열어봐",
            "읽어봐",
        ],
        "KW_GREP_SEARCH": [
            "grep",
            "search for",
            "find occurrences",
            "find references",
            "look for",
            "where is",
            "keyword",
            "symbol",
            "검색",
            "찾아",
            "찾기",
            "참조",
            "어디",
        ],
        "KW_GLOB_PATTERN": [
            "glob",
            "pattern",
            "*.py",
            "*.json",
            "*.csv",
            "*.txt",
            "*.md",
            "**/",
            "matching files",
            "file extension",
            "패턴",
            "확장자",
            "파일들",
        ],
        "KW_LIST_DIRECTORY": [
            "list directory",
            "list files",
            "show files",
            "directory contents",
            "folder contents",
            "ls ",
            "tree",
            "what files",
            "목록",
            "디렉토리",
            "폴더",
            "파일 목록",
            "구조",
        ],
        "KW_RUN_TESTS": [
            "pytest",
            "unittest",
            "unit test",
            "run tests",
            "test suite",
            "npm test",
            "yarn test",
            "cargo test",
            "go test",
            "make test",
            "테스트",
            "시험",
        ],
        "KW_LINT_TYPECHECK": [
            "lint",
            "typecheck",
            "type check",
            "mypy",
            "ruff",
            "flake8",
            "eslint",
            "tsc",
            "pyright",
            "prettier",
            "린트",
            "타입체크",
            "타입 체크",
            "정적 검사",
        ],
        "KW_RUN_BASH": [
            "bash",
            "shell",
            "terminal",
            "command",
            "execute",
            "run command",
            "install",
            "chmod",
            "python script",
            "명령",
            "실행",
            "터미널",
            "설치",
        ],
        "KW_ASK_USER": [
            "clarify",
            "ask user",
            "confirm",
            "ambiguous",
            "need more information",
            "which one",
            "should i",
            "확인",
            "물어",
            "모호",
            "어느",
            "선택",
        ],
        "KW_PLAN_TASK": [
            "plan",
            "steps",
            "approach",
            "break down",
            "todo",
            "strategy",
            "outline",
            "계획",
            "단계",
            "전략",
            "정리",
        ],
        "KW_WEB_SEARCH": [
            "web",
            "internet",
            "online",
            "browse",
            "latest",
            "current",
            "up to date",
            "recent",
            "search online",
            "웹",
            "인터넷",
            "최신",
            "최근",
            "검색해",
            "찾아봐",
        ],
        "KW_APPLY_PATCH": [
            "patch",
            "diff",
            "apply patch",
            "fix bug",
            "change code",
            "패치",
            "수정",
            "고쳐",
            "버그",
        ],
        "KW_EDIT_FILE": [
            "edit file",
            "rewrite",
            "replace",
            "update file",
            "modify file",
            "파일 수정",
            "수정해",
            "바꿔",
            "교체",
            "업데이트",
        ],
        "KW_WRITE_FILE": [
            "create file",
            "write file",
            "new file",
            "save as",
            "파일 생성",
            "새 파일",
            "작성",
            "저장",
        ],
    }
    return " ".join(name for name, needles in groups.items() if any(needle in t for needle in needles))


def keyword_tokens(text):
    t = str(text).lower()
    groups = {
        "KW_READ_FILE": [
            "read file",
            "open file",
            "view file",
            "show file",
            "inspect file",
            "cat ",
            "print file",
            "check file contents",
            "\ud30c\uc77c",
            "\uc5f4\uc5b4",
            "\uc77d",
            "\ubcf4\uc5ec",
            "\ud655\uc778",
            "\ub0b4\uc6a9 \ud655\uc778",
        ],
        "KW_GREP_SEARCH": [
            "grep",
            "search for",
            "find occurrences",
            "find references",
            "look for",
            "where is",
            "keyword",
            "symbol",
            "\uac80\uc0c9",
            "\ucc3e",
            "\ucc3e\uae30",
            "\ucc38\uc870",
            "\uc704\uce58",
        ],
        "KW_GLOB_PATTERN": [
            "glob",
            "pattern",
            "*.py",
            "*.json",
            "*.csv",
            "*.txt",
            "*.md",
            "**/",
            "matching files",
            "file extension",
            "\ud328\ud134",
            "\ud655\uc7a5\uc790",
            "\ud30c\uc77c\ub4e4",
        ],
        "KW_LIST_DIRECTORY": [
            "list directory",
            "list files",
            "show files",
            "directory contents",
            "folder contents",
            "ls ",
            "tree",
            "what files",
            "\ubaa9\ub85d",
            "\ub514\ub809\ud1a0\ub9ac",
            "\ud3f4\ub354",
            "\ud30c\uc77c \ubaa9\ub85d",
            "\uad6c\uc870",
        ],
        "KW_RUN_TESTS": [
            "pytest",
            "unittest",
            "unit test",
            "run tests",
            "test suite",
            "npm test",
            "yarn test",
            "cargo test",
            "go test",
            "make test",
            "happy path",
            "profile tests",
            "\ud14c\uc2a4\ud2b8",
            "\uc2e4\ud5d8",
        ],
        "KW_LINT_TYPECHECK": [
            "lint",
            "typecheck",
            "type check",
            "type-check",
            "mypy",
            "ruff",
            "flake8",
            "eslint",
            "tsc",
            "pyright",
            "prettier",
            "\ub9b0\ud2b8",
            "\ud0c0\uc785\uccb4\ud06c",
            "\ud0c0\uc785 \uccb4\ud06c",
            "\uc815\uc801 \uac80\uc0ac",
        ],
        "KW_RUN_BASH": [
            "bash",
            "shell",
            "terminal",
            "command",
            "execute",
            "run command",
            "install",
            "chmod",
            "python script",
            "docker",
            "\uba85\ub839",
            "\uc2e4\ud589",
            "\ud130\ubbf8\ub110",
            "\uc124\uce58",
        ],
        "KW_ASK_USER": [
            "clarify",
            "ask user",
            "confirm",
            "ambiguous",
            "need more information",
            "which one",
            "should i",
            "\ud655\uc778",
            "\ubb3c\uc5b4",
            "\ubaa8\ud638",
            "\uc5b4\ub290",
            "\uc120\ud0dd",
        ],
        "KW_PLAN_TASK": [
            "plan",
            "steps",
            "approach",
            "break down",
            "todo",
            "strategy",
            "outline",
            "\uacc4\ud68d",
            "\ub2e8\uacc4",
            "\uc804\ub7b5",
            "\uc815\ub9ac",
        ],
        "KW_WEB_SEARCH": [
            "web",
            "internet",
            "online",
            "browse",
            "latest",
            "up to date",
            "recent",
            "search online",
            "\uc6f9",
            "\uc778\ud130\ub137",
            "\ucd5c\uc2e0",
            "\ucd5c\uadfc",
            "\uac80\uc0c9\ud574",
            "\ucc3e\uc544\ubd10",
        ],
        "KW_APPLY_PATCH": [
            "patch",
            "diff",
            "apply patch",
            "fix bug",
            "change code",
            "\ud328\uce58",
            "\uc218\uc815",
            "\uace0\uccd0",
            "\ubc84\uadf8",
        ],
        "KW_EDIT_FILE": [
            "edit file",
            "rewrite",
            "replace",
            "update file",
            "modify file",
            "\ud30c\uc77c \uc218\uc815",
            "\uc218\uc815\ud574",
            "\ubc14\uafd4",
            "\uad50\uccb4",
            "\uc5c5\ub370\uc774\ud2b8",
        ],
        "KW_WRITE_FILE": [
            "create file",
            "write file",
            "new file",
            "save as",
            "\ud30c\uc77c \uc0dd\uc131",
            "\uc0c8 \ud30c\uc77c",
            "\uc791\uc131",
            "\uc800\uc7a5",
        ],
    }
    return " ".join(name for name, needles in groups.items() if any(needle in t for needle in needles))


def jm_text(row):
    current = prompt_text(row)
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    history = row.get("history") or []

    actions = [fast_compact(item.get("name"), 80) for item in history if item.get("name")]

    user_msgs = []
    result_summaries = []
    action_args = []
    for item in reversed(history):
        role = item.get("role", "")
        if len(user_msgs) < 4 and role == "user" and item.get("content"):
            user_msgs.append(fast_compact(item.get("content"), 500))
        if len(result_summaries) < 4 and item.get("result_summary"):
            result_summaries.append(fast_compact(item.get("result_summary"), 250))
        if len(action_args) < 4:
            args = item.get("args")
            if isinstance(args, dict):
                action_args.append(" ".join(f"{k}_{fast_compact(v, 200)}" for k, v in args.items()))
            elif args:
                action_args.append(fast_compact(args, 300))
        if len(user_msgs) >= 4 and len(result_summaries) >= 4 and len(action_args) >= 4:
            break
    user_msgs.reverse()
    result_summaries.reverse()
    action_args.reverse()

    last_action = actions[-1] if actions else "none"
    last2_action = "_".join(actions[-2:]) if len(actions) >= 2 else last_action
    last3_action = "_".join(actions[-3:]) if len(actions) >= 3 else last2_action
    action_bigrams = [f"{a}__{b}" for a, b in zip(actions[:-1], actions[1:])]
    action_trigrams = [f"{a}__{b}__{c}" for a, b, c in zip(actions[:-2], actions[1:-1], actions[2:])]

    open_files = [fast_compact(x, 120) for x in (ws.get("open_files") or [])]
    open_file_text = " ".join(open_files[-8:])
    open_ext_text = " ".join(f"ext_{Path(str(x)).suffix.lower() or 'no_ext'}" for x in open_files)

    lang_mix_items = []
    for key, value in (ws.get("language_mix") or {}).items():
        try:
            lang_mix_items.append(f"lang_{fast_compact(key, 40)}_{round(float(value), 2)}")
        except Exception:
            lang_mix_items.append(f"lang_{fast_compact(key, 40)}_unknown")

    meta_text = " ".join(
        [
            "user_tier_" + fast_compact(meta.get("user_tier"), 80),
            "language_pref_" + fast_compact(meta.get("language_pref"), 80),
            "git_dirty_" + str(ws.get("git_dirty", "unknown")),
            "ci_" + fast_compact(ws.get("last_ci_status"), 80),
            "budget_" + bucket_le(meta.get("budget_tokens_remaining"), [1000, 5000, 10000, 30000, 70000, 150000]),
            "turn_" + bucket_le(meta.get("turn_index"), [1, 3, 5, 10, 20, 40]),
            "elapsed_" + bucket_le(meta.get("elapsed_session_sec"), [60, 300, 900, 1800, 3600, 7200]),
            "loc_" + bucket_le(ws.get("loc"), [100, 1000, 5000, 20000, 100000]),
            " ".join(lang_mix_items),
            open_file_text,
            open_ext_text,
        ]
    )

    combined_for_keywords = " ".join(
        [
            current,
            " ".join(user_msgs),
            " ".join(result_summaries),
            " ".join(action_args),
            open_file_text,
        ]
    )

    return "\n".join(
        [
            "CURRENT_PROMPT " + current,
            "CURRENT_PROMPT2 " + current,
            "CURRENT_PROMPT3 " + current,
            "RECENT_USER_MESSAGES " + " ".join(user_msgs),
            "ALL_ACTIONS " + " ".join(actions),
            "LAST_ACTION " + last_action,
            "LAST2_ACTION " + last2_action,
            "LAST3_ACTION " + last3_action,
            "ACTION_BIGRAMS " + " ".join(action_bigrams),
            "ACTION_TRIGRAMS " + " ".join(action_trigrams),
            "RECENT_RESULTS " + " ".join(result_summaries),
            "RECENT_ARGS " + " ".join(action_args),
            "RECENT_ARGS2 " + " ".join(action_args),
            "KEYWORDS " + keyword_tokens(combined_for_keywords),
            "META " + meta_text,
        ]
    )


def story_text(row, with_state=False):
    parts = []
    if with_state:
        parts.append("STATE " + " ".join(state_tokens(row)))

    actions = previous_actions(row)
    if actions:
        parts.append("PREVIOUS_ACTION_SEQUENCE " + " ".join(actions))

    user_turns = []
    tool_args = []
    results = []
    for item in (row.get("history") or [])[-12:]:
        role = item.get("role")
        if role == "user" and item.get("content"):
            user_turns.append(compact(item.get("content"), 500))
        if item.get("args"):
            args = item.get("args")
            if isinstance(args, dict):
                tool_args.extend(f"{k}={compact(v, 200)}" for k, v in args.items())
            else:
                tool_args.append(compact(args, 300))
        if item.get("result_summary"):
            results.append(compact(item.get("result_summary"), 250))

    parts.extend(
        [
            "PREVIOUS_USER_REQUESTS " + " ".join(user_turns[-5:]),
            "PREVIOUS_TOOL_ARGS " + " ".join(tool_args[-12:]),
            "PREVIOUS_RESULTS " + " ".join(results[-8:]),
            "CURRENT_PROMPT " + prompt_text(row),
        ]
    )
    return "\n".join(parts)


FIVE_VIEW_NAMES = (
    "current_prompt",
    "history_user_text",
    "action_names_sequence",
    "action_args_and_paths",
    "result_summary",
)


def five_view_texts(row):
    history = row.get("history") or []
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}

    user_turns = []
    actions = []
    args_and_paths = []
    results = []
    for item in history:
        if item.get("role") == "user" and item.get("content"):
            user_turns.append(compact(item.get("content"), 700))

        name = item.get("name")
        if name:
            actions.append(fast_compact(name, 80))

        args = item.get("args")
        if isinstance(args, dict):
            args_and_paths.extend(f"{key}={compact(value, 250)}" for key, value in args.items())
        elif args:
            args_and_paths.append(compact(args, 350))

        if item.get("result_summary"):
            results.append(compact(item.get("result_summary"), 400))

    action_bigrams = [f"{a}__{b}" for a, b in zip(actions[:-1], actions[1:])]
    action_trigrams = [f"{a}__{b}__{c}" for a, b, c in zip(actions[:-2], actions[1:-1], actions[2:])]
    last_action = actions[-1] if actions else "none"
    last2_action = "__".join(actions[-2:]) if len(actions) >= 2 else last_action
    last3_action = "__".join(actions[-3:]) if len(actions) >= 3 else last2_action

    open_files = [compact(path, 240) for path in (ws.get("open_files") or [])]
    open_exts = [Path(str(path)).suffix.lower() or "no_ext" for path in open_files]
    args_view = " ".join(args_and_paths[-16:] + open_files[-12:] + [f"ext_{ext}" for ext in open_exts[-12:]])

    return {
        "current_prompt": prompt_text(row),
        "history_user_text": " ".join(user_turns[-8:]),
        "action_names_sequence": " ".join(
            [
                "ALL_ACTIONS",
                " ".join(actions),
                "LAST_ACTION",
                last_action,
                "LAST2_ACTION",
                last2_action,
                "LAST3_ACTION",
                last3_action,
                "ACTION_BIGRAMS",
                " ".join(action_bigrams),
                "ACTION_TRIGRAMS",
                " ".join(action_trigrams),
            ]
        ),
        "action_args_and_paths": args_view,
        "result_summary": " ".join(results[-10:]),
    }


def build_view_texts(rows):
    views = {name: [] for name in FIVE_VIEW_NAMES}
    for row in rows:
        row_views = five_view_texts(row)
        for name in FIVE_VIEW_NAMES:
            views[name].append(row_views[name])
    return views


def build_texts(rows, mode):
    if mode == "prompt":
        return [prompt_text(row) for row in rows]
    if mode == "story":
        return [story_text(row, with_state=False) for row in rows]
    if mode == "story_state":
        return [story_text(row, with_state=True) for row in rows]
    if mode == "jm_text":
        return [jm_text(row) for row in rows]
    raise ValueError(f"unknown text mode: {mode}")


def make_vectorizer(kind, args):
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=args.word_min_df,
            max_features=args.word_max_features,
            sublinear_tf=True,
        )
    if kind == "word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=args.word_min_df,
                        max_features=args.word_max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=args.char_min_df,
                        max_features=args.char_max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    if kind == "jm_word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=args.jm_word_min_df,
                        max_features=args.jm_word_max_features,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=args.jm_char_min_df,
                        max_features=args.jm_char_max_features,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown vectorizer kind: {kind}")


def make_word_tfidf(*, ngram_range, min_df, max_features, token_pattern=None):
    kwargs = {
        "analyzer": "word",
        "ngram_range": ngram_range,
        "min_df": min_df,
        "max_features": max_features,
        "sublinear_tf": True,
    }
    if token_pattern is not None:
        kwargs["token_pattern"] = token_pattern
    return TfidfVectorizer(**kwargs)


def make_char_tfidf(*, ngram_range, min_df, max_features):
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def make_five_view_vectorizers(args):
    token_pattern = r"(?u)\b[\w./\\:-]+\b"
    return [
        (
            "current_prompt",
            FeatureUnion(
                [
                    (
                        "word",
                        make_word_tfidf(
                            ngram_range=(1, 2),
                            min_df=args.view_word_min_df,
                            max_features=args.view_current_word_max_features,
                        ),
                    ),
                    (
                        "char_wb",
                        make_char_tfidf(
                            ngram_range=(3, 5),
                            min_df=args.view_char_min_df,
                            max_features=args.view_current_char_max_features,
                        ),
                    ),
                ]
            ),
        ),
        (
            "history_user_text",
            FeatureUnion(
                [
                    (
                        "word",
                        make_word_tfidf(
                            ngram_range=(1, 2),
                            min_df=args.view_word_min_df,
                            max_features=args.view_history_word_max_features,
                        ),
                    ),
                    (
                        "char_wb",
                        make_char_tfidf(
                            ngram_range=(3, 5),
                            min_df=args.view_char_min_df,
                            max_features=args.view_history_char_max_features,
                        ),
                    ),
                ]
            ),
        ),
        (
            "action_names_sequence",
            make_word_tfidf(
                ngram_range=(1, 4),
                min_df=args.view_action_min_df,
                max_features=args.view_action_max_features,
                token_pattern=token_pattern,
            ),
        ),
        (
            "action_args_and_paths",
            FeatureUnion(
                [
                    (
                        "word",
                        make_word_tfidf(
                            ngram_range=(1, 2),
                            min_df=args.view_word_min_df,
                            max_features=args.view_args_word_max_features,
                            token_pattern=token_pattern,
                        ),
                    ),
                    (
                        "char_wb",
                        make_char_tfidf(
                            ngram_range=(3, 6),
                            min_df=args.view_char_min_df,
                            max_features=args.view_args_char_max_features,
                        ),
                    ),
                ]
            ),
        ),
        (
            "result_summary",
            FeatureUnion(
                [
                    (
                        "word",
                        make_word_tfidf(
                            ngram_range=(1, 2),
                            min_df=args.view_word_min_df,
                            max_features=args.view_result_word_max_features,
                        ),
                    ),
                    (
                        "char_wb",
                        make_char_tfidf(
                            ngram_range=(3, 6),
                            min_df=args.view_char_min_df,
                            max_features=args.view_result_char_max_features,
                        ),
                    ),
                ]
            ),
        ),
    ]


def safe_fit_transform(vectorizer, train_texts, valid_texts):
    try:
        return vectorizer.fit_transform(train_texts), vectorizer.transform(valid_texts)
    except ValueError as exc:
        if "empty vocabulary" not in str(exc).lower():
            raise
        return (
            sparse.csr_matrix((len(train_texts), 0), dtype=np.float32),
            sparse.csr_matrix((len(valid_texts), 0), dtype=np.float32),
        )


def build_five_view_features(train_rows, valid_rows, args, train_views=None, valid_views=None):
    if train_views is None:
        train_views = build_view_texts(train_rows)
    if valid_views is None:
        valid_views = build_view_texts(valid_rows)

    train_parts = []
    valid_parts = []
    for view_name, vectorizer in make_five_view_vectorizers(args):
        x_train, x_valid = safe_fit_transform(vectorizer, train_views[view_name], valid_views[view_name])
        train_parts.append(x_train)
        valid_parts.append(x_valid)
    return sparse.hstack(train_parts, format="csr"), sparse.hstack(valid_parts, format="csr")


def state_dict(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    history = row.get("history") or []
    current = prompt_text(row)
    results = [
        compact(item.get("result_summary"), 500)
        for item in history
        if item.get("result_summary")
    ]
    result_text = " ".join(results).lower()
    actions = previous_actions(row)
    action_counts = Counter(actions)
    open_files = ws.get("open_files") or []
    language_mix = ws.get("language_mix") or {}
    language_ranked = []
    if isinstance(language_mix, dict):
        try:
            language_ranked = [
                str(key)
                for key, _ in sorted(
                    language_mix.items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
            ]
        except Exception:
            language_ranked = [str(key) for key in language_mix.keys()]

    has_path_in_prompt = bool(
        re.search(
            r"([A-Za-z]:[\\/]|[./\\][\w./\\-]+|\b[\w.-]+\.(py|js|ts|tsx|json|csv|md|txt|yaml|yml|ipynb)\b)",
            current,
            flags=re.IGNORECASE,
        )
    )
    has_error_keyword = any(
        term in (current + " " + result_text).lower()
        for term in ("error", "failed", "fail", "exception", "traceback", "오류", "실패", "에러")
    )

    feats = {
        "user_tier=" + compact(meta.get("user_tier"), 80): 1,
        "language_pref=" + compact(meta.get("language_pref"), 80): 1,
        "last_ci_status=" + compact(ws.get("last_ci_status"), 80): 1,
        "git_dirty=" + str(ws.get("git_dirty", "")): 1,
        "turn_bin=" + bucket(meta.get("turn_index"), 1): 1,
        "budget_bin=" + bucket(meta.get("budget_tokens_remaining"), 10000): 1,
        "elapsed_bin=" + bucket(meta.get("elapsed_session_sec"), 60): 1,
        "loc_bin=" + bucket(ws.get("loc"), 5000): 1,
        "history_len": len(history),
        "history_length_bin=" + bucket(len(history), 2): 1,
        "prompt_len_bin=" + bucket(len(prompt_text(row)), 40): 1,
        "open_files_count": len(open_files),
        "has_failed_in_last_result": int("fail" in result_text or "failed" in result_text),
        "has_error_keyword": int(has_error_keyword),
        "has_path_in_prompt": int(has_path_in_prompt),
        "has_code_block_in_prompt": int("```" in current),
        "has_question_mark": int("?" in current or "？" in current),
    }
    for file_name in open_files:
        feats["open_file=" + compact(file_name, 120)] = 1
        suffix = Path(str(file_name)).suffix.lower() or "no_ext"
        feats["open_ext=" + suffix] = 1
        feats["open_file_extension=" + suffix] = 1
    for idx, lang in enumerate(language_ranked[:2], start=1):
        feats[f"workspace_language_top{idx}=" + compact(lang, 40)] = 1
    for lang in language_mix.keys() if isinstance(language_mix, dict) else []:
        feats["lang_mix=" + compact(lang, 40)] = 1

    if actions:
        feats["last_action=" + actions[-1]] = 1
        feats["first_action=" + actions[0]] = 1
        feats["action_count"] = len(actions)
        if len(actions) >= 2:
            feats["last2_actions=" + ">".join(actions[-2:])] = 1
        if len(actions) >= 3:
            feats["last3_actions=" + ">".join(actions[-3:])] = 1
        for action in actions[-8:]:
            feats["recent_action=" + action] = feats.get("recent_action=" + action, 0) + 1
    else:
        feats["last_action=NONE"] = 1

    for action in ("read_file", "grep_search", "edit_file", "run_tests"):
        feats[f"count_{action}"] = action_counts.get(action, 0)
    return feats


def build_state_features(train_rows, valid_rows):
    vectorizer = DictVectorizer(sparse=True)
    x_train = vectorizer.fit_transform([state_dict(row) for row in train_rows])
    x_valid = vectorizer.transform([state_dict(row) for row in valid_rows])
    return x_train, x_valid


PRIOR_SLOTS = ("global", "last", "first", "last2", "last3", "turn")


def prior_key_slots(row):
    actions = previous_actions(row)
    slots = {"global": "global"}
    if actions:
        slots["last"] = "last=" + actions[-1]
        slots["first"] = "first=" + actions[0]
        slots["last2"] = "last2=" + ">".join(actions[-2:]) if len(actions) >= 2 else "last2=NONE"
        slots["last3"] = "last3=" + ">".join(actions[-3:]) if len(actions) >= 3 else "last3=NONE"
    else:
        slots["last"] = "last=NONE"
        slots["first"] = "first=NONE"
        slots["last2"] = "last2=NONE"
        slots["last3"] = "last3=NONE"
    meta = row.get("session_meta") or {}
    slots["turn"] = "turn=" + bucket(meta.get("turn_index"), 1)
    return slots


def build_transition_prior(train_rows, y_train, target_rows, classes, smoothing=1.0):
    class_to_idx = {label: i for i, label in enumerate(classes)}
    n_classes = len(classes)
    counts = defaultdict(lambda: np.zeros(n_classes, dtype=np.float64))
    global_counts = np.zeros(n_classes, dtype=np.float64)

    for row, label in zip(train_rows, y_train):
        label_idx = class_to_idx[label]
        global_counts[label_idx] += 1
        for slot, key in prior_key_slots(row).items():
            counts[f"{slot}:{key}"][label_idx] += 1

    global_prob = (global_counts + smoothing) / (global_counts.sum() + smoothing * n_classes)
    blocks = []
    for row in target_rows:
        row_blocks = []
        slots = prior_key_slots(row)
        for slot in PRIOR_SLOTS:
            key = slots[slot]
            c = counts.get(f"{slot}:{key}")
            if c is None or c.sum() == 0:
                prob = global_prob
            else:
                prob = (c + smoothing) / (c.sum() + smoothing * n_classes)
            row_blocks.append(prob)
        blocks.append(np.concatenate(row_blocks))
    return sparse.csr_matrix(np.vstack(blocks))


def make_classifier(args):
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.classifier == "logreg":
        clf = LogisticRegression(
            C=args.logreg_c,
            solver=args.logreg_solver,
            class_weight=class_weight,
            max_iter=args.max_iter,
            tol=args.tol,
            random_state=args.seed,
        )
        if args.logreg_solver == "liblinear":
            return OneVsRestClassifier(clf)
        return clf

    return SGDClassifier(
        loss=args.loss,
        alpha=args.alpha,
        penalty="l2",
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        random_state=args.seed,
        n_jobs=1,
    )


def as_probability(clf, x, classes):
    if hasattr(clf, "predict_proba"):
        raw = clf.predict_proba(x)
    else:
        raw = clf.decision_function(x)
        if raw.ndim == 1:
            raw = np.vstack([-raw, raw]).T
        raw = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(raw)
        raw = exp / exp.sum(axis=1, keepdims=True)

    aligned = np.zeros((x.shape[0], len(classes)), dtype=np.float32)
    for source_idx, label in enumerate(clf.classes_):
        target_idx = np.where(classes == label)[0][0]
        aligned[:, target_idx] = raw[:, source_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def build_fold_matrix(
    model_name,
    train_rows,
    valid_rows,
    y_train,
    classes,
    args,
    train_texts=None,
    valid_texts=None,
    train_views=None,
    valid_views=None,
):
    spec = MODEL_PRESETS[model_name]
    train_parts = []
    valid_parts = []

    if spec.get("views"):
        x_train_views, x_valid_views = build_five_view_features(
            train_rows,
            valid_rows,
            args,
            train_views=train_views,
            valid_views=valid_views,
        )
        train_parts.append(x_train_views)
        valid_parts.append(x_valid_views)

    if spec["text"]:
        vectorizer = make_vectorizer(spec["vectorizer"], args)
        if train_texts is None:
            train_texts = build_texts(train_rows, spec["text"])
        if valid_texts is None:
            valid_texts = build_texts(valid_rows, spec["text"])
        train_parts.append(vectorizer.fit_transform(train_texts))
        valid_parts.append(vectorizer.transform(valid_texts))

    if spec["state"]:
        x_train_state, x_valid_state = build_state_features(train_rows, valid_rows)
        train_parts.append(x_train_state)
        valid_parts.append(x_valid_state)

    if spec["prior"]:
        x_train_prior = build_transition_prior(train_rows, y_train, train_rows, classes)
        x_valid_prior = build_transition_prior(train_rows, y_train, valid_rows, classes)
        train_parts.append(x_train_prior)
        valid_parts.append(x_valid_prior)

    if not train_parts:
        raise ValueError(f"model {model_name} has no features")
    if len(train_parts) == 1:
        return train_parts[0], valid_parts[0]
    return sparse.hstack(train_parts, format="csr"), sparse.hstack(valid_parts, format="csr")


def make_splitter(args, dataset):
    n_groups = len(set(dataset.groups))
    n_splits = min(args.n_splits, n_groups)
    if args.splitter == "stratified_group":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        return splitter.split(dataset.rows, dataset.y, dataset.groups)
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(dataset.rows, dataset.y, dataset.groups)


def write_metrics(out_dir, name, y_true, y_pred, labels):
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T
    report_df.to_csv(out_dir / f"{name}_class_report.csv", encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")

    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def artifact_name(model_name):
    return MODEL_PRESETS[model_name].get("artifact", model_name)


def predict_with_class_bias(proba, classes, bias):
    scores = np.log(np.clip(proba, 1e-9, 1.0)) + bias.reshape(1, -1)
    return classes[scores.argmax(axis=1)]


def search_class_biases(proba, y_true, classes, args):
    if args.bias_grid_step <= 0:
        raise ValueError("--bias-grid-step must be positive")
    grid = np.arange(
        args.bias_grid_min,
        args.bias_grid_max + args.bias_grid_step / 2.0,
        args.bias_grid_step,
        dtype=np.float32,
    )
    logp = np.log(np.clip(proba, 1e-9, 1.0))
    bias = np.zeros(len(classes), dtype=np.float32)
    best_pred = classes[logp.argmax(axis=1)]
    best_score = float(f1_score(y_true, best_pred, labels=classes, average="macro", zero_division=0))

    for _ in range(args.bias_grid_passes):
        improved = False
        for class_idx in range(len(classes)):
            original = float(bias[class_idx])
            local_best_value = original
            local_best_score = best_score
            for value in grid:
                bias[class_idx] = float(value)
                pred = classes[(logp + bias.reshape(1, -1)).argmax(axis=1)]
                score = float(f1_score(y_true, pred, labels=classes, average="macro", zero_division=0))
                if score > local_best_score:
                    local_best_score = score
                    local_best_value = float(value)
            bias[class_idx] = local_best_value
            if local_best_score > best_score:
                best_score = local_best_score
                best_pred = classes[(logp + bias.reshape(1, -1)).argmax(axis=1)]
                improved = True
        if not improved:
            break

    return {
        "pred": best_pred,
        "bias": bias.copy(),
        "macro_f1": best_score,
    }


def run_model(model_name, dataset, splits, args, out_dir, text_cache_by_mode=None):
    n = len(dataset.rows)
    oof_proba = np.zeros((n, len(dataset.classes)), dtype=np.float32)
    oof_pred = np.empty(n, dtype=object)
    fold_rows = []
    spec = MODEL_PRESETS[model_name]
    all_texts = None
    all_views = None
    if spec["text"]:
        if text_cache_by_mode is not None and spec["text"] in text_cache_by_mode:
            all_texts = text_cache_by_mode[spec["text"]]
        else:
            print(f"{model_name} build {spec['text']} texts...", flush=True)
            all_texts = np.asarray(build_texts(dataset.rows, spec["text"]), dtype=object)
            if text_cache_by_mode is not None:
                text_cache_by_mode[spec["text"]] = all_texts
    if spec.get("views"):
        cache_key = "views:five_view"
        if text_cache_by_mode is not None and cache_key in text_cache_by_mode:
            all_views = text_cache_by_mode[cache_key]
        else:
            print(f"{model_name} build 5 view texts...", flush=True)
            all_views = {
                name: np.asarray(values, dtype=object)
                for name, values in build_view_texts(dataset.rows).items()
            }
            if text_cache_by_mode is not None:
                text_cache_by_mode[cache_key] = all_views

    for fold, (train_idx, valid_idx) in enumerate(splits, start=1):
        t0 = time.time()
        train_rows = [dataset.rows[i] for i in train_idx]
        valid_rows = [dataset.rows[i] for i in valid_idx]
        y_train = dataset.y[train_idx]
        y_valid = dataset.y[valid_idx]
        train_texts = all_texts[train_idx] if all_texts is not None else None
        valid_texts = all_texts[valid_idx] if all_texts is not None else None
        train_views = (
            {name: values[train_idx] for name, values in all_views.items()}
            if all_views is not None
            else None
        )
        valid_views = (
            {name: values[valid_idx] for name, values in all_views.items()}
            if all_views is not None
            else None
        )

        x_train, x_valid = build_fold_matrix(
            model_name,
            train_rows,
            valid_rows,
            y_train,
            dataset.classes,
            args,
            train_texts=train_texts,
            valid_texts=valid_texts,
            train_views=train_views,
            valid_views=valid_views,
        )
        clf = make_classifier(args)
        clf.fit(x_train, y_train)

        valid_proba = as_probability(clf, x_valid, dataset.classes)
        raw_pred = dataset.classes[valid_proba.argmax(axis=1)]
        valid_pred = raw_pred
        bias_result = None
        if args.bias_grid_search:
            bias_result = search_class_biases(valid_proba, y_valid, dataset.classes, args)
            valid_pred = bias_result["pred"]
        oof_proba[valid_idx] = valid_proba
        oof_pred[valid_idx] = valid_pred

        metrics = {
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "n_features": int(x_train.shape[1]),
            "raw_macro_f1": float(f1_score(y_valid, raw_pred, average="macro")),
            "raw_accuracy": float(accuracy_score(y_valid, raw_pred)),
            "macro_f1": float(f1_score(y_valid, valid_pred, average="macro")),
            "accuracy": float(accuracy_score(y_valid, valid_pred)),
            "seconds": round(time.time() - t0, 2),
        }
        if bias_result is not None:
            metrics["bias_macro_f1"] = float(bias_result["macro_f1"])
            metrics["class_bias"] = json.dumps(
                {str(label): float(value) for label, value in zip(dataset.classes, bias_result["bias"])},
                ensure_ascii=False,
            )
        fold_rows.append(metrics)
        print(
            f"{model_name} fold={fold} "
            f"raw_macro_f1={metrics['raw_macro_f1']:.6f} "
            f"macro_f1={metrics['macro_f1']:.6f} "
            f"acc={metrics['accuracy']:.6f} "
            f"features={metrics['n_features']} "
            f"sec={metrics['seconds']}",
            flush=True,
        )

    artifact = artifact_name(model_name)
    np.save(out_dir / f"oof_{artifact}.npy", oof_proba)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{artifact}_folds.csv", index=False)
    metrics = write_metrics(out_dir, artifact, dataset.y, oof_pred, dataset.classes)
    metrics["folds"] = fold_rows
    metrics["note"] = MODEL_PRESETS[model_name]["note"]
    metrics["artifact"] = f"oof_{artifact}.npy"
    metrics["bias_grid_search"] = bool(args.bias_grid_search)
    print(f"{model_name} OOF macro_f1={metrics['macro_f1']:.6f} acc={metrics['accuracy']:.6f}", flush=True)
    return oof_proba, oof_pred, metrics


def save_oof_contract(out_dir, dataset, split_list, args):
    fold_ids = np.full(len(dataset.rows), -1, dtype=np.int16)
    for fold_idx, (_, valid_idx) in enumerate(split_list):
        fold_ids[valid_idx] = fold_idx
    if np.any(fold_ids < 0):
        missing = int(np.sum(fold_ids < 0))
        raise ValueError(f"fold assignment missing for {missing} rows")

    np.save(out_dir / "y_true.npy", np.asarray(dataset.y, dtype=str))
    np.save(out_dir / "classes.npy", np.asarray(dataset.classes, dtype=str))
    np.save(out_dir / "sample_ids.npy", np.asarray(dataset.ids, dtype=str))
    np.save(out_dir / "session_ids.npy", np.asarray(dataset.groups, dtype=str))
    np.save(out_dir / "fold_ids.npy", fold_ids)

    manifest = {
        "row_order": "train.jsonl loading order after optional max_sessions filter",
        "class_order": "classes.npy column order for every oof_*.npy file",
        "fold_ids": "0-based fold index; each row is predicted by the model trained without that session_id group",
        "n_rows": int(len(dataset.rows)),
        "n_classes": int(len(dataset.classes)),
        "n_folds": int(len(split_list)),
        "splitter": args.splitter,
        "group_key": "session_id parsed from id before -step_",
        "classes": list(map(str, dataset.classes)),
    }
    with open(out_dir / "oof_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return fold_ids


def score_proba(y_true, classes, proba):
    pred = classes[proba.argmax(axis=1)]
    return float(f1_score(y_true, pred, average="macro")), pred


def search_ensemble_weights(model_names, probas, y_true, classes, args):
    stack = np.stack(probas).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    candidates = []

    n_models = len(model_names)
    equal = np.ones(n_models, dtype=np.float32) / n_models
    candidates.append(("equal", equal))
    for i, name in enumerate(model_names):
        w = np.zeros(n_models, dtype=np.float32)
        w[i] = 1.0
        candidates.append((f"single:{name}", w))

    for trial in range(args.weight_search_trials):
        concentration = np.full(n_models, args.weight_search_alpha, dtype=np.float64)
        w = rng.dirichlet(concentration).astype(np.float32)
        candidates.append((f"dirichlet:{trial}", w))

    rows = []
    best = None
    for label, weights in candidates:
        proba = np.tensordot(weights, stack, axes=(0, 0))
        macro_f1, _ = score_proba(y_true, classes, proba)
        row = {
            "candidate": label,
            "macro_f1": macro_f1,
            **{f"w_{name}": float(weight) for name, weight in zip(model_names, weights)},
        }
        rows.append(row)
        if best is None or macro_f1 > best["macro_f1"]:
            best = row

    best_weights = np.array([best[f"w_{name}"] for name in model_names], dtype=np.float32)
    best_proba = np.tensordot(best_weights, stack, axes=(0, 0))
    _, best_pred = score_proba(y_true, classes, best_proba)
    return best, best_weights, best_proba, best_pred, pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=list(MODEL_PRESETS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--splitter", choices=["group", "stratified_group"], default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--word-max-features", type=int, default=220000)
    parser.add_argument("--char-max-features", type=int, default=80000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=3)
    parser.add_argument("--jm-word-max-features", type=int, default=60000)
    parser.add_argument("--jm-char-max-features", type=int, default=30000)
    parser.add_argument("--jm-word-min-df", type=int, default=2)
    parser.add_argument("--jm-char-min-df", type=int, default=2)
    parser.add_argument("--view-current-word-max-features", type=int, default=60000)
    parser.add_argument("--view-current-char-max-features", type=int, default=30000)
    parser.add_argument("--view-history-word-max-features", type=int, default=50000)
    parser.add_argument("--view-history-char-max-features", type=int, default=25000)
    parser.add_argument("--view-action-max-features", type=int, default=25000)
    parser.add_argument("--view-args-word-max-features", type=int, default=40000)
    parser.add_argument("--view-args-char-max-features", type=int, default=40000)
    parser.add_argument("--view-result-word-max-features", type=int, default=30000)
    parser.add_argument("--view-result-char-max-features", type=int, default=30000)
    parser.add_argument("--view-word-min-df", type=int, default=2)
    parser.add_argument("--view-char-min-df", type=int, default=2)
    parser.add_argument("--view-action-min-df", type=int, default=1)
    parser.add_argument("--classifier", choices=["logreg", "sgd"], default="logreg")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--logreg-c", type=float, default=4.0)
    parser.add_argument("--logreg-solver", choices=["lbfgs", "liblinear", "saga"], default="saga")
    parser.add_argument("--loss", default="hinge", choices=["hinge", "log_loss", "modified_huber"])
    parser.add_argument("--alpha", type=float, default=3e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.set_defaults(bias_grid_search=True)
    parser.add_argument("--bias-grid-search", dest="bias_grid_search", action="store_true")
    parser.add_argument("--no-bias-grid-search", dest="bias_grid_search", action="store_false")
    parser.add_argument("--bias-grid-min", type=float, default=-0.8)
    parser.add_argument("--bias-grid-max", type=float, default=0.8)
    parser.add_argument("--bias-grid-step", type=float, default=0.1)
    parser.add_argument("--bias-grid-passes", type=int, default=2)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--weight-search-trials", type=int, default=500)
    parser.add_argument("--weight-search-alpha", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("experiments/oof") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config["models"] = list(args.models)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("load dataset...", flush=True)
    dataset = load_dataset(args.data_dir, max_sessions=args.max_sessions)
    print(
        f"rows={len(dataset.rows)} sessions={len(set(dataset.groups))} "
        f"classes={len(dataset.classes)} class_counts={dict(Counter(dataset.y))}",
        flush=True,
    )
    pd.Series(dataset.y).value_counts().rename_axis("class").reset_index(name="count").to_csv(
        out_dir / "label_counts.csv", index=False, encoding="utf-8-sig"
    )

    split_list = list(make_splitter(args, dataset))
    fold_ids = save_oof_contract(out_dir, dataset, split_list, args)
    pd.DataFrame(
        [
            {
                "fold": i + 1,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "train_sessions": len(set(dataset.groups[train_idx])),
                "valid_sessions": len(set(dataset.groups[valid_idx])),
                "valid_macro_majority": float(
                    f1_score(
                        dataset.y[valid_idx],
                        np.repeat(Counter(dataset.y[train_idx]).most_common(1)[0][0], len(valid_idx)),
                        average="macro",
                    )
                ),
            }
            for i, (train_idx, valid_idx) in enumerate(split_list)
        ]
    ).to_csv(out_dir / "folds.csv", index=False)

    summary = {}
    oof_predictions = pd.DataFrame(
        {"id": dataset.ids, "session_id": dataset.groups, "fold_id": fold_ids, "true": dataset.y}
    )
    probas = []
    proba_names = []
    text_cache_by_mode = {}

    for model_name in args.models:
        oof_proba, oof_pred, metrics = run_model(
            model_name,
            dataset,
            split_list,
            args,
            out_dir,
            text_cache_by_mode=text_cache_by_mode,
        )
        summary[model_name] = metrics
        artifact = artifact_name(model_name)
        oof_predictions[f"{artifact}_pred"] = oof_pred
        probas.append(oof_proba)
        proba_names.append(artifact)

    if probas and not args.no_ensemble:
        ensemble_proba = np.mean(probas, axis=0)
        ensemble_pred = dataset.classes[ensemble_proba.argmax(axis=1)]
        np.save(out_dir / "oof_ensemble_mean.npy", ensemble_proba)
        oof_predictions["ensemble_mean_pred"] = ensemble_pred
        summary["ensemble_mean"] = write_metrics(out_dir, "ensemble_mean", dataset.y, ensemble_pred, dataset.classes)
        print(
            f"ensemble_mean OOF macro_f1={summary['ensemble_mean']['macro_f1']:.6f} "
            f"acc={summary['ensemble_mean']['accuracy']:.6f}",
            flush=True,
        )

        best, weights, weighted_proba, weighted_pred, search_df = search_ensemble_weights(
            proba_names, probas, dataset.y, dataset.classes, args
        )
        np.save(out_dir / "oof_ensemble_weighted.npy", weighted_proba)
        search_df.to_csv(out_dir / "ensemble_search.csv", index=False)
        oof_predictions["ensemble_weighted_pred"] = weighted_pred
        summary["ensemble_weighted"] = write_metrics(
            out_dir, "ensemble_weighted", dataset.y, weighted_pred, dataset.classes
        )
        summary["ensemble_weighted"]["weights"] = {
            name: float(weight) for name, weight in zip(proba_names, weights)
        }
        summary["ensemble_weighted"]["search_best_candidate"] = best["candidate"]
        with open(out_dir / "ensemble_weights.json", "w", encoding="utf-8") as f:
            json.dump(summary["ensemble_weighted"], f, ensure_ascii=False, indent=2)
        print(
            f"ensemble_weighted OOF macro_f1={summary['ensemble_weighted']['macro_f1']:.6f} "
            f"acc={summary['ensemble_weighted']['accuracy']:.6f} "
            f"candidate={best['candidate']}",
            flush=True,
        )

    oof_predictions.to_csv(out_dir / "oof_predictions.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
