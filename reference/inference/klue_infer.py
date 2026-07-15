from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer


MODEL_DIR = Path("model")
TEST_JSONL = Path("data") / "test.jsonl"
SAMPLE_SUBMISSION = Path("data") / "sample_submission.csv"
OUTPUT_CSV = Path("output") / "submission.csv"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "128"))
USE_FP16 = os.environ.get("USE_FP16", "1") != "0"

ACTION_LABELS = [
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
]
ACTION_SHORT = {
    "apply_patch": "ap",
    "ask_user": "ask",
    "edit_file": "edit",
    "glob_pattern": "glob",
    "grep_search": "grep",
    "lint_or_typecheck": "lint",
    "list_directory": "ls",
    "plan_task": "plan",
    "read_file": "read",
    "respond_only": "resp",
    "run_bash": "bash",
    "run_tests": "test",
    "web_search": "web",
    "write_file": "write",
}
USER_TIER_VALUES = ["free", "pro", "enterprise", "unknown"]
LANGUAGE_PREF_VALUES = ["en", "ko", "unknown", "other"]
CI_STATUS_VALUES = ["passed", "failed", "none", "unknown", "other"]
PRIMARY_LANG_VALUES = [
    "py",
    "ts",
    "tsx",
    "js",
    "java",
    "kt",
    "rs",
    "go",
    "yaml",
    "sh",
    "md",
    "html",
    "css",
    "json",
    "sql",
    "vue",
    "toml",
    "dockerfile",
    "ipynb",
    "tf",
    "swift",
    "xml",
    "none",
    "other",
]


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return str(value).strip()


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def compact_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in sorted(value):
            compacted = compact_value(value[key])
            if compacted:
                parts.append(f"{key}={compacted}")
        return " ".join(parts)
    if isinstance(value, list):
        return " ".join(part for part in (compact_value(item) for item in value) if part)
    return str(value)


def bucket_num(value: Any, bins: list[float]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(number):
        return "unknown"
    for bound in bins:
        if number <= bound:
            return f"le_{str(bound).replace('.', '_')}"
    return f"gt_{str(bins[-1]).replace('.', '_')}"


def basename_token(path: Any) -> str:
    text = safe_str(path)
    if not text:
        return ""
    return Path(text).name.replace(" ", "_")


def extension_token(path: Any) -> str:
    text = safe_str(path)
    if "." not in text:
        return ""
    ext = text.rsplit(".", 1)[-1].lower().strip()
    return f"ext_{ext}" if ext else ""


def compact_count(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    if value <= 4:
        return "3_4"
    if value <= 8:
        return "5_8"
    return "gt_8"


def short_action_name(action: str) -> str:
    return ACTION_SHORT.get(action, action.replace("_", ""))


def format_action_trace(history: Any, current_prompt: Any) -> str:
    items = safe_list(history)
    action_names: list[str] = []
    user_turns = 0
    result_texts: list[str] = []

    for item in items:
        if not isinstance(item, dict):
            continue
        role = safe_str(item.get("role"))
        if role == "user":
            user_turns += 1
        elif role == "assistant_action":
            action_names.append(safe_str(item.get("name")) or "unknown_action")
            result = compact_value(item.get("result_summary", ""))
            if result:
                result_texts.append(result.lower())

    short_actions = [short_action_name(action) for action in action_names]
    last_action = short_actions[-1] if short_actions else "none"
    last2 = "_".join(short_actions[-2:]) if len(short_actions) >= 2 else last_action
    last3 = "_".join(short_actions[-3:]) if len(short_actions) >= 3 else last2
    action_tail = short_actions[-6:]
    counts = Counter(action_names)
    joined_results = " ".join(result_texts[-8:])

    failure_terms = ["fail", "failed", "error", "traceback", "exception", "timeout", "missing", "not found"]
    failed_count = sum(joined_results.count(term) for term in failure_terms)
    patch_count = sum(counts[action] for action in ["apply_patch", "edit_file", "write_file"])
    test_count = sum(counts[action] for action in ["run_tests", "lint_or_typecheck"])

    tokens = [
        f"last_action={last_action}",
        f"last2_actions={last2}",
        f"last3_actions={last3}",
        "tail=" + " ".join(action_tail),
        f"na={compact_count(len(action_names))}",
        f"nu={compact_count(user_turns)}",
        f"hlen={compact_count(len(items))}",
        f"fail={compact_count(failed_count)}",
        f"patch={compact_count(patch_count)}",
        f"tests={compact_count(test_count)}",
        "plen=" + bucket_num(len(compact_value(current_prompt).split()), [4, 8, 12, 20, 32, 64]),
    ]
    for action in ACTION_LABELS:
        count = counts[action]
        if count:
            tokens.append(f"c_{short_action_name(action)}={compact_count(count)}")
    return " ".join(token for token in tokens if token)


def format_neural_meta(session_meta: Any) -> str:
    meta = safe_dict(session_meta)
    workspace = safe_dict(meta.get("workspace"))
    tokens: list[str] = []

    tokens.append(f"user_tier={safe_str(meta.get('user_tier')) or 'unknown'}")
    tokens.append(f"language_pref={safe_str(meta.get('language_pref')) or 'unknown'}")
    tokens.append(
        "budget="
        + bucket_num(meta.get("budget_tokens_remaining"), [1000, 5000, 10000, 30000, 70000, 150000])
    )
    tokens.append("turn=" + bucket_num(meta.get("turn_index"), [1, 3, 5, 10, 20, 40]))
    tokens.append("elapsed=" + bucket_num(meta.get("elapsed_session_sec"), [60, 300, 900, 1800, 3600, 7200]))
    tokens.append("loc=" + bucket_num(workspace.get("loc"), [100, 1000, 5000, 20000, 100000]))
    tokens.append(f"git_dirty={safe_str(workspace.get('git_dirty')) or 'unknown'}")
    tokens.append(f"last_ci_status={safe_str(workspace.get('last_ci_status')) or 'unknown'}")

    language_mix = safe_dict(workspace.get("language_mix"))
    for key in sorted(language_mix):
        tokens.append(f"lang={key}")
        tokens.append(f"lang_{key}=" + bucket_num(language_mix.get(key), [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]))

    open_files = safe_list(workspace.get("open_files"))
    tokens.append("open_files_count=" + bucket_num(len(open_files), [0, 1, 2, 4, 8, 16]))
    for path in open_files[-8:]:
        base = basename_token(path)
        ext = extension_token(path)
        if base:
            tokens.append(f"file={base}")
        if ext:
            tokens.append(ext)
    return " ".join(tokens)


def make_text_v9_current_first_trace(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "[CURRENT_PROMPT]",
            compact_value(row.get("current_prompt", "")),
            "",
            "[ACTION_TRACE]",
            format_action_trace(row.get("history", []), row.get("current_prompt", "")),
            "",
            "[SESSION_META]",
            format_neural_meta(row.get("session_meta", {})),
        ]
    ).strip()


def collect_history_stats(history: Any) -> dict[str, Any]:
    action_names: list[str] = []
    user_turns = 0
    result_texts: list[str] = []
    for item in safe_list(history):
        if not isinstance(item, dict):
            continue
        role = safe_str(item.get("role"))
        if role == "user":
            user_turns += 1
        elif role == "assistant_action":
            action_names.append(safe_str(item.get("name")) or "unknown_action")
            result = compact_value(item.get("result_summary", ""))
            if result:
                result_texts.append(result.lower())

    counts = Counter(action_names)
    joined_results = " ".join(result_texts[-8:])
    failure_terms = ["fail", "failed", "error", "traceback", "exception", "timeout", "missing", "not found"]
    failed_count = sum(joined_results.count(term) for term in failure_terms)
    patch_count = sum(counts[action] for action in ["apply_patch", "edit_file", "write_file"])
    test_count = sum(counts[action] for action in ["run_tests", "lint_or_typecheck"])
    return {
        "action_names": action_names,
        "counts": counts,
        "user_turns": user_turns,
        "history_len": len(safe_list(history)),
        "failed_count": failed_count,
        "patch_count": patch_count,
        "test_count": test_count,
    }


def bucket_or_other(value: str, allowed: list[str], *, default: str = "unknown") -> str:
    value = safe_str(value).lower() or default
    return value if value in allowed else "other"


def one_hot(value: str, values: list[str]) -> list[float]:
    return [1.0 if value == candidate else 0.0 for candidate in values]


def scaled_log1p(value: Any, denom: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if math.isnan(number) or number < 0:
        number = 0.0
    return min(1.0, math.log1p(number) / denom)


def primary_language(language_mix: Any) -> str:
    mix = safe_dict(language_mix)
    if not mix:
        return "none"
    best_key = max(mix, key=lambda key: float(mix.get(key) or 0.0))
    best_key = safe_str(best_key).lower()
    return best_key if best_key in PRIMARY_LANG_VALUES else "other"


def feature_names() -> list[str]:
    names: list[str] = []
    names.extend([f"count_{action}" for action in ACTION_LABELS])
    names.extend([f"last_{action}" for action in ACTION_LABELS])
    names.extend([f"prev_{action}" for action in ACTION_LABELS])
    names.extend([f"user_tier_{value}" for value in USER_TIER_VALUES])
    names.extend([f"language_pref_{value}" for value in LANGUAGE_PREF_VALUES])
    names.extend([f"ci_{value}" for value in CI_STATUS_VALUES])
    names.extend([f"primary_lang_{value}" for value in PRIMARY_LANG_VALUES])
    names.extend(
        [
            "num_actions_log",
            "num_user_turns_log",
            "budget_log",
            "turn_index_log",
            "elapsed_log",
            "workspace_loc_log",
            "git_dirty",
            "open_files_count_log",
            "current_prompt_words_log",
            "current_prompt_chars_log",
            "history_len_log",
            "failed_count_log",
            "patch_count_log",
            "test_count_log",
        ]
    )
    return names


FEATURE_NAMES = feature_names()


def make_structured_features(row: dict[str, Any]) -> list[float]:
    history_stats = collect_history_stats(row.get("history", []))
    actions: list[str] = history_stats["action_names"]
    counts: Counter[str] = history_stats["counts"]
    meta = safe_dict(row.get("session_meta"))
    workspace = safe_dict(meta.get("workspace"))
    current_prompt = compact_value(row.get("current_prompt", ""))

    features: list[float] = []
    features.extend([scaled_log1p(counts[action], math.log1p(12.0)) for action in ACTION_LABELS])
    last_action = actions[-1] if actions else "__none__"
    prev_action = actions[-2] if len(actions) >= 2 else "__none__"
    features.extend([1.0 if last_action == action else 0.0 for action in ACTION_LABELS])
    features.extend([1.0 if prev_action == action else 0.0 for action in ACTION_LABELS])

    user_tier = bucket_or_other(meta.get("user_tier"), USER_TIER_VALUES)
    language_pref = bucket_or_other(meta.get("language_pref"), LANGUAGE_PREF_VALUES)
    ci_status = bucket_or_other(workspace.get("last_ci_status"), CI_STATUS_VALUES)
    features.extend(one_hot(user_tier, USER_TIER_VALUES))
    features.extend(one_hot(language_pref, LANGUAGE_PREF_VALUES))
    features.extend(one_hot(ci_status, CI_STATUS_VALUES))
    features.extend(one_hot(primary_language(workspace.get("language_mix")), PRIMARY_LANG_VALUES))

    features.extend(
        [
            scaled_log1p(len(actions), math.log1p(16.0)),
            scaled_log1p(history_stats["user_turns"], math.log1p(16.0)),
            scaled_log1p(meta.get("budget_tokens_remaining"), math.log1p(200000.0)),
            scaled_log1p(meta.get("turn_index"), math.log1p(64.0)),
            scaled_log1p(meta.get("elapsed_session_sec"), math.log1p(10000.0)),
            scaled_log1p(workspace.get("loc"), math.log1p(150000.0)),
            1.0 if safe_str(workspace.get("git_dirty")).lower() == "true" else 0.0,
            scaled_log1p(len(safe_list(workspace.get("open_files"))), math.log1p(24.0)),
            scaled_log1p(len(current_prompt.split()), math.log1p(128.0)),
            scaled_log1p(len(current_prompt), math.log1p(1000.0)),
            scaled_log1p(history_stats["history_len"], math.log1p(32.0)),
            scaled_log1p(history_stats["failed_count"], math.log1p(16.0)),
            scaled_log1p(history_stats["patch_count"], math.log1p(16.0)),
            scaled_log1p(history_stats["test_count"], math.log1p(16.0)),
        ]
    )
    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError(f"Feature size mismatch: {len(features)} != {len(FEATURE_NAMES)}")
    return features


class KlueStructuredClassifier(torch.nn.Module):
    def __init__(self, structured_config: dict[str, Any]) -> None:
        super().__init__()
        encoder_config_dict = dict(structured_config["encoder_config"])
        model_type = encoder_config_dict.pop("model_type")
        encoder_config = AutoConfig.for_model(model_type, **encoder_config_dict)
        self.encoder = AutoModel.from_config(encoder_config)
        feature_dim = int(structured_config["feature_dim"])
        hidden_dim = int(structured_config["hidden_dim"])
        dropout = float(structured_config["dropout"])
        encoder_hidden = int(self.encoder.config.hidden_size)
        pooled_dim = encoder_hidden * 2 + feature_dim
        self.feature_norm = torch.nn.LayerNorm(feature_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.LayerNorm(pooled_dim),
            torch.nn.Linear(pooled_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, max(128, hidden_dim // 3)),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(max(128, hidden_dim // 3), int(structured_config["num_labels"])),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, features=None, **_: Any):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**kwargs)
        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]
        if attention_mask is None:
            mean = hidden.mean(dim=1)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        feature_tensor = self.feature_norm(features.to(hidden.dtype))
        pooled = torch.cat([cls, mean, feature_tensor], dim=1)
        return SimpleNamespace(logits=self.classifier(pooled))


def load_quantized_state(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
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


def predict_klue_proba_subset(
    rows: list[dict[str, Any]],
    model_dir: Path,
    device: torch.device,
    batch_size: int = 64,
    label_order: list[str] | None = None,
) -> np.ndarray:
    if not rows:
        width = len(label_order) if label_order is not None else len(ACTION_LABELS)
        return np.zeros((0, width), dtype=np.float32)

    model_dir = Path(model_dir)
    structured_config = json.loads((model_dir / "structured_config.json").read_text(encoding="utf-8"))
    text_version = structured_config.get("text_version")
    if text_version != "v9_current_first_trace":
        raise ValueError(f"Unsupported KLUE text_version: {text_version}")
    if structured_config.get("feature_names") != FEATURE_NAMES:
        raise ValueError("KLUE feature names do not match inference wrapper.")

    max_length = int(structured_config.get("max_length", 384))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = KlueStructuredClassifier(structured_config)
    state_path = model_dir / "model_int8.pt"
    if not state_path.exists():
        state_path = model_dir / "model.pt"
    model.load_state_dict(load_quantized_state(state_path, device))
    model.to(device)
    if device.type == "cuda" and USE_FP16:
        model.half()
    model.eval()

    id2label = {int(key): value for key, value in structured_config["id2label"].items()}
    model_labels = [id2label[i] for i in range(len(id2label))]
    if label_order is None:
        label_order = model_labels
    label_to_out = {label: idx for idx, label in enumerate(label_order)}
    model_to_out = [label_to_out.get(label, None) for label in model_labels]

    texts = [make_text_v9_current_first_trace(row) for row in rows]
    features = [make_structured_features(row) for row in rows]
    output = np.zeros((len(rows), len(label_order)), dtype=np.float32)

    index = 0
    cur_batch = int(batch_size)
    with torch.inference_mode():
        while index < len(rows):
            batch_texts = texts[index : index + cur_batch]
            batch_features = features[index : index + cur_batch]
            encoded = tokenizer(
                batch_texts,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoded["features"] = torch.tensor(batch_features, dtype=torch.float32, device=device)
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda" and USE_FP16):
                    logits = model(**encoded).logits
            except RuntimeError as exc:
                if device.type == "cuda" and "out of memory" in str(exc).lower() and cur_batch > 1:
                    torch.cuda.empty_cache()
                    cur_batch = max(1, cur_batch // 2)
                    print(f"KLUE cuda oom; retry with batch_size={cur_batch}")
                    continue
                raise
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            for model_idx, out_idx in enumerate(model_to_out):
                if out_idx is not None:
                    output[index : index + len(batch_texts), out_idx] = probs[:, model_idx]
            index += len(batch_texts)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output
