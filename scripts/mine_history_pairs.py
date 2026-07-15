#!/usr/bin/env python3
"""Mine missing historical user/action pairs without copying future state.

The input is a JSONL file whose rows contain an ``id``, ``history`` and optional
``session_meta``. History events are expected to alternate loosely between user
messages (``role=user``, ``content=...``) and assistant tool/action events
(``name=<label>``). The script reconstructs earlier supervised rows and writes a
JSONL/CSV pair.

Mining must be run inside a training fold. Never mine the full dataset and split
afterward: another window from the same root session can leak into validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable


LABELS = {
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
}

STEP_RE = re.compile(r"^(?P<root>.+)-step_(?P<step>\d+)$")


def root_session(sample_id: str) -> str:
    match = STEP_RE.match(str(sample_id))
    return match.group("root") if match else str(sample_id)


def canonical_key(session: str, step: int) -> tuple[str, int]:
    return root_session(session), int(step)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(row)
    return rows


def load_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"id", "action"}.issubset(reader.fieldnames):
            raise ValueError("label CSV must contain id and action columns")
        labels = {str(row["id"]): str(row["action"]) for row in reader}
    unknown = sorted(set(labels.values()) - LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    return labels


def safe_session_meta(meta: Any, preserve_dynamic: bool = False) -> dict[str, Any]:
    """Keep stable metadata; drop state observed after the reconstructed step."""

    if not isinstance(meta, dict):
        return {}
    if preserve_dynamic:
        return deepcopy(meta)

    stable_keys = {
        "language_pref",
        "user_tier",
        "primary_language",
        "language_mix",
        "top_lang",
    }
    return {key: deepcopy(value) for key, value in meta.items() if key in stable_keys}


def event_signature(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def historical_pairs(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield candidate pairs using only events before each target action."""

    history = row.get("history") or []
    if not isinstance(history, list):
        return

    action_count = 0
    pending_user: tuple[int, dict[str, Any]] | None = None
    for index, raw_event in enumerate(history):
        if not isinstance(raw_event, dict):
            continue
        event = deepcopy(raw_event)

        if event.get("role") == "user" and str(event.get("content") or "").strip():
            pending_user = (index, event)
            continue

        action = str(event.get("name") or "")
        if action not in LABELS:
            continue

        action_count += 1
        if pending_user is None:
            continue

        user_index, user_event = pending_user
        prompt = str(user_event.get("content") or "").strip()
        if not prompt:
            continue

        # The target user message and target action are excluded. Only genuinely
        # prior events become history for the reconstructed state.
        prior_history = deepcopy(history[:user_index])
        yield {
            "step": action_count,
            "prompt": prompt,
            "action": action,
            "history": prior_history,
            "source_event_index": index,
            "source_event_signature": event_signature(event),
        }
        pending_user = None


def mine_rows(
    rows: list[dict[str, Any]],
    known_labels: dict[str, str] | None = None,
    preserve_dynamic_meta: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    known_labels = known_labels or {}
    candidates: dict[tuple[str, int], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)

    for source_row in rows:
        source_id = str(source_row.get("id") or "")
        if not source_id:
            continue
        session = root_session(source_id)
        for pair in historical_pairs(source_row):
            candidates[canonical_key(session, pair["step"])].append((source_row, pair))

    mined_rows: list[dict[str, Any]] = []
    mined_labels: list[dict[str, str]] = []
    conflicts: list[dict[str, Any]] = []
    overlap_checked = 0
    overlap_mismatches = 0

    for (session, step), observations in sorted(candidates.items()):
        sample_id = f"{session}-step_{step:02d}"
        observed_labels = Counter(pair["action"] for _, pair in observations)
        observed_prompts = Counter(pair["prompt"] for _, pair in observations)

        if len(observed_labels) != 1 or len(observed_prompts) != 1:
            conflicts.append(
                {
                    "id": sample_id,
                    "labels": dict(observed_labels),
                    "prompt_count": len(observed_prompts),
                }
            )
            continue

        action = next(iter(observed_labels))
        if sample_id in known_labels:
            overlap_checked += 1
            if known_labels[sample_id] != action:
                overlap_mismatches += 1
                conflicts.append(
                    {
                        "id": sample_id,
                        "labels": {"known": known_labels[sample_id], "mined": action},
                        "prompt_count": len(observed_prompts),
                    }
                )
            continue

        # Prefer the shortest source window that still contains the pair. This
        # minimizes accidental dependence on later session state.
        source_row, pair = min(observations, key=lambda item: len(item[0].get("history") or []))
        reconstructed = {
            "id": sample_id,
            "current_prompt": pair["prompt"],
            "history": pair["history"],
            "session_meta": safe_session_meta(
                source_row.get("session_meta"), preserve_dynamic=preserve_dynamic_meta
            ),
            "mining_meta": {
                "source_id": str(source_row.get("id")),
                "source_event_index": pair["source_event_index"],
                "source_event_signature": pair["source_event_signature"],
                "dynamic_meta_preserved": bool(preserve_dynamic_meta),
            },
        }
        mined_rows.append(reconstructed)
        mined_labels.append({"id": sample_id, "action": action})

    report = {
        "source_rows": len(rows),
        "candidate_keys": len(candidates),
        "mined_rows": len(mined_rows),
        "conflict_keys": len(conflicts),
        "overlap_checked": overlap_checked,
        "overlap_mismatches": overlap_mismatches,
        "dynamic_meta_preserved": bool(preserve_dynamic_meta),
        "label_counts": dict(Counter(item["action"] for item in mined_labels)),
        "conflicts": conflicts,
    }
    return mined_rows, mined_labels, report


def write_outputs(
    rows: list[dict[str, Any]], labels: list[dict[str, str]], report: dict[str, Any], out: Path
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with out.with_suffix(".labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "action"])
        writer.writeheader()
        writer.writerows(labels)
    with out.with_suffix(".report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output stem without extension")
    parser.add_argument(
        "--preserve-dynamic-meta",
        action="store_true",
        help="copy all session_meta fields; unsafe unless each historical state is independently known",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.train_jsonl)
    labels = load_labels(args.train_labels) if args.train_labels else {}
    mined_rows, mined_labels, report = mine_rows(
        rows, labels, preserve_dynamic_meta=args.preserve_dynamic_meta
    )
    if report["overlap_mismatches"]:
        raise SystemExit(
            f"aborted: {report['overlap_mismatches']} mined labels disagree with known labels"
        )
    write_outputs(mined_rows, mined_labels, report, args.out)
    print(json.dumps({key: value for key, value in report.items() if key != "conflicts"}, indent=2))


if __name__ == "__main__":
    main()
