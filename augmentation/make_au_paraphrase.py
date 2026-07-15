from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """당신은 한국어 소프트웨어 개발 대화의 의역기입니다.
사용자의 발화를 의미와 요청 종류가 바뀌지 않는 다른 문장으로 바꾸세요.

규칙:
1. 지시어(다시, 그, 아까, 방금, 이전, 위에, 이거, 저거)는 반드시 원형 그대로 유지합니다.
2. 파일명, 경로, 함수명, 클래스명, 명령어, 백틱 안 코드 토큰은 글자 하나도 바꾸지 않습니다.
3. 요청 종류가 바뀌면 안 됩니다. 읽기 요청은 읽기 요청으로, 검색 요청은 검색 요청으로, 실행 요청은 실행 요청으로, 수정 요청은 수정 요청으로 유지합니다.
4. 한국어는 한국어로, 영어 섞인 문장은 영어 토큰을 그대로 유지합니다.
5. 원문이 영어 문장이면 의역도 영어로 작성합니다. 번역 금지.
6. 어순, 어투, 높임만 자연스럽게 변주합니다.
7. 출력은 의역 문장 하나만 반환합니다."""

DEIXIS = ["다시", "아까", "방금", "그때", "이전", "위에", "이거", "저거", "그거", "한번더", "한번 더", "한 번 더"]
FILE_RE = re.compile(r"[\w./-]+\.[A-Za-z]{1,5}")
BACKTICK_RE = re.compile(r"`([^`]+)`")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: Any) -> str:
    return SPACE_RE.sub(" ", str(text or "").strip())


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", str(text or "")))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_label_map(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "id" not in reader.fieldnames or "action" not in reader.fieldnames:
            raise ValueError(f"{path} must contain id,action columns")
        return {row["id"]: row["action"] for row in reader}


def save_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def cache_name(row_id: str) -> str:
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return f"para_{digest}.json"


def extract_deixis(text: str) -> list[str]:
    return [tok for tok in DEIXIS if tok in text]


def extract_file_tokens(text: str) -> list[str]:
    return sorted(set(FILE_RE.findall(text)))


def extract_backtick_tokens(text: str) -> list[str]:
    return sorted(set(BACKTICK_RE.findall(text)))


def protect_code_dots(text: str) -> str:
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text)
    protected = re.sub(r"\.(?=[A-Za-z_]\w*\()", "<DOT>", protected)
    protected = re.sub(r"(?<!\s)\.(?=[A-Za-z_])", "<DOT>", protected)
    for token in extract_file_tokens(text):
        protected = protected.replace(token, token.replace(".", "<DOT>"))
    return protected


def sentence_count(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    parts = [p.strip() for p in re.split(r"[.!?。？！]+", protect_code_dots(text)) if p.strip()]
    return max(1, len(parts))


def strip_model_output(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[-*\d.)\s]+", "", text).strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return normalize_text(text)


def contains_any(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(kw.lower() in low for kw in keywords)


def action_preserved(action: str, paraphrase: str) -> tuple[bool, str]:
    text = paraphrase.lower()
    checks: dict[str, list[str]] = {
        "run_tests": ["run", "test", "suite", "pytest", "jest", "vitest", "테스트", "돌려", "실행", "태워"],
        "run_bash": ["run", "execute", "cmd", "command", "bash", "build", "install", "vet", "dbt", "pod", "돌려", "실행", "빌드"],
        "lint_or_typecheck": ["lint", "type", "typecheck", "mypy", "tsc", "ruff", "정적", "타입"],
        "read_file": ["open", "read", "show", "look", "열어", "열어서", "보여", "읽어", "확인"],
        "grep_search": ["find", "search", "grep", "where", "reference", "usage", "찾아", "검색", "어디", "호출", "참조"],
        "glob_pattern": ["glob", "pattern", "files", "파일들", "패턴", "훑어"],
        "list_directory": ["list", "tree", "directory", "folder", "목록", "구조", "폴더", "디렉"],
        "edit_file": ["fix", "edit", "change", "update", "patch", "rename", "wrap", "add", "고쳐", "수정", "추가", "바꿔", "연결"],
        "apply_patch": ["patch", "edit", "change", "fix", "update", "rename", "refactor", "패치", "수정", "정리", "손봐"],
        "write_file": ["create", "write", "scaffold", "make", "fresh", "생성", "작성", "만들"],
        "ask_user": ["뭐부터", "해야 하나", "막막", "물어", "확인해", "질문"],
        "plan_task": ["plan", "계획", "방향", "먼저", "순서", "정리"],
        "respond_only": ["요약", "마무리", "여기까지", "고마워", "수고"],
        "web_search": ["search", "web", "latest", "요즘", "국룰", "비교", "검색"],
    }
    keywords = checks.get(action)
    if not keywords:
        return True, "no_action_check"
    if contains_any(text, keywords):
        return True, "OK"
    return False, f"action_drift:{action}"


@dataclass
class QCResult:
    ok: bool
    reason: str
    len_ratio: float
    deixis_kept: bool
    filetok_kept: bool
    backtick_kept: bool
    action_kept: bool
    sentence_count: int
    same_text: bool


def qc(original: str, paraphrase: str, action: str) -> QCResult:
    original = normalize_text(original)
    paraphrase = normalize_text(paraphrase)
    if not paraphrase:
        return QCResult(False, "empty", 0.0, False, False, False, False, 0, False)

    len_ratio = len(paraphrase) / max(1, len(original))
    min_ratio, max_ratio = (0.4, 4.0) if len(original) <= 12 else (0.5, 2.2)
    same_text = normalize_text(original).lower() == normalize_text(paraphrase).lower()
    language_kept = has_hangul(original) == has_hangul(paraphrase)
    missing_deixis = [tok for tok in extract_deixis(original) if tok not in paraphrase]
    missing_files = [tok for tok in extract_file_tokens(original) if tok not in paraphrase]
    missing_backticks = [tok for tok in extract_backtick_tokens(original) if tok not in paraphrase]
    action_ok, action_reason = action_preserved(action, paraphrase)
    n_sent = sentence_count(paraphrase)

    reasons: list[str] = []
    if same_text:
        reasons.append("same_text")
    if len_ratio < min_ratio or len_ratio > max_ratio:
        reasons.append("len_ratio")
    if not language_kept:
        reasons.append("language_mismatch")
    if missing_deixis:
        reasons.append("missing_deixis:" + "|".join(missing_deixis))
    if missing_files:
        reasons.append("missing_filetok:" + "|".join(missing_files))
    if missing_backticks:
        reasons.append("missing_backtick:" + "|".join(missing_backticks))
    if n_sent > 3:
        reasons.append("too_many_sentences")
    risk_reasons: list[str] = []
    if not action_ok:
        risk_reasons.append("action_risk")

    return QCResult(
        ok=not reasons,
        reason="OK" if not reasons and not risk_reasons else ";".join(reasons + risk_reasons),
        len_ratio=len_ratio,
        deixis_kept=not missing_deixis,
        filetok_kept=not missing_files,
        backtick_kept=not missing_backticks,
        action_kept=action_ok,
        sentence_count=n_sent,
        same_text=same_text,
    )


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout: int) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_openai(prompt: str, action: str, row_id: str, model: str, temperature: float, timeout: int, api_url: str | None) -> dict[str, Any]:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    url = api_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"action label: {action}\n"
                    f"id: {row_id}\n"
                    f"original current_prompt:\n{prompt}\n\n"
                    "의역 문장 하나만 출력하세요."
                ),
            },
        ],
    }
    raw = post_json(url, payload, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, timeout)
    text = raw["choices"][0]["message"]["content"]
    return {"text": strip_model_output(text), "raw_response": raw}


def call_anthropic(prompt: str, action: str, row_id: str, model: str, temperature: float, timeout: int, api_url: str | None) -> dict[str, Any]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    url = api_url or "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 256,
        "temperature": temperature,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": (
                    f"action label: {action}\n"
                    f"id: {row_id}\n"
                    f"original current_prompt:\n{prompt}\n\n"
                    "의역 문장 하나만 출력하세요."
                ),
            }
        ],
    }
    raw = post_json(
        url,
        payload,
        {"x-api-key": key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"},
        timeout,
    )
    text = "".join(part.get("text", "") for part in raw.get("content", []) if part.get("type") == "text")
    return {"text": strip_model_output(text), "raw_response": raw}


def call_local(prompt: str, action: str, row_id: str) -> dict[str, Any]:
    text = normalize_text(prompt)
    replacements = [
        ("좀", "한번"),
        ("보여줘", "열어서 확인해줘"),
        ("찾아줘", "검색해서 확인해줘"),
        ("돌려봐", "실행해서 확인해봐"),
        ("고쳐", "수정해"),
        ("추가해줘", "넣어줘"),
        ("요약해줘", "정리해줘"),
        ("show me", "pull up"),
        ("find", "look for"),
        ("run", "execute"),
        ("fix", "patch"),
    ]
    out = text
    for src, dst in replacements:
        flags = re.IGNORECASE if src.isascii() else 0
        candidate = re.sub(re.escape(src), dst, out, count=1, flags=flags)
        if candidate != out:
            out = candidate
            break
    if normalize_text(out).lower() == normalize_text(text).lower():
        out = f"please {text}" if not has_hangul(text) else f"{text} 확인 부탁해"
    return {"text": strip_model_output(out), "raw_response": {"provider": "local", "id": row_id, "action": action}}


def generate_one(args: argparse.Namespace, prompt: str, action: str, row_id: str) -> dict[str, Any]:
    if args.provider == "openai":
        return call_openai(prompt, action, row_id, args.model, args.temperature, args.timeout, args.api_url)
    if args.provider == "anthropic":
        return call_anthropic(prompt, action, row_id, args.model, args.temperature, args.timeout, args.api_url)
    if args.provider == "local":
        return call_local(prompt, action, row_id)
    raise ValueError(args.provider)


def default_model(provider: str, model: str | None) -> str:
    if model:
        return model
    if provider == "openai":
        return "gpt-4o-mini"
    if provider == "anthropic":
        return "claude-3-5-sonnet-latest"
    return "local-rule-draft"


def load_existing_review(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def load_existing_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["id"]: row for row in read_jsonl(path)}


def load_existing_labels(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def write_report(
    path: Path,
    total_train_rows: int,
    target_au_rows: int,
    accepted_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    output_paths: dict[str, str],
) -> None:
    reject_reasons = Counter(row["qc_reason"] for row in review_rows if row["qc"] != "OK")
    accepted_actions = Counter(row["action"] for row in label_rows)
    rejected_actions = Counter(row["action"] for row in review_rows if row["qc"] != "OK")
    report = {
        "total_train_rows": total_train_rows,
        "target_au_rows": target_au_rows,
        "generated_count": len(review_rows),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(review_rows) - len(accepted_rows),
        "reject_reason_counts": dict(reject_reasons),
        "action_distribution_accepted": dict(accepted_actions),
        "action_distribution_rejected": dict(rejected_actions),
        "output_paths": output_paths,
        "note": "50 row sample pass is not performance proof; run 3fold CV before using full paraphrases for final training.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_all(
    out_dir: Path,
    target_order: list[str],
    accepted_by_id: dict[str, dict[str, Any]],
    labels_by_id: dict[str, dict[str, Any]],
    review_by_id: dict[str, dict[str, Any]],
    total_train_rows: int,
    target_au_rows: int,
) -> None:
    accepted = [accepted_by_id[row_id] for row_id in target_order if row_id in accepted_by_id]
    labels = [labels_by_id[row_id] for row_id in target_order if row_id in labels_by_id]
    reviews = [review_by_id[row_id] for row_id in target_order if row_id in review_by_id]
    jsonl_path = out_dir / "au_para.jsonl"
    labels_path = out_dir / "au_para.labels.csv"
    review_path = out_dir / "au_para_review.csv"
    report_path = out_dir / "para_report.json"
    write_jsonl(jsonl_path, accepted)
    save_csv(labels_path, labels, ["id", "action"])
    save_csv(
        review_path,
        reviews,
        [
            "id",
            "row_index",
            "action",
            "qc",
            "qc_reason",
            "len_ratio",
            "deixis_kept",
            "filetok_kept",
            "backtick_kept",
            "action_kept",
            "sentence_count",
            "cache_file",
            "original",
            "paraphrase",
        ],
    )
    write_report(
        report_path,
        total_train_rows,
        target_au_rows,
        accepted,
        labels,
        reviews,
        {
            "jsonl": str(jsonl_path),
            "labels": str(labels_path),
            "review": str(review_path),
            "report": str(report_path),
        },
    )


def append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AU(sess_au) paraphrase augmentation rows.")
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/original/train.jsonl"))
    parser.add_argument("--labels-csv", type=Path, default=Path("data/original/train_labels.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/au_para"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--logs-dir", type=Path, default=Path("logs"))
    parser.add_argument("--provider", choices=["anthropic", "openai", "local"], default="anthropic")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--id-prefix", default="sess_au")
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.model = default_model(args.provider, args.model)
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    labels = load_label_map(args.labels_csv)
    all_rows = read_jsonl(args.train_jsonl)
    au_rows = [row for row in all_rows if str(row.get("id", "")).startswith(args.id_prefix)]
    selected_rows = au_rows[args.start_index :]
    if args.limit is not None:
        selected_rows = selected_rows[: args.limit]
    target_order = [str(row["id"]) for row in au_rows]

    accepted_by_id = load_existing_jsonl_by_id(args.out_dir / "au_para.jsonl") if args.resume else {}
    labels_by_id = load_existing_labels(args.out_dir / "au_para.labels.csv") if args.resume else {}
    review_by_id = load_existing_review(args.out_dir / "au_para_review.csv") if args.resume else {}

    processed_since_save = 0
    log_path = args.logs_dir / "au_para_batches.log"
    append_log(
        log_path,
        {
            "event": "start",
            "time": datetime.now().isoformat(timespec="seconds"),
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "total_train_rows": len(all_rows),
            "target_au_rows_total": len(au_rows),
            "selected_rows": len(selected_rows),
            "start_index": args.start_index,
            "limit": args.limit,
            "resume": args.resume,
        },
    )

    for local_index, row in enumerate(selected_rows):
        row_id = str(row["id"])
        row_index = args.start_index + local_index
        if args.resume and row_id in review_by_id:
            continue
        if row_id not in labels:
            raise KeyError(f"missing label for {row_id}")
        action = labels[row_id]
        original_prompt = normalize_text(row.get("current_prompt", ""))
        cache_path = args.cache_dir / cache_name(row_id)

        generated: dict[str, Any] | None = None
        if cache_path.exists() and not args.overwrite_cache:
            generated = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    generated_payload = generate_one(args, original_prompt, action, row_id)
                    generated = {
                        "id": row_id,
                        "action": action,
                        "provider": args.provider,
                        "model": args.model,
                        "temperature": args.temperature,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "original": original_prompt,
                        "text": generated_payload["text"],
                        "raw_response": generated_payload["raw_response"],
                    }
                    cache_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")
                    break
                except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
                    last_error = repr(exc)
                    time.sleep(min(2 * attempt, 10))
            if generated is None:
                generated = {
                    "id": row_id,
                    "action": action,
                    "provider": args.provider,
                    "model": args.model,
                    "temperature": args.temperature,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "original": original_prompt,
                    "text": "",
                    "error": last_error or "unknown_error",
                    "raw_response": {},
                }
                cache_path.write_text(json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8")

        paraphrase = strip_model_output(generated.get("text", ""))
        q = qc(original_prompt, paraphrase, action)
        review_by_id[row_id] = {
            "id": row_id,
            "row_index": row_index,
            "action": action,
            "qc": "OK" if q.ok else "FAIL",
            "qc_reason": q.reason,
            "len_ratio": round(q.len_ratio, 4),
            "deixis_kept": q.deixis_kept,
            "filetok_kept": q.filetok_kept,
            "backtick_kept": q.backtick_kept,
            "action_kept": q.action_kept,
            "sentence_count": q.sentence_count,
            "cache_file": str(cache_path),
            "original": original_prompt,
            "paraphrase": paraphrase,
        }
        if q.ok:
            new_row = deepcopy(row)
            new_row["current_prompt"] = paraphrase
            accepted_by_id[row_id] = new_row
            labels_by_id[row_id] = {"id": row_id, "action": action}
        else:
            accepted_by_id.pop(row_id, None)
            labels_by_id.pop(row_id, None)

        processed_since_save += 1
        if processed_since_save >= args.batch_size:
            save_all(args.out_dir, target_order, accepted_by_id, labels_by_id, review_by_id, len(all_rows), len(au_rows))
            append_log(
                log_path,
                {
                    "event": "batch_saved",
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "processed_reviews": len(review_by_id),
                    "accepted": len(accepted_by_id),
                    "rejected": len(review_by_id) - len(accepted_by_id),
                    "last_row_index": row_index,
                },
            )
            processed_since_save = 0
        if args.sleep > 0 and args.provider != "local":
            time.sleep(args.sleep)

    save_all(args.out_dir, target_order, accepted_by_id, labels_by_id, review_by_id, len(all_rows), len(au_rows))
    append_log(
        log_path,
        {
            "event": "finish",
            "time": datetime.now().isoformat(timespec="seconds"),
            "processed_reviews": len(review_by_id),
            "accepted": len(accepted_by_id),
            "rejected": len(review_by_id) - len(accepted_by_id),
        },
    )
    print((args.out_dir / "para_report.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
