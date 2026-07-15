import argparse
import copy
import csv
import json
import re
from collections import Counter
from pathlib import Path


DEIXIS_TERMS = [
    "다시",
    "아까",
    "방금",
    "그때",
    "이전",
    "위에",
    "이거",
    "저거",
    "그거",
    "한번더",
    "한번 더",
    "한 번 더",
]

FILE_TOKEN_RE = re.compile(r"[\w./-]+\.[a-zA-Z]{1,5}")
BACKTICK_RE = re.compile(r"`([^`]+)`")
HANGUL_RE = re.compile(r"[가-힣]")

ACTION_KEYWORDS = {
    "run_tests": re.compile(r"테스트|돌려|실행|run|test|suite|pytest|jest", re.I),
    "lint_or_typecheck": re.compile(r"lint|type|check|타입|검사|정적", re.I),
    "read_file": re.compile(r"열어|보자|봐|보여|확인|open|read|show|look", re.I),
    "grep_search": re.compile(r"찾아|검색|어디|위치|grep|find|search|where", re.I),
    "edit_file": re.compile(r"고쳐|수정|바꿔|추가|넣어|패치|작성|만들|fix|change|add|update|patch|rename|write", re.I),
    "write_file": re.compile(r"고쳐|수정|바꿔|추가|넣어|패치|작성|만들|fix|change|add|update|patch|rename|write", re.I),
    "apply_patch": re.compile(r"고쳐|수정|바꿔|추가|넣어|패치|작성|만들|fix|change|add|update|patch|rename|write", re.I),
    "ask_user": re.compile(r"물어|질문|어느|뭐가|어떻게|맞나|should|which|what|\?", re.I),
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def load_labels(path):
    return {str(r["id"]): str(r["action"]) for r in read_csv(path)}


def load_rows(train_jsonl):
    out = {}
    with Path(train_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("id", ""))] = row
    return out


def has_hangul(text):
    return bool(HANGUL_RE.search(text or ""))


def tokens_preserved(pattern, original, paraphrase):
    orig = set(pattern.findall(original or ""))
    para = set(pattern.findall(paraphrase or ""))
    return orig.issubset(para)


def deixis_preserved(original, paraphrase):
    original = original or ""
    paraphrase = paraphrase or ""
    for term in DEIXIS_TERMS:
        if term in original and term not in paraphrase:
            return False
    return True


def sentence_count(text):
    text = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", text or "")
    text = FILE_TOKEN_RE.sub("<FILE>", text)
    parts = [p.strip() for p in re.split(r"[.!?。！？]+", text) if p.strip()]
    return len(parts)


def len_ratio_ok(original, paraphrase):
    olen = max(len(original or ""), 1)
    ratio = len(paraphrase or "") / olen
    if len(original or "") <= 12:
        return 0.4 <= ratio <= 4.0, ratio
    return 0.5 <= ratio <= 2.2, ratio


def action_kept(action, paraphrase):
    pattern = ACTION_KEYWORDS.get(action)
    if pattern is None:
        return True
    return bool(pattern.search(paraphrase or ""))


def qc_row(original, paraphrase, action):
    hard = []
    paraphrase = paraphrase or ""
    if not paraphrase.strip():
        hard.append("empty")
    if paraphrase == (original or ""):
        hard.append("same_text")
    if any(ch in paraphrase for ch in ["\n", "\r", "\t"]):
        hard.append("control_char")

    deixis_ok = deixis_preserved(original, paraphrase)
    filetok_ok = tokens_preserved(FILE_TOKEN_RE, original, paraphrase)
    backtick_ok = tokens_preserved(BACKTICK_RE, original, paraphrase)
    lang_ok = has_hangul(original) == has_hangul(paraphrase)
    length_ok, ratio = len_ratio_ok(original, paraphrase)
    sent_ok = sentence_count(paraphrase) <= 3
    action_ok = action_kept(action, paraphrase)

    if not deixis_ok:
        hard.append("deixis_lost")
    if not filetok_ok:
        hard.append("filetok_lost")
    if not backtick_ok:
        hard.append("backtick_lost")
    if not lang_ok:
        hard.append("lang_mismatch")
    if not length_ok:
        hard.append("len_ratio")
    if not sent_ok:
        hard.append("too_many_sentences")

    reasons = list(hard)
    if not action_ok:
        reasons.append("action_risk")
    return {
        "qc": "OK" if not hard else "REJECT",
        "qc_reason": "OK" if not reasons else "|".join(reasons),
        "len_ratio": ratio,
        "deixis_kept": deixis_ok,
        "filetok_kept": filetok_ok,
        "backtick_kept": backtick_ok,
        "lang_kept": lang_ok,
        "action_kept": action_ok,
        "hard_reasons": hard,
    }


def batch_number(path):
    m = re.search(r"batch_(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"bad batch filename: {path}")
    return int(m.group(1))


def validate_structure(batch_rows, para_rows):
    if len(batch_rows) != len(para_rows):
        return False, f"row_count {len(batch_rows)} != {len(para_rows)}"
    for i, (base, para) in enumerate(zip(batch_rows, para_rows)):
        for key in ["idx", "id", "action", "original"]:
            if str(base.get(key, "")) != str(para.get(key, "")):
                return False, f"row {i} {key} mismatch"
    return True, "OK"


def assemble(args):
    labels = load_labels(args.labels_csv)
    raw_rows = load_rows(args.train_jsonl)
    batch_dir = Path(args.batch_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    original_batches = sorted(batch_dir.glob("batch_[0-9][0-9][0-9].csv"))
    review = []
    accepted_rows = []
    label_rows = []
    batch_errors = []
    reject_reasons = Counter()
    accepted_actions = Counter()
    rejected_actions = Counter()
    filled = 0
    valid_batches = 0
    structural_rejected_rows = 0

    for batch_path in original_batches:
        num = batch_number(batch_path)
        para_path = batch_dir / f"batch_{num:03d}_para.csv"
        if not para_path.exists():
            continue
        filled += 1
        batch_rows = read_csv(batch_path)
        para_rows = read_csv(para_path)
        ok, message = validate_structure(batch_rows, para_rows)
        if not ok:
            structural_rejected_rows += len(batch_rows)
            batch_errors.append({"batch": num, "reason": message})
            continue
        valid_batches += 1

        for base, para in zip(batch_rows, para_rows):
            row_id = str(base["id"])
            action = str(base["action"])
            if labels.get(row_id) != action:
                raise ValueError(f"label mismatch against labels csv: {row_id}")
            if row_id not in raw_rows:
                raise ValueError(f"missing raw train row: {row_id}")
            original = str(base["original"])
            paraphrase = str(para.get("paraphrase") or "").strip()
            qc = qc_row(original, paraphrase, action)

            review.append(
                {
                    "id": row_id,
                    "batch": f"{num:03d}",
                    "action": action,
                    "qc": qc["qc"],
                    "qc_reason": qc["qc_reason"],
                    "len_ratio": f"{qc['len_ratio']:.4f}",
                    "deixis_kept": qc["deixis_kept"],
                    "filetok_kept": qc["filetok_kept"],
                    "backtick_kept": qc["backtick_kept"],
                    "lang_kept": qc["lang_kept"],
                    "action_kept": qc["action_kept"],
                    "original": original,
                    "paraphrase": paraphrase,
                }
            )

            if qc["qc"] == "OK":
                new_row = copy.deepcopy(raw_rows[row_id])
                new_row["current_prompt"] = paraphrase
                accepted_rows.append(new_row)
                label_rows.append({"id": row_id, "action": action})
                accepted_actions[action] += 1
            else:
                rejected_actions[action] += 1
                for reason in qc["hard_reasons"]:
                    reject_reasons[reason] += 1

    with (out_dir / "au_para.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in accepted_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_csv(out_dir / "au_para.labels.csv", label_rows, ["id", "action"])
    write_csv(
        out_dir / "au_para_review.csv",
        review,
        [
            "id",
            "batch",
            "action",
            "qc",
            "qc_reason",
            "len_ratio",
            "deixis_kept",
            "filetok_kept",
            "backtick_kept",
            "lang_kept",
            "action_kept",
            "original",
            "paraphrase",
        ],
    )

    report = {
        "target_au_rows": sum(len(read_csv(p)) for p in original_batches),
        "batches_total": len(original_batches),
        "batches_filled": filled,
        "batches_valid": valid_batches,
        "accepted": len(accepted_rows),
        "rejected": sum(1 for r in review if r["qc"] != "OK"),
        "structural_rejected_rows": structural_rejected_rows,
        "reject_reason_counts": dict(reject_reasons),
        "action_distribution_accepted": dict(accepted_actions),
        "action_distribution_rejected": dict(rejected_actions),
        "batch_errors": batch_errors,
        "note": "Only sess_au paraphrases are assembled. action_risk is a soft review flag, not a hard reject.",
    }
    with (out_dir / "para_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Assemble and QC filled AU paraphrase batches.")
    ap.add_argument("--train-jsonl", default="data/original/train.jsonl")
    ap.add_argument("--labels-csv", default="data/original/train_labels.csv")
    ap.add_argument("--batch-dir", default="data/au_para_batches")
    ap.add_argument("--out-dir", default="data/au_para")
    args = ap.parse_args()
    report = assemble(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
