from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from train_inspect4_specialist_proto import inspect_short_text, seed_everything


FULL_LABELS = np.asarray(
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
INSPECT_LABELS = np.asarray(
    ["read_file", "grep_search", "glob_pattern", "list_directory"], dtype=object
)
INSPECT_SET = set(INSPECT_LABELS.tolist())
INSPECT_DESCRIPTIONS = {
    "read_file": (
        "open a known file path and inspect that file's contents; "
        "a specific file is already identified; 지정된 파일 경로를 열어 내용을 읽기"
    ),
    "grep_search": (
        "search text, symbols, references, or usages inside file contents; "
        "파일 내용에서 문자열 심볼 참조 사용처 검색"
    ),
    "glob_pattern": (
        "find file paths by filename, extension, wildcard, or glob pattern; "
        "파일명 확장자 와일드카드 패턴으로 파일 경로 찾기"
    ),
    "list_directory": (
        "list immediate directory children or inspect folder and project structure; "
        "디렉터리 하위 항목과 폴더 구조 나열"
    ),
}


def load_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_sim_rows(data_dir: Path):
    with (data_dir / "train_labels.csv").open(encoding="utf-8-sig", newline="") as handle:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(handle)}
    rows = []
    y = []
    for row in load_jsonl(data_dir / "train.jsonl"):
        rid = str(row.get("id"))
        if rid.startswith("sess_sim"):
            rows.append(row)
            y.append(labels[rid])
    return rows, np.asarray(y, dtype=object)


def load_fold_map(oof_dir: Path):
    mapping = {}
    for fold in range(5):
        ids = np.load(oof_dir / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
        for rid in ids:
            if rid in mapping:
                raise RuntimeError(f"duplicate OOF id: {rid}")
            mapping[rid] = fold
    return mapping


def candidate_pair_text(state_text: str, label: str):
    return "\n".join(
        [
            "query: score this candidate for the next inspection action",
            "[CANDIDATE]",
            f"action={label}",
            f"meaning={INSPECT_DESCRIPTIONS[label]}",
            "[WORKFLOW_STATE]",
            state_text,
        ]
    )


class CandidateRowDataset(Dataset):
    def __init__(self, rows, labels, feature_fn):
        self.rows = list(rows)
        self.labels = np.asarray(labels, dtype=object)
        self.states = [inspect_short_text(row, 0) for row in self.rows]
        self.features = np.vstack([feature_fn(row) for row in self.rows]).astype(np.float32)
        self.targets = np.asarray(
            [INSPECT_LABELS.tolist().index(label) if label in INSPECT_SET else -1 for label in self.labels],
            dtype=np.int64,
        )
        self.ids = np.asarray([str(row["id"]) for row in self.rows], dtype=object)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return {
            "state": self.states[index],
            "features": self.features[index],
            "target": int(self.targets[index]),
            "id": str(self.ids[index]),
        }


def make_collate(tokenizer, max_length):
    def collate(batch):
        pair_texts = []
        features = []
        for item in batch:
            for label in INSPECT_LABELS:
                pair_texts.append(candidate_pair_text(item["state"], str(label)))
                features.append(item["features"])
        encoded = tokenizer(
            pair_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["features"] = torch.tensor(np.vstack(features), dtype=torch.float32)
        encoded["labels"] = torch.tensor([item["target"] for item in batch], dtype=torch.long)
        encoded["ids"] = [item["id"] for item in batch]
        return encoded

    return collate


class CandidateConditionedRanker(nn.Module):
    def __init__(self, model_dir: Path, feature_dim: int, dropout: float):
        super().__init__()
        config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_dir, config=config, local_files_only=True)
        hidden = int(config.hidden_size)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden + feature_dim),
            nn.Linear(hidden + feature_dim, 384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 96),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(96, 1),
        )

    @staticmethod
    def pool_mean(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask, features, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        mean = self.pool_mean(output.last_hidden_state, attention_mask)
        mean = F.normalize(mean.float(), p=2, dim=-1).to(output.last_hidden_state.dtype)
        feats = self.feature_norm(features.to(mean.dtype))
        scores = self.scorer(torch.cat([mean, feats], dim=-1)).squeeze(-1)
        if scores.numel() % len(INSPECT_LABELS) != 0:
            raise RuntimeError("candidate scores cannot be reshaped into four actions")
        return scores.view(-1, len(INSPECT_LABELS))


def model_inputs(batch):
    keys = {"input_ids", "attention_mask", "token_type_ids", "features"}
    return {key: value for key, value in batch.items() if key in keys}


@torch.no_grad()
def infer_ranker(model, loader, device):
    model.eval()
    probs = []
    ids = []
    targets = []
    for batch in loader:
        ids.extend(batch.pop("ids"))
        labels = batch.pop("labels")
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(**model_inputs(batch))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        targets.append(labels.numpy())
    return (
        np.asarray(ids, dtype=object),
        np.concatenate(targets).astype(np.int64),
        np.vstack(probs).astype(np.float32),
    )


def load_base_fold_proba(args, fold, valid_ids):
    a_dir = Path(args.base_a_oof_dir)
    a_ids = np.load(a_dir / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
    a_proba = np.load(a_dir / f"oof_fold{fold}_proba.npy").astype(np.float32)

    b_dir = Path(args.base_b_oof_root) / (
        f"sim0783_header3tok_mined_aligned_aufold{fold + 1}_e2_seed42_20260708_171135"
    )
    b_ids = np.load(b_dir / "valid_sample_ids.npy", allow_pickle=True).astype(str)
    b_proba = np.load(b_dir / "valid_proba.npy").astype(np.float32)
    b_classes = np.load(b_dir / "classes.npy", allow_pickle=True).astype(str)
    if b_classes.tolist() != FULL_LABELS.tolist():
        raise RuntimeError(f"unexpected B class order: {b_classes.tolist()}")

    a_index = {rid: index for index, rid in enumerate(a_ids)}
    b_index = {rid: index for index, rid in enumerate(b_ids)}
    missing_a = [rid for rid in valid_ids if rid not in a_index]
    missing_b = [rid for rid in valid_ids if rid not in b_index]
    if missing_a or missing_b:
        raise RuntimeError(f"base OOF alignment failed: missing_a={len(missing_a)} missing_b={len(missing_b)}")
    a = np.vstack([a_proba[a_index[rid]] for rid in valid_ids]).astype(np.float32)
    b = np.vstack([b_proba[b_index[rid]] for rid in valid_ids]).astype(np.float32)
    return (0.60 * a + 0.40 * b).astype(np.float32)


def step_number(row):
    rid = str(row.get("id", ""))
    try:
        return int(rid.rsplit("-step_", 1)[1])
    except Exception:
        return -1


def evaluate_blends(rows, y_labels, base_proba, ranker_proba, epoch):
    full_to_idx = {label: index for index, label in enumerate(FULL_LABELS)}
    inspect_full_idx = np.asarray([full_to_idx[label] for label in INSPECT_LABELS], dtype=np.int64)
    y_full = np.asarray([full_to_idx[label] for label in y_labels], dtype=np.int64)
    base_pred = base_proba.argmax(axis=1)
    true_inspect = np.asarray([label in INSPECT_SET for label in y_labels], dtype=bool)
    y_inspect = np.asarray(
        [INSPECT_LABELS.tolist().index(label) if label in INSPECT_SET else -1 for label in y_labels],
        dtype=np.int64,
    )

    base_cond = base_proba[:, inspect_full_idx]
    inspect_mass = base_cond.sum(axis=1, keepdims=True)
    base_cond = base_cond / np.maximum(inspect_mass, 1e-8)
    base_internal_pred = base_cond.argmax(axis=1)
    ranker_pred = ranker_proba.argmax(axis=1)

    base_macro = float(f1_score(y_full, base_pred, labels=np.arange(len(FULL_LABELS)), average="macro"))
    base_acc = float(accuracy_score(y_full, base_pred))
    base_inspect_macro = float(f1_score(y_inspect[true_inspect], base_internal_pred[true_inspect], average="macro"))
    ranker_inspect_macro = float(f1_score(y_inspect[true_inspect], ranker_pred[true_inspect], average="macro"))
    ranker_inspect_acc = float(accuracy_score(y_inspect[true_inspect], ranker_pred[true_inspect]))

    rows_out = []
    gate_masks = {"top1_inspect": np.isin(base_pred, inspect_full_idx)}
    for threshold in [0.45, 0.50, 0.60, 0.70]:
        gate_masks[f"inspect_mass_ge_{threshold:.2f}"] = inspect_mass[:, 0] >= threshold

    for gate_name, gate in gate_masks.items():
        for alpha in [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
            mixed_cond = (1.0 - alpha) * base_cond + alpha * ranker_proba
            final = base_proba.copy()
            final[np.ix_(gate, inspect_full_idx)] = (
                inspect_mass[gate] * mixed_cond[gate]
            )
            pred = final.argmax(axis=1)
            macro = float(f1_score(y_full, pred, labels=np.arange(len(FULL_LABELS)), average="macro"))
            acc = float(accuracy_score(y_full, pred))
            rows_out.append(
                {
                    "epoch": int(epoch),
                    "gate": gate_name,
                    "alpha": float(alpha),
                    "gated_rows": int(gate.sum()),
                    "macro_f1": macro,
                    "macro_delta": macro - base_macro,
                    "accuracy": acc,
                    "accuracy_delta": acc - base_acc,
                    "changed_predictions": int((pred != base_pred).sum()),
                }
            )

    step_metrics = []
    steps = np.asarray([step_number(row) for row in rows], dtype=np.int16)
    for step in [1, 2, 3, 4, 5, 6]:
        mask = true_inspect & (steps == step)
        if not mask.any():
            continue
        step_metrics.append(
            {
                "epoch": int(epoch),
                "step": int(step),
                "rows": int(mask.sum()),
                "base_internal_macro_f1": float(
                    f1_score(y_inspect[mask], base_internal_pred[mask], labels=np.arange(4), average="macro")
                ),
                "ranker_macro_f1": float(
                    f1_score(y_inspect[mask], ranker_pred[mask], labels=np.arange(4), average="macro")
                ),
                "base_internal_acc": float(accuracy_score(y_inspect[mask], base_internal_pred[mask])),
                "ranker_acc": float(accuracy_score(y_inspect[mask], ranker_pred[mask])),
            }
        )

    fixed = next(row for row in rows_out if row["gate"] == "top1_inspect" and row["alpha"] == 0.30)
    best = max(rows_out, key=lambda row: row["macro_f1"])
    summary = {
        "epoch": int(epoch),
        "base_macro_f1": base_macro,
        "base_accuracy": base_acc,
        "true_inspect_rows": int(true_inspect.sum()),
        "base_internal_inspect_macro_f1": base_inspect_macro,
        "ranker_inspect_macro_f1": ranker_inspect_macro,
        "ranker_inspect_accuracy": ranker_inspect_acc,
        "fixed_top1_alpha_0.30": fixed,
        "best_sweep": best,
    }
    report = classification_report(
        y_inspect[true_inspect],
        ranker_pred[true_inspect],
        labels=np.arange(4),
        target_names=INSPECT_LABELS.tolist(),
        digits=4,
        zero_division=0,
    )
    return summary, rows_out, step_metrics, report


def save_model(model, tokenizer, out_dir: Path, epoch: int, summary):
    model_dir = out_dir / f"epoch_{epoch}_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "model.pt")
    tokenizer.save_pretrained(model_dir)
    (model_dir / "ranker_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument(
        "--code-dir",
        default="reference/training",
    )
    parser.add_argument(
        "--base-a-oof-dir",
        default="artifacts/oof/sim_a",
    )
    parser.add_argument("--base-b-oof-root", default="artifacts/oof/sim_b")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--encoder-lr", type=float, default=1.5e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed + args.fold)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    from run_domain_input_global_oof import structured_features

    all_rows, all_y = load_sim_rows(Path(args.data_dir))
    fold_map = load_fold_map(Path(args.base_a_oof_dir))
    folds = np.asarray([fold_map[str(row["id"])] for row in all_rows], dtype=np.int16)
    train_mask = (folds != args.fold) & np.asarray([label in INSPECT_SET for label in all_y], dtype=bool)
    valid_mask = folds == args.fold
    train_rows = [all_rows[index] for index in np.where(train_mask)[0]]
    train_y = all_y[train_mask]
    valid_rows = [all_rows[index] for index in np.where(valid_mask)[0]]
    valid_y = all_y[valid_mask]

    train_roots = {str(row["id"]).split("-step_", 1)[0] for row in train_rows}
    valid_roots = {str(row["id"]).split("-step_", 1)[0] for row in valid_rows}
    overlap = train_roots & valid_roots
    if overlap:
        raise RuntimeError(f"train/valid root session leakage: {len(overlap)}")

    run_summary = {
        "fold": args.fold,
        "train_true_inspect_rows": len(train_rows),
        "train_candidate_sequences_per_epoch": len(train_rows) * 4,
        "valid_all_rows": len(valid_rows),
        "valid_candidate_sequences_per_epoch": len(valid_rows) * 4,
        "train_label_counts": dict(Counter(train_y.tolist())),
        "valid_label_counts": dict(Counter(valid_y.tolist())),
        "root_session_overlap": 0,
    }
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        valid_ids = [str(row["id"]) for row in valid_rows]
        base_proba = load_base_fold_proba(args, args.fold, valid_ids)
        print(
            json.dumps(
                {
                    "dry_run": "ok",
                    "base_oof_shape": list(base_proba.shape),
                    "base_oof_row_sums": [
                        float(base_proba.sum(axis=1).min()),
                        float(base_proba.sum(axis=1).max()),
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=True)
    train_ds = CandidateRowDataset(train_rows, train_y, structured_features)
    valid_ds = CandidateRowDataset(valid_rows, valid_y, structured_features)
    collate_train = make_collate(tokenizer, args.max_length)
    collate_valid = make_collate(tokenizer, args.max_length)
    generator = torch.Generator().manual_seed(args.seed + args.fold)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_train,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_valid,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    model = CandidateConditionedRanker(
        Path(args.model_dir), feature_dim=train_ds.features.shape[1], dropout=args.dropout
    ).to(device)

    counts = np.bincount(train_ds.targets, minlength=4).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts, 1.0) / 4.0
    class_weights = np.clip(class_weights, 0.6, 2.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"class_weights={class_weights.tolist()}", flush=True)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {
                "params": list(model.feature_norm.parameters()) + list(model.scorer.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, updates_per_epoch * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    base_proba = load_base_fold_proba(args, args.fold, valid_ds.ids.astype(str).tolist())

    history = []
    all_sweeps = []
    all_step_metrics = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        started = time.time()
        for step, batch in enumerate(train_loader, start=1):
            batch.pop("ids")
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.fp16)):
                logits = model(**model_inputs(batch))
                loss = F.cross_entropy(logits.float(), labels, weight=class_weights_tensor)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        valid_ids, _, ranker_proba = infer_ranker(model, valid_loader, device)
        if valid_ids.tolist() != valid_ds.ids.astype(str).tolist():
            raise RuntimeError("validation ranker output order mismatch")
        summary, sweep, step_metrics, report = evaluate_blends(
            valid_rows, valid_y, base_proba, ranker_proba, epoch
        )
        summary["train_loss"] = float(np.mean(losses))
        summary["seconds"] = round(time.time() - started, 2)
        history.append(summary)
        all_sweeps.extend(sweep)
        all_step_metrics.extend(step_metrics)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

        pd.DataFrame(history).to_json(
            out_dir / "epoch_summary.jsonl", orient="records", lines=True, force_ascii=False
        )
        pd.DataFrame(all_sweeps).to_csv(out_dir / "blend_sweep.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(all_step_metrics).to_csv(
            out_dir / "step_metrics.csv", index=False, encoding="utf-8-sig"
        )
        (out_dir / f"epoch_{epoch}_class_report.txt").write_text(report, encoding="utf-8")
        np.save(out_dir / f"epoch_{epoch}_valid_ranker_proba.npy", ranker_proba)
        save_model(model, tokenizer, out_dir, epoch, {**run_summary, **summary, **vars(args)})

    best_fixed = max(history, key=lambda item: item["fixed_top1_alpha_0.30"]["macro_f1"])
    best_sweep = max(
        (row for item in history for row in [item["best_sweep"]]), key=lambda row: row["macro_f1"]
    )
    final = {"run": run_summary, "best_fixed": best_fixed, "best_sweep": best_sweep}
    (out_dir / "final_summary.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
