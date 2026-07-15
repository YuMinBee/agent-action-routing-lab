import argparse
import csv
import json
import math
import re
import random
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
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_domain_input_global_oof import (
    FULL_LABELS,
    TextStructuredGlobalModel,
    build_au_text,
    build_sim_text,
    bucket_num,
    compact,
    last_result,
    last_result_status,
    load_jsonl,
    previous_actions,
    recent_cmds,
    recent_paths,
    recent_patterns,
    recent_targets,
    result_type,
    session_id,
    state_short_text,
    structured_features,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


LABEL_TO_GROUP = {
    "read_file": "inspect",
    "grep_search": "inspect",
    "glob_pattern": "inspect",
    "list_directory": "inspect",
    "edit_file": "modify",
    "apply_patch": "modify",
    "write_file": "modify",
    "run_tests": "validate",
    "run_bash": "validate",
    "lint_or_typecheck": "validate",
    "plan_task": "reason",
    "ask_user": "reason",
    "respond_only": "reason",
    "web_search": "reason",
}


def primary_lang(row):
    ws = ((row.get("session_meta") or {}).get("workspace") or {})
    mix = ws.get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return str(max(mix.items(), key=lambda item: item[1] if isinstance(item[1], (int, float)) else 0)[0]).lower()


def last_user(row):
    for item in reversed(row.get("history") or []):
        if item.get("role") == "user":
            return compact(item.get("content"), 500)
    return ""


def arg_text(item, limit=160):
    args = item.get("args")
    if not args:
        return "none"
    if isinstance(args, dict):
        return compact(json.dumps(args, ensure_ascii=False, sort_keys=True), limit)
    return compact(args, limit)


def action_events(row, limit=14):
    events = []
    for item in row.get("history") or []:
        name = item.get("name")
        if not name:
            continue
        events.append(
            " ".join(
                [
                    f"act={name}",
                    f"group={LABEL_TO_GROUP.get(name, 'other')}",
                    f"args={arg_text(item, 140)}",
                    f"res_type={result_type({'history': [item]})}",
                    f"res={compact(item.get('result_summary'), 120)}",
                ]
            )
        )
    return events[-limit:]


def action_summary(row):
    actions = previous_actions(row)
    groups = [LABEL_TO_GROUP.get(action, "other") for action in actions]
    counts = Counter(actions)
    group_counts = Counter(groups)
    bigrams = [f"{a}>{b}" for a, b in zip(actions, actions[1:])]
    group_bigrams = [f"{a}>{b}" for a, b in zip(groups, groups[1:])]
    return " ".join(
        [
            f"tail3={'>'.join(actions[-3:]) or 'START'}",
            f"tail6={'>'.join(actions[-6:]) or 'START'}",
            f"group_tail6={'>'.join(groups[-6:]) or 'START'}",
            "bigrams=" + " ".join(bigrams[-8:]),
            "group_bigrams=" + " ".join(group_bigrams[-8:]),
            "counts=" + " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
            "group_counts=" + " ".join(f"{k}:{v}" for k, v in sorted(group_counts.items())),
        ]
    )


def au_augmented_text(row, variant):
    prompt = compact(row.get("current_prompt"), 700)
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = " ".join(str(x) for x in (ws.get("open_files") or [])[-8:])
    base_header = f"query: [SRC=au] [AUG=v{variant}] [STEP={meta.get('turn_index', 'na')}]"
    events = action_events(row)
    if variant == 0:
        return build_au_text(row)
    if variant == 1:
        return "\n".join(
            [
                base_header,
                "[CURRENT_PROMPT]",
                prompt,
                "[HISTORY_ACTION_TRACE]",
                "\n".join(events) if events else "START",
                "[ACTION_SUMMARY]",
                action_summary(row),
                "[STATE]",
                state_short_text(row),
            ]
        )
    if variant == 2:
        return "\n".join(
            [
                base_header,
                "[ACTION_SUMMARY_ONLY]",
                action_summary(row),
                "[LAST_USER]",
                last_user(row),
                "[LAST_RESULT]",
                f"status={last_result_status(last_result(row))} type={result_type(row)} hint={compact(last_result(row), 180)}",
                "[TOOLS]",
                " ".join(
                    [
                        "paths=" + " ".join(recent_paths(row, 10)),
                        "patterns=" + " ".join(recent_patterns(row, 8)),
                        "cmds=" + " ".join(recent_cmds(row, 6)),
                        "targets=" + " ".join(recent_targets(row, 6)),
                    ]
                ),
            ]
        )
    if variant == 3:
        return "\n".join(
            [
                base_header,
                "[COMPACT_AU_ROUTE]",
                action_summary(row),
                "[PROMPT_LIGHT]",
                compact(prompt, 260),
                "[WORKSPACE]",
                f"lang={primary_lang(row)} open={open_files} state={state_short_text(row)}",
            ]
        )
    return "\n".join(
        [
            base_header,
            "[DIRECT]",
            build_au_text(row),
            "[TRACE_REPEAT_LIGHT]",
            action_summary(row),
        ]
    )


def sim_augmented_text(row, variant):
    prompt = compact(row.get("current_prompt"), 900)
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    open_files = " ".join(str(x) for x in (ws.get("open_files") or [])[-8:])
    base_header = f"query: [SRC=sim] [AUG=v{variant}] [STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}]"
    events = action_events(row)
    if variant == 0:
        return build_sim_text(row)
    if variant == 1:
        return "\n".join(
            [
                base_header,
                "[CURRENT_PROMPT]",
                prompt,
                "[HISTORY_ACTION_TRACE]",
                "\n".join(events) if events else "START",
                "[ACTION_SUMMARY]",
                action_summary(row),
                "[STATE]",
                state_short_text(row),
                "[PATHS]",
                " ".join(recent_paths(row, 10)),
            ]
        )
    if variant == 2:
        return "\n".join(
            [
                base_header,
                "[ACTION_SUMMARY_ONLY]",
                action_summary(row),
                "[LAST_USER]",
                last_user(row),
                "[LAST_RESULT]",
                f"status={last_result_status(last_result(row))} type={result_type(row)} hint={compact(last_result(row), 180)}",
                "[TOOLS]",
                " ".join(
                    [
                        "paths=" + " ".join(recent_paths(row, 10)),
                        "patterns=" + " ".join(recent_patterns(row, 8)),
                        "cmds=" + " ".join(recent_cmds(row, 6)),
                        "targets=" + " ".join(recent_targets(row, 6)),
                    ]
                ),
            ]
        )
    if variant == 3:
        return "\n".join(
            [
                base_header,
                "[COMPACT_SIM_ROUTE]",
                action_summary(row),
                "[PROMPT_LIGHT]",
                compact(prompt, 320),
                "[WORKSPACE]",
                f"lang={primary_lang(row)} open={open_files} state={state_short_text(row)}",
            ]
        )
    return "\n".join(
        [
            base_header,
            "[DIRECT]",
            build_sim_text(row),
            "[TRACE_REPEAT_LIGHT]",
            action_summary(row),
        ]
    )


def base_text(row, input_src):
    if input_src == "sim":
        return build_sim_text(row)
    return build_au_text(row)


def augmented_text(row, variant, input_src):
    if input_src == "sim":
        return sim_augmented_text(row, variant)
    return au_augmented_text(row, variant)


class AugDataset(Dataset):
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


def train_model(model, train_loader, train_eval_loader, valid_loader, y_train_eval, y_valid, args, device, save_dir, tokenizer):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    ce = nn.CrossEntropyLoss()
    best = {"valid_macro_f1": -1.0, "valid_accuracy": -1.0, "epoch": -1, "proba": None}
    best_state = None
    rows = []
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
        train_proba = predict(model, train_eval_loader, device)
        valid_proba = predict(model, valid_loader, device)
        train_pred = FULL_LABELS[train_proba.argmax(axis=1)]
        valid_pred = FULL_LABELS[valid_proba.argmax(axis=1)]
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_macro_f1": float(f1_score(y_train_eval, train_pred, labels=FULL_LABELS, average="macro", zero_division=0)),
            "train_accuracy": float(accuracy_score(y_train_eval, train_pred)),
            "valid_macro_f1": float(f1_score(y_valid, valid_pred, labels=FULL_LABELS, average="macro", zero_division=0)),
            "valid_accuracy": float(accuracy_score(y_valid, valid_pred)),
            "seconds": round(time.time() - t0, 2),
        }
        rows.append(row)
        print(
            f"  epoch={epoch} loss={row['loss']:.5f} train_macro_f1={row['train_macro_f1']:.6f} "
            f"train_acc={row['train_accuracy']:.6f} valid_macro_f1={row['valid_macro_f1']:.6f} "
            f"valid_acc={row['valid_accuracy']:.6f} sec={row['seconds']:.1f}",
            flush=True,
        )
        if row["valid_macro_f1"] > best["valid_macro_f1"]:
            best = {
                "valid_macro_f1": row["valid_macro_f1"],
                "valid_accuracy": row["valid_accuracy"],
                "epoch": epoch,
                "proba": valid_proba,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, save_dir / "model.pt")
            tokenizer.save_pretrained(save_dir)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best, rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--fold-dir", default="experiments/oof/hierarchical_story_state_transition_sgd_targetctx_20260702_rerun")
    parser.add_argument("--model-dir", default="models/multilingual-e5-small")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-name", default="au_augmented_8_2")
    parser.add_argument("--input-src", choices=["au", "sim"], default="au")
    parser.add_argument("--id-prefix", default="sess_au")
    parser.add_argument("--valid-fold", type=int, default=4, help="0-based fold id used as the untouched 20 percent valid split")
    parser.add_argument("--augment-copies", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/oof/{args.artifact_name}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data_dir = Path(args.data_dir)
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    all_rows = [
        row
        for row in load_jsonl(data_dir / "train.jsonl")
        if not args.id_prefix or str(row["id"]).startswith(args.id_prefix)
    ]
    ids = np.asarray([row["id"] for row in all_rows], dtype=object)
    sessions = np.asarray([session_id(row["id"]) for row in all_rows], dtype=object)
    y = np.asarray([labels[row["id"]] for row in all_rows], dtype=object)

    fold_dir = Path(args.fold_dir)
    ref_ids = np.load(fold_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    ref_folds = np.load(fold_dir / "fold_ids.npy")
    fold_map = {sample_id: int(fold) for sample_id, fold in zip(ref_ids, ref_folds)}
    folds = np.asarray([fold_map[sample_id] for sample_id in ids], dtype=np.int16)
    valid_mask = folds == args.valid_fold
    train_mask = ~valid_mask
    train_idx = np.where(train_mask)[0]
    valid_idx = np.where(valid_mask)[0]

    base_train_texts = [base_text(all_rows[i], args.input_src) for i in train_idx]
    base_train_features = np.vstack([structured_features(all_rows[i]) for i in train_idx])
    base_train_y = y[train_idx]
    valid_texts = [base_text(all_rows[i], args.input_src) for i in valid_idx]
    valid_features = np.vstack([structured_features(all_rows[i]) for i in valid_idx])
    valid_y = y[valid_idx]

    train_texts = []
    train_features = []
    train_y = []
    aug_base_ids = []
    aug_variant = []
    for i in train_idx:
        feature = structured_features(all_rows[i])
        for variant in range(args.augment_copies + 1):
            train_texts.append(augmented_text(all_rows[i], variant, args.input_src))
            train_features.append(feature)
            train_y.append(labels[ids[i]])
            aug_base_ids.append(ids[i])
            aug_variant.append(variant)
    train_features = np.vstack(train_features)
    train_y = np.asarray(train_y, dtype=object)
    aug_base_ids = np.asarray(aug_base_ids, dtype=object)

    train_sessions = set(sessions[train_idx])
    valid_sessions = set(sessions[valid_idx])
    leak_report = {
        "input_src": args.input_src,
        "id_prefix": args.id_prefix,
        "source_rows": int(len(all_rows)),
        "au_rows": int(len(all_rows)) if args.input_src == "au" else 0,
        "sim_rows": int(len(all_rows)) if args.input_src == "sim" else 0,
        "valid_fold_0_based": int(args.valid_fold),
        "train_original_rows": int(len(train_idx)),
        "valid_original_rows": int(len(valid_idx)),
        "train_augmented_rows": int(len(train_texts)),
        "valid_augmented_rows": 0,
        "train_valid_id_overlap": int(len(set(ids[train_idx]) & set(ids[valid_idx]))),
        "valid_id_used_as_aug_base": int(len(set(ids[valid_idx]) & set(aug_base_ids))),
        "train_valid_session_overlap": int(len(train_sessions & valid_sessions)),
        "direct_label_field_insertions": 0,
    }
    with open(out_dir / "leakage_check.json", "w", encoding="utf-8") as f:
        json.dump(leak_report, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"base_id": aug_base_ids, "variant": aug_variant, "label": train_y}).to_csv(
        out_dir / "train_augmented_index.csv", index=False, encoding="utf-8-sig"
    )

    action_to_idx = {label: idx for idx, label in enumerate(FULL_LABELS)}
    y_train_idx = np.asarray([action_to_idx[label] for label in train_y], dtype=np.int64)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(json.dumps(leak_report, ensure_ascii=False), flush=True)
    print("valid label counts:", Counter(valid_y.tolist()), flush=True)

    collate = make_collate(tokenizer, args.max_length)
    train_loader = DataLoader(
        AugDataset(train_texts, train_features, y_train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=collate,
        num_workers=0,
    )
    train_eval_loader = DataLoader(
        AugDataset(base_train_texts, base_train_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    valid_loader = DataLoader(
        AugDataset(valid_texts, valid_features),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )
    model = TextStructuredGlobalModel(args.model_dir, feature_dim=train_features.shape[1])
    model.to(device)
    best, epoch_rows = train_model(
        model,
        train_loader,
        train_eval_loader,
        valid_loader,
        base_train_y,
        valid_y,
        args,
        device,
        out_dir / "model",
        tokenizer,
    )
    pred = FULL_LABELS[best["proba"].argmax(axis=1)]
    report = classification_report(valid_y, pred, labels=FULL_LABELS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "valid_class_report.csv", encoding="utf-8-sig")
    pd.DataFrame(confusion_matrix(valid_y, pred, labels=FULL_LABELS), index=FULL_LABELS, columns=FULL_LABELS).to_csv(
        out_dir / "valid_confusion_matrix.csv", encoding="utf-8-sig"
    )
    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
    np.save(out_dir / "classes.npy", FULL_LABELS.astype(str))
    np.save(out_dir / "valid_sample_ids.npy", ids[valid_idx].astype(str))
    np.save(out_dir / "valid_proba.npy", best["proba"])
    summary = {
        "best_epoch": int(best["epoch"]),
        "best_valid_macro_f1": float(best["valid_macro_f1"]),
        "best_valid_accuracy": float(best["valid_accuracy"]),
        "leakage_check": leak_report,
        "lowest_class_f1": [
            {"label": label, "f1": float(report[label]["f1-score"])} for label in sorted(FULL_LABELS, key=lambda x: report[x]["f1-score"])
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
