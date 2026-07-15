import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_au_augmented_8_2 import AugDataset, make_collate, model_inputs
from run_domain_input_global_oof import FULL_LABELS, TextStructuredGlobalModel
from train_augmented_full import build_augmented_dataset


def parse_variant_weight_map(value):
    if not value:
        return {}
    weights = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"bad --variant-weights item: {item}; expected VARIANT:WEIGHT")
        key, weight = item.split(":", 1)
        weights[int(key.strip())] = float(weight.strip())
    return weights


def weights_for_variants(variants, variant_weights):
    weight_map = parse_variant_weight_map(variant_weights)
    weights = np.asarray([float(weight_map.get(int(v), 1.0)) for v in variants], dtype=np.float32)
    mean = float(weights.mean()) if len(weights) else 1.0
    if mean > 0:
        weights = weights / mean
    return weights


class WeightedAugDataset(Dataset):
    def __init__(self, texts, features, y, sample_weights):
        self.texts = list(texts)
        self.features = np.asarray(features, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {
            "text": self.texts[idx],
            "features": self.features[idx],
            "label": int(self.y[idx]),
            "sample_weight": float(self.sample_weights[idx]),
        }


def make_weighted_collate(tokenizer, max_length):
    base_collate = make_collate(tokenizer, max_length)

    def collate(batch):
        encoded = base_collate(batch)
        encoded["sample_weights"] = torch.tensor(
            [item["sample_weight"] for item in batch],
            dtype=torch.float32,
        )
        return encoded

    return collate


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def root_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_cols(df):
    id_col = "id" if "id" in df.columns else "sample_id"
    candidates = ["label", "action", "target", "next_action"]
    label_col = None
    for c in candidates:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        non_id = [c for c in df.columns if c != id_col]
        if len(non_id) != 1:
            raise ValueError(f"Cannot infer label column from columns={list(df.columns)}")
        label_col = non_id[0]
    return id_col, label_col


def load_rows_and_labels(data_dir, id_prefix):
    data_dir = Path(data_dir)
    rows_all = load_jsonl(data_dir / "train.jsonl")
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    id_col, label_col = infer_cols(labels_df)
    label_map = dict(zip(labels_df[id_col].astype(str), labels_df[label_col].astype(str)))

    rows, y = [], []
    for r in rows_all:
        sid = str(r.get("id"))
        if not sid.startswith(id_prefix):
            continue
        if sid not in label_map:
            continue
        rows.append(r)
        y.append(label_map[sid])
    return rows, np.asarray(y, dtype=object)


def load_mined_rows(jsonl_path, labels_path, id_prefix):
    rows_all = load_jsonl(jsonl_path)
    labels_df = pd.read_csv(labels_path)
    id_col, label_col = infer_cols(labels_df)
    label_map = dict(zip(labels_df[id_col].astype(str), labels_df[label_col].astype(str)))

    rows, y = [], []
    for r in rows_all:
        sid = str(r.get("id"))
        if not sid.startswith(id_prefix):
            continue
        if sid not in label_map:
            continue
        rows.append(r)
        y.append(label_map[sid])
    return rows, np.asarray(y, dtype=object)


def make_groupkfold(n_splits, seed):
    try:
        return GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    except TypeError:
        print("[WARN] GroupKFold shuffle/random_state unsupported. Using non-shuffled GroupKFold.", flush=True)
        return GroupKFold(n_splits=n_splits)


def make_fold_indices(original_rows, original_y, roots, args):
    if args.fold_dir:
        fold_dir = Path(args.fold_dir)
        ref_ids = np.load(fold_dir / "sample_ids.npy", allow_pickle=True).astype(str)
        ref_folds = np.load(fold_dir / "fold_ids.npy").astype(np.int64)
        fold_map = {sample_id: int(fold) for sample_id, fold in zip(ref_ids, ref_folds)}
        ids = np.asarray([str(r["id"]) for r in original_rows], dtype=object)
        missing = [sample_id for sample_id in ids if sample_id not in fold_map]
        if missing:
            raise ValueError(
                f"{len(missing)} original rows are missing from fold_dir={fold_dir}. "
                f"examples={missing[:5]}"
            )

        folds = np.asarray([fold_map[sample_id] for sample_id in ids], dtype=np.int64)
        root_to_folds = defaultdict(set)
        for root, fold in zip(roots, folds):
            root_to_folds[str(root)].add(int(fold))
        split_roots = {root: sorted(vals) for root, vals in root_to_folds.items() if len(vals) > 1}
        if split_roots:
            examples = list(split_roots.items())[:5]
            raise ValueError(f"root/session appears in multiple folds: examples={examples}")

        fold_indices = [np.where(folds == fold_id)[0] for fold_id in range(args.folds)]
        fold_of_root = {root: next(iter(vals)) for root, vals in root_to_folds.items()}
        fold_source = {
            "mode": "fold_dir",
            "fold_dir": str(fold_dir),
            "fold_counts": {str(i): int(len(idx)) for i, idx in enumerate(fold_indices)},
        }
        return fold_indices, fold_of_root, fold_source

    gkf = make_groupkfold(args.folds, args.seed)
    fold_indices = []
    fold_of_root = {}
    dummy_x = np.zeros(len(original_rows))
    for fold_id, (_, valid_idx) in enumerate(gkf.split(dummy_x, original_y, groups=roots)):
        valid_idx = np.asarray(valid_idx)
        fold_indices.append(valid_idx)
        for idx in valid_idx:
            fold_of_root[str(roots[idx])] = fold_id
    fold_source = {
        "mode": "groupkfold",
        "fold_counts": {str(i): int(len(idx)) for i, idx in enumerate(fold_indices)},
    }
    return fold_indices, fold_of_root, fold_source


@torch.no_grad()
def evaluate(model, loader, device, classes):
    model.eval()
    probs = []
    trues = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**model_inputs(batch))
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs.append(p)
        trues.extend(batch["labels"].cpu().numpy().tolist())

    P = np.concatenate(probs, axis=0)
    pred = P.argmax(axis=1)

    f1 = f1_score(
        [classes[i] for i in trues],
        [classes[i] for i in pred],
        average="macro",
    )
    return float(f1), P, np.asarray(trues, dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--input-src", choices=["sim", "au"], required=True)
    p.add_argument("--id-prefix", required=True)
    p.add_argument("--mined-enriched-jsonl", required=True)
    p.add_argument("--mined-enriched-labels", required=True)
    p.add_argument("--fold-dir", default="", help="Use existing sample_ids.npy/fold_ids.npy folds instead of GroupKFold")
    p.add_argument("--augment-copies", type=int, default=4)
    p.add_argument("--variant-weights", default="")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold-offset", type=int, default=0)
    p.add_argument("--max-folds", type=int, default=5)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--use-fast", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save-best-state", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    seed_everything(args.seed)

    original_rows, original_y = load_rows_and_labels(args.data_dir, args.id_prefix)
    mined_rows, mined_y = load_mined_rows(
        args.mined_enriched_jsonl,
        args.mined_enriched_labels,
        args.id_prefix,
    )

    roots = np.asarray([root_id(r["id"]) for r in original_rows], dtype=object)
    original_root_set = set(roots.tolist())

    fold_indices, fold_of_root, fold_source = make_fold_indices(
        original_rows,
        original_y,
        roots,
        args,
    )

    # mined는 같은 root가 validation fold에 있으면 train에서 제외
    mined_folds = np.asarray(
        [fold_of_root.get(root_id(r["id"]), -1) for r in mined_rows],
        dtype=np.int64,
    )

    mined_index = pd.DataFrame(
        {
            "id": [r["id"] for r in mined_rows],
            "root_id": [root_id(r["id"]) for r in mined_rows],
            "label": mined_y,
            "fold": mined_folds,
            "root_in_original": [root_id(r["id"]) in original_root_set for r in mined_rows],
        }
    )
    mined_index.to_csv(out_dir / "mined_index.csv", index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "input_src": args.input_src,
                "id_prefix": args.id_prefix,
                "original_rows": len(original_rows),
                "mined_rows": len(mined_rows),
                "valid_mined_rows": int((mined_folds >= 0).sum()),
                "folds": args.folds,
                "fold_source": fold_source,
                "epochs": args.epochs,
                "augment_copies": args.augment_copies,
                "max_length": args.max_length,
                "seed": args.seed,
                "mined_fold_counts": {
                    str(k): int(v)
                    for k, v in pd.Series(mined_folds).value_counts().sort_index().items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    if args.dry_run:
        dry = {
            "dry_run": True,
            "input_src": args.input_src,
            "id_prefix": args.id_prefix,
            "original_rows": int(len(original_rows)),
            "mined_rows": int(len(mined_rows)),
            "folds": int(args.folds),
            "fold_source": fold_source,
            "mined_fold_counts": {
                str(k): int(v)
                for k, v in pd.Series(mined_folds).value_counts().sort_index().items()
            },
        }
        with open(out_dir / "dry_run_summary.json", "w", encoding="utf-8") as f:
            json.dump(dry, f, ensure_ascii=False, indent=2)
        print(json.dumps(dry, ensure_ascii=False, indent=2), flush=True)
        return

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        use_fast=args.use_fast,
    )

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)

    classes = list(FULL_LABELS)
    label_to_idx = {label: i for i, label in enumerate(classes)}

    best_rows = []
    start = args.fold_offset
    end = min(args.folds, args.fold_offset + args.max_folds)

    for fold_id in range(start, end):
        seed_everything(args.seed + fold_id)

        valid_idx = set(fold_indices[fold_id].tolist())
        valid_roots = set(str(roots[i]) for i in valid_idx)

        train_rows, train_y = [], []
        valid_rows, valid_y = [], []
        train_roots = set()

        for i, (r, y) in enumerate(zip(original_rows, original_y)):
            if i in valid_idx:
                valid_rows.append(r)
                valid_y.append(y)
            else:
                train_rows.append(r)
                train_y.append(y)
                train_roots.add(str(roots[i]))

        overlap = train_roots & valid_roots
        if overlap:
            raise ValueError(f"train/valid root overlap in fold {fold_id}: examples={list(overlap)[:5]}")

        mined_train_count = 0
        mined_train_roots = set()
        for r, y, mf in zip(mined_rows, mined_y, mined_folds):
            if mf >= 0 and mf != fold_id:
                train_rows.append(r)
                train_y.append(y)
                mined_train_count += 1
                mined_train_roots.add(root_id(r["id"]))

        mined_overlap = mined_train_roots & valid_roots
        if mined_overlap:
            raise ValueError(
                f"mined train root overlaps valid roots in fold {fold_id}: "
                f"examples={list(mined_overlap)[:5]}"
            )

        train_y = np.asarray(train_y, dtype=object)
        valid_y = np.asarray(valid_y, dtype=object)

        train_texts, train_features, train_y_aug, _, train_variants = build_augmented_dataset(
            train_rows,
            train_y,
            args.input_src,
            args.augment_copies,
        )
        train_sample_weights = weights_for_variants(train_variants, args.variant_weights)

        valid_texts, valid_features, valid_y_aug, valid_base_ids, _ = build_augmented_dataset(
            valid_rows,
            valid_y,
            args.input_src,
            0,
        )

        y_train_idx = np.asarray([label_to_idx[y] for y in train_y_aug], dtype=np.int64)
        y_valid_idx = np.asarray([label_to_idx[y] for y in valid_y_aug], dtype=np.int64)

        print(
            f"[fold {fold_id}] train_original={len(train_rows) - mined_train_count} "
            f"train_mined={mined_train_count} train_base={len(train_rows)} "
            f"train_aug={len(train_texts)} valid_original={len(valid_rows)}",
            flush=True,
        )
        if args.variant_weights:
            print(
                "[fold {fold}] variant_weight_mean=".format(fold=fold_id)
                + " ".join(
                    f"v{k}:{float(train_sample_weights[train_variants == k].mean()):.4f}"
                    for k in sorted(set(int(v) for v in train_variants))
                ),
                flush=True,
            )

        train_loader = DataLoader(
            WeightedAugDataset(train_texts, train_features, y_train_idx, train_sample_weights),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + fold_id),
            collate_fn=make_weighted_collate(tokenizer, args.max_length),
            num_workers=0,
        )

        valid_loader = DataLoader(
            AugDataset(valid_texts, valid_features, y_valid_idx),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=make_collate(tokenizer, args.max_length),
            num_workers=0,
        )

        model = TextStructuredGlobalModel(
            args.model_dir,
            feature_dim=train_features.shape[1],
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
        warmup_steps = int(total_steps * args.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda" and args.fp16),
        )

        ce = nn.CrossEntropyLoss(reduction="none" if args.variant_weights else "mean")

        metrics = []
        best = {
            "fold": fold_id,
            "best_epoch": None,
            "valid_macro_f1": -1.0,
        }

        for epoch in range(1, args.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            t0 = time.time()

            for batch_idx, batch in enumerate(train_loader, start=1):
                batch = {k: v.to(device) for k, v in batch.items()}

                with torch.amp.autocast(
                    "cuda",
                    enabled=(device.type == "cuda" and args.fp16),
                ):
                    logits = model(**model_inputs(batch))
                    if args.variant_weights:
                        per_sample_loss = ce(logits, batch["labels"])
                        loss = (per_sample_loss * batch["sample_weights"]).mean() / args.grad_accum
                    else:
                        loss = ce(logits, batch["labels"]) / args.grad_accum

                scaler.scale(loss).backward()
                losses.append(float(loss.detach().cpu()) * args.grad_accum)

                if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            val_f1, val_proba, val_true = evaluate(model, valid_loader, device, classes)

            row = {
                "fold": fold_id,
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else None,
                "valid_macro_f1": val_f1,
                "seconds": round(time.time() - t0, 2),
            }
            metrics.append(row)

            print(
                f"[fold {fold_id}] epoch={epoch} loss={row['loss']:.5f} "
                f"valid_macro_f1={val_f1:.6f} sec={row['seconds']:.1f}",
                flush=True,
            )

            if val_f1 > best["valid_macro_f1"]:
                best = {
                    "fold": fold_id,
                    "best_epoch": epoch,
                    "valid_macro_f1": val_f1,
                }

                np.save(out_dir / f"oof_fold{fold_id}_proba.npy", val_proba)
                np.save(out_dir / f"oof_fold{fold_id}_true.npy", val_true)
                np.save(
                    out_dir / f"oof_fold{fold_id}_ids.npy",
                    np.asarray(valid_base_ids, dtype=object),
                )

                if args.save_best_state:
                    torch.save(
                        model.state_dict(),
                        out_dir / f"fold{fold_id}_best_state.pt",
                    )

        pd.DataFrame(metrics).to_csv(
            out_dir / f"fold_{fold_id}_metrics.csv",
            index=False,
            encoding="utf-8-sig",
        )
        best_rows.append(best)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(out_dir / "cv_best.csv", index=False, encoding="utf-8-sig")

    all_metrics = []
    for p in sorted(out_dir.glob("fold_*_metrics.csv")):
        all_metrics.append(pd.read_csv(p))

    all_df = pd.concat(all_metrics, ignore_index=True)

    epoch_summary = (
        all_df.groupby("epoch")["valid_macro_f1"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .sort_values("epoch")
    )

    epoch_summary.to_csv(
        out_dir / "epoch_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "best_epoch_by_mean": int(
            epoch_summary.sort_values("mean", ascending=False).iloc[0]["epoch"]
        ),
        "best_mean_macro_f1": float(epoch_summary["mean"].max()),
        "best_by_fold_mean": float(best_df["valid_macro_f1"].mean()),
        "best_by_fold_std": float(best_df["valid_macro_f1"].std(ddof=0)),
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== epoch_summary ===", flush=True)
    print(epoch_summary.to_string(index=False), flush=True)
    print("=== cv_best ===", flush=True)
    print(best_df.to_string(index=False), flush=True)
    print("=== summary ===", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
