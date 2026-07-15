import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from run_au_augmented_8_2 import AugDataset, augmented_text, make_collate, model_inputs
from run_domain_input_global_oof import FULL_LABELS, TextStructuredGlobalModel, load_jsonl, structured_features


def load_domain_rows(data_dir, id_prefix):
    data_dir = Path(data_dir)
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    rows = [row for row in load_jsonl(data_dir / "train.jsonl") if str(row["id"]).startswith(id_prefix)]
    ids = np.asarray([row["id"] for row in rows], dtype=object)
    y = np.asarray([labels[row["id"]] for row in rows], dtype=object)
    return rows, ids, y


def build_augmented_dataset(rows, y, input_src, augment_copies):
    texts = []
    features = []
    labels = []
    base_ids = []
    variants = []
    for row, label in zip(rows, y):
        feature = structured_features(row)
        for variant in range(augment_copies + 1):
            texts.append(augmented_text(row, variant, input_src))
            features.append(feature)
            labels.append(label)
            base_ids.append(row["id"])
            variants.append(variant)
    return (
        texts,
        np.vstack(features).astype(np.float32),
        np.asarray(labels, dtype=object),
        np.asarray(base_ids, dtype=object),
        np.asarray(variants, dtype=np.int16),
    )


def save_checkpoint(model, tokenizer, out_dir, epoch):
    save_dir = out_dir / f"epoch_{epoch}_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir)
    return save_dir


def train_full(model, train_loader, args, device, tokenizer, out_dir):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    ce = nn.CrossEntropyLoss()
    save_epochs = {int(item.strip()) for item in args.save_epochs.split(",") if item.strip()}
    epoch_rows = []
    saved = {}
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
        row = {
            "epoch": int(epoch),
            "loss": float(np.mean(losses)) if losses else None,
            "seconds": round(time.time() - t0, 2),
        }
        epoch_rows.append(row)
        print(f"  epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
        if epoch in save_epochs:
            saved[str(epoch)] = str(save_checkpoint(model, tokenizer, out_dir, epoch))
            print(f"  saved_epoch={epoch} dir={saved[str(epoch)]}", flush=True)
    if str(args.epochs) not in saved:
        saved[str(args.epochs)] = str(save_checkpoint(model, tokenizer, out_dir, args.epochs))
    pd.DataFrame(epoch_rows).to_csv(out_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
    return {"final_epoch": int(args.epochs), "final_loss": epoch_rows[-1]["loss"], "saved": saved}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-name", default="augmented_full")
    parser.add_argument("--input-src", choices=["sim", "au"], required=True)
    parser.add_argument("--id-prefix", required=True)
    parser.add_argument("--augment-copies", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save-epochs", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=448)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"outputs/models/{args.artifact_name}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    rows, ids, y = load_domain_rows(args.data_dir, args.id_prefix)
    texts, features, y_aug, base_ids, variants = build_augmented_dataset(rows, y, args.input_src, args.augment_copies)
    label_to_idx = {label: idx for idx, label in enumerate(FULL_LABELS)}
    y_idx = np.asarray([label_to_idx[label] for label in y_aug], dtype=np.int64)
    label_counts = pd.Series(y).value_counts().sort_index()
    aug_counts = pd.Series(y_aug).value_counts().sort_index()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"input_src={args.input_src} id_prefix={args.id_prefix} original_rows={len(rows)} "
        f"augmented_rows={len(texts)} max_length={args.max_length} feature_dim={features.shape[1]}",
        flush=True,
    )
    print(label_counts.to_string(), flush=True)

    train_loader = DataLoader(
        AugDataset(texts, features, y_idx),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length),
        num_workers=0,
    )
    model = TextStructuredGlobalModel(args.model_dir, feature_dim=features.shape[1])
    model.to(device)
    result = train_full(model, train_loader, args, device, tokenizer, out_dir)

    pd.DataFrame({"base_id": base_ids, "variant": variants, "label": y_aug}).to_csv(
        out_dir / "train_augmented_index.csv", index=False, encoding="utf-8-sig"
    )
    label_counts.to_csv(out_dir / "label_counts_original.csv", header=["count"])
    aug_counts.to_csv(out_dir / "label_counts_augmented.csv", header=["count"])
    np.save(out_dir / "classes.npy", FULL_LABELS.astype(str))
    np.save(out_dir / "train_sample_ids.npy", ids.astype(str))
    summary = {
        "train": result,
        "input_src": args.input_src,
        "id_prefix": args.id_prefix,
        "original_rows": int(len(rows)),
        "augmented_rows": int(len(texts)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
