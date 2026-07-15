from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
EXACT_CODE = REPO_ROOT / "reference" / "training"
B_ROOT = REPO_ROOT / "artifacts" / "oof" / "sim_b"
A_ROOT = REPO_ROOT / "artifacts" / "oof" / "sim_a"
OUT_ROOT = REPO_ROOT / "outputs" / "analysis" / "selective_multiview_tta"
MODEL_CONFIG_DIR = (
    REPO_ROOT / "models" / "sim_b"
)

sys.path.insert(0, str(EXACT_CODE))
from run_au_augmented_8_2 import (  # noqa: E402
    AugDataset,
    augmented_text,
    make_collate,
    model_inputs,
)
from run_domain_input_global_oof import (  # noqa: E402
    FULL_LABELS,
    structured_features,
)


class InferenceTextStructuredGlobalModel(nn.Module):
    def __init__(self, model_dir, feature_dim=93, dropout=0.2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_config(self.config)
        hidden = int(self.config.hidden_size)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2 + feature_dim),
            nn.Linear(hidden * 2 + feature_dim, 768),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(256, len(FULL_LABELS)),
        )

    @staticmethod
    def pool_mean(output, attention_mask):
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        return (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask, features, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        cls = output.last_hidden_state[:, 0]
        mean = self.pool_mean(output, attention_mask)
        features = self.feature_norm(features.to(mean.dtype))
        return self.classifier(torch.cat([cls, mean, features], dim=-1))


def load_rows() -> dict[str, dict]:
    path = REPO_ROOT / "data" / "raw" / "train.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["id"])] = row
    return rows


def load_labels() -> dict[str, str]:
    path = REPO_ROOT / "data" / "raw" / "train_labels.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {str(row["id"]): str(row["action"]) for row in csv.DictReader(handle)}


def predict_variant(model, tokenizer, rows, features, variant, device, batch_size):
    texts = [augmented_text(row, variant, "sim") for row in rows]
    loader = DataLoader(
        AugDataset(texts, features),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate(tokenizer, 512),
        num_workers=0,
    )
    chunks = []
    started = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**model_inputs(batch))
            chunks.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
            if batch_idx % 100 == 0 or batch_idx == len(loader):
                print(
                    f"variant={variant} batch={batch_idx}/{len(loader)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
    return np.vstack(chunks).astype(np.float32)


def align_a(ids: np.ndarray, fold: int) -> np.ndarray:
    a_ids = np.load(A_ROOT / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
    a_proba = np.load(A_ROOT / f"oof_fold{fold}_proba.npy").astype(np.float32)
    lookup = {sample_id: idx for idx, sample_id in enumerate(a_ids)}
    missing = [sample_id for sample_id in ids if sample_id not in lookup]
    if missing:
        raise ValueError(f"A OOF IDs missing in fold {fold}: {missing[:3]}")
    return np.stack([a_proba[lookup[sample_id]] for sample_id in ids])


def margin(proba: np.ndarray) -> np.ndarray:
    top2 = np.partition(proba, -2, axis=1)[:, -2:]
    return top2.max(axis=1) - top2.min(axis=1)


def evaluate(fold_payloads):
    y = np.concatenate([item["y"] for item in fold_payloads])
    pa = np.concatenate([item["pa"] for item in fold_payloads])
    p0 = np.concatenate([item["p0"] for item in fold_payloads])
    p1 = np.concatenate([item["p1"] for item in fold_payloads])
    p3 = np.concatenate([item["p3"] for item in fold_payloads])
    folds = np.concatenate(
        [np.full(len(item["y"]), item["fold"], dtype=np.int16) for item in fold_payloads]
    )

    base = 0.60 * pa + 0.40 * p0
    base_pred = base.argmax(axis=1)
    base_f1 = f1_score(y, base_pred, labels=np.arange(len(FULL_LABELS)), average="macro")
    base_acc = accuracy_score(y, base_pred)
    base_margin = margin(base)

    thresholds = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 1.01]
    weights = []
    for w1 in np.arange(0.05, 0.41, 0.05):
        for w3 in np.arange(0.05, 0.31, 0.05):
            w0 = 1.0 - float(w1) - float(w3)
            if w0 >= 0.30:
                weights.append((w0, float(w1), float(w3)))

    results = []
    for threshold in thresholds:
        gate = base_margin < threshold
        for w0, w1, w3 in weights:
            b_tta = w0 * p0 + w1 * p1 + w3 * p3
            final = base.copy()
            final[gate] = 0.60 * pa[gate] + 0.40 * b_tta[gate]
            pred = final.argmax(axis=1)
            results.append(
                {
                    "threshold": threshold,
                    "gate_rows": int(gate.sum()),
                    "gate_rate": float(gate.mean()),
                    "w0": w0,
                    "w1": w1,
                    "w3": w3,
                    "macro_f1": float(
                        f1_score(
                            y,
                            pred,
                            labels=np.arange(len(FULL_LABELS)),
                            average="macro",
                            zero_division=0,
                        )
                    ),
                    "accuracy": float(accuracy_score(y, pred)),
                }
            )

    results.sort(key=lambda row: row["macro_f1"], reverse=True)
    best = results[0]
    best["delta_macro_f1"] = best["macro_f1"] - base_f1

    per_fold = []
    for fold in sorted(set(folds.tolist())):
        mask = folds == fold
        per_fold.append(
            {
                "fold": int(fold),
                "base_macro_f1": float(
                    f1_score(
                        y[mask],
                        base_pred[mask],
                        labels=np.arange(len(FULL_LABELS)),
                        average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    summary = {
        "folds": sorted(set(folds.tolist())),
        "rows": int(len(y)),
        "base_macro_f1": float(base_f1),
        "base_accuracy": float(base_acc),
        "best": best,
        "top10": results[:10],
        "per_fold_base": per_fold,
    }
    return summary, results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()

    folds = [int(item) for item in args.folds.split(",") if item.strip()]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    row_map = load_rows()
    label_map = load_labels()
    label_to_idx = {str(label): idx for idx, label in enumerate(FULL_LABELS)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} folds={folds} batch_size={args.batch_size}", flush=True)

    payloads = []
    for fold in folds:
        fold_dir = B_ROOT / (
            f"sim0783_header3tok_mined_aligned_aufold{fold + 1}_e2_seed42_20260708_171135"
        )
        ids = np.load(fold_dir / "valid_sample_ids.npy", allow_pickle=True).astype(str)
        p0 = np.load(fold_dir / "valid_proba.npy").astype(np.float32)
        y = np.asarray([label_to_idx[label_map[sample_id]] for sample_id in ids], dtype=np.int64)
        rows = [row_map[sample_id] for sample_id in ids]
        features = np.vstack([structured_features(row) for row in rows]).astype(np.float32)
        pa = align_a(ids, fold)

        p1_path = OUT_ROOT / f"fold{fold}_variant1_proba.npy"
        p3_path = OUT_ROOT / f"fold{fold}_variant3_proba.npy"
        if args.reuse and p1_path.exists() and p3_path.exists():
            p1 = np.load(p1_path)
            p3 = np.load(p3_path)
        else:
            model_dir = fold_dir / "model"
            tokenizer = AutoTokenizer.from_pretrained(
                model_dir, local_files_only=True, use_fast=False
            )
            model = InferenceTextStructuredGlobalModel(
                MODEL_CONFIG_DIR, feature_dim=features.shape[1]
            )
            if device.type == "cuda":
                model.half()
            state = torch.load(model_dir / "model.pt", map_location="cpu")
            model.load_state_dict(state)
            model.to(device)
            print(f"fold={fold} rows={len(rows)} model={model_dir}", flush=True)
            p1 = predict_variant(
                model, tokenizer, rows, features, 1, device, args.batch_size
            )
            np.save(p1_path, p1)
            p3 = predict_variant(
                model, tokenizer, rows, features, 3, device, args.batch_size
            )
            np.save(p3_path, p3)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        payloads.append(
            {"fold": fold, "y": y, "pa": pa, "p0": p0, "p1": p1, "p3": p3}
        )

    summary, results = evaluate(payloads)
    suffix = "_".join(str(fold) for fold in folds)
    (OUT_ROOT / f"summary_folds_{suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT_ROOT / f"sweep_folds_{suffix}.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
