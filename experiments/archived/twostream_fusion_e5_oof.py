import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from multihead_e5_oof import (
    FULL_LABELS,
    GROUPS,
    GROUP_TO_LABELS,
    LABEL_TO_GROUP,
    RESPOND_LABELS,
    compute_loss,
    load_all_rows,
    make_label_arrays,
    to_action_proba,
)


STREAM_MAX_LENGTH = {
    "semantic": 224,
    "context": 320,
}

FUSION_WEIGHTS = {
    "group": (1.2, 1.0),
    "inspect": (0.5, 2.2),
    "modify": (1.2, 1.4),
    "validate": (0.7, 2.2),
    "reason": (2.1, 0.5),
    "global": (1.0, 1.0),
    "respond": (2.3, 0.3),
}

FUSION_WEIGHT_PRESETS = {
    "fixed": FUSION_WEIGHTS,
    "aggro": {
        "group": (1.1, 1.1),
        "inspect": (0.35, 2.4),
        "modify": (1.1, 1.6),
        "validate": (0.55, 2.5),
        "reason": (2.4, 0.35),
        "global": (1.0, 1.0),
        "respond": (2.6, 0.2),
    },
}


def split_sections(text):
    sections = {}
    current = None
    buf = []
    for line in str(text).splitlines():
        if line.startswith("[") and line.endswith("]"):
            if current is not None:
                sections[current] = "\n".join(buf).strip()
            current = line.strip("[]")
            buf = []
        elif current is not None:
            buf.append(line)
    if current is not None:
        sections[current] = "\n".join(buf).strip()
    return sections


def stream_texts_from_common(text):
    sections = split_sections(text)
    semantic = "\n".join(
        [
            "[PROMPT]",
            sections.get("PROMPT", ""),
            "[LAST_USER]",
            sections.get("LAST_USER", ""),
            "[HINTS]",
            sections.get("HINTS", ""),
        ]
    )
    context = "\n".join(
        [
            "[FLOW]",
            sections.get("FLOW", ""),
            "[ARGS]",
            sections.get("ARGS", ""),
            "[RESULT]",
            sections.get("RESULT", ""),
            "[STATE]",
            sections.get("STATE", ""),
        ]
    )
    return semantic, context


class MultiViewDataset(Dataset):
    def __init__(self, streams, y_action=None, y_group=None, y_local=None, y_respond=None):
        self.streams = list(streams)
        self.y_action = y_action
        self.y_group = y_group
        self.y_local = y_local
        self.y_respond = y_respond

    def __len__(self):
        return len(self.streams)

    def __getitem__(self, idx):
        sem, context = self.streams[idx]
        item = {"semantic": sem, "context": context}
        if self.y_action is not None:
            item["action_label"] = int(self.y_action[idx])
            item["group_label"] = int(self.y_group[idx])
            item["local_label"] = int(self.y_local[idx])
            item["respond_label"] = int(self.y_respond[idx])
        return item


def make_multiview_collate(tokenizer):
    def encode(texts, max_length):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def collate(batch):
        out = {}
        for name in ["semantic", "context"]:
            encoded = encode([item[name] for item in batch], STREAM_MAX_LENGTH[name])
            for key, value in encoded.items():
                out[f"{name}_{key}"] = value
        if "action_label" in batch[0]:
            out["action_labels"] = torch.tensor([item["action_label"] for item in batch], dtype=torch.long)
            out["group_labels"] = torch.tensor([item["group_label"] for item in batch], dtype=torch.long)
            out["local_labels"] = torch.tensor([item["local_label"] for item in batch], dtype=torch.long)
            out["respond_labels"] = torch.tensor([item["respond_label"] for item in batch], dtype=torch.long)
        return out

    return collate


class HeadMLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.net(x)


class FusionClassifier(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 768),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(768, out_features),
        )

    def forward(self, x):
        return self.net(x)


def inv_softplus(values):
    values = torch.as_tensor(values, dtype=torch.float32).clamp_min(1e-4)
    return torch.log(torch.expm1(values))


class TwoStreamE5(nn.Module):
    def __init__(
        self,
        model_dir,
        dropout=0.2,
        gradient_checkpointing=False,
        fusion_weights=None,
        learnable_fusion=False,
    ):
        super().__init__()
        fusion_weights = fusion_weights or FUSION_WEIGHTS
        self.learnable_fusion = bool(learnable_fusion)
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        hidden = int(getattr(self.config, "hidden_size"))
        fused = hidden * 2
        if self.learnable_fusion:
            self.fusion_logits = nn.ParameterDict(
                {name: nn.Parameter(inv_softplus(weights)) for name, weights in fusion_weights.items()}
            )
        else:
            for name, weights in fusion_weights.items():
                self.register_buffer(f"{name}_weights", torch.tensor(weights, dtype=torch.float32), persistent=False)
        fused = hidden * 4
        self.group_head = FusionClassifier(fused, 768, len(GROUPS), dropout)
        self.global_head = FusionClassifier(fused, 1536, len(FULL_LABELS), dropout)
        self.respond_head = FusionClassifier(fused, 768, len(RESPOND_LABELS), dropout)
        fusion_hidden = {
            "inspect": 2048,
            "modify": 1536,
            "validate": 2048,
            "reason": 1536,
        }
        self.specialist_heads = nn.ModuleDict(
            {
                group: FusionClassifier(fused, fusion_hidden[group], len(labels), dropout)
                for group, labels in GROUP_TO_LABELS.items()
            }
        )

    def pool(self, output, attention_mask):
        token_embeddings = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def encode_stream(self, prefix, kwargs):
        model_inputs = {
            "input_ids": kwargs[f"{prefix}_input_ids"],
            "attention_mask": kwargs[f"{prefix}_attention_mask"],
        }
        token_type_key = f"{prefix}_token_type_ids"
        if token_type_key in kwargs:
            model_inputs["token_type_ids"] = kwargs[token_type_key]
        output = self.encoder(**model_inputs)
        return F.normalize(self.pool(output, model_inputs["attention_mask"]), p=2, dim=-1)

    def fuse(self, name, z_semantic, z_context):
        if self.learnable_fusion:
            weights = F.softplus(self.fusion_logits[name]).to(z_semantic.dtype)
        else:
            weights = getattr(self, f"{name}_weights").to(z_semantic.dtype)
        z = torch.cat(
            [
                weights[0] * z_semantic,
                weights[1] * z_context,
                z_semantic * z_context,
                torch.abs(z_semantic - z_context),
            ],
            dim=-1,
        )
        return z

    def fusion_weight_values(self):
        out = {}
        names = list(FUSION_WEIGHT_PRESETS["fixed"].keys())
        for name in names:
            if self.learnable_fusion:
                weights = F.softplus(self.fusion_logits[name])
            else:
                weights = getattr(self, f"{name}_weights")
            out[name] = [float(x) for x in weights.detach().cpu().tolist()]
        return out

    def forward(self, **kwargs):
        z_semantic = self.encode_stream("semantic", kwargs)
        z_context = self.encode_stream("context", kwargs)
        z_group = self.fuse("group", z_semantic, z_context)
        z_global = self.fuse("global", z_semantic, z_context)
        z_respond = self.fuse("respond", z_semantic, z_context)
        return {
            "group": self.group_head(z_group),
            "global": self.global_head(z_global),
            "respond": self.respond_head(z_respond),
            "specialist": {
                group: self.specialist_heads[group](self.fuse(group, z_semantic, z_context))
                for group in GROUP_TO_LABELS
            },
        }


def model_input_keys(batch):
    keys = set()
    for prefix in ["semantic", "context"]:
        keys.update({f"{prefix}_input_ids", f"{prefix}_attention_mask", f"{prefix}_token_type_ids"})
    return {k: v for k, v in batch.items() if k in keys}


def predict(model, loader, device, global_blend):
    model.eval()
    action_chunks = []
    group_chunks = []
    global_chunks = []
    respond_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(**model_input_keys(batch))
            action_chunks.append(to_action_proba(outputs, global_blend).detach().cpu().numpy())
            group_chunks.append(torch.softmax(outputs["group"], dim=-1).detach().cpu().numpy())
            global_chunks.append(torch.softmax(outputs["global"], dim=-1).detach().cpu().numpy())
            respond_chunks.append(torch.softmax(outputs["respond"], dim=-1).detach().cpu().numpy())
    return np.vstack(action_chunks), np.vstack(group_chunks), np.vstack(global_chunks), np.vstack(respond_chunks)


def train_fold(model, train_loader, valid_loader, y_valid, args, device, save_dir=None, tokenizer=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    best = {
        "macro_f1": -1.0,
        "epoch": -1,
        "proba": None,
        "group_proba": None,
        "global_proba": None,
        "respond_proba": None,
    }
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.fp16)):
                outputs = model(**model_input_keys(batch))
                loss = compute_loss(outputs, batch, args) / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if batch_idx % args.grad_accum == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        valid_proba, valid_group_proba, valid_global_proba, valid_respond_proba = predict(
            model, valid_loader, device, args.valid_global_blend
        )
        valid_pred = FULL_LABELS[valid_proba.argmax(axis=1)]
        macro_f1 = float(f1_score(y_valid, valid_pred, labels=FULL_LABELS, average="macro", zero_division=0))
        acc = float(accuracy_score(y_valid, valid_pred))
        print(
            f"  epoch={epoch} loss={np.mean(losses):.5f} valid_macro_f1={macro_f1:.6f} "
            f"acc={acc:.6f} sec={time.time() - t0:.1f}",
            flush=True,
        )
        if macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": macro_f1,
                "accuracy": acc,
                "epoch": epoch,
                "proba": valid_proba,
                "group_proba": valid_group_proba,
                "global_proba": valid_global_proba,
                "respond_proba": valid_respond_proba,
            }
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_dir / "model.pt")
                if tokenizer is not None:
                    tokenizer.save_pretrained(save_dir)
    if best_state is not None:
        model.load_state_dict(best_state)
    if hasattr(model, "fusion_weight_values"):
        best["fusion_weights"] = model.fusion_weight_values()
    return best


def write_metrics(out_dir, name, y_true, pred, labels):
    report = classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / f"{name}_class_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y_true, pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / f"{name}_confusion_matrix.csv", encoding="utf-8-sig")
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--fold-dir", default="experiments/oof/hierarchical_story_state_transition_sgd_targetctx_20260702_rerun")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--artifact-name", default="twostream_e5_mlp")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--group-loss-weight", type=float, default=0.3)
    parser.add_argument("--spec-loss-weight", type=float, default=1.0)
    parser.add_argument("--respond-loss-weight", type=float, default=0.3)
    parser.add_argument("--global-loss-weight", type=float, default=0.5)
    parser.add_argument("--valid-global-blend", type=float, default=0.0)
    parser.add_argument("--folds", default="3,4,5", help="Comma-separated 1-based fold numbers")
    parser.add_argument("--fusion-preset", default="aggro", choices=sorted(FUSION_WEIGHT_PRESETS))
    parser.add_argument("--learnable-fusion", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-fast", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--save-fold-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/oof/{args.artifact_name}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        config = vars(args).copy()
        config["stream_max_length"] = STREAM_MAX_LENGTH
        config["fusion_preset"] = args.fusion_preset
        config["fusion_weights"] = FUSION_WEIGHT_PRESETS[args.fusion_preset]
        config["learnable_fusion"] = bool(args.learnable_fusion)
        json.dump(config, f, ensure_ascii=False, indent=2)

    rows, ids, sessions, y_action, y_group, common_texts = load_all_rows(args.data_dir)
    streams = [stream_texts_from_common(text) for text in common_texts]
    fold_dir = Path(args.fold_dir)
    fold_ids = np.load(fold_dir / "fold_ids.npy")
    fold_ids_ref = np.load(fold_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    if not np.array_equal(ids.astype(str), fold_ids_ref):
        raise ValueError("sample_ids do not align with fold_dir")
    y_action_idx, y_group_idx, y_local_idx, y_respond_idx = make_label_arrays(y_action, y_group)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"rows={len(streams)} folds={dict(zip(*np.unique(fold_ids, return_counts=True)))} "
        f"stream_max_length={STREAM_MAX_LENGTH}",
        flush=True,
    )

    folds = sorted(np.unique(fold_ids))
    requested = {int(item.strip()) - 1 for item in args.folds.split(",") if item.strip()}
    folds = [fold for fold in folds if int(fold) in requested]
    if not folds:
        raise ValueError("No folds selected")

    oof_proba = np.zeros((len(streams), len(FULL_LABELS)), dtype=np.float32)
    oof_group = np.zeros((len(streams), len(GROUPS)), dtype=np.float32)
    oof_global = np.zeros((len(streams), len(FULL_LABELS)), dtype=np.float32)
    oof_respond = np.zeros((len(streams), len(RESPOND_LABELS)), dtype=np.float32)
    filled = np.zeros(len(streams), dtype=bool)
    fold_rows = []
    collate = make_multiview_collate(tokenizer)
    for fold in folds:
        t0 = time.time()
        valid_idx = np.where(fold_ids == fold)[0]
        train_idx = np.where(fold_ids != fold)[0]
        print(f"fold={int(fold)+1} train={len(train_idx)} valid={len(valid_idx)}", flush=True)
        model = TwoStreamE5(
            args.model_dir,
            gradient_checkpointing=args.gradient_checkpointing,
            fusion_weights=FUSION_WEIGHT_PRESETS[args.fusion_preset],
            learnable_fusion=args.learnable_fusion,
        )
        model.to(device)
        train_loader = DataLoader(
            MultiViewDataset(
                [streams[i] for i in train_idx],
                y_action_idx[train_idx],
                y_group_idx[train_idx],
                y_local_idx[train_idx],
                y_respond_idx[train_idx],
            ),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        valid_loader = DataLoader(
            MultiViewDataset(
                [streams[i] for i in valid_idx],
                y_action_idx[valid_idx],
                y_group_idx[valid_idx],
                y_local_idx[valid_idx],
                y_respond_idx[valid_idx],
            ),
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        save_dir = out_dir / f"fold_{int(fold)+1}_model" if args.save_fold_models else None
        best = train_fold(model, train_loader, valid_loader, y_action[valid_idx], args, device, save_dir, tokenizer)
        if "fusion_weights" in best:
            with open(out_dir / f"fold_{int(fold)+1}_fusion_weights.json", "w", encoding="utf-8") as f:
                json.dump(best["fusion_weights"], f, ensure_ascii=False, indent=2)
        oof_proba[valid_idx] = best["proba"]
        oof_group[valid_idx] = best["group_proba"]
        oof_global[valid_idx] = best["global_proba"]
        oof_respond[valid_idx] = best["respond_proba"]
        filled[valid_idx] = True
        fold_rows.append(
            {
                "fold": int(fold) + 1,
                "best_epoch": int(best["epoch"]),
                "macro_f1": float(best["macro_f1"]),
                "accuracy": float(best["accuracy"]),
                "fusion_weights": json.dumps(best.get("fusion_weights", {}), ensure_ascii=False),
                "seconds": round(time.time() - t0, 2),
            }
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    covered = filled
    np.save(out_dir / "classes.npy", FULL_LABELS.astype(str))
    np.save(out_dir / "group_classes.npy", GROUPS.astype(str))
    np.save(out_dir / "sample_ids.npy", ids.astype(str))
    np.save(out_dir / "session_ids.npy", sessions.astype(str))
    np.save(out_dir / "y_true.npy", y_action.astype(str))
    np.save(out_dir / "y_group.npy", y_group.astype(str))
    np.save(out_dir / "fold_ids.npy", fold_ids)
    np.save(out_dir / "filled.npy", filled)
    np.save(out_dir / f"oof_{args.artifact_name}.npy", oof_proba)
    np.save(out_dir / f"oof_{args.artifact_name}_group.npy", oof_group)
    np.save(out_dir / f"oof_{args.artifact_name}_global.npy", oof_global)
    np.save(out_dir / f"oof_{args.artifact_name}_respond.npy", oof_respond)
    pd.DataFrame(fold_rows).to_csv(out_dir / f"{args.artifact_name}_folds.csv", index=False)

    y_eval = y_action[covered]
    pred = FULL_LABELS[oof_proba[covered].argmax(axis=1)]
    metrics = write_metrics(out_dir, args.artifact_name, y_eval, pred, FULL_LABELS)
    global_pred = FULL_LABELS[oof_global[covered].argmax(axis=1)]
    global_metrics = write_metrics(out_dir, f"{args.artifact_name}_global", y_eval, global_pred, FULL_LABELS)
    blend_rows = []
    for blend in np.linspace(0.0, 0.8, 17):
        blended = (1.0 - blend) * oof_proba[covered] + blend * oof_global[covered]
        blend_pred = FULL_LABELS[blended.argmax(axis=1)]
        blend_rows.append(
            {
                "global_blend": float(blend),
                "macro_f1": float(f1_score(y_eval, blend_pred, labels=FULL_LABELS, average="macro", zero_division=0)),
                "accuracy": float(accuracy_score(y_eval, blend_pred)),
            }
        )
    blend_df = pd.DataFrame(blend_rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    blend_df.to_csv(out_dir / f"{args.artifact_name}_blend_search.csv", index=False)
    summary = {
        "multihead": metrics,
        "global": global_metrics,
        "best_blend": blend_df.iloc[0].to_dict(),
        "folds": fold_rows,
        "covered_rows": int(covered.sum()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"saved: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
