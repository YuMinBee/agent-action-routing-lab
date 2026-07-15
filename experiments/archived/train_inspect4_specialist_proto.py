import argparse
import csv
import json
import math
import os
import random
import shutil
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


INSPECT_LABELS = np.array(["read_file", "grep_search", "glob_pattern", "list_directory"], dtype=object)


def seed_everything(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_jsonl(path: Path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def compact(value, limit=500):
    if value is None:
        return ""
    import re

    text = str(value).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def load_original_inspect_rows(data_dir: Path, id_prefix: str):
    with open(data_dir / "train_labels.csv", encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    rows = []
    y = []
    for row in load_jsonl(data_dir / "train.jsonl"):
        rid = str(row.get("id"))
        label = labels.get(rid)
        if rid.startswith(id_prefix) and label in set(INSPECT_LABELS):
            rows.append(row)
            y.append(label)
    return rows, np.asarray(y, dtype=object)


def load_extra_inspect_rows(jsonl_path: Path, labels_path: Path, id_prefix: str):
    if not jsonl_path or not labels_path or not jsonl_path.exists() or not labels_path.exists():
        return [], np.asarray([], dtype=object)
    with open(labels_path, encoding="utf-8-sig", newline="") as f:
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    rows = []
    y = []
    for row in load_jsonl(jsonl_path):
        rid = str(row.get("id"))
        label = labels.get(rid)
        if rid.startswith(id_prefix) and label in set(INSPECT_LABELS):
            rows.append(row)
            y.append(label)
    return rows, np.asarray(y, dtype=object)


def assistant_actions(row):
    return [item for item in row.get("history") or [] if item.get("name")]


def previous_actions(row):
    return [str(item.get("name")) for item in assistant_actions(row)]


def last_result(row):
    for item in reversed(row.get("history") or []):
        if item.get("result_summary"):
            return compact(item.get("result_summary"), 500)
    return ""


def last_user(row):
    for item in reversed(row.get("history") or []):
        if item.get("role") == "user":
            return compact(item.get("content"), 500)
    return ""


def last_action_args(row, limit=260):
    for item in reversed(row.get("history") or []):
        if item.get("name"):
            args = item.get("args")
            if isinstance(args, dict):
                return compact(json.dumps(args, ensure_ascii=False, sort_keys=True), limit)
            return compact(args, limit)
    return "none"


def result_status(text):
    low = str(text or "").lower()
    if not low:
        return "none"
    if any(k in low for k in ["failed", "failure", "error", "traceback", "exit code 1"]):
        return "fail"
    if any(k in low for k in ["no match", "0 match", "no result", "0 result", "empty", "not found"]):
        return "empty"
    return "ok"


def bucket_num(value, cuts, prefix="le"):
    try:
        x = float(value)
    except Exception:
        return "na"
    for cut in cuts:
        if x <= cut:
            return f"{prefix}_{cut}"
    return f"gt_{cuts[-1]}"


def prompt_length_bucket(prompt):
    n = len(str(prompt or "").split())
    if n <= 5:
        return "le5"
    if n <= 10:
        return "le10"
    if n <= 18:
        return "le18"
    if n <= 30:
        return "le30"
    return "gt30"


def budget_bucket(meta):
    return bucket_num((meta or {}).get("budget_tokens_remaining"), [5000, 12000, 20000], prefix="le")


def last_result_bucket(text):
    return result_status(text)


def result_type(row):
    actions = previous_actions(row)
    last = actions[-1] if actions else "NONE"
    res = last_result(row).lower()
    if last == "read_file" or " read " in f" {res} ":
        return "read"
    if last == "grep_search" or "matches" in res or "occurrences" in res:
        return "grep"
    if last == "glob_pattern":
        return "glob"
    if last == "list_directory" or "entries" in res:
        return "list"
    if last in {"run_tests", "run_bash", "lint_or_typecheck"}:
        return "exec"
    return "other"


def recent_values_from_args(row, keys, limit=8):
    values = []
    for item in reversed(row.get("history") or []):
        args = item.get("args")
        if not isinstance(args, dict):
            continue
        for key in keys:
            value = args.get(key)
            if not value:
                continue
            if isinstance(value, list):
                values.extend(compact(v, 120) for v in value)
            else:
                values.append(compact(value, 160))
        if len(values) >= limit:
            break
    return dedupe_tail(values, limit)


def dedupe_tail(values, limit=8):
    out = []
    seen = set()
    for value in reversed(values):
        value = compact(value, 180)
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return list(reversed(out[-limit:]))


def recent_paths(row, limit=10):
    import re

    path_re = re.compile(
        r"[\w./\\-]+\.(?:py|js|jsx|ts|tsx|json|md|txt|ya?ml|toml|go|rs|java|kt|cpp|c|h|css|html|sql|sh|ps1)",
        re.I,
    )
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    values = [str(path) for path in ws.get("open_files") or []]
    values.extend(recent_values_from_args(row, ["path", "paths", "file", "files"], limit=limit))
    for item in reversed(row.get("history") or []):
        chunks = [item.get("result_summary")]
        args = item.get("args")
        if isinstance(args, dict):
            chunks.extend(args.values())
        elif args:
            chunks.append(args)
        for chunk in chunks:
            if chunk:
                values.extend(path_re.findall(str(chunk)))
        if len(values) >= limit * 2:
            break
    return dedupe_tail(values, limit)


def recent_patterns(row, limit=8):
    import re

    glob_re = re.compile(r"(?:\*\*?[/.\w-]*|[/.\w-]*\*\*?|[/.\w-]*\*[/.\w-]*|\{[^}]+\}|\[[^\]]+\])")
    values = recent_values_from_args(row, ["pattern", "query", "glob"], limit=limit)
    values.extend(glob_re.findall(compact(row.get("current_prompt"), 800)))
    return dedupe_tail(values, limit)


def open_files(row, limit=10):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    return [str(x) for x in (ws.get("open_files") or [])[-limit:]]


def flow_text(row):
    actions = previous_actions(row)
    bigrams = [f"{a}>{b}" for a, b in zip(actions, actions[1:])]
    counts = Counter(actions)
    return " ".join(
        [
            f"last={actions[-1] if actions else 'START'}",
            f"prev={actions[-2] if len(actions) >= 2 else 'START'}",
            f"tail3={'>'.join(actions[-3:]) or 'START'}",
            f"tail6={'>'.join(actions[-6:]) or 'START'}",
            "bigrams=" + " ".join(bigrams[-8:]),
            "counts=" + " ".join(f"{k}:{v}" for k, v in sorted(counts.items())),
        ]
    )


def inspect_short_text(row, variant: int):
    prompt = compact(row.get("current_prompt"), 650)
    res = last_result(row)
    paths = " ".join(recent_paths(row, 10)) or "none"
    patterns = " ".join(recent_patterns(row, 8)) or "none"
    opened = " ".join(open_files(row, 10)) or "none"
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    header = (
        "query: choose inspection action from read_file grep_search glob_pattern list_directory "
        f"[SRC=sim] [AUG=v{variant}] "
        f"[STEP={bucket_num(meta.get('turn_index'), [2, 5, 10, 20], prefix='le')}] "
        f"[PLEN={prompt_length_bucket(row.get('current_prompt'))}] "
        f"[BUDGET={budget_bucket(meta)}] "
        f"[LASTRES={last_result_bucket(res)}]"
    )
    common = [
        header,
        "[PROMPT]",
        prompt,
        "[LAST_ACTION]",
        flow_text(row),
        "[LAST_RESULT]",
        f"status={result_status(res)} type={result_type(row)} hint={compact(res, 260)}",
        "[OPEN_FILES]",
        opened,
        "[PATHS]",
        paths,
        "[PATTERNS]",
        patterns,
    ]
    if variant == 0:
        return "\n".join(common)
    if variant == 1:
        return "\n".join(
            [
                header,
                "[PROMPT]",
                prompt,
                "[LAST_USER]",
                last_user(row),
                "[LAST]",
                f"args={last_action_args(row)} result={compact(res, 320)}",
                "[OPEN_AND_PATHS]",
                f"open={opened} paths={paths} patterns={patterns}",
            ]
        )
    if variant == 2:
        return "\n".join(
            [
                header,
                "[FLOW_FIRST]",
                flow_text(row),
                "[RESULT]",
                f"status={result_status(res)} type={result_type(row)} hint={compact(res, 320)}",
                "[TOOL_HINTS]",
                f"paths={paths} patterns={patterns} open={opened}",
                "[PROMPT_LIGHT]",
                compact(prompt, 260),
            ]
        )
    if variant == 3:
        return "\n".join(
            [
                header,
                "[DECISION_HINT]",
                "file content/reference/usage => grep_search | file path/open file => read_file | wildcard/find files => glob_pattern | folder/tree/list => list_directory",
                "[PROMPT]",
                prompt,
                "[STATE]",
                f"open_files_count={len(open_files(row, 999))} git_dirty={ws.get('git_dirty', 'na')} top_lang={top_lang(row)}",
                "[PATHS_PATTERNS]",
                f"{paths} {patterns}",
            ]
        )
    return "\n".join(
        [
            header,
            "[COMPACT]",
            f"prompt={compact(prompt, 420)}",
            f"flow={flow_text(row)}",
            f"result_status={result_status(res)} result_type={result_type(row)}",
            f"open={opened}",
            f"paths={paths}",
            f"patterns={patterns}",
        ]
    )


def top_lang(row):
    meta = row.get("session_meta") or {}
    ws = meta.get("workspace") or {}
    mix = ws.get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return str(max(mix.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)[0]).lower()


def parse_view_weights(text):
    out = {}
    if not text:
        return out
    for part in str(text).split(","):
        if not part.strip():
            continue
        k, v = part.split(":", 1)
        out[int(k.strip())] = float(v.strip())
    return out


def build_dataset(rows, y, structured_features, views, view_weights):
    label_to_idx = {label: i for i, label in enumerate(INSPECT_LABELS)}
    texts, feats, labels, weights, ids, vars_ = [], [], [], [], [], []
    for row, label in zip(rows, y):
        feature = structured_features(row).astype(np.float32)
        for v in views:
            texts.append(inspect_short_text(row, v))
            feats.append(feature)
            labels.append(label_to_idx[str(label)])
            weights.append(float(view_weights.get(v, 1.0)))
            ids.append(str(row["id"]))
            vars_.append(v)
    return (
        texts,
        np.vstack(feats).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
        np.asarray(ids, dtype=object),
        np.asarray(vars_, dtype=np.int16),
    )


class InspectDataset(Dataset):
    def __init__(self, texts, features, labels=None, weights=None):
        self.texts = list(texts)
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = labels
        self.weights = weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {"text": self.texts[idx], "features": self.features[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        if self.weights is not None:
            item["weight"] = float(self.weights[idx])
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
        if "weight" in batch[0]:
            encoded["weights"] = torch.tensor([item["weight"] for item in batch], dtype=torch.float32)
        return encoded

    return collate


class InspectProtoModel(nn.Module):
    def __init__(self, model_dir, feature_dim, proto_init, dropout=0.15, use_cls=False):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(model_dir, config=self.config, local_files_only=True)
        hidden = int(getattr(self.config, "hidden_size"))
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.use_cls = use_cls
        self.prototypes = nn.Parameter(torch.tensor(proto_init, dtype=torch.float32))
        rep_dim = hidden + feature_dim + 4 + (hidden if use_cls else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(rep_dim),
            nn.Linear(rep_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.75),
            nn.Linear(128, 4),
        )
        self.proto_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def pool_mean(self, output, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(output.last_hidden_state.dtype)
        return (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

    def forward(self, input_ids, attention_mask, features, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        output = self.encoder(**kwargs)
        cls = output.last_hidden_state[:, 0]
        mean = self.pool_mean(output, attention_mask)
        mean_norm = F.normalize(mean.float(), p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes.float(), p=2, dim=-1)
        proto_sim = mean_norm @ proto_norm.T
        parts = [mean_norm.to(mean.dtype), self.feature_norm(features.to(mean.dtype)), proto_sim.to(mean.dtype)]
        if self.use_cls:
            parts.insert(1, cls)
        logits = self.head(torch.cat(parts, dim=-1))
        return logits + self.proto_scale.to(logits.dtype) * proto_sim.to(logits.dtype)


def model_inputs(batch):
    return {k: v for k, v in batch.items() if k in {"input_ids", "attention_mask", "token_type_ids", "features"}}


@torch.no_grad()
def init_prototypes(model_dir, tokenizer, rows, y, structured_features, max_length, batch_size, device, feature_dim):
    tmp = InspectProtoModel(
        model_dir,
        feature_dim=feature_dim,
        proto_init=np.random.normal(0, 0.02, size=(4, AutoConfig.from_pretrained(model_dir, local_files_only=True).hidden_size)).astype(np.float32),
        dropout=0.0,
        use_cls=False,
    ).to(device)
    tmp.eval()
    texts = [inspect_short_text(row, 0) for row in rows]
    feats = np.vstack([structured_features(row) for row in rows]).astype(np.float32)
    ds = InspectDataset(texts, feats, np.zeros(len(texts), dtype=np.int64))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=make_collate(tokenizer, max_length), num_workers=0)
    reps = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            output = tmp.encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                **({"token_type_ids": batch["token_type_ids"]} if "token_type_ids" in batch else {}),
            )
            mean = tmp.pool_mean(output, batch["attention_mask"])
            reps.append(F.normalize(mean.float(), p=2, dim=-1).cpu().numpy())
    reps = np.vstack(reps).astype(np.float32)
    proto = []
    y = np.asarray(y, dtype=object)
    for label in INSPECT_LABELS:
        mask = y == label
        vec = reps[mask].mean(axis=0) if mask.any() else reps.mean(axis=0)
        vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
        proto.append(vec.astype(np.float32))
    del tmp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.vstack(proto)


def save_model(model, tokenizer, out_dir: Path, epoch: int, meta):
    save_dir = out_dir / f"epoch_{epoch}_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, save_dir / "model.pt")
    tokenizer.save_pretrained(save_dir)
    np.save(save_dir / "classes.npy", INSPECT_LABELS.astype(str))
    with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return save_dir


def train(args):
    seed_everything(args.seed)
    sys.path.insert(0, str(Path(args.code_dir).resolve()))
    from run_domain_input_global_oof import structured_features

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    original_rows, original_y = load_original_inspect_rows(Path(args.data_dir), args.id_prefix)
    extra_rows, extra_y = load_extra_inspect_rows(Path(args.extra_jsonl), Path(args.extra_labels), args.id_prefix)
    rows = original_rows + extra_rows
    y = np.concatenate([original_y, extra_y])
    views = [int(x) for x in args.views.split(",") if x.strip()]
    view_weights = parse_view_weights(args.view_weights)

    texts, feats, labels, weights, ids, variants = build_dataset(rows, y, structured_features, views, view_weights)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, use_fast=args.use_fast)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    summary_base = {
        "original_inspect_rows": int(len(original_rows)),
        "extra_inspect_rows": int(len(extra_rows)),
        "total_inspect_rows": int(len(rows)),
        "augmented_rows": int(len(texts)),
        "views": views,
        "view_weights": {str(k): float(v) for k, v in view_weights.items()},
        "label_counts": {str(k): int(v) for k, v in pd.Series(y).value_counts().sort_index().items()},
        "feature_dim": int(feats.shape[1]),
        "max_length": int(args.max_length),
        "seed": int(args.seed),
    }
    print(json.dumps(summary_base, ensure_ascii=False, indent=2), flush=True)
    print(f"device={device}", flush=True)
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    if args.dry_run:
        return

    proto_rows, proto_y = rows, y
    if args.proto_init_max_per_class > 0:
        rng = np.random.default_rng(args.seed)
        keep = []
        for label in INSPECT_LABELS:
            idx = np.where(y == label)[0]
            if len(idx) > args.proto_init_max_per_class:
                idx = rng.choice(idx, size=args.proto_init_max_per_class, replace=False)
            keep.extend(idx.tolist())
        keep = sorted(keep)
        proto_rows = [rows[i] for i in keep]
        proto_y = y[keep]
    print(f"proto_init_rows={len(proto_rows)} per_class={dict(Counter(proto_y.tolist()))}", flush=True)
    proto_init = init_prototypes(
        args.model_dir,
        tokenizer,
        proto_rows,
        proto_y,
        structured_features,
        args.max_length,
        args.eval_batch_size,
        device,
        feats.shape[1],
    )

    model = InspectProtoModel(
        args.model_dir,
        feature_dim=feats.shape[1],
        proto_init=proto_init,
        dropout=args.dropout,
        use_cls=args.use_cls,
    ).to(device)

    ds = InspectDataset(texts, feats, labels, weights)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, args.max_length),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(args.seed),
    )
    class_weight = None
    if args.class_weight == "balanced":
        cnt = np.bincount(labels, minlength=4).astype(np.float32)
        cw = cnt.sum() / np.maximum(cnt, 1) / 4.0
        cw = np.clip(cw, 1.0 / args.max_class_weight, args.max_class_weight)
        class_weight = torch.tensor(cw, dtype=torch.float32, device=device)
        print(f"class_weight={cw.tolist()}", flush=True)
    ce = nn.CrossEntropyLoss(weight=class_weight, reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(loader) / args.grad_accum) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * args.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.fp16))
    save_epochs = {int(x.strip()) for x in args.save_epochs.split(",") if x.strip()} or {args.epochs}

    hist = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.fp16)):
                logits = model(**model_inputs(batch))
                loss_vec = ce(logits, batch["labels"]) * batch["weights"].to(logits.dtype)
                loss = loss_vec.mean() / args.grad_accum
            scaler.scale(loss).backward()
            losses.append(float(loss.detach().cpu()) * args.grad_accum)
            if step % args.grad_accum == 0 or step == len(loader):
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(time.time() - t0, 2)}
        hist.append(row)
        print(f"epoch={epoch} loss={row['loss']:.5f} sec={row['seconds']:.1f}", flush=True)
        if epoch in save_epochs:
            meta = {**summary_base, "epoch": epoch, "history": hist, "classes": INSPECT_LABELS.tolist()}
            save_dir = save_model(model, tokenizer, out_dir, epoch, meta)
            print(f"saved_epoch={epoch} dir={save_dir}", flush=True)

    pd.DataFrame(hist).to_csv(out_dir / "epoch_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"id": ids, "variant": variants, "label": [INSPECT_LABELS[i] for i in labels], "weight": weights}).to_csv(
        out_dir / "train_augmented_index.csv", index=False, encoding="utf-8-sig"
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({**summary_base, "history": hist}, f, ensure_ascii=False, indent=2)
    print(f"done out_dir={out_dir}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="open/data")
    p.add_argument("--model-dir", default="models/multilingual-e5-base")
    p.add_argument("--code-dir", default="code")
    p.add_argument("--extra-jsonl", default="data/pi_exchange_seed42/mined_plus_exchange.jsonl")
    p.add_argument("--extra-labels", default="data/pi_exchange_seed42/mined_plus_exchange.labels.csv")
    p.add_argument("--output-dir", default="outputs/models/inspect4_proto_specialist")
    p.add_argument("--id-prefix", default="sess_sim")
    p.add_argument("--views", default="0,1,2,3,4")
    p.add_argument("--view-weights", default="0:1.4,1:1.1,2:1.0,3:1.0,4:0.7")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--save-epochs", default="2,3,4")
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=384)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.06)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    p.add_argument("--max-class-weight", type=float, default=2.0)
    p.add_argument("--proto-init-max-per-class", type=int, default=6000)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--use-fast", action="store_true")
    p.add_argument("--use-cls", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
