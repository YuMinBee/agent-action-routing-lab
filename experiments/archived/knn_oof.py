import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from transformers import AutoModel, AutoTokenizer


FULL_LABELS = np.array(
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

INSPECT_LABELS = np.array(["glob_pattern", "grep_search", "list_directory", "read_file"], dtype=object)


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_labels(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}


def session_id(sample_id):
    return str(sample_id).split("-step_")[0]


def step_num(sample_id):
    m = re.search(r"-step_(\d+)$", str(sample_id))
    return int(m.group(1)) if m else -1


def step_group(step):
    if step <= 1:
        return "s1"
    if step <= 3:
        return "s2_3"
    if step <= 6:
        return "s4_6"
    if step <= 10:
        return "s7_10"
    return "s11p"


def previous_actions(row):
    return [str(item.get("name")) for item in row.get("history") or [] if item.get("name")]


def load_submit_builder(path):
    path = Path(path)
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location("submit_text_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_texts(rows, builder):
    texts = []
    for row in rows:
        sample_id = str(row["id"])
        input_src = "au" if sample_id.startswith("sess_au") else "sim"
        texts.append(builder.build_text(row, input_src=input_src))
    return texts


def make_metadata(rows, y):
    records = []
    for row, label in zip(rows, y):
        sample_id = str(row["id"])
        actions = previous_actions(row)
        step = step_num(sample_id)
        records.append(
            {
                "id": sample_id,
                "action": str(label),
                "session_id": session_id(sample_id),
                "src": "au" if sample_id.startswith("sess_au") else "sim",
                "step": step,
                "step_group": step_group(step),
                "last_action": actions[-1] if actions else "none",
                "prev_action": actions[-2] if len(actions) >= 2 else "none",
                "history_len": len(actions),
            }
        )
    return pd.DataFrame(records)


def mean_pool(output, attention_mask):
    token_embeddings = output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def encode_texts(texts, model_dir, max_length, batch_size, device, fp16):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    if fp16 and device.type == "cuda":
        model.half()
    model.eval()

    chunks = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and fp16)):
                output = model(**encoded)
                emb = mean_pool(output, encoded["attention_mask"])
                emb = F.normalize(emb, p=2, dim=-1)
            chunks.append(emb.float().cpu().numpy())
            if start and start % (batch_size * 20) == 0:
                print(f"encoded {start}/{len(texts)} rows in {time.time() - t0:.1f}s", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(chunks).astype(np.float32)


def align_class_matrix(proba, src_classes, dst_classes):
    out = np.zeros((proba.shape[0], len(dst_classes)), dtype=np.float32)
    src_map = {str(label): idx for idx, label in enumerate(src_classes)}
    for dst_idx, label in enumerate(dst_classes):
        out[:, dst_idx] = proba[:, src_map[str(label)]]
    return out


def load_cls_oof(args, ids):
    if args.cls_proba_file:
        path = Path(args.cls_proba_file)
        if not path.is_absolute() and not path.exists() and args.cls_oof_dir:
            path = Path(args.cls_oof_dir) / path
        proba = np.load(path).astype(np.float32)
        if proba.shape[1] == len(FULL_LABELS):
            return proba
        if args.cls_oof_dir:
            classes_path = Path(args.cls_oof_dir) / "classes.npy"
            if classes_path.exists():
                classes = np.load(classes_path, allow_pickle=True).astype(str)
                return align_class_matrix(proba, classes, FULL_LABELS)
        raise ValueError("--cls-proba-file must have 14 columns in FULL_LABELS order, or provide --cls-oof-dir/classes.npy")

    if args.cls_oof_dir:
        oof_dir = Path(args.cls_oof_dir)
        classes = np.load(oof_dir / "classes.npy", allow_pickle=True).astype(str)
        sample_ids = np.load(oof_dir / "sample_ids.npy", allow_pickle=True).astype(str)
        if not np.array_equal(sample_ids.astype(str), ids.astype(str)):
            raise ValueError("cls_oof_dir sample_ids do not align with train.jsonl")
        if args.cls_artifact:
            main = np.load(oof_dir / f"oof_{args.cls_artifact}.npy")
            if args.cls_global_blend > 0:
                global_path = oof_dir / f"oof_{args.cls_artifact}_global.npy"
                global_proba = np.load(global_path)
                main = (1.0 - args.cls_global_blend) * main + args.cls_global_blend * global_proba
            return align_class_matrix(main, classes, FULL_LABELS)
        if args.cls_proba_file:
            path = Path(args.cls_proba_file)
            if not path.is_absolute():
                path = oof_dir / path
            return align_class_matrix(np.load(path), classes, FULL_LABELS)
    return None


def label_indices(y):
    mapping = {label: idx for idx, label in enumerate(FULL_LABELS)}
    return np.asarray([mapping[str(label)] for label in y], dtype=np.int64)


def knn_proba(neigh_idx, sims, y_train_idx, n_classes, k, temperature):
    idx = neigh_idx[:, :k]
    sim = sims[:, :k]
    weights = np.exp((sim - sim.max(axis=1, keepdims=True)) / max(temperature, 1e-6))
    out = np.zeros((idx.shape[0], n_classes), dtype=np.float32)
    neigh_labels = y_train_idx[idx]
    for row in range(idx.shape[0]):
        np.add.at(out[row], neigh_labels[row], weights[row])
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-8)
    return out


def build_knn_oof_sklearn(emb, y_idx, groups, n_splits, max_k, n_jobs):
    n = emb.shape[0]
    neigh_idx_all = np.zeros((n, max_k), dtype=np.int32)
    sims_all = np.zeros((n, max_k), dtype=np.float32)
    fold_ids = np.full(n, -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(emb, y_idx, groups=groups), start=0):
        t0 = time.time()
        nn = NearestNeighbors(n_neighbors=max_k, metric="cosine", algorithm="brute", n_jobs=n_jobs)
        nn.fit(emb[train_idx])
        dist, idx = nn.kneighbors(emb[valid_idx], return_distance=True)
        neigh_idx_all[valid_idx] = train_idx[idx]
        sims_all[valid_idx] = 1.0 - dist
        fold_ids[valid_idx] = fold
        print(f"knn fold={fold + 1} train={len(train_idx)} valid={len(valid_idx)} sec={time.time() - t0:.1f}", flush=True)
    return neigh_idx_all, sims_all, fold_ids


def build_knn_oof_torch(emb, y_idx, groups, n_splits, max_k, device, chunk_size):
    n = emb.shape[0]
    neigh_idx_all = np.zeros((n, max_k), dtype=np.int32)
    sims_all = np.zeros((n, max_k), dtype=np.float32)
    fold_ids = np.full(n, -1, dtype=np.int16)
    splitter = GroupKFold(n_splits=n_splits)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(emb, y_idx, groups=groups), start=0):
        t0 = time.time()
        train_tensor = torch.from_numpy(emb[train_idx]).to(device=device, dtype=dtype).T.contiguous()
        for start in range(0, len(valid_idx), chunk_size):
            chunk_idx = valid_idx[start : start + chunk_size]
            query_tensor = torch.from_numpy(emb[chunk_idx]).to(device=device, dtype=dtype)
            sims = query_tensor @ train_tensor
            vals, inds = torch.topk(sims, k=max_k, dim=1, largest=True, sorted=True)
            neigh_idx_all[chunk_idx] = train_idx[inds.cpu().numpy()]
            sims_all[chunk_idx] = vals.float().cpu().numpy()
            del query_tensor, sims, vals, inds
        del train_tensor
        if device.type == "cuda":
            torch.cuda.empty_cache()
        fold_ids[valid_idx] = fold
        print(f"torch knn fold={fold + 1} train={len(train_idx)} valid={len(valid_idx)} sec={time.time() - t0:.1f}", flush=True)
    return neigh_idx_all, sims_all, fold_ids


def metrics(name, y_true, proba, labels=FULL_LABELS):
    pred = labels[proba.argmax(axis=1)]
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


def subset_metric_rows(name, y_true, pred, meta):
    rows = []
    for key, values in {
        "src": ["sim", "au"],
        "step_group": ["s1", "s2_3", "s4_6", "s7_10", "s11p"],
    }.items():
        for value in values:
            mask = meta[key].to_numpy() == value
            if mask.any():
                rows.append(
                    {
                        "name": name,
                        "subset": f"{key}={value}",
                        "rows": int(mask.sum()),
                        "macro_f1": float(f1_score(y_true[mask], pred[mask], labels=FULL_LABELS, average="macro", zero_division=0)),
                        "accuracy": float(accuracy_score(y_true[mask], pred[mask])),
                    }
                )
    mask = meta["step"].to_numpy() != 1
    rows.append(
        {
            "name": name,
            "subset": "step!=1",
            "rows": int(mask.sum()),
            "macro_f1": float(f1_score(y_true[mask], pred[mask], labels=FULL_LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true[mask], pred[mask])),
        }
    )
    inspect_mask = np.isin(y_true, INSPECT_LABELS)
    rows.append(
        {
            "name": name,
            "subset": "true_inspect4",
            "rows": int(inspect_mask.sum()),
            "macro_f1": float(f1_score(y_true[inspect_mask], pred[inspect_mask], labels=INSPECT_LABELS, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true[inspect_mask], pred[inspect_mask])),
        }
    )
    return rows


def confusion_pairs(y_true, pred):
    pairs = {}
    for src, dst in [
        ("read_file", "grep_search"),
        ("grep_search", "read_file"),
        ("glob_pattern", "list_directory"),
        ("list_directory", "glob_pattern"),
    ]:
        pairs[f"{src}_to_{dst}"] = int(np.sum((y_true == src) & (pred == dst)))
    return pairs


def inspect_rerank(cls_proba, knn_proba_arr, alpha, threshold):
    inspect_idx = np.asarray([int(np.where(FULL_LABELS == label)[0][0]) for label in INSPECT_LABELS])
    cls_mass = cls_proba[:, inspect_idx].sum(axis=1)
    cls_top = FULL_LABELS[cls_proba.argmax(axis=1)]
    mask = (cls_mass >= threshold) | np.isin(cls_top, INSPECT_LABELS)
    cls_local = cls_proba[:, inspect_idx] / np.maximum(cls_mass[:, None], 1e-8)
    knn_mass = knn_proba_arr[:, inspect_idx].sum(axis=1)
    knn_local = knn_proba_arr[:, inspect_idx] / np.maximum(knn_mass[:, None], 1e-8)
    local = alpha * cls_local + (1.0 - alpha) * knn_local
    out = cls_proba.copy()
    rows = np.where(mask)[0]
    out[rows[:, None], inspect_idx[None, :]] = local[rows] * cls_mass[rows, None]
    return out, int(mask.sum())


def evaluate_and_save(out_dir, name, y_true, proba):
    pred = FULL_LABELS[proba.argmax(axis=1)]
    pd.DataFrame(classification_report(y_true, pred, labels=FULL_LABELS, output_dict=True, zero_division=0)).T.to_csv(
        out_dir / f"{name}_class_report.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(confusion_matrix(y_true, pred, labels=FULL_LABELS), index=FULL_LABELS, columns=FULL_LABELS).to_csv(
        out_dir / f"{name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--submit-script", default="experiments/submissions/submit_pi_exchange_preserve_sim_seed42oldau_20260707_202407/script.py")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default="experiments/analysis/e5_knn_oof")
    parser.add_argument("--embedding-file", default="")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ks", default="8,16,32,64,128")
    parser.add_argument("--temps", default="0.03,0.05,0.07,0.1,0.15,0.2")
    parser.add_argument("--blend-alphas", default="0.98,0.95,0.9,0.85,0.8,0.75,0.7,0.6,0.5")
    parser.add_argument("--inspect-thresholds", default="0.35,0.45,0.5,0.6,0.7")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--knn-backend", choices=["torch", "sklearn"], default="torch")
    parser.add_argument("--knn-chunk-size", type=int, default=1024)
    parser.add_argument("--cls-oof-dir", default="experiments/oof/multihead_e5base_respbin_global512_e7_20260703_014338")
    parser.add_argument("--cls-artifact", default="multihead_e5base_respbin_global512_e7_20260703_014338")
    parser.add_argument("--cls-global-blend", type=float, default=0.45)
    parser.add_argument("--cls-proba-file", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    data_dir = Path(args.data_dir)
    rows = load_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    ids = np.asarray([str(row["id"]) for row in rows], dtype=str)
    y = np.asarray([labels[sample_id] for sample_id in ids], dtype=str)
    y_idx = label_indices(y)
    groups = np.asarray([session_id(sample_id) for sample_id in ids], dtype=str)
    meta = make_metadata(rows, y)
    meta.to_csv(out_dir / "joined_metadata.csv", index=False, encoding="utf-8-sig")

    if args.embedding_file and Path(args.embedding_file).exists():
        emb = np.load(args.embedding_file).astype(np.float32)
        print(f"loaded embeddings {args.embedding_file} {emb.shape}", flush=True)
    else:
        builder = load_submit_builder(args.submit_script)
        texts = build_texts(rows, builder)
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        print(f"encoding rows={len(texts)} model={args.model_dir} device={device}", flush=True)
        emb = encode_texts(texts, args.model_dir, args.max_length, args.batch_size, device, args.fp16)
        emb_path = out_dir / "e5_embeddings.npy"
        np.save(emb_path, emb.astype(np.float16))
        print(f"saved embeddings {emb_path}", flush=True)

    cls_proba = load_cls_oof(args, ids)
    if cls_proba is not None:
        if cls_proba.shape != (len(y), len(FULL_LABELS)):
            raise ValueError(f"cls_proba shape mismatch: got={cls_proba.shape} expected={(len(y), len(FULL_LABELS))}")
        np.save(out_dir / "p_cls_oof.npy", cls_proba.astype(np.float32))

    ks = [int(item) for item in args.ks.split(",") if item.strip()]
    temps = [float(item) for item in args.temps.split(",") if item.strip()]
    alphas = [float(item) for item in args.blend_alphas.split(",") if item.strip()]
    thresholds = [float(item) for item in args.inspect_thresholds.split(",") if item.strip()]
    max_k = max(ks)
    if args.knn_backend == "torch":
        knn_device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        neigh_idx, sims, fold_ids = build_knn_oof_torch(
            emb,
            y_idx,
            groups,
            args.n_splits,
            max_k,
            knn_device,
            args.knn_chunk_size,
        )
    else:
        neigh_idx, sims, fold_ids = build_knn_oof_sklearn(emb, y_idx, groups, args.n_splits, max_k, args.n_jobs)
    np.save(out_dir / "fold_ids.npy", fold_ids)

    rows_out = []
    subset_rows = []
    best = {"macro_f1": -1.0}
    best_knn = {"macro_f1": -1.0}
    best_inspect = {"macro_f1": -1.0}

    if cls_proba is not None:
        cls_pred = evaluate_and_save(out_dir, "base_cls", y, cls_proba)
        base_row = metrics("base_cls", y, cls_proba)
        base_row.update(confusion_pairs(y, cls_pred))
        rows_out.append(base_row)
        subset_rows.extend(subset_metric_rows("base_cls", y, cls_pred, meta))

    for k in ks:
        for temp in temps:
            p_knn = np.zeros((len(y), len(FULL_LABELS)), dtype=np.float32)
            for fold in sorted(np.unique(fold_ids)):
                valid = np.where(fold_ids == fold)[0]
                train_labels = y_idx
                p_knn[valid] = knn_proba(neigh_idx[valid], sims[valid], train_labels, len(FULL_LABELS), k, temp)

            name = f"knn_k{k}_t{temp:g}"
            pred = FULL_LABELS[p_knn.argmax(axis=1)]
            row = metrics(name, y, p_knn)
            row.update({"k": k, "temp": temp, "mode": "knn"})
            row.update(confusion_pairs(y, pred))
            rows_out.append(row)
            if row["macro_f1"] > best_knn["macro_f1"]:
                best_knn = dict(row)
                np.save(out_dir / "p_knn_best.npy", p_knn.astype(np.float32))
                evaluate_and_save(out_dir, "best_knn", y, p_knn)

            if cls_proba is None:
                continue

            for alpha in alphas:
                blended = alpha * cls_proba + (1.0 - alpha) * p_knn
                blend_name = f"blend_a{alpha:g}_k{k}_t{temp:g}"
                blend_pred = FULL_LABELS[blended.argmax(axis=1)]
                blend_row = metrics(blend_name, y, blended)
                blend_row.update({"k": k, "temp": temp, "alpha": alpha, "mode": "full_blend"})
                blend_row.update(confusion_pairs(y, blend_pred))
                rows_out.append(blend_row)
                if blend_row["macro_f1"] > best["macro_f1"]:
                    best = dict(blend_row)
                    np.save(out_dir / "p_blend_best.npy", blended.astype(np.float32))
                    evaluate_and_save(out_dir, "best_blend", y, blended)

                for threshold in thresholds:
                    reranked, mask_count = inspect_rerank(cls_proba, p_knn, alpha, threshold)
                    rerank_name = f"inspect_rerank_a{alpha:g}_k{k}_t{temp:g}_thr{threshold:g}"
                    rerank_pred = FULL_LABELS[reranked.argmax(axis=1)]
                    rerank_row = metrics(rerank_name, y, reranked)
                    rerank_row.update(
                        {
                            "k": k,
                            "temp": temp,
                            "alpha": alpha,
                            "threshold": threshold,
                            "mode": "inspect_rerank",
                            "rerank_rows": mask_count,
                        }
                    )
                    rerank_row.update(confusion_pairs(y, rerank_pred))
                    rows_out.append(rerank_row)
                    if rerank_row["macro_f1"] > best_inspect["macro_f1"]:
                        best_inspect = dict(rerank_row)
                        np.save(out_dir / "p_inspect_rerank_best.npy", reranked.astype(np.float32))
                        evaluate_and_save(out_dir, "best_inspect_rerank", y, reranked)

    results = pd.DataFrame(rows_out).sort_values(["macro_f1", "accuracy"], ascending=False)
    results.to_csv(out_dir / "sweep_results.csv", index=False, encoding="utf-8-sig")

    for candidate_name, proba_file in [
        ("best_knn", "p_knn_best.npy"),
        ("best_blend", "p_blend_best.npy"),
        ("best_inspect_rerank", "p_inspect_rerank_best.npy"),
    ]:
        path = out_dir / proba_file
        if path.exists():
            pred = FULL_LABELS[np.load(path).argmax(axis=1)]
            subset_rows.extend(subset_metric_rows(candidate_name, y, pred, meta))
    pd.DataFrame(subset_rows).to_csv(out_dir / "subset_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "rows": int(len(y)),
        "sessions": int(len(np.unique(groups))),
        "label_counts": Counter(y).most_common(),
        "best_knn": best_knn,
        "best_full_blend": best,
        "best_inspect_rerank": best_inspect,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(results.head(25).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
