import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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


ACTION_PROTOTYPES = {
    "apply_patch": [
        "passage: action=apply_patch | apply a small precise patch to existing files after the target change is clear",
        "passage: action=apply_patch | use when editing code with a diff-style patch is the next concrete step",
        "passage: action=apply_patch | modify one or more known files by inserting, replacing, or deleting lines",
        "passage: action=apply_patch | implement the requested fix now after reading enough context",
        "passage: action=apply_patch | make surgical source changes rather than only inspecting or explaining",
        "passage: action=apply_patch | update tests, config, docs, or code with a patch operation",
        "passage: action=apply_patch | user asked to change code and the relevant location is already identified",
        "passage: action=apply_patch | proceed from diagnosis to concrete file modification",
    ],
    "ask_user": [
        "passage: action=ask_user | ask a clarifying question because required information is missing",
        "passage: action=ask_user | use when multiple risky interpretations remain and local context cannot decide",
        "passage: action=ask_user | request user choice before destructive, ambiguous, or externally dependent action",
        "passage: action=ask_user | wait for user input because progress depends on their preference or secret",
        "passage: action=ask_user | clarify which file, environment, behavior, or option the user wants",
        "passage: action=ask_user | user instruction is under-specified and guessing would be unsafe",
    ],
    "edit_file": [
        "passage: action=edit_file | edit a known file to implement a requested code or text change",
        "passage: action=edit_file | make direct changes in a specific existing file after it has been identified",
        "passage: action=edit_file | update source code, config, or documentation in place",
        "passage: action=edit_file | use when user asks to tweak, fix, rename, refactor, or add behavior in a file",
        "passage: action=edit_file | continue by modifying the file currently being discussed",
        "passage: action=edit_file | perform straightforward file edits rather than searching more",
        "passage: action=edit_file | apply a code change when the path and intended edit are known",
        "passage: action=edit_file | change existing implementation details inside a repository file",
    ],
    "glob_pattern": [
        "passage: action=glob_pattern | find files by filename, extension, wildcard, or path pattern",
        "passage: action=glob_pattern | use when looking for candidate files matching patterns such as *.py or **/*.tsx",
        "passage: action=glob_pattern | locate project files by name like package.json, Dockerfile, pyproject, or workflow yaml",
        "passage: action=glob_pattern | discover which files exist before reading a specific one",
        "passage: action=glob_pattern | search for paths not text contents",
        "passage: action=glob_pattern | user asks where files are or which files match a naming pattern",
        "passage: action=glob_pattern | enumerate files across the repository using a glob expression",
        "passage: action=glob_pattern | find all files with a given extension or basename",
        "passage: action=glob_pattern | look up candidate source, test, config, or workflow files by path",
        "passage: action=glob_pattern | use when a wildcard file pattern is the natural next step",
    ],
    "grep_search": [
        "passage: action=grep_search | search text or symbol across repository files",
        "passage: action=grep_search | locate where a function, class, variable, phrase, or error message appears",
        "passage: action=grep_search | use when the target is unknown and user asks where something is used",
        "passage: action=grep_search | find references, usages, imports, definitions, or matching strings",
        "passage: action=grep_search | search inside files rather than list filenames",
        "passage: action=grep_search | continue by finding occurrences after a user mentions an identifier",
        "passage: action=grep_search | inspect codebase-wide matches before opening a file",
        "passage: action=grep_search | use regex or literal text to find relevant lines",
        "passage: action=grep_search | user asks where a behavior, flag, route, store action, or helper lives",
        "passage: action=grep_search | search for an error message, log text, API name, or config key",
    ],
    "lint_or_typecheck": [
        "passage: action=lint_or_typecheck | run static validation such as lint, typecheck, build, mypy, ruff, eslint, or tsc",
        "passage: action=lint_or_typecheck | verify code quality or type correctness after edits",
        "passage: action=lint_or_typecheck | use when user asks for linting, type errors, or compile-time checks",
        "passage: action=lint_or_typecheck | run non-test validation command focused on types or formatting rules",
        "passage: action=lint_or_typecheck | check whether changes pass static analysis before finishing",
        "passage: action=lint_or_typecheck | confirm no lint/type/build regressions after implementation",
    ],
    "list_directory": [
        "passage: action=list_directory | inspect directory contents or project folder structure",
        "passage: action=list_directory | use when the next step is to list files inside a known directory",
        "passage: action=list_directory | show top-level repository layout or contents of a folder",
        "passage: action=list_directory | explore a directory before choosing which file to read",
        "passage: action=list_directory | user asks what is in a folder, tree, or project area",
        "passage: action=list_directory | list immediate children of a path rather than search text",
        "passage: action=list_directory | inspect nearby files in a known package or module directory",
        "passage: action=list_directory | use ls or directory listing to understand structure",
    ],
    "plan_task": [
        "passage: action=plan_task | make or update a step-by-step plan before doing implementation",
        "passage: action=plan_task | user asks to break work into steps or outline an approach",
        "passage: action=plan_task | plan a multi-step coding task before touching files",
        "passage: action=plan_task | use when strategy, sequencing, or decomposition is the next action",
        "passage: action=plan_task | organize a workflow after receiving a complex request",
        "passage: action=plan_task | decide an implementation plan based on current context",
    ],
    "read_file": [
        "passage: action=read_file | read a known file path",
        "passage: action=read_file | inspect contents of a file already identified by search or open state",
        "passage: action=read_file | continue by opening the specific file mentioned in previous result",
        "passage: action=read_file | user names a file and wants to see or understand its contents",
        "passage: action=read_file | open source, config, test, log, or document file before editing",
        "passage: action=read_file | after grep or glob found a candidate path, read that file",
        "passage: action=read_file | inspect exact implementation in a specific path",
        "passage: action=read_file | read the file currently referenced by the conversation",
        "passage: action=read_file | use when a path is known and line-level content matters",
        "passage: action=read_file | check existing code before modifying it",
    ],
    "respond_only": [
        "passage: action=respond_only | answer the user directly without using tools or changing files",
        "passage: action=respond_only | summarize results after work is complete",
        "passage: action=respond_only | provide explanation, conclusion, or final answer from available context",
        "passage: action=respond_only | user asks a question that can be answered from known information",
        "passage: action=respond_only | no more repository action is needed and response should be sent",
        "passage: action=respond_only | acknowledge completion or report findings",
    ],
    "run_bash": [
        "passage: action=run_bash | run a shell command for inspection, generation, setup, or diagnostics",
        "passage: action=run_bash | execute command-line tools that are not specifically tests or linters",
        "passage: action=run_bash | use terminal to inspect git, environment, package metadata, or script output",
        "passage: action=run_bash | run build helpers, data scripts, or repository commands",
        "passage: action=run_bash | user asks to run a command or check command output",
        "passage: action=run_bash | use when filesystem or process information is best obtained from shell",
        "passage: action=run_bash | execute a quick local command before deciding next step",
    ],
    "run_tests": [
        "passage: action=run_tests | run unit tests, integration tests, pytest, jest, vitest, or test suite",
        "passage: action=run_tests | verify behavior after code changes by executing tests",
        "passage: action=run_tests | user asks to run tests or reproduce a failing test",
        "passage: action=run_tests | execute a test command rather than static lint or typecheck",
        "passage: action=run_tests | confirm bug fix with relevant automated tests",
        "passage: action=run_tests | run targeted tests for a changed module or feature",
    ],
    "web_search": [
        "passage: action=web_search | search the internet for current, external, or documentation information",
        "passage: action=web_search | use when user asks to look up latest docs, standards, prices, news, or APIs",
        "passage: action=web_search | verify facts outside the local repository",
        "passage: action=web_search | find official documentation or examples online before coding",
        "passage: action=web_search | research best practices or current library behavior",
        "passage: action=web_search | query the web because local files cannot answer the question",
    ],
    "write_file": [
        "passage: action=write_file | create a new file with complete content",
        "passage: action=write_file | overwrite or generate a full file rather than patch small sections",
        "passage: action=write_file | produce a new script, config, document, or artifact file",
        "passage: action=write_file | use when requested output is a fresh file on disk",
        "passage: action=write_file | write generated content into a specified path",
        "passage: action=write_file | create missing source, test, data, or documentation file",
    ],
}


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
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_module(path, module_name):
    path = Path(path)
    sys.path.insert(0, str(path.parent.resolve()))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_query_texts(rows, builder, submit_script):
    if builder == "inspect_focus":
        module = load_module("experiments/run_inspect4_e5small_oof.py", "inspect4_text_builder")
        return [module.build_inspect_text(row) for row in rows]
    if builder == "submission":
        module = load_module(submit_script, "submission_text_builder")
        texts = []
        for row in rows:
            input_src = "au" if str(row.get("id", "")).startswith("sess_au") else "sim"
            texts.append(module.build_text(row, input_src=input_src))
        return texts
    raise ValueError(f"unknown query_builder={builder}")


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
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and fp16)):
                output = model(**encoded)
                emb = mean_pool(output, encoded["attention_mask"])
                emb = F.normalize(emb, p=2, dim=-1)
            chunks.append(emb.float().cpu().numpy())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(chunks)


def action_prototype_texts(classes):
    texts = []
    owners = []
    for label in classes:
        prototypes = ACTION_PROTOTYPES.get(str(label))
        if not prototypes:
            raise ValueError(f"missing prototypes for {label}")
        texts.extend(prototypes)
        owners.extend([str(label)] * len(prototypes))
    return texts, np.asarray(owners, dtype=object)


def max_action_scores(query_emb, proto_emb, proto_owner, classes):
    sim = query_emb @ proto_emb.T
    scores = np.zeros((query_emb.shape[0], len(classes)), dtype=np.float32)
    for idx, label in enumerate(classes):
        mask = proto_owner == str(label)
        scores[:, idx] = sim[:, mask].max(axis=1)
    return scores


def softmax_np(x, temperature):
    z = x / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def minmax_np(x):
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-6)


def inspect_rerank(base_proba, proto_proba, labels, alpha, threshold):
    inspect_idx = np.asarray([int(np.where(labels == label)[0][0]) for label in INSPECT_LABELS if label in labels])
    if len(inspect_idx) != len(INSPECT_LABELS):
        return base_proba.copy(), 0
    base_mass = base_proba[:, inspect_idx].sum(axis=1)
    base_top = labels[base_proba.argmax(axis=1)]
    mask = (base_mass >= threshold) | np.isin(base_top, INSPECT_LABELS)
    base_local = base_proba[:, inspect_idx] / np.maximum(base_mass[:, None], 1e-8)
    proto_mass = proto_proba[:, inspect_idx].sum(axis=1)
    proto_local = proto_proba[:, inspect_idx] / np.maximum(proto_mass[:, None], 1e-8)
    local = alpha * base_local + (1.0 - alpha) * proto_local
    out = base_proba.copy()
    rows = np.where(mask)[0]
    out[rows[:, None], inspect_idx[None, :]] = local[rows] * base_mass[rows, None]
    return out, int(mask.sum())


def metric_row(name, y_true, pred, labels, changed=None):
    row = {
        "name": name,
        "macro_f1": float(f1_score(y_true, pred, labels=labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }
    if changed is not None:
        row["changed_rows"] = int(changed)
    return row


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


def source_metrics(ids, y_true, pred, labels):
    rows = []
    ids = np.asarray(ids).astype(str)
    for source, mask in [
        ("au", np.char.startswith(ids, "sess_au")),
        ("sim", np.char.startswith(ids, "sess_sim")),
    ]:
        if mask.any():
            row = metric_row(source, y_true[mask], pred[mask], labels)
            row["rows"] = int(mask.sum())
            rows.append(row)
    return rows


def infer_oof_file(oof_dir):
    candidates = sorted(Path(oof_dir).glob("oof*.npy"))
    candidates = [path for path in candidates if "group" not in path.name and "respond" not in path.name]
    if len(candidates) != 1:
        raise ValueError(f"pass --oof-proba-file explicitly; candidates={candidates}")
    return candidates[0].resolve()


def load_oof(args):
    oof_dir = Path(args.oof_dir)
    proba_path = Path(args.oof_proba_file) if args.oof_proba_file else infer_oof_file(oof_dir)
    if not proba_path.is_absolute():
        proba_path = oof_dir / proba_path
    proba = np.load(proba_path)
    classes = np.load(oof_dir / "classes.npy", allow_pickle=True).astype(str)
    sample_ids = np.load(oof_dir / "sample_ids.npy", allow_pickle=True).astype(str)
    if len(sample_ids) != proba.shape[0]:
        raise ValueError(f"sample_ids/proba length mismatch: {len(sample_ids)} vs {proba.shape[0]}")
    y_true_path = oof_dir / "y_true.npy"
    y_true = np.load(y_true_path, allow_pickle=True).astype(str) if y_true_path.exists() else None
    if y_true is not None and len(y_true) != len(sample_ids):
        y_true = None
    return proba, classes, sample_ids, y_true, proba_path


def select_rows(data_dir, sample_ids, y_true):
    data_dir = Path(data_dir)
    rows = load_jsonl(data_dir / "train.jsonl")
    labels = load_labels(data_dir / "train_labels.csv")
    by_id = {str(row["id"]): row for row in rows}
    selected_rows = []
    selected_y = []
    for sample_id in sample_ids:
        row = by_id.get(str(sample_id))
        if row is None:
            raise ValueError(f"sample_id not found in data_dir: {sample_id}")
        selected_rows.append(row)
        selected_y.append(labels[str(sample_id)])
    selected_y = np.asarray(selected_y, dtype=str)
    if y_true is not None and not np.array_equal(selected_y.astype(str), y_true.astype(str)):
        raise ValueError("OOF y_true does not align with labels from data_dir")
    return selected_rows, selected_y


def save_eval(out_dir, name, y_true, pred, labels):
    pd.DataFrame(classification_report(y_true, pred, labels=labels, output_dict=True, zero_division=0)).T.to_csv(
        out_dir / f"{name}_class_report.csv",
        encoding="utf-8-sig",
    )
    pd.DataFrame(confusion_matrix(y_true, pred, labels=labels), index=labels, columns=labels).to_csv(
        out_dir / f"{name}_confusion_matrix.csv",
        encoding="utf-8-sig",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="experiments/analysis/session_source_split/au_data")
    parser.add_argument("--oof-dir", required=True)
    parser.add_argument("--oof-proba-file", default="")
    parser.add_argument("--model-dir", default="models/multilingual-e5-base")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--query-builder", choices=["inspect_focus", "submission"], default="inspect_focus")
    parser.add_argument("--submit-script", default="experiments/submissions/old0779_nomined_seed42_sim2_exactau8_submit_20260707_134838/script.py")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--proto-max-length", type=int, default=96)
    parser.add_argument("--proto-batch-size", type=int, default=64)
    parser.add_argument("--alphas", default="0.95,0.9,0.85,0.8,0.75,0.7")
    parser.add_argument("--temps", default="0.03,0.05,0.07,0.1,0.15")
    parser.add_argument("--normalizers", default="softmax,minmax")
    parser.add_argument("--inspect-thresholds", default="0.25,0.35,0.45,0.55")
    parser.add_argument("--query-embedding-file", default="")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    oof_proba, classes, sample_ids, y_true_oof, proba_path = load_oof(args)
    selected_rows, y_true = select_rows(args.data_dir, sample_ids, y_true_oof)
    labels = np.asarray(classes, dtype=str)
    out_dir = Path(args.output_dir) if args.output_dir else Path("experiments/analysis/e5_prototype_rerank") / Path(args.oof_dir).name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    query_texts = build_query_texts(selected_rows, args.query_builder, args.submit_script)
    proto_texts, proto_owner = action_prototype_texts(labels)

    print(
        f"rows={len(query_texts)} classes={list(labels)} model={args.model_dir} device={device} "
        f"query_builder={args.query_builder} oof={proba_path}",
        flush=True,
    )
    if args.query_embedding_file and Path(args.query_embedding_file).exists():
        query_emb = np.load(args.query_embedding_file).astype(np.float32)
        if query_emb.shape[0] != len(query_texts):
            raise ValueError(f"query embedding row mismatch: {query_emb.shape[0]} vs {len(query_texts)}")
        print(f"loaded query embeddings {args.query_embedding_file} {query_emb.shape}", flush=True)
    else:
        query_emb = encode_texts(query_texts, args.model_dir, args.max_length, args.batch_size, device, args.fp16)
    proto_emb = encode_texts(proto_texts, args.model_dir, args.proto_max_length, args.proto_batch_size, device, args.fp16)
    proto_scores = max_action_scores(query_emb, proto_emb, proto_owner, labels)
    np.save(out_dir / "prototype_scores.npy", proto_scores.astype(np.float32))
    np.save(out_dir / "sample_ids.npy", sample_ids.astype(str))
    np.save(out_dir / "classes.npy", labels.astype(str))

    base_pred = labels[oof_proba.argmax(axis=1)]
    proto_pred = labels[proto_scores.argmax(axis=1)]
    rows = []
    base_row = metric_row("base_classifier", y_true, base_pred, labels, changed=0)
    base_row.update(confusion_pairs(y_true, base_pred))
    rows.append(base_row)
    proto_row = metric_row("prototype_raw_argmax", y_true, proto_pred, labels, changed=int(np.sum(proto_pred != base_pred)))
    proto_row.update(confusion_pairs(y_true, proto_pred))
    rows.append(proto_row)

    best = {"macro_f1": -1.0}
    normalizers = [item.strip() for item in args.normalizers.split(",") if item.strip()]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    temps = [float(item) for item in args.temps.split(",") if item.strip()]
    thresholds = [float(item) for item in args.inspect_thresholds.split(",") if item.strip()]
    for normalizer in normalizers:
        if normalizer == "minmax":
            proto_norm = minmax_np(proto_scores)
            temp_values = [math.nan]
        elif normalizer == "softmax":
            proto_norm = None
            temp_values = temps
        else:
            raise ValueError(f"unknown normalizer={normalizer}")
        for temp in temp_values:
            if normalizer == "softmax":
                proto_norm = softmax_np(proto_scores, temp)
            for alpha in alphas:
                blended = alpha * oof_proba + (1.0 - alpha) * proto_norm
                pred = labels[blended.argmax(axis=1)]
                row = metric_row(
                    f"blend_{normalizer}_a{alpha:g}_t{temp:g}",
                    y_true,
                    pred,
                    labels,
                    changed=int(np.sum(pred != base_pred)),
                )
                row["alpha"] = alpha
                row["temperature"] = temp
                row["normalizer"] = normalizer
                row["mode"] = "full_blend"
                row.update(confusion_pairs(y_true, pred))
                rows.append(row)
                if row["macro_f1"] > best["macro_f1"]:
                    best = dict(row)
                    best["pred"] = pred
                    best["proto_norm"] = proto_norm

                for threshold in thresholds:
                    reranked, changed_rows = inspect_rerank(oof_proba, proto_norm, labels, alpha, threshold)
                    rerank_pred = labels[reranked.argmax(axis=1)]
                    rerank_row = metric_row(
                        f"inspect_rerank_{normalizer}_a{alpha:g}_t{temp:g}_thr{threshold:g}",
                        y_true,
                        rerank_pred,
                        labels,
                        changed=int(np.sum(rerank_pred != base_pred)),
                    )
                    rerank_row["alpha"] = alpha
                    rerank_row["temperature"] = temp
                    rerank_row["normalizer"] = normalizer
                    rerank_row["threshold"] = threshold
                    rerank_row["mode"] = "inspect_rerank"
                    rerank_row["rerank_rows"] = changed_rows
                    rerank_row.update(confusion_pairs(y_true, rerank_pred))
                    rows.append(rerank_row)
                    if rerank_row["macro_f1"] > best["macro_f1"]:
                        best = dict(rerank_row)
                        best["pred"] = rerank_pred
                        best["proto_norm"] = proto_norm

    results = pd.DataFrame(rows).sort_values(["macro_f1", "accuracy"], ascending=False)
    results.to_csv(out_dir / "sweep_results.csv", index=False, encoding="utf-8-sig")
    save_eval(out_dir, "base_classifier", y_true, base_pred, labels)
    save_eval(out_dir, "prototype_raw_argmax", y_true, proto_pred, labels)
    if "pred" in best:
        save_eval(out_dir, "best_blend", y_true, best["pred"], labels)
        np.save(out_dir / "best_pred.npy", best["pred"].astype(str))

    source_rows = []
    source_rows.extend(source_metrics(sample_ids, y_true, base_pred, labels))
    if "pred" in best:
        best_sources = source_metrics(sample_ids, y_true, best["pred"], labels)
        for row in best_sources:
            row["name"] = "best_" + row["name"]
        source_rows.extend(best_sources)
    pd.DataFrame(source_rows).to_csv(out_dir / "source_metrics.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data_dir": args.data_dir,
        "oof_dir": args.oof_dir,
        "oof_proba_file": str(proba_path),
        "model_dir": args.model_dir,
        "query_builder": args.query_builder,
        "rows": int(len(sample_ids)),
        "classes": labels.tolist(),
        "base": {key: value for key, value in base_row.items() if key != "name"},
        "prototype_raw": {key: value for key, value in proto_row.items() if key != "name"},
        "best": {key: value for key, value in best.items() if key not in {"pred", "proto_norm"}},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with open(out_dir / "prototypes.json", "w", encoding="utf-8") as f:
        json.dump({label: ACTION_PROTOTYPES[label] for label in labels}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(results.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
