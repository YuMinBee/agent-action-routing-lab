import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_oof import (  # noqa: E402
    MODEL_PRESETS,
    as_probability,
    build_fold_matrix,
    build_texts,
    load_dataset,
    make_classifier,
    make_splitter,
    save_oof_contract,
    score_proba,
    write_metrics,
)

try:
    from text_rule_router import apply_rule_gate, route_record
except Exception:  # pragma: no cover - router is optional for this experiment
    apply_rule_gate = None
    route_record = None


GROUP_TO_LABELS = {
    "inspect": ["read_file", "grep_search", "glob_pattern", "list_directory"],
    "modify": ["edit_file", "apply_patch", "write_file"],
    "validate": ["run_tests", "lint_or_typecheck", "run_bash"],
    "reason": ["plan_task", "ask_user", "respond_only", "web_search"],
}

LABEL_TO_GROUP = {
    label: group
    for group, labels in GROUP_TO_LABELS.items()
    for label in labels
}

GROUPS = np.asarray(list(GROUP_TO_LABELS), dtype=object)


def labels_to_groups(labels):
    return np.asarray([LABEL_TO_GROUP[label] for label in labels], dtype=object)


def normalize_proba(proba):
    proba = np.clip(proba, 1e-12, None)
    return proba / proba.sum(axis=1, keepdims=True)


def fit_group_clfs(x_train, y_train, args):
    group_clf = make_classifier(args)
    y_group = labels_to_groups(y_train)
    group_clf.fit(x_train, y_group)

    action_clfs = {}
    for group, labels in GROUP_TO_LABELS.items():
        mask = np.isin(y_train, labels)
        clf = make_classifier(args)
        clf.fit(x_train[mask], y_train[mask])
        action_clfs[group] = clf
    return group_clf, action_clfs


def hierarchical_probability(group_clf, action_clfs, x_valid, classes):
    group_proba_raw = group_clf.predict_proba(x_valid)
    group_proba = np.zeros((x_valid.shape[0], len(GROUPS)), dtype=np.float32)
    group_to_idx = {group: idx for idx, group in enumerate(GROUPS)}
    for source_idx, group in enumerate(group_clf.classes_):
        group_proba[:, group_to_idx[group]] = group_proba_raw[:, source_idx]

    out = np.zeros((x_valid.shape[0], len(classes)), dtype=np.float32)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for group, clf in action_clfs.items():
        raw = clf.predict_proba(x_valid)
        g_prob = group_proba[:, group_to_idx[group]][:, None]
        for source_idx, label in enumerate(clf.classes_):
            if label in class_to_idx:
                out[:, class_to_idx[label]] = g_prob[:, 0] * raw[:, source_idx]
    return normalize_proba(out), group_proba


def write_group_metrics(out_dir, y_true, group_pred):
    y_group = labels_to_groups(y_true)
    report = classification_report(y_group, group_pred, labels=GROUPS, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out_dir / "group_stage_class_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(y_group, group_pred, labels=GROUPS)
    pd.DataFrame(cm, index=GROUPS, columns=GROUPS).to_csv(
        out_dir / "group_stage_confusion_matrix.csv",
        encoding="utf-8-sig",
    )
    return {
        "macro_f1": float(f1_score(y_group, group_pred, average="macro")),
        "accuracy": float(accuracy_score(y_group, group_pred)),
    }


def evaluate_router_gate(name, proba, rows, y_true, classes):
    if apply_rule_gate is None or route_record is None:
        return []

    out = []
    for outw in [1.0, 0.8, 0.5, 0.3, 0.15]:
        gated = np.zeros_like(proba)
        for i, row in enumerate(rows):
            route = route_record(row)
            p_dict = {label: float(proba[i, j]) for j, label in enumerate(classes)}
            g_dict = apply_rule_gate(
                p_dict,
                route,
                hard_override=True,
                out_of_candidate_weight=outw,
            )
            gated[i] = [g_dict[label] for label in classes]
        gated = normalize_proba(gated)
        pred = classes[gated.argmax(axis=1)]
        out.append(
            {
                "name": f"{name}_router_outw_{outw}",
                "macro_f1": float(f1_score(y_true, pred, average="macro")),
                "accuracy": float(accuracy_score(y_true, pred)),
            }
        )
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="open/data")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", default="story_state_transition", choices=list(MODEL_PRESETS))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--splitter", choices=["group", "stratified_group"], default="group")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--word-max-features", type=int, default=220000)
    parser.add_argument("--char-max-features", type=int, default=80000)
    parser.add_argument("--word-min-df", type=int, default=2)
    parser.add_argument("--char-min-df", type=int, default=3)
    parser.add_argument("--jm-word-max-features", type=int, default=60000)
    parser.add_argument("--jm-char-max-features", type=int, default=30000)
    parser.add_argument("--jm-word-min-df", type=int, default=2)
    parser.add_argument("--jm-char-min-df", type=int, default=3)
    parser.add_argument("--classifier", choices=["logreg", "sgd"], default="logreg")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--logreg-c", type=float, default=4.0)
    parser.add_argument("--logreg-solver", default="saga")
    parser.add_argument("--loss", default="log_loss")
    parser.add_argument("--alpha", type=float, default=3e-6)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"experiments/oof/hierarchical_{args.model}_{stamp}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.data_dir, max_sessions=args.max_sessions)
    split_list = list(make_splitter(args, dataset))
    save_oof_contract(out_dir, dataset, split_list, args)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    spec = MODEL_PRESETS[args.model]
    all_texts = None
    if spec["text"]:
        print(f"build {spec['text']} texts...", flush=True)
        all_texts = np.asarray(build_texts(dataset.rows, spec["text"]), dtype=object)

    n = len(dataset.rows)
    flat_proba = np.zeros((n, len(dataset.classes)), dtype=np.float32)
    hier_proba = np.zeros((n, len(dataset.classes)), dtype=np.float32)
    group_oof_proba = np.zeros((n, len(GROUPS)), dtype=np.float32)
    group_oof_pred = np.empty(n, dtype=object)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(split_list, start=1):
        t0 = time.time()
        train_rows = [dataset.rows[i] for i in train_idx]
        valid_rows = [dataset.rows[i] for i in valid_idx]
        y_train = dataset.y[train_idx]
        y_valid = dataset.y[valid_idx]
        train_texts = all_texts[train_idx] if all_texts is not None else None
        valid_texts = all_texts[valid_idx] if all_texts is not None else None

        x_train, x_valid = build_fold_matrix(
            args.model,
            train_rows,
            valid_rows,
            y_train,
            dataset.classes,
            args,
            train_texts=train_texts,
            valid_texts=valid_texts,
        )

        flat_clf = make_classifier(args)
        flat_clf.fit(x_train, y_train)
        fold_flat = as_probability(flat_clf, x_valid, dataset.classes)

        group_clf, action_clfs = fit_group_clfs(x_train, y_train, args)
        fold_hier, fold_group = hierarchical_probability(group_clf, action_clfs, x_valid, dataset.classes)

        flat_proba[valid_idx] = fold_flat
        hier_proba[valid_idx] = fold_hier
        group_oof_proba[valid_idx] = fold_group
        group_oof_pred[valid_idx] = GROUPS[fold_group.argmax(axis=1)]

        flat_pred = dataset.classes[fold_flat.argmax(axis=1)]
        hier_pred = dataset.classes[fold_hier.argmax(axis=1)]
        group_pred = GROUPS[fold_group.argmax(axis=1)]
        y_group = labels_to_groups(y_valid)
        metrics = {
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "n_features": int(x_train.shape[1]),
            "flat_macro_f1": float(f1_score(y_valid, flat_pred, average="macro")),
            "flat_accuracy": float(accuracy_score(y_valid, flat_pred)),
            "hier_macro_f1": float(f1_score(y_valid, hier_pred, average="macro")),
            "hier_accuracy": float(accuracy_score(y_valid, hier_pred)),
            "group_macro_f1": float(f1_score(y_group, group_pred, average="macro")),
            "group_accuracy": float(accuracy_score(y_group, group_pred)),
            "seconds": round(time.time() - t0, 2),
        }
        fold_rows.append(metrics)
        print(
            f"fold={fold} flat={metrics['flat_macro_f1']:.6f} "
            f"hier={metrics['hier_macro_f1']:.6f} "
            f"group={metrics['group_macro_f1']:.6f} "
            f"features={metrics['n_features']} sec={metrics['seconds']}",
            flush=True,
        )

    np.save(out_dir / "oof_flat.npy", flat_proba)
    np.save(out_dir / "oof_hierarchical.npy", hier_proba)
    np.save(out_dir / "oof_group.npy", group_oof_proba)
    pd.DataFrame(fold_rows).to_csv(out_dir / "hierarchical_folds.csv", index=False)

    flat_pred = dataset.classes[flat_proba.argmax(axis=1)]
    hier_pred = dataset.classes[hier_proba.argmax(axis=1)]
    flat_metrics = write_metrics(out_dir, "flat", dataset.y, flat_pred, dataset.classes)
    hier_metrics = write_metrics(out_dir, "hierarchical", dataset.y, hier_pred, dataset.classes)
    group_metrics = write_group_metrics(out_dir, dataset.y, group_oof_pred)

    compare_rows = [
        {
            "name": "flat",
            "macro_f1": flat_metrics["macro_f1"],
            "accuracy": flat_metrics["accuracy"],
        },
        {
            "name": "hierarchical",
            "macro_f1": hier_metrics["macro_f1"],
            "accuracy": hier_metrics["accuracy"],
        },
    ]
    compare_rows.extend(evaluate_router_gate("flat", flat_proba, dataset.rows, dataset.y, dataset.classes))
    compare_rows.extend(evaluate_router_gate("hierarchical", hier_proba, dataset.rows, dataset.y, dataset.classes))
    compare_df = pd.DataFrame(compare_rows).sort_values("macro_f1", ascending=False)
    compare_df.to_csv(out_dir / "hierarchical_compare.csv", index=False)

    summary = {
        "flat": flat_metrics,
        "hierarchical": hier_metrics,
        "group_stage": group_metrics,
        "compare": compare_df.to_dict(orient="records"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(compare_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
