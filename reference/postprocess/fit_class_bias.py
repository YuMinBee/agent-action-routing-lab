from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


LABELS = np.asarray(
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


def normalize_truth(values):
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    mapping = {label: idx for idx, label in enumerate(LABELS)}
    return np.asarray([mapping[str(value)] for value in values], dtype=np.int64)


def log_proba(proba):
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    return np.log(np.clip(proba, 1e-9, 1.0))


def macro_f1_fast(y, pred, n_classes=14):
    encoded = y * n_classes + pred
    cm = np.bincount(encoded, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    scores = np.divide(2.0 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(scores.mean())


def fit_bias(proba, y, rounds=3, seed=42):
    lp = log_proba(proba)
    bias = np.zeros(len(LABELS), dtype=np.float64)
    best = macro_f1_fast(y, lp.argmax(axis=1))
    grids = [
        np.linspace(-0.60, 0.60, 13),
        np.linspace(-0.20, 0.20, 9),
        np.linspace(-0.08, 0.08, 9),
    ]
    rng = np.random.default_rng(seed)
    for grid in grids:
        for _ in range(rounds):
            improved = False
            for class_idx in rng.permutation(len(LABELS)):
                current = bias[class_idx]
                best_value = current
                best_score = best
                for delta in grid:
                    bias[class_idx] = current + float(delta)
                    score = macro_f1_fast(y, (lp + bias).argmax(axis=1))
                    if score > best_score + 1e-8:
                        best_score = score
                        best_value = bias[class_idx]
                bias[class_idx] = best_value
                if best_score > best + 1e-9:
                    best = best_score
                    improved = True
            if not improved:
                break
    bias -= bias.mean()
    return bias, best


def read_zip_npy(archive, member):
    return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)


def load_exact_stack(args):
    anchor_ids = np.load(args.anchor_dir / "ids.npy", allow_pickle=True).astype(str)
    y_true = normalize_truth(np.load(args.anchor_dir / "y_true.npy", allow_pickle=True))
    proba = np.load(args.anchor_dir / "p_cls_oof_787858.npy").astype(np.float32)
    folds = np.full(len(anchor_ids), -1, dtype=np.int64)
    index = {sample_id: idx for idx, sample_id in enumerate(anchor_ids)}

    for fold in range(5):
        fold_ids = np.load(args.sim_fold_dir / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
        for sample_id in fold_ids:
            folds[index[sample_id]] = fold

    para_by_id = {}
    plus_by_id = {}
    true_by_id = {}
    fold_by_id = {}
    with zipfile.ZipFile(args.au_plus_zip) as archive:
        for fold in range(5):
            para_ids = np.load(args.au_para_dir / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
            para_true = normalize_truth(
                np.load(args.au_para_dir / f"oof_fold{fold}_true.npy", allow_pickle=True)
            )
            para_proba = np.load(args.au_para_dir / f"oof_fold{fold}_proba.npy").astype(np.float32)
            prefix = args.au_plus_prefix
            plus_ids = read_zip_npy(archive, f"{prefix}/oof_fold{fold}_ids.npy").astype(str)
            plus_true = normalize_truth(read_zip_npy(archive, f"{prefix}/oof_fold{fold}_true.npy"))
            plus_proba = read_zip_npy(archive, f"{prefix}/oof_fold{fold}_proba.npy").astype(np.float32)
            plus_index = {sample_id: idx for idx, sample_id in enumerate(plus_ids)}
            for row_idx, sample_id in enumerate(para_ids):
                other = plus_index[sample_id]
                if para_true[row_idx] != plus_true[other]:
                    raise RuntimeError(f"AU truth mismatch for {sample_id}")
                para_by_id[sample_id] = para_proba[row_idx]
                plus_by_id[sample_id] = plus_proba[other]
                true_by_id[sample_id] = int(para_true[row_idx])
                fold_by_id[sample_id] = fold

    au_mask = np.char.startswith(anchor_ids, "sess_au")
    for row_idx in np.where(au_mask)[0]:
        sample_id = anchor_ids[row_idx]
        proba[row_idx] = 0.5 * para_by_id[sample_id] + 0.5 * plus_by_id[sample_id]
        folds[row_idx] = fold_by_id[sample_id]
        if y_true[row_idx] != true_by_id[sample_id]:
            raise RuntimeError(f"anchor truth mismatch for {sample_id}")
    if (folds < 0).any():
        missing = anchor_ids[folds < 0][:10].tolist()
        raise RuntimeError(f"missing fold assignments: {missing}")
    return anchor_ids, y_true, proba, folds


def crossfit_domain(proba, y, folds, domain, seed):
    baseline_pred = proba.argmax(axis=1)
    baseline = f1_score(y, baseline_pred, average="macro")
    raw_biases = []
    for fold in range(5):
        train = folds != fold
        bias, _ = fit_bias(proba[train], y[train], seed=seed + fold)
        raw_biases.append(bias)

    rows = []
    for shrink in [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]:
        pred = baseline_pred.copy()
        fold_deltas = []
        for fold in range(5):
            valid = folds == fold
            adjusted = log_proba(proba[valid]) + shrink * raw_biases[fold]
            pred[valid] = adjusted.argmax(axis=1)
            base_fold = f1_score(y[valid], baseline_pred[valid], average="macro")
            new_fold = f1_score(y[valid], pred[valid], average="macro")
            fold_deltas.append(new_fold - base_fold)
        score = f1_score(y, pred, average="macro")
        delta = np.asarray(fold_deltas)
        rows.append(
            {
                "domain": domain,
                "shrink": shrink,
                "macro_f1": score,
                "delta_macro_f1": score - baseline,
                "accuracy": accuracy_score(y, pred),
                "changed": int((pred != baseline_pred).sum()),
                "positive_folds": int((delta > 0).sum()),
                "mean_fold_delta": float(delta.mean()),
                "min_fold_delta": float(delta.min()),
                "fold_deltas": json.dumps([round(float(value), 8) for value in delta]),
            }
        )
    sweep = pd.DataFrame(rows).sort_values(
        ["positive_folds", "mean_fold_delta", "min_fold_delta"],
        ascending=[False, False, False],
    )
    eligible = sweep[
        (sweep["delta_macro_f1"] > 0)
        & (sweep["positive_folds"] >= 4)
        & (sweep["min_fold_delta"] >= -0.0005)
    ]
    selected = eligible.iloc[0] if not eligible.empty else sweep[sweep["shrink"] == 0.0].iloc[0]
    full_bias, in_sample = fit_bias(proba, y, seed=seed + 100)
    return sweep, selected.to_dict(), full_bias, baseline, in_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--anchor-dir",
        type=Path,
        default=Path("artifacts/oof/anchor"),
    )
    parser.add_argument(
        "--sim-fold-dir",
        type=Path,
        default=Path("artifacts/oof/sim_a"),
    )
    parser.add_argument(
        "--au-para-dir",
        type=Path,
        default=Path("artifacts/oof/au_paraphrase"),
    )
    parser.add_argument(
        "--au-plus-zip",
        type=Path,
        default=Path("artifacts/oof/au_plus_mined.zip"),
    )
    parser.add_argument(
        "--au-plus-prefix",
        default=(
            "au_para_plus_seed42_cv_20260705_oldmined229_e4_package_0709/"
            "cv_results/au_cv_para_plus_oldmined229_5fold_e8_seed42_len448"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/postprocess/class_bias"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ids, y, proba, folds = load_exact_stack(args)
    au_mask = np.char.startswith(ids, "sess_au")
    sim_mask = ~au_mask
    exact = {
        "all": f1_score(y, proba.argmax(axis=1), average="macro"),
        "sim": f1_score(y[sim_mask], proba[sim_mask].argmax(axis=1), average="macro"),
        "au": f1_score(y[au_mask], proba[au_mask].argmax(axis=1), average="macro"),
    }

    domain_results = {}
    all_sweeps = []
    scaled_bias = {}
    for domain, mask in [("sim", sim_mask), ("au", au_mask)]:
        sweep, selected, full_bias, baseline, in_sample = crossfit_domain(
            proba[mask], y[mask], folds[mask], domain, args.seed
        )
        all_sweeps.append(sweep)
        shrink = float(selected["shrink"])
        scaled_bias[domain] = full_bias * shrink
        domain_results[domain] = {
            "baseline_macro_f1": baseline,
            "selected": selected,
            "full_fit_unshrunk_macro_f1": in_sample,
            "unscaled_bias": {label: float(value) for label, value in zip(LABELS, full_bias)},
            "scaled_bias": {
                label: float(value) for label, value in zip(LABELS, scaled_bias[domain])
            },
        }

    adjusted_pred = np.empty(len(y), dtype=np.int64)
    adjusted_pred[sim_mask] = (
        log_proba(proba[sim_mask]) + scaled_bias["sim"]
    ).argmax(axis=1)
    adjusted_pred[au_mask] = (
        log_proba(proba[au_mask]) + scaled_bias["au"]
    ).argmax(axis=1)
    deployment_oof = {
        "macro_f1": f1_score(y, adjusted_pred, average="macro"),
        "delta_macro_f1": f1_score(y, adjusted_pred, average="macro") - exact["all"],
        "accuracy": accuracy_score(y, adjusted_pred),
        "changed": int((adjusted_pred != proba.argmax(axis=1)).sum()),
    }
    config = {
        "labels": LABELS.tolist(),
        "bias_sim": scaled_bias["sim"].tolist(),
        "bias_au": scaled_bias["au"].tolist(),
        "space": "log_probability",
        "fit": "domain-specific 5-fold crossfit shrink selection, full OOF final bias",
    }
    (args.output_dir / "class_bias.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(all_sweeps, ignore_index=True).to_csv(
        args.output_dir / "shrink_sweep.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "rows": len(ids),
        "exact_anchor": exact,
        "domains": domain_results,
        "deployment_oof_diagnostic": deployment_oof,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nShrink sweep")
    print(pd.concat(all_sweeps, ignore_index=True).to_string(index=False))


if __name__ == "__main__":
    main()
