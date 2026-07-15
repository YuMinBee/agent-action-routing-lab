from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


LABELS = [
    "apply_patch", "ask_user", "edit_file", "glob_pattern", "grep_search",
    "lint_or_typecheck", "list_directory", "plan_task", "read_file",
    "respond_only", "run_bash", "run_tests", "web_search", "write_file",
]
BUNDLE = Path("artifacts/repro_bundle")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def load_complete_oof(root: Path, kind: str):
    proba_by_id = {}
    true_by_id = {}
    for fold in range(5):
        if kind == "sim_a":
            ids = np.load(root / f"oof_fold{fold}_ids.npy", allow_pickle=True).astype(str)
            proba = np.load(root / f"oof_fold{fold}_proba.npy").astype(np.float64)
            truth = np.load(root / f"oof_fold{fold}_true.npy").astype(int)
            true_by_id.update({sid: int(y) for sid, y in zip(ids, truth)})
        else:
            fold_dir = root / f"fold{fold}"
            ids = np.load(fold_dir / "valid_sample_ids.npy", allow_pickle=True).astype(str)
            proba = np.load(fold_dir / "valid_proba.npy").astype(np.float64)
        proba_by_id.update({sid: row for sid, row in zip(ids, proba)})
    return proba_by_id, true_by_id


def load_folds():
    oof_root = BUNDLE / "oof"
    sim_a, truth = load_complete_oof(oof_root / "sim_pi_exchange_5fold", "sim_a")
    sim_b, _ = load_complete_oof(oof_root / "simB_5fold", "sim_b")
    bias = np.asarray(json.loads(
        (BUNDLE / "postprocess/output/class_bias.json").read_text(encoding="utf-8")
    )["bias_sim"], dtype=np.float64)
    label_to_id = {label: i for i, label in enumerate(LABELS)}
    folds = []
    for fold in (0, 1):
        table = pd.read_csv(oof_root / "klue_v9_fold01" / f"fold{fold}_oof_predictions.csv")
        ids = table["id"].astype(str).to_numpy()
        a = np.stack([sim_a[sid] for sid in ids])
        b = np.stack([sim_b[sid] for sid in ids])
        y = np.asarray([truth[sid] for sid in ids], dtype=np.int64)
        expected = np.asarray([label_to_id[x] for x in table["true_action"]], dtype=np.int64)
        assert np.array_equal(y, expected)
        klue = softmax(table[[f"logit_{x}" for x in LABELS]].to_numpy(dtype=np.float64))
        raw = 0.60 * a + 0.40 * b
        raw /= raw.sum(axis=1, keepdims=True)
        biased = softmax(np.log(np.clip(raw, 1e-9, 1.0)) + bias)
        folds.append({"a": a, "b": b, "k": klue, "raw": raw, "y": y, "base": biased.argmax(1), "bias": bias})
    return folds


def score_folds(folds, predictions):
    base_scores = [f1_score(f["y"], f["base"], average="macro") for f in folds]
    scores = [f1_score(f["y"], p, average="macro") for f, p in zip(folds, predictions)]
    joined_base = f1_score(
        np.concatenate([f["y"] for f in folds]),
        np.concatenate([f["base"] for f in folds]),
        average="macro",
    )
    joined = f1_score(
        np.concatenate([f["y"] for f in folds]), np.concatenate(predictions), average="macro"
    )
    return joined, joined - joined_base, [s - b for s, b in zip(scores, base_scores)]


def main():
    folds = load_folds()
    rows = []
    margins = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20)
    betas = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0, 1.5, 2.0)
    for mode in ("pair_prob", "pair_log", "soft_all_prob", "soft_all_log"):
        for margin_limit in margins:
            for beta in betas:
                predictions = []
                changed = 0
                for f in folds:
                    a, b, k, raw, base = f["a"], f["b"], f["k"], f["raw"], f["base"]
                    pred_a, pred_b = a.argmax(1), b.argmax(1)
                    top2 = np.sort(np.partition(raw, -2, axis=1)[:, -2:], axis=1)
                    candidate = (pred_a != pred_b) & ((top2[:, 1] - top2[:, 0]) < margin_limit)
                    pred = base.copy()
                    idx = np.flatnonzero(candidate)
                    if mode.startswith("pair"):
                        cand_a, cand_b = pred_a[idx], pred_b[idx]
                        if mode == "pair_prob":
                            score_a = raw[idx, cand_a] + beta * k[idx, cand_a]
                            score_b = raw[idx, cand_b] + beta * k[idx, cand_b]
                        else:
                            score_a = np.log(np.clip(raw[idx, cand_a], 1e-9, 1.0)) + beta * np.log(np.clip(k[idx, cand_a], 1e-9, 1.0))
                            score_b = np.log(np.clip(raw[idx, cand_b], 1e-9, 1.0)) + beta * np.log(np.clip(k[idx, cand_b], 1e-9, 1.0))
                        pred[idx] = np.where(score_a >= score_b, cand_a, cand_b)
                    else:
                        if mode == "soft_all_prob":
                            fused = raw[idx] + beta * k[idx]
                            fused /= fused.sum(axis=1, keepdims=True)
                            fused = softmax(np.log(np.clip(fused, 1e-9, 1.0)) + f["bias"])
                        else:
                            fused = softmax(np.log(np.clip(raw[idx], 1e-9, 1.0)) + beta * np.log(np.clip(k[idx], 1e-9, 1.0)) + f["bias"])
                        pred[idx] = fused.argmax(1)
                    changed += int((pred != base).sum())
                    predictions.append(pred)
                score, delta, fold_delta = score_folds(folds, predictions)
                rows.append({
                    "mode": mode, "margin": margin_limit, "beta": beta,
                    "changed": changed, "score": score, "delta": delta,
                    "fold0_delta": fold_delta[0], "fold1_delta": fold_delta[1],
                    "both_positive": fold_delta[0] > 0 and fold_delta[1] > 0,
                })

    output = pd.DataFrame(rows).sort_values(["both_positive", "delta"], ascending=[False, False])
    out_dir = Path(__file__).resolve().parent / "klue_pairwise_ensemble_20260713"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_dir / "sweep.csv", index=False)
    print(output.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
