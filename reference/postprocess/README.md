# Postprocessing Reference

- `fit_class_bias.py` fits domain-specific additive offsets in log-probability space using OOF predictions.
- `sweep_klue_pairwise.py` tests conservative KLUE tie-break gates.
- `klue_tiebreak_oof_sweep.csv` preserves the local sweep table used to choose the final margin/confidence pair.

Never fit these parameters on in-sample full-train predictions. Every probability used for fitting must be OOF-aligned by sample id.
