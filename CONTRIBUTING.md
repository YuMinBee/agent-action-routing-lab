# Contributing

Small, evidence-backed improvements are welcome.

Before opening a pull request:

1. Do not add competition data, generated prompts, model weights, credentials, or personal paths.
2. Use root-session grouped validation for any reported score.
3. State whether validation rows were augmented.
4. Report mean, standard deviation, and per-fold results rather than one selected fold.
5. Run `python scripts/audit_repository.py` and the unit tests.

For experiment reports, include the baseline, one changed variable, OOF artifact alignment method, runtime, and inference-cost impact.
