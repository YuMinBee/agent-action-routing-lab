# Training Reference

These files are curated from the competition workspace rather than presented as a single turnkey command.

- `run_domain_input_global_oof.py`: shared E5 + 93 structured features, source-aware input and OOF training.
- `run_au_augmented_8_2.py`: AU GroupKFold/holdout training with train-only augmentation.
- `train_augmented_full.py`: full-data component training after epoch selection.
- `train_cv_pi_exchange_weighted.py`: weighted multi-view and auxiliary-data experiments used by a strong SIM branch.

Read each `--help` output before running. The original project evolved quickly, so component-specific epoch, max length, augmentation, and local checkpoint paths must be supplied explicitly.

The public contract is:

1. Keep root sessions intact across folds.
2. Augment only the training partition.
3. Record seed and input hashes.
4. Pair the resulting checkpoint with the same builder at inference.
