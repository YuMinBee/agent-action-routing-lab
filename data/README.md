# Data Placement

Competition data are not distributed in this repository. Place authorized copies under `data/raw/`.

```text
data/
`-- raw/
    |-- train.jsonl
    |-- train_labels.csv
    |-- test.jsonl
    `-- sample_submission.csv
```

Generated assets should live under ignored directories such as:

```text
data/generated/mined_enriched.jsonl
data/generated/mined_enriched_labels.csv
data/generated/au_paraphrase.jsonl
data/generated/au_paraphrase_labels.csv
```

Never commit row-level competition data, prompts, session history, or generated derivatives unless the original license explicitly permits redistribution.
