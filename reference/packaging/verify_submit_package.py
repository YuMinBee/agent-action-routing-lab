import subprocess
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoConfig, AutoTokenizer


APP = Path("/app")


def require_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    print(f"OK file {path}")


def main():
    required = [
        APP / "script.py",
        APP / "requirements.txt",
        APP / "model" / "sim" / "config.json",
        APP / "model" / "sim" / "model_int8.pt",
        APP / "model" / "sim" / "tokenizer.json",
        APP / "model" / "sim" / "tokenizer_config.json",
        APP / "model" / "au" / "config.json",
        APP / "model" / "au" / "model_int8.pt",
        APP / "model" / "au" / "tokenizer.json",
        APP / "model" / "au" / "tokenizer_config.json",
        APP / "data" / "test.jsonl",
        APP / "data" / "sample_submission.csv",
    ]
    for path in required:
        require_file(path)

    for name in ["sim", "au"]:
        model_dir = APP / "model" / name
        AutoConfig.from_pretrained(model_dir, local_files_only=True)
        AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=True)
        print(f"OK hf local load {model_dir}")

    subprocess.run([sys.executable, "script.py"], cwd=APP, check=True)

    sample = pd.read_csv(APP / "data" / "sample_submission.csv")
    sub = pd.read_csv(APP / "output" / "submission.csv")
    if list(sub.columns) != list(sample.columns):
        raise AssertionError((list(sub.columns), list(sample.columns)))
    if len(sub) != len(sample):
        raise AssertionError((len(sub), len(sample)))
    if sub["id"].tolist() != sample["id"].tolist():
        raise AssertionError("submission ids do not match sample")
    if sub["action"].isna().any():
        raise AssertionError("submission contains NA action")

    print(f"SUBMISSION_OK shape={sub.shape}")
    print(sub.head().to_string(index=False))
    print("FINAL_VERIFY_OK")


if __name__ == "__main__":
    main()
