#!/usr/bin/env python3
"""Audit id/session overlap between train and validation fold manifests."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


STEP_RE = re.compile(r"-step_\d+$")


def root_session(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def read_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [str(json.loads(line)["id"]) for line in handle if line.strip()]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain an id column")
        return [str(row["id"]) for row in reader]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--valid", type=Path, required=True)
    args = parser.parse_args()

    train_ids = set(read_ids(args.train))
    valid_ids = set(read_ids(args.valid))
    train_sessions = {root_session(value) for value in train_ids}
    valid_sessions = {root_session(value) for value in valid_ids}

    report = {
        "train_rows": len(train_ids),
        "valid_rows": len(valid_ids),
        "id_overlap": len(train_ids & valid_ids),
        "session_overlap": len(train_sessions & valid_sessions),
    }
    print(json.dumps(report, indent=2))
    if report["id_overlap"] or report["session_overlap"]:
        raise SystemExit("leakage detected")


if __name__ == "__main__":
    main()
