#!/usr/bin/env python3
"""Fail on private paths, secrets, large artifacts, or malformed JSON."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 20 * 1024 * 1024
IGNORED_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
BINARY_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
    ".safetensors",
    ".onnx",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".parquet",
    ".zip",
    ".7z",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".csv",
    ".ps1",
    ".sh",
    ".gitignore",
}

PRIVATE_PATH = re.compile(
    r"(?:[A-Za-z]:[\\/](?:Users|home)[\\/][^\\/\s]+|/home/[^/\s]+|/Users/[^/\s]+)",
    re.IGNORECASE,
)
GENERIC_DRIVE_PATH = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/]")
TOKEN_PATTERNS = [
    re.compile("gh" + r"[opsu]_[A-Za-z0-9]{20,}"),
    re.compile("sk" + r"-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]


def tracked_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in IGNORED_DIRS for part in path.parts):
            continue
        files.append(path)
    return files


def main() -> None:
    problems: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"large file ({size} bytes): {relative}")
        if path.suffix.lower() in BINARY_SUFFIXES:
            problems.append(f"forbidden artifact: {relative}")

        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                problems.append(f"invalid JSON: {relative}: {exc}")

        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {
            ".gitignore",
            "LICENSE",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"non-UTF8 text: {relative}")
            continue

        # The audit source necessarily contains the detection regex itself.
        if path.resolve() != Path(__file__).resolve() and PRIVATE_PATH.search(text):
            problems.append(f"personal absolute path: {relative}")
        if path.resolve() != Path(__file__).resolve() and GENERIC_DRIVE_PATH.search(text):
            problems.append(f"machine-specific drive path: {relative}")
        for pattern in TOKEN_PATTERNS:
            if pattern.search(text):
                problems.append(f"possible secret token: {relative}")

    if problems:
        print("Repository audit failed:")
        for problem in sorted(set(problems)):
            print(f"- {problem}")
        raise SystemExit(1)

    print(f"PASS: audited {len(tracked_files())} files; no private or oversized artifacts found")


if __name__ == "__main__":
    main()
