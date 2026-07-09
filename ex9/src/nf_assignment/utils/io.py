"""Config, JSONL, manifest, and path utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a dictionary."""

    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write JSON with stable formatting."""

    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries as newline-delimited JSON."""

    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            json.dump(row, handle, sort_keys=True)
            handle.write("\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV file into dictionaries."""

    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]
) -> None:
    """Write a list of dictionaries as CSV."""

    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
