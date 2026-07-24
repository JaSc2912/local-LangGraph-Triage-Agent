from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from ftfy import fix_text

from insurance_triage.schemas import Ticket, TriageResult

REQUIRED_COLUMNS = {"subject", "body", "language"}
ID_COLUMNS = ("ticket_id", "ticketid", "id")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")


def normalize_column_name(name: object) -> str:
    normalized = str(name).strip().casefold()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def normalize_ticket_text(subject: object, body: object) -> str:
    parts: list[str] = []
    for value in (subject, body):
        if value is None or pd.isna(value):
            continue
        text = fix_text(html.unescape(str(value)))
        text = HTML_TAG_RE.sub(" ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = HORIZONTAL_SPACE_RE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.splitlines())
        text = EXCESS_NEWLINES_RE.sub("\n\n", text)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _read_header(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, nrows=0, encoding="utf-8-sig")
    return {normalize_column_name(column): str(column) for column in frame.columns}


def validate_dataset(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    try:
        columns = _read_header(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        frame = pd.read_csv(path, nrows=0, encoding="latin-1")
        columns = {normalize_column_name(column): str(column) for column in frame.columns}
    missing = REQUIRED_COLUMNS - columns.keys()
    if missing:
        available = ", ".join(sorted(columns))
        required = ", ".join(sorted(missing))
        raise ValueError(
            f"Dataset '{path}' is missing required columns: {required}. "
            f"Available columns: {available}."
        )
    return columns


def discover_dataset(input_path: Path | None, input_dir: Path) -> Path:
    if input_path is not None:
        path = input_path.expanduser().resolve()
        validate_dataset(path)
        return path

    if not input_dir.exists():
        raise FileNotFoundError(
            f"No dataset directory found at '{input_dir}'. Download the Kaggle CSV into "
            "data/raw/ or pass --input."
        )

    candidates: list[Path] = []
    for path in input_dir.rglob("*.csv"):
        try:
            validate_dataset(path)
        except (FileNotFoundError, UnicodeDecodeError, ValueError, pd.errors.ParserError):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No compatible CSV found below '{input_dir}'. Expected subject, body, "
            "and language columns."
        )
    return max(candidates, key=lambda path: (path.stat().st_size, str(path)))


def _read_dataset(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"Could not decode dataset: {path}")


def load_tickets(
    path: Path,
    *,
    language: str,
    limit: int,
    seed: int,
) -> list[Ticket]:
    if limit < 1:
        raise ValueError("Limit must be at least 1.")

    validate_dataset(path)
    frame = _read_dataset(path)
    frame = frame.rename(columns={column: normalize_column_name(column) for column in frame})

    normalized_language = language.strip().casefold()
    frame["language"] = frame["language"].fillna("").astype(str).str.strip().str.casefold()
    frame = frame.loc[frame["language"] == normalized_language].copy()
    if frame.empty:
        raise ValueError(f"No tickets found for language '{language}'.")

    frame["subject"] = frame["subject"].fillna("").astype(str)
    frame["body"] = frame["body"].fillna("").astype(str)
    frame = frame.drop_duplicates(subset=["subject", "body"], keep="first")
    if len(frame) > limit:
        frame = frame.sample(n=limit, random_state=seed)
    frame = frame.sort_index()

    id_column = next((column for column in ID_COLUMNS if column in frame.columns), None)
    tickets: list[Ticket] = []
    for source_index, row in frame.iterrows():
        ticket_id = (
            str(row[id_column]) if id_column and pd.notna(row[id_column]) else str(source_index)
        )
        tickets.append(
            Ticket(
                ticket_id=ticket_id,
                subject=row["subject"],
                body=row["body"],
                language=row["language"],
                source_index=int(source_index) if isinstance(source_index, int) else None,
            )
        )
    return tickets


def write_results(results: Iterable[TriageResult], output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_csv_row() for result in results]
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False, encoding="utf-8")
    temporary_path.replace(output_path)
    return output_path


def write_json(data: object, output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path
