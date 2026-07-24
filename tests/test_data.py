from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from insurance_triage.data import (
    discover_dataset,
    load_tickets,
    normalize_ticket_text,
    write_results,
)
from insurance_triage.schemas import (
    NextAction,
    ProcessingStatus,
    Topic,
    TriageResult,
    Urgency,
)


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_normalize_ticket_text_preserves_meaning() -> None:
    text = normalize_ticket_text(
        " Login&nbsp;problem ", "<p>I   cannot log in.</p>\n\n\nPlease help."
    )

    assert text == "Login problem\n\nI cannot log in.\n\nPlease help."


def test_normalize_ticket_text_repairs_common_mojibake() -> None:
    text = normalize_ticket_text("Policy update â€“ urgent", "I canâ€™t open the document.")

    assert text == "Policy update – urgent\n\nI can't open the document."


def test_discovery_chooses_largest_compatible_csv(tmp_path: Path) -> None:
    small = tmp_path / "small.csv"
    large = tmp_path / "large.csv"
    invalid = tmp_path / "invalid.csv"
    rows = [{"Subject": "A", "Body": "B", "Language": "en"}]
    _write_dataset(small, rows)
    _write_dataset(large, rows * 3)
    _write_dataset(invalid, [{"message": "not compatible"}])

    assert discover_dataset(None, tmp_path) == large


def test_loader_filters_deduplicates_and_samples_deterministically(tmp_path: Path) -> None:
    dataset = tmp_path / "tickets.csv"
    _write_dataset(
        dataset,
        [
            {"id": 1, "subject": "A", "body": "First", "language": "en"},
            {"id": 2, "subject": "A", "body": "First", "language": "en"},
            {"id": 3, "subject": "B", "body": "Second", "language": "de"},
            {"id": 4, "subject": "C", "body": "Third", "language": "en"},
            {"id": 5, "subject": "D", "body": "Fourth", "language": "en"},
        ],
    )

    first = load_tickets(dataset, language="en", limit=2, seed=42)
    second = load_tickets(dataset, language="en", limit=2, seed=42)

    assert [ticket.ticket_id for ticket in first] == [ticket.ticket_id for ticket in second]
    assert len(first) == 2


def test_loader_rejects_missing_language(tmp_path: Path) -> None:
    dataset = tmp_path / "tickets.csv"
    _write_dataset(dataset, [{"subject": "A", "body": "B", "language": "de"}])

    with pytest.raises(ValueError, match="No tickets"):
        load_tickets(dataset, language="en", limit=1, seed=42)


def test_result_csv_serializes_enums_and_lists(tmp_path: Path) -> None:
    result = TriageResult(
        ticket_id="1",
        text_snippet="Payment failed",
        topic=Topic.BILLING,
        topic_confidence=0.9,
        urgency=Urgency.MEDIUM,
        urgency_confidence=0.8,
        next_action=NextAction.BILLING,
        needs_more_information=True,
        missing_information=["invoice number"],
        clarification_questions=["Which invoice is affected?"],
        model_profile="quality",
        model_name="model",
        processing_status=ProcessingStatus.SUCCESS,
        latency_ms=10,
    )

    output = write_results([result], tmp_path / "result.csv")
    row = pd.read_csv(output).iloc[0]

    assert row["topic"] == Topic.BILLING.value
    assert row["missing_information"] == "invoice number"
