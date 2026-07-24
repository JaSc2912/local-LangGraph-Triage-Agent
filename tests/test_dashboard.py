from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from insurance_triage.dashboard import comparison_frame, results_frame
from insurance_triage.schemas import (
    NextAction,
    ProcessingStatus,
    Topic,
    TriageResult,
    Urgency,
)


def test_results_frame_serializes_dashboard_rows() -> None:
    result = TriageResult(
        ticket_id="42",
        text_snippet="Portal login fails",
        topic=Topic.TECHNICAL,
        topic_confidence=0.9,
        urgency=Urgency.MEDIUM,
        urgency_confidence=0.8,
        next_action=NextAction.TECHNICAL,
        needs_more_information=False,
        model_profile="compact",
        model_name="test-model",
        processing_status=ProcessingStatus.SUCCESS,
        latency_ms=123,
    )

    frame = results_frame([result])

    assert frame.iloc[0]["topic"] == Topic.TECHNICAL.value
    assert frame.iloc[0]["latency_ms"] == 123


def test_comparison_frame_calculates_rates() -> None:
    frame = comparison_frame(
        {
            "profiles": {
                "quality": {
                    "model": "large",
                    "ticket_count": 20,
                    "error_count": 1,
                    "reviewed_count": 5,
                    "needs_more_information_count": 10,
                    "latency_ms": {"mean": 5000, "p95": 7000},
                }
            }
        }
    )

    row = frame.iloc[0]
    assert row["review_rate"] == 0.25
    assert row["missing_info_rate"] == 0.5
    assert row["p95_latency_ms"] == 7000


def test_dashboard_renders_without_runtime_errors() -> None:
    dashboard_path = Path(__file__).parents[1] / "src" / "insurance_triage" / "dashboard.py"

    app = AppTest.from_file(str(dashboard_path)).run(timeout=15)

    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        "Live-Triage",
        "Quality-Sample",
        "Profilvergleich",
    ]
