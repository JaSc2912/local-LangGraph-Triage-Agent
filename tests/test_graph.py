from __future__ import annotations

from conftest import FakeModelClient

from insurance_triage.config import Settings, resolve_profile
from insurance_triage.runner import BatchRunner
from insurance_triage.schemas import (
    MissingInfoAssessment,
    NextAction,
    ProcessingStatus,
    ReviewAssessment,
    Ticket,
    Topic,
    TopicAssessment,
    Urgency,
    UrgencyAssessment,
)


def _confident_client(
    *,
    topic: Topic = Topic.TECHNICAL,
    urgency: Urgency = Urgency.LOW,
    needs_info: bool = False,
) -> FakeModelClient:
    return FakeModelClient(
        {
            TopicAssessment: TopicAssessment(
                topic=topic,
                confidence=0.95,
                evidence="cannot log in",
            ),
            UrgencyAssessment: UrgencyAssessment(
                urgency=urgency,
                confidence=0.90,
                risk_signals=["explicit risk"] if urgency == Urgency.HIGH else [],
            ),
            MissingInfoAssessment: MissingInfoAssessment(
                needs_more_information=needs_info,
                missing_information=["error message"] if needs_info else [],
                clarification_questions=["What error appears?"] if needs_info else [],
            ),
        }
    )


def test_confident_technical_ticket_skips_reviewer() -> None:
    client = _confident_client()
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    result = runner.process_ticket(
        Ticket(ticket_id="1", subject="Login", body="I cannot log in to the online portal.")
    )

    assert result.next_action == NextAction.TECHNICAL
    assert result.reviewed_by_llm is False
    assert ReviewAssessment not in client.calls


def test_low_confidence_ticket_uses_semantic_reviewer() -> None:
    client = FakeModelClient(
        {
            TopicAssessment: TopicAssessment(
                topic=Topic.OTHER,
                confidence=0.4,
                evidence="damaged laptop",
            ),
            UrgencyAssessment: UrgencyAssessment(
                urgency=Urgency.MEDIUM,
                confidence=0.9,
            ),
            MissingInfoAssessment: MissingInfoAssessment(
                needs_more_information=True,
                missing_information=["incident date"],
                clarification_questions=["When did it happen?"],
                ticket_is_ambiguous=True,
            ),
            ReviewAssessment: ReviewAssessment(
                topic=Topic.CLAIMS,
                topic_confidence=0.9,
                urgency=Urgency.MEDIUM,
                urgency_confidence=0.85,
                needs_more_information=False,
                missing_information=[],
                clarification_questions=[],
                evidence="damaged laptop",
                notes="The message describes damage.",
            ),
        }
    )
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    result = runner.process_ticket(
        Ticket(ticket_id="2", subject="Damage", body="My laptop was damaged yesterday.")
    )

    assert result.topic == Topic.CLAIMS
    assert result.next_action == NextAction.CLAIM
    assert result.reviewed_by_llm is True
    assert ReviewAssessment in client.calls


def test_explicit_risk_term_can_only_promote_urgency() -> None:
    client = _confident_client(urgency=Urgency.LOW)
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    result = runner.process_ticket(
        Ticket(
            ticket_id="3",
            subject="Security",
            body="This appears to be an account takeover and I cannot access the portal.",
        )
    )

    assert result.urgency == Urgency.HIGH
    assert result.next_action == NextAction.ESCALATE


def test_high_urgency_precedes_missing_information() -> None:
    client = _confident_client(urgency=Urgency.HIGH, needs_info=True)
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    result = runner.process_ticket(
        Ticket(ticket_id="4", subject="Fraud", body="There is an active fraudulent payment.")
    )

    assert result.next_action == NextAction.ESCALATE


def test_empty_ticket_does_not_call_model() -> None:
    client = FakeModelClient({})
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    result = runner.process_ticket(Ticket(ticket_id="5"))

    assert result.processing_status == ProcessingStatus.INCOMPLETE
    assert result.next_action == NextAction.ASK_FOR_INFO
    assert client.calls == []


def test_model_failure_becomes_error_result_and_batch_continues() -> None:
    client = FakeModelClient({})
    runner = BatchRunner(resolve_profile("compact"), Settings(), client)

    results = runner.run_tickets(
        [
            Ticket(ticket_id="broken", body="A real ticket that needs model inference."),
            Ticket(ticket_id="empty"),
        ]
    )

    assert results[0].processing_status == ProcessingStatus.ERROR
    assert results[0].topic is None
    assert results[1].processing_status == ProcessingStatus.INCOMPLETE
