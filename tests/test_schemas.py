from __future__ import annotations

import pytest
from pydantic import ValidationError

from insurance_triage.schemas import (
    MissingInfoAssessment,
    ReviewAssessment,
    Topic,
    TopicAssessment,
    Urgency,
    UrgencyAssessment,
)


@pytest.mark.parametrize(
    ("schema", "field"),
    [(TopicAssessment, "confidence"), (UrgencyAssessment, "confidence")],
)
def test_percentage_confidence_is_normalized(schema, field: str) -> None:
    values = {field: 85}
    if schema is TopicAssessment:
        values["topic"] = Topic.POLICY
    else:
        values["urgency"] = Urgency.LOW

    assessment = schema.model_validate(values)

    assert assessment.confidence == 0.85


def test_review_percentage_confidences_are_normalized() -> None:
    assessment = ReviewAssessment(
        topic=Topic.POLICY,
        topic_confidence=75,
        urgency=Urgency.LOW,
        urgency_confidence=90,
        needs_more_information=False,
        missing_information=[],
        clarification_questions=[],
    )

    assert assessment.topic_confidence == 0.75
    assert assessment.urgency_confidence == 0.9


@pytest.mark.parametrize("schema", [MissingInfoAssessment, ReviewAssessment])
def test_missing_information_requires_a_concrete_question(schema) -> None:
    values = {
        "needs_more_information": True,
        "missing_information": ["policy number"],
        "clarification_questions": [],
    }
    if schema is ReviewAssessment:
        values.update(
            topic=Topic.POLICY,
            topic_confidence=0.8,
            urgency=Urgency.LOW,
            urgency_confidence=0.8,
        )

    with pytest.raises(ValidationError, match="clarification_questions"):
        schema.model_validate(values)


@pytest.mark.parametrize("schema", [MissingInfoAssessment, ReviewAssessment])
def test_complete_assessment_rejects_orphaned_missing_information(schema) -> None:
    values = {
        "needs_more_information": False,
        "missing_information": ["policy number"],
        "clarification_questions": [],
    }
    if schema is ReviewAssessment:
        values.update(
            topic=Topic.POLICY,
            topic_confidence=0.8,
            urgency=Urgency.LOW,
            urgency_confidence=0.8,
        )

    with pytest.raises(ValidationError, match="must be empty"):
        schema.model_validate(values)
