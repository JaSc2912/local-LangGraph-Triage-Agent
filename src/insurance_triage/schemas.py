from __future__ import annotations

from enum import StrEnum
from typing import Self, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _normalize_confidence(value: object) -> object:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 < value <= 100:
        return value / 100
    return value


class Topic(StrEnum):
    POLICY = "Policy / Contract"
    CLAIMS = "Claims / Damage"
    BILLING = "Billing / Payment"
    TECHNICAL = "Technical / Online Access"
    OTHER = "Other"


class Urgency(StrEnum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class NextAction(StrEnum):
    FAQ = "Send standard FAQ or self-service link"
    CLAIM = "Create or update a claim"
    BILLING = "Forward to billing team"
    TECHNICAL = "Forward to technical support"
    ESCALATE = "Escalate to human supervisor"
    ASK_FOR_INFO = "Ask for more information"


class ProcessingStatus(StrEnum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    ERROR = "error"


class Ticket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    subject: str = ""
    body: str = ""
    language: str = ""
    source_index: int | None = None


class TopicAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Topic
    confidence: float = Field(ge=0.0, le=1.0)
    secondary_topic: Topic | None = None
    evidence: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=300)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        return _normalize_confidence(value)


class UrgencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    risk_signals: list[str] = Field(default_factory=list, max_length=5)
    notes: str = Field(default="", max_length=300)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        return _normalize_confidence(value)


class MissingInfoAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    needs_more_information: bool
    missing_information: list[str] = Field(max_length=5)
    clarification_questions: list[str] = Field(max_length=2)
    ticket_is_ambiguous: bool = False
    multiple_issues: bool = False
    notes: str = Field(default="", max_length=300)

    @field_validator("clarification_questions")
    @classmethod
    def limit_questions(cls, questions: list[str]) -> list[str]:
        return questions[:2]

    @model_validator(mode="after")
    def validate_information_consistency(self) -> Self:
        if self.needs_more_information:
            if not self.missing_information:
                raise ValueError(
                    "missing_information must name at least one item when more "
                    "information is needed"
                )
            if not self.clarification_questions:
                raise ValueError(
                    "clarification_questions must contain at least one concrete question "
                    "when more information is needed"
                )
        elif self.missing_information or self.clarification_questions:
            raise ValueError(
                "missing_information and clarification_questions must be empty when no "
                "more information is needed"
            )
        return self


class ReviewAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: Topic
    topic_confidence: float = Field(ge=0.0, le=1.0)
    urgency: Urgency
    urgency_confidence: float = Field(ge=0.0, le=1.0)
    needs_more_information: bool
    missing_information: list[str] = Field(max_length=5)
    clarification_questions: list[str] = Field(max_length=2)
    changed_fields: list[str] = Field(default_factory=list, max_length=5)
    evidence: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=300)

    @field_validator("topic_confidence", "urgency_confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, value: object) -> object:
        return _normalize_confidence(value)

    @field_validator("clarification_questions")
    @classmethod
    def limit_questions(cls, questions: list[str]) -> list[str]:
        return questions[:2]

    @model_validator(mode="after")
    def validate_information_consistency(self) -> Self:
        if self.needs_more_information:
            if not self.missing_information:
                raise ValueError(
                    "missing_information must name at least one item when more "
                    "information is needed"
                )
            if not self.clarification_questions:
                raise ValueError(
                    "clarification_questions must contain at least one concrete question "
                    "when more information is needed"
                )
        elif self.missing_information or self.clarification_questions:
            raise ValueError(
                "missing_information and clarification_questions must be empty when no "
                "more information is needed"
            )
        return self


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticket_id: str
    text_snippet: str
    topic: Topic | None
    topic_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    urgency: Urgency | None
    urgency_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    next_action: NextAction | None
    needs_more_information: bool | None
    missing_information: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    reviewed_by_llm: bool = False
    model_profile: str
    model_name: str
    processing_status: ProcessingStatus
    latency_ms: int = Field(ge=0)
    notes: str = ""

    def to_csv_row(self) -> dict[str, object]:
        return {
            "ticket_id": self.ticket_id,
            "text_snippet": self.text_snippet,
            "topic": self.topic.value if self.topic else "",
            "topic_confidence": self.topic_confidence,
            "urgency": self.urgency.value if self.urgency else "",
            "urgency_confidence": self.urgency_confidence,
            "next_action": self.next_action.value if self.next_action else "",
            "needs_more_information": self.needs_more_information,
            "missing_information": " | ".join(self.missing_information),
            "clarification_questions": " | ".join(self.clarification_questions),
            "reviewed_by_llm": self.reviewed_by_llm,
            "model_profile": self.model_profile,
            "model_name": self.model_name,
            "processing_status": self.processing_status.value,
            "latency_ms": self.latency_ms,
            "notes": self.notes,
        }


class TriageState(TypedDict, total=False):
    ticket: Ticket
    started_at: float
    normalized_text: str
    topic_assessment: TopicAssessment
    urgency_assessment: UrgencyAssessment
    missing_info_assessment: MissingInfoAssessment
    review_required: bool
    review_reasons: list[str]
    reviewed_by_llm: bool
    matched_risk_terms: list[str]
    next_action: NextAction
    processing_status: ProcessingStatus
    notes: list[str]
    result: TriageResult
