from __future__ import annotations

import re
from time import perf_counter
from typing import Literal

from langgraph.graph import END, START, StateGraph

from insurance_triage.config import ModelProfile, Settings
from insurance_triage.data import normalize_ticket_text
from insurance_triage.model_client import ModelClient
from insurance_triage.prompts import (
    missing_info_messages,
    review_messages,
    topic_messages,
    urgency_messages,
)
from insurance_triage.schemas import (
    MissingInfoAssessment,
    NextAction,
    ProcessingStatus,
    ReviewAssessment,
    Topic,
    TopicAssessment,
    TriageResult,
    TriageState,
    Urgency,
    UrgencyAssessment,
)


def _append_notes(state: TriageState, *notes: str) -> list[str]:
    combined = list(state.get("notes", []))
    combined.extend(note.strip() for note in notes if note and note.strip())
    return combined


def _snippet(text: str, max_length: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 1].rstrip() + "…"


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term).replace(r"\ ", r"\s+")
    return bool(re.search(rf"(?<!\w){escaped}(?!\w)", text, flags=re.IGNORECASE))


def build_triage_graph(
    model_client: ModelClient,
    profile: ModelProfile,
    settings: Settings,
):
    def preprocess(state: TriageState) -> dict[str, object]:
        ticket = state["ticket"]
        normalized = normalize_ticket_text(ticket.subject, ticket.body)
        return {
            "normalized_text": normalized,
            "reviewed_by_llm": False,
            "notes": list(state.get("notes", [])),
        }

    def route_after_preprocess(state: TriageState) -> Literal["incomplete", "classify"]:
        return "classify" if state["normalized_text"] else "incomplete"

    def incomplete_handler(state: TriageState) -> dict[str, object]:
        return {
            "topic_assessment": TopicAssessment(
                topic=Topic.OTHER,
                confidence=0.0,
                evidence="",
                notes="The ticket contains no usable text.",
            ),
            "urgency_assessment": UrgencyAssessment(
                urgency=Urgency.LOW,
                confidence=0.0,
                notes="Urgency cannot be inferred from an empty ticket.",
            ),
            "missing_info_assessment": MissingInfoAssessment(
                needs_more_information=True,
                missing_information=["description of the problem or request"],
                clarification_questions=[
                    "Please describe the problem or request you need help with."
                ],
                ticket_is_ambiguous=True,
                notes="The ticket is empty.",
            ),
            "next_action": NextAction.ASK_FOR_INFO,
            "processing_status": ProcessingStatus.INCOMPLETE,
            "notes": _append_notes(state, "Empty ticket handled without a model call."),
        }

    def topic_specialist(state: TriageState) -> dict[str, object]:
        text = state["normalized_text"][: settings.model_input_chars]
        assessment = model_client.invoke(topic_messages(text), TopicAssessment, profile)
        return {
            "topic_assessment": assessment,
            "notes": _append_notes(state, assessment.notes),
        }

    def urgency_specialist(state: TriageState) -> dict[str, object]:
        text = state["normalized_text"][: settings.model_input_chars]
        assessment = model_client.invoke(urgency_messages(text), UrgencyAssessment, profile)
        return {
            "urgency_assessment": assessment,
            "notes": _append_notes(state, assessment.notes),
        }

    def missing_info_specialist(state: TriageState) -> dict[str, object]:
        text = state["normalized_text"][: settings.model_input_chars]
        topic = state["topic_assessment"].topic
        assessment = model_client.invoke(
            missing_info_messages(text, topic),
            MissingInfoAssessment,
            profile,
        )
        return {
            "missing_info_assessment": assessment,
            "notes": _append_notes(state, assessment.notes),
        }

    def validation_gate(state: TriageState) -> dict[str, object]:
        topic = state["topic_assessment"]
        urgency = state["urgency_assessment"]
        missing = state["missing_info_assessment"]
        reasons: list[str] = []

        if topic.confidence < settings.confidence_threshold:
            reasons.append("low topic confidence")
        if urgency.confidence < settings.confidence_threshold:
            reasons.append("low urgency confidence")
        if topic.secondary_topic is not None and topic.secondary_topic != topic.topic:
            reasons.append("secondary topic detected")
        if topic.topic == Topic.OTHER:
            reasons.append("topic classified as Other")
        if urgency.urgency == Urgency.HIGH and not urgency.risk_signals:
            reasons.append("high urgency without an explicit risk signal")
        if missing.ticket_is_ambiguous:
            reasons.append("ticket marked as ambiguous")
        if missing.multiple_issues:
            reasons.append("multiple issues detected")
        if missing.needs_more_information and not missing.missing_information:
            reasons.append("missing-information decision has no missing fields")

        return {
            "review_required": bool(reasons),
            "review_reasons": reasons,
            "notes": _append_notes(
                state,
                f"Semantic review requested: {', '.join(reasons)}." if reasons else "",
            ),
        }

    def route_after_validation(state: TriageState) -> Literal["review", "guardrail"]:
        return "review" if state["review_required"] else "guardrail"

    def semantic_reviewer(state: TriageState) -> dict[str, object]:
        text = state["normalized_text"][: settings.model_input_chars]
        reviewed = model_client.invoke(
            review_messages(
                text,
                state["topic_assessment"],
                state["urgency_assessment"],
                state["missing_info_assessment"],
                state.get("review_reasons", []),
            ),
            ReviewAssessment,
            profile,
        )
        topic = TopicAssessment(
            topic=reviewed.topic,
            confidence=reviewed.topic_confidence,
            evidence=reviewed.evidence,
            notes=reviewed.notes,
        )
        urgency = UrgencyAssessment(
            urgency=reviewed.urgency,
            confidence=reviewed.urgency_confidence,
            risk_signals=state["urgency_assessment"].risk_signals,
            notes=reviewed.notes,
        )
        missing = MissingInfoAssessment(
            needs_more_information=reviewed.needs_more_information,
            missing_information=reviewed.missing_information,
            clarification_questions=reviewed.clarification_questions,
            notes=reviewed.notes,
        )
        return {
            "topic_assessment": topic,
            "urgency_assessment": urgency,
            "missing_info_assessment": missing,
            "reviewed_by_llm": True,
            "notes": _append_notes(state, reviewed.notes),
        }

    def risk_guardrail(state: TriageState) -> dict[str, object]:
        text = state["normalized_text"]
        matched = [term for term in settings.risk_terms if _contains_term(text, term)]
        urgency = state["urgency_assessment"]
        notes = list(state.get("notes", []))
        if matched and urgency.urgency != Urgency.HIGH:
            urgency = urgency.model_copy(
                update={
                    "urgency": Urgency.HIGH,
                    "risk_signals": list(dict.fromkeys([*urgency.risk_signals, *matched]))[:5],
                    "notes": "Urgency promoted by the explicit risk guardrail.",
                }
            )
            notes = _append_notes(
                state,
                f"Risk guardrail promoted urgency for: {', '.join(matched)}.",
            )
        return {
            "urgency_assessment": urgency,
            "matched_risk_terms": matched,
            "notes": notes,
        }

    def deterministic_router(state: TriageState) -> dict[str, object]:
        topic = state["topic_assessment"].topic
        urgency = state["urgency_assessment"].urgency
        needs_info = state["missing_info_assessment"].needs_more_information

        if urgency == Urgency.HIGH:
            action = NextAction.ESCALATE
        elif needs_info:
            action = NextAction.ASK_FOR_INFO
        elif topic == Topic.CLAIMS:
            action = NextAction.CLAIM
        elif topic == Topic.BILLING:
            action = NextAction.BILLING
        elif topic == Topic.TECHNICAL:
            action = NextAction.TECHNICAL
        elif topic == Topic.POLICY:
            action = NextAction.FAQ
        elif urgency == Urgency.MEDIUM:
            action = NextAction.ESCALATE
        else:
            action = NextAction.FAQ

        return {
            "next_action": action,
            "processing_status": ProcessingStatus.SUCCESS,
        }

    def finalize(state: TriageState) -> dict[str, object]:
        ticket = state["ticket"]
        topic = state["topic_assessment"]
        urgency = state["urgency_assessment"]
        missing = state["missing_info_assessment"]
        elapsed_ms = max(0, round((perf_counter() - state["started_at"]) * 1000))
        unique_notes = list(dict.fromkeys(note for note in state.get("notes", []) if note))
        result = TriageResult(
            ticket_id=ticket.ticket_id,
            text_snippet=_snippet(state.get("normalized_text", ""), settings.snippet_length),
            topic=topic.topic,
            topic_confidence=topic.confidence,
            urgency=urgency.urgency,
            urgency_confidence=urgency.confidence,
            next_action=state["next_action"],
            needs_more_information=missing.needs_more_information,
            missing_information=missing.missing_information,
            clarification_questions=missing.clarification_questions,
            reviewed_by_llm=state.get("reviewed_by_llm", False),
            model_profile=profile.name,
            model_name=profile.model,
            processing_status=state["processing_status"],
            latency_ms=elapsed_ms,
            notes=" ".join(unique_notes)[:1000],
        )
        return {"result": result}

    builder = StateGraph(TriageState)
    builder.add_node("preprocess", preprocess)
    builder.add_node("incomplete_handler", incomplete_handler)
    builder.add_node("topic_specialist", topic_specialist)
    builder.add_node("urgency_specialist", urgency_specialist)
    builder.add_node("missing_info_specialist", missing_info_specialist)
    builder.add_node("validation_gate", validation_gate)
    builder.add_node("semantic_reviewer", semantic_reviewer)
    builder.add_node("risk_guardrail", risk_guardrail)
    builder.add_node("deterministic_router", deterministic_router)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "preprocess")
    builder.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {"incomplete": "incomplete_handler", "classify": "topic_specialist"},
    )
    builder.add_edge("incomplete_handler", "finalize")
    builder.add_edge("topic_specialist", "urgency_specialist")
    builder.add_edge("urgency_specialist", "missing_info_specialist")
    builder.add_edge("missing_info_specialist", "validation_gate")
    builder.add_conditional_edges(
        "validation_gate",
        route_after_validation,
        {"review": "semantic_reviewer", "guardrail": "risk_guardrail"},
    )
    builder.add_edge("semantic_reviewer", "risk_guardrail")
    builder.add_edge("risk_guardrail", "deterministic_router")
    builder.add_edge("deterministic_router", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile()
