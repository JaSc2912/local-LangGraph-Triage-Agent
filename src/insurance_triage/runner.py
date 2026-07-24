from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from math import ceil
from time import perf_counter

from insurance_triage.config import ModelProfile, Settings
from insurance_triage.data import normalize_ticket_text
from insurance_triage.graph import build_triage_graph
from insurance_triage.model_client import ModelClient, OllamaModelClient
from insurance_triage.schemas import ProcessingStatus, Ticket, TriageResult

ProgressCallback = Callable[[int, int, TriageResult], None]


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, ceil(len(ordered) * fraction) - 1))
    return ordered[index]


class BatchRunner:
    def __init__(
        self,
        profile: ModelProfile,
        settings: Settings,
        model_client: ModelClient | None = None,
    ) -> None:
        self.profile = profile
        self.settings = settings
        self.model_client = model_client or OllamaModelClient(settings.ollama_base_url)
        self.graph = build_triage_graph(self.model_client, profile, settings)

    def preflight(self) -> None:
        self.model_client.preflight(self.profile)

    def process_ticket(self, ticket: Ticket) -> TriageResult:
        started_at = perf_counter()
        try:
            final_state = self.graph.invoke({"ticket": ticket, "started_at": started_at})
            return final_state["result"]
        except Exception as exc:
            normalized = normalize_ticket_text(ticket.subject, ticket.body)
            return TriageResult(
                ticket_id=ticket.ticket_id,
                text_snippet=" ".join(normalized.split())[: self.settings.snippet_length],
                topic=None,
                topic_confidence=None,
                urgency=None,
                urgency_confidence=None,
                next_action=None,
                needs_more_information=None,
                reviewed_by_llm=False,
                model_profile=self.profile.name,
                model_name=self.profile.model,
                processing_status=ProcessingStatus.ERROR,
                latency_ms=max(0, round((perf_counter() - started_at) * 1000)),
                notes=f"{type(exc).__name__}: {exc}"[:1000],
            )

    def run_tickets(
        self,
        tickets: Sequence[Ticket],
        progress_callback: ProgressCallback | None = None,
    ) -> list[TriageResult]:
        results: list[TriageResult] = []
        total = len(tickets)
        for index, ticket in enumerate(tickets, start=1):
            result = self.process_ticket(ticket)
            results.append(result)
            if progress_callback:
                progress_callback(index, total, result)
        return results

    @staticmethod
    def summarize(results: Sequence[TriageResult]) -> dict[str, object]:
        latencies = [result.latency_ms for result in results]
        statuses = Counter(result.processing_status.value for result in results)
        topics = Counter(result.topic.value for result in results if result.topic)
        urgencies = Counter(result.urgency.value for result in results if result.urgency)
        actions = Counter(result.next_action.value for result in results if result.next_action)
        return {
            "ticket_count": len(results),
            "successful_count": statuses[ProcessingStatus.SUCCESS.value],
            "incomplete_count": statuses[ProcessingStatus.INCOMPLETE.value],
            "error_count": statuses[ProcessingStatus.ERROR.value],
            "reviewed_count": sum(result.reviewed_by_llm for result in results),
            "needs_more_information_count": sum(
                result.needs_more_information is True for result in results
            ),
            "latency_ms": {
                "mean": round(sum(latencies) / len(latencies)) if latencies else 0,
                "p50": _percentile(latencies, 0.50),
                "p95": _percentile(latencies, 0.95),
            },
            "topic_distribution": dict(sorted(topics.items())),
            "urgency_distribution": dict(sorted(urgencies.items())),
            "action_distribution": dict(sorted(actions.items())),
        }
