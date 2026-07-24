from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from insurance_triage.config import Settings, resolve_profile
from insurance_triage.data import discover_dataset, load_tickets, write_json
from insurance_triage.runner import BatchRunner
from insurance_triage.schemas import ProcessingStatus, TriageResult

console = Console()


def _agreement(
    quality: list[TriageResult],
    compact: list[TriageResult],
) -> dict[str, object]:
    compact_by_id = {result.ticket_id: result for result in compact}
    pairs = [
        (result, compact_by_id[result.ticket_id])
        for result in quality
        if result.ticket_id in compact_by_id
        and result.processing_status != ProcessingStatus.ERROR
        and compact_by_id[result.ticket_id].processing_status != ProcessingStatus.ERROR
    ]
    count = len(pairs)

    def rate(predicate) -> float:
        if not count:
            return 0.0
        return round(sum(predicate(left, right) for left, right in pairs) / count, 4)

    return {
        "comparable_tickets": count,
        "topic_agreement": rate(lambda left, right: left.topic == right.topic),
        "urgency_agreement": rate(lambda left, right: left.urgency == right.urgency),
        "action_agreement": rate(lambda left, right: left.next_action == right.next_action),
    }


def compare_command(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", help="Dataset CSV. Auto-discovered below data/raw by default."),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", help="Comparison JSON output."),
    ] = None,
    language: Annotated[str, typer.Option("--language")] = "en",
    limit: Annotated[int, typer.Option("--limit", min=1)] = 25,
    seed: Annotated[int, typer.Option("--seed")] = 42,
) -> None:
    """Compare quality and compact on the same deterministic ticket sample."""
    base = Settings()
    settings = replace(
        base,
        language=language,
        comparison_limit=limit,
        seed=seed,
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", base.ollama_base_url),
    )
    try:
        dataset_path = discover_dataset(input_path, settings.input_dir)
        tickets = load_tickets(
            dataset_path,
            language=settings.language,
            limit=settings.comparison_limit,
            seed=settings.seed,
        )
        results: dict[str, list[TriageResult]] = {}
        summaries: dict[str, dict[str, object]] = {}
        for profile_name in ("quality", "compact"):
            profile = resolve_profile(profile_name)
            runner = BatchRunner(profile, settings)
            runner.preflight()
            console.print(f"Warming [bold]{profile_name}[/bold] (excluded from metrics)…")
            runner.process_ticket(tickets[0])
            console.print(f"Running [bold]{profile_name}[/bold] on {len(tickets)} tickets…")
            profile_results = runner.run_tickets(tickets)
            results[profile_name] = profile_results
            summaries[profile_name] = runner.summarize(profile_results)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    comparison = {
        "dataset": dataset_path.name,
        "language": settings.language,
        "limit": settings.comparison_limit,
        "seed": settings.seed,
        "warmup_ticket_excluded": True,
        "profiles": {
            name: {
                "model": resolve_profile(name).model,
                **summaries[name],
            }
            for name in ("quality", "compact")
        },
        "agreement": _agreement(results["quality"], results["compact"]),
    }
    destination = output_path or settings.comparison_output
    written = write_json(comparison, destination)
    console.print(f"[bold green]Comparison:[/bold green] {written}")


def main() -> None:
    typer.run(compare_command)
