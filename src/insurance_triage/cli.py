from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from insurance_triage.config import Settings, resolve_profile
from insurance_triage.data import discover_dataset, load_tickets, write_json, write_results
from insurance_triage.runner import BatchRunner

console = Console()


def run_command(
    profile_name: Annotated[
        str,
        typer.Argument(help="Local model profile: quality or compact."),
    ],
    input_path: Annotated[
        Path | None,
        typer.Option("--input", help="Dataset CSV. Auto-discovered below data/raw by default."),
    ] = None,
    output_path: Annotated[
        Path | None,
        typer.Option("--output", help="Result CSV. Derived from the profile by default."),
    ] = None,
    language: Annotated[
        str,
        typer.Option("--language", help="Dataset language code."),
    ] = "en",
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, help="Maximum number of unique tickets."),
    ] = 200,
    seed: Annotated[
        int,
        typer.Option("--seed", help="Random sampling seed."),
    ] = 42,
) -> None:
    """Run the local insurance support ticket triage workflow."""
    try:
        profile = resolve_profile(profile_name)
        base = Settings()
        settings = replace(
            base,
            language=language,
            limit=limit,
            seed=seed,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", base.ollama_base_url),
        )
        dataset_path = discover_dataset(input_path, settings.input_dir)
        tickets = load_tickets(
            dataset_path,
            language=settings.language,
            limit=settings.limit,
            seed=settings.seed,
        )
        destination = output_path or settings.default_output(profile.name)
        runner = BatchRunner(profile, settings)
        runner.preflight()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc

    table = Table(show_header=False, box=None)
    table.add_row("Profile", f"[bold]{profile.name}[/bold] ({profile.model})")
    table.add_row("Dataset", dataset_path.name)
    table.add_row("Tickets", str(len(tickets)))
    table.add_row("Language", settings.language)
    table.add_row("Output", str(destination))
    console.print(table)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("Triaging tickets", total=len(tickets))

        def update_progress(index: int, total: int, _result: object) -> None:
            progress.update(task_id, completed=index, total=total)

        results = runner.run_tickets(tickets, progress_callback=update_progress)

    result_path = write_results(results, destination)
    summary = {
        "profile": profile.name,
        "model": profile.model,
        "dataset": dataset_path.name,
        "language": settings.language,
        "limit": settings.limit,
        "seed": settings.seed,
        **runner.summarize(results),
    }
    summary_path = result_path.with_name(f"{result_path.stem}_summary.json")
    write_json(summary, summary_path)

    console.print(f"[bold green]Results:[/bold green] {result_path}")
    console.print(f"[bold green]Summary:[/bold green] {summary_path}")
    if summary["error_count"]:
        console.print(
            f"[bold yellow]Completed with {summary['error_count']} ticket errors.[/bold yellow]"
        )
        raise typer.Exit(code=1)


def main() -> None:
    typer.run(run_command)
