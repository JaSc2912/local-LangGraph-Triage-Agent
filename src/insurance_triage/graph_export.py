from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer
from pydantic import BaseModel
from rich.console import Console

from insurance_triage.config import Settings, resolve_profile
from insurance_triage.graph import build_triage_graph

console = Console()


class GraphOnlyModelClient:
    def preflight(self, _profile: object) -> None:
        return None

    def invoke(
        self,
        _messages: object,
        _output_schema: type[BaseModel],
        _profile: object,
    ) -> NoReturn:
        raise RuntimeError("The graph-only model client cannot run inference.")


def graph_command(
    output_path: Annotated[
        Path | None,
        typer.Option("--output", help="Mermaid output path."),
    ] = None,
) -> None:
    """Export the workflow as Mermaid text without loading a model."""
    settings = Settings()
    profile = resolve_profile("quality")
    graph = build_triage_graph(GraphOnlyModelClient(), profile, settings)
    destination = (output_path or settings.graph_output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(graph.get_graph().draw_mermaid() + "\n", encoding="utf-8")
    console.print(f"[bold green]Graph:[/bold green] {destination}")


def main() -> None:
    typer.run(graph_command)
