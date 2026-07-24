from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelProfile:
    name: str
    model: str
    num_ctx: int
    num_predict: int
    temperature: float = 0.0
    think: bool = False
    keep_alive: str = "30m"
    timeout_seconds: float = 240.0
    retries: int = 2


MODEL_PROFILES: dict[str, ModelProfile] = {
    "quality": ModelProfile(
        name="quality",
        model="qwen3.5:35b-a3b-q4_K_M",
        num_ctx=4096,
        num_predict=512,
        timeout_seconds=300.0,
    ),
    "compact": ModelProfile(
        name="compact",
        model="qwen3.5:9b",
        num_ctx=4096,
        num_predict=384,
        timeout_seconds=180.0,
    ),
}


@dataclass(frozen=True, slots=True)
class Settings:
    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("outputs")
    graph_output: Path = Path("docs/triage_graph.mmd")
    comparison_output: Path = Path("outputs/profile_comparison.json")
    language: str = "en"
    limit: int = 200
    comparison_limit: int = 25
    seed: int = 42
    snippet_length: int = 240
    model_input_chars: int = 6000
    confidence_threshold: float = 0.70
    ollama_base_url: str = "http://localhost:11434"
    risk_terms: tuple[str, ...] = field(
        default=(
            "account takeover",
            "compromised account",
            "data loss",
            "data breach",
            "fraud",
            "fraudulent",
            "unauthorized payment",
            "unauthorised payment",
            "injury",
            "injured",
            "fire",
            "flood",
            "total outage",
            "complete outage",
            "system outage",
        )
    )

    def default_output(self, profile_name: str) -> Path:
        return self.output_dir / f"triage_results_{profile_name}.csv"


def resolve_profile(name: str) -> ModelProfile:
    normalized = name.strip().casefold()
    try:
        return MODEL_PROFILES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PROFILES))
        raise ValueError(f"Unknown profile '{name}'. Choose one of: {choices}.") from exc
