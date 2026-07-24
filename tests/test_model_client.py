from __future__ import annotations

from types import SimpleNamespace

import pytest

from insurance_triage.config import resolve_profile
from insurance_triage.model_client import OllamaModelClient
from insurance_triage.schemas import Topic, TopicAssessment


class StubOllamaClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.chat_calls = 0
        self.chat_kwargs: list[dict[str, object]] = []
        self.shown_models: list[str] = []

    def show(self, model: str) -> None:
        self.shown_models.append(model)

    def chat(self, **kwargs):
        self.chat_kwargs.append(kwargs)
        content = self.contents[min(self.chat_calls, len(self.contents) - 1)]
        self.chat_calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=content))


def test_preflight_checks_exact_profile_model() -> None:
    stub = StubOllamaClient(["{}"])
    client = OllamaModelClient("http://localhost:11434", client=stub)
    profile = resolve_profile("compact")

    client.preflight(profile)

    assert stub.shown_models == [profile.model]


def test_invalid_structured_output_is_retried() -> None:
    valid = TopicAssessment(
        topic=Topic.TECHNICAL,
        confidence=0.9,
        evidence="login",
    ).model_dump_json()
    stub = StubOllamaClient(["not json", valid])
    client = OllamaModelClient("http://localhost:11434", client=stub)

    result = client.invoke([], TopicAssessment, resolve_profile("compact"))

    assert result.topic == Topic.TECHNICAL
    assert stub.chat_calls == 2
    assert stub.chat_kwargs[0]["format"] == TopicAssessment.model_json_schema()
    assert stub.chat_kwargs[1]["format"] == "json"
    retry_messages = stub.chat_kwargs[1]["messages"]
    assert "Return ONLY one JSON object" in retry_messages[-1]["content"]


def test_exhausted_retries_raise_actionable_error() -> None:
    stub = StubOllamaClient(["not json"])
    client = OllamaModelClient("http://localhost:11434", client=stub)

    with pytest.raises(RuntimeError, match="3 attempts.*Last error"):
        client.invoke([], TopicAssessment, resolve_profile("compact"))
