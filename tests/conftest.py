from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from insurance_triage.config import ModelProfile

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class FakeModelClient:
    def __init__(self, responses: dict[type[BaseModel], BaseModel | list[BaseModel]]) -> None:
        self.responses: dict[type[BaseModel], list[BaseModel]] = {}
        for schema, value in responses.items():
            self.responses[schema] = list(value) if isinstance(value, list) else [value]
        self.calls: list[type[BaseModel]] = []
        self.call_counts: defaultdict[type[BaseModel], int] = defaultdict(int)
        self.preflight_profiles: list[str] = []

    def preflight(self, profile: ModelProfile) -> None:
        self.preflight_profiles.append(profile.name)

    def invoke(
        self,
        _messages: Sequence[dict[str, str]],
        output_schema: type[SchemaT],
        _profile: ModelProfile,
    ) -> SchemaT:
        self.calls.append(output_schema)
        index = self.call_counts[output_schema]
        self.call_counts[output_schema] += 1
        available = self.responses.get(output_schema)
        if not available:
            raise AssertionError(f"No fake response configured for {output_schema.__name__}")
        response = available[min(index, len(available) - 1)]
        return output_schema.model_validate(response.model_dump())
