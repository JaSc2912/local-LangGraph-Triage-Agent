from __future__ import annotations

import json
from collections.abc import Sequence
from time import sleep
from typing import Protocol, TypeVar

from ollama import Client, ResponseError
from pydantic import BaseModel, ValidationError

from insurance_triage.config import ModelProfile

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class ModelClient(Protocol):
    def preflight(self, profile: ModelProfile) -> None: ...

    def invoke(
        self,
        messages: Sequence[dict[str, str]],
        output_schema: type[SchemaT],
        profile: ModelProfile,
    ) -> SchemaT: ...


class OllamaModelClient:
    def __init__(self, base_url: str, client: Client | None = None) -> None:
        self._client = client or Client(host=base_url)

    def preflight(self, profile: ModelProfile) -> None:
        try:
            self._client.show(profile.model)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama model '{profile.model}' is unavailable. Start Ollama and run: "
                f"ollama pull {profile.model}"
            ) from exc

    def invoke(
        self,
        messages: Sequence[dict[str, str]],
        output_schema: type[SchemaT],
        profile: ModelProfile,
    ) -> SchemaT:
        last_error: Exception | None = None
        for attempt in range(profile.retries + 1):
            try:
                request_messages = list(messages)
                output_format: str | dict[str, object] = output_schema.model_json_schema()
                if attempt > 0:
                    compact_schema = json.dumps(
                        output_schema.model_json_schema(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    request_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Return ONLY one JSON object matching this JSON Schema exactly. "
                                "Do not use Markdown or commentary. JSON Schema:\n"
                                f"{compact_schema}"
                            ),
                        }
                    )
                    output_format = "json"
                response = self._client.chat(
                    model=profile.model,
                    messages=request_messages,
                    format=output_format,
                    stream=False,
                    think=profile.think,
                    keep_alive=profile.keep_alive,
                    options={
                        "temperature": profile.temperature,
                        "num_ctx": profile.num_ctx,
                        "num_predict": profile.num_predict,
                    },
                )
                content = response.message.content
                return output_schema.model_validate_json(content)
            except (ResponseError, ValidationError, ValueError, TypeError) as exc:
                last_error = exc
                if attempt < profile.retries:
                    sleep(0.25 * (attempt + 1))

        error_detail = " ".join(str(last_error).split())[:400] if last_error else "unknown error"
        raise RuntimeError(
            f"Model '{profile.model}' failed to produce a valid "
            f"{output_schema.__name__} response after {profile.retries + 1} attempts. "
            f"Last error: {error_detail}"
        ) from last_error
