from __future__ import annotations

import pytest

from insurance_triage.config import Settings, resolve_profile


def test_quality_profile_is_optimized_for_3090() -> None:
    profile = resolve_profile("QUALITY")

    assert profile.model == "qwen3.5:35b-a3b-q4_K_M"
    assert profile.num_ctx == 4096
    assert profile.think is False


def test_compact_profile_uses_same_public_shape() -> None:
    profile = resolve_profile("compact")

    assert profile.model == "qwen3.5:9b"
    assert profile.num_ctx == 4096


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="quality"):
        resolve_profile("large")


def test_default_output_is_derived_from_profile() -> None:
    assert Settings().default_output("quality").as_posix() == "outputs/triage_results_quality.csv"
