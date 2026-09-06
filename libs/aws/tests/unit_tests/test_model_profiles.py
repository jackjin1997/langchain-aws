"""Tests for AWS model profile augmentations."""

from pathlib import Path

import pytest

PROFILE_AUGMENTATIONS = (
    Path(__file__).parents[2] / "langchain_aws/data/profile_augmentations.toml"
)


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-sonnet-5",
        "au.anthropic.claude-sonnet-5",
        "eu.anthropic.claude-sonnet-5",
        "global.anthropic.claude-sonnet-5",
        "jp.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-5",
    ],
)
def test_claude_sonnet_5_structured_output_augmentation(model_id: str) -> None:
    """Claude Sonnet 5 profiles explicitly retain structured output support."""
    contents = PROFILE_AUGMENTATIONS.read_text()
    header = f'[overrides."{model_id}"]'
    _, separator, remainder = contents.partition(header)
    assert separator

    section = remainder.partition("\n[")[0]
    assert "structured_output = true" in section.splitlines()
