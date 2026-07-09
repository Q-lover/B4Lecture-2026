"""Condition feature specification helpers for speech experiments."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

CONTENT_CONDITIONS = frozenset({"hubert_soft", "ppg"})
WORLD_AUX_CONDITION = "world_aux"


@dataclass(frozen=True)
class ConditionSpec:
    """A condition specified as cached component feature names."""

    components: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate that component names are non-empty and unique."""

        if not self.components:
            raise ValueError("condition must contain at least one component.")
        invalid = [name for name in self.components if not name]
        if invalid:
            raise ValueError("condition components must be non-empty strings.")
        duplicates = sorted(
            {name for name in self.components if self.components.count(name) > 1}
        )
        if duplicates:
            raise ValueError(f"condition components must be unique: {duplicates}")

    @property
    def name(self) -> str:
        """Stable display and metadata name for this component combination."""

        return "+".join(self.components)

    @property
    def uses_world_aux(self) -> bool:
        """Return whether WORLD F0/AP auxiliary features are part of the condition."""

        return WORLD_AUX_CONDITION in self.components

    def single_content_condition(self) -> str:
        """Return the one content condition used for VC source extraction."""

        content = [name for name in self.components if name in CONTENT_CONDITIONS]
        if len(content) != 1:
            raise ValueError(
                "VC condition must contain exactly one content component "
                f"from {sorted(CONTENT_CONDITIONS)}, got {list(self.components)}"
            )
        unsupported = sorted(
            set(self.components) - CONTENT_CONDITIONS - {WORLD_AUX_CONDITION}
        )
        if unsupported:
            raise ValueError(f"unsupported VC condition components: {unsupported}")
        return content[0]


ConditionInput = str | Sequence[str] | ConditionSpec


def parse_condition_spec(
    value: ConditionInput | None,
    *,
    default: ConditionInput | None = None,
) -> ConditionSpec:
    """Normalize a CLI/config condition value into component names.

    Strings may use either comma or plus separators.
    """

    if value is None:
        if default is None:
            raise ValueError("condition is required.")
        value = default
    if isinstance(value, ConditionSpec):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("condition must not be empty.")
        components = tuple(
            part.strip() for part in re.split(r"[,+]", text) if part.strip()
        )
    else:
        components = tuple(str(part).strip() for part in value)
    return ConditionSpec(components)
