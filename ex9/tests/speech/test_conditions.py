"""Speech condition specification tests."""

from __future__ import annotations

import pytest
from nf_assignment.speech.conditions import parse_condition_spec


def test_parse_condition_spec_accepts_component_list_and_delimited_string() -> None:
    from_list = parse_condition_spec(["hubert_soft", "world_aux"])
    from_comma = parse_condition_spec("hubert_soft,world_aux")
    from_plus = parse_condition_spec("hubert_soft+world_aux")

    assert from_list.components == ("hubert_soft", "world_aux")
    assert from_comma.components == from_list.components
    assert from_plus.components == from_list.components
    assert from_list.name == "hubert_soft+world_aux"
    assert from_list.single_content_condition() == "hubert_soft"
    assert from_list.uses_world_aux is True


def test_parse_condition_spec_rejects_legacy_combined_name_for_vc() -> None:
    spec = parse_condition_spec("hubert_soft_world_aux")

    assert spec.components == ("hubert_soft_world_aux",)
    with pytest.raises(ValueError, match="exactly one content component"):
        spec.single_content_condition()
