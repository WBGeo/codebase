"""G.1: shared GOLEM MOOSE/HIT block parser regression tests."""
import pytest
from core.simulation_components.simulation_packages.golem.golem_input_parser import (
    locate_blocks, top_block, direct_children, block_body,
)


def test_modern_and_legacy_nested_blocks():
    text = "[Materials]\n  [./rock]\n  [../]\n  [fluid]\n  []\n[]\n"
    assert direct_children(text, "Materials") == ["rock", "fluid"]
    lines, parent = top_block(text, "Materials")
    assert parent.start == 0 and parent.end == len(lines) - 1
    assert "[./rock]" in block_body(text, "Materials")


def test_brackets_in_comments_and_multiline_quoted_values_are_not_blocks():
    text = """[Outputs]
  # [fake]
  label = '[not_a_block]'
  points = '
[also_not_a_block]
[]
  '
  csv = true # [ignored]
[]
"""
    lines, block = top_block(text, "Outputs")
    assert block.end == len(lines) - 1
    assert direct_children(text, "Outputs") == []
    assert "csv = true" in block_body(text, "Outputs")


def test_comment_after_structural_token_is_ignored():
    text = "[Outputs] # comment\n  [./csv] # comment\n  [../]\n[] # done\n"
    assert direct_children(text, "Outputs") == ["csv"]


@pytest.mark.parametrize("content, message", [
    ("[Outputs]\n csv = true\n", "Unterminated"),
    ("[]\n", "Unexpected"),
    ("[Outputs]\n[]\n[]\n", "Unexpected"),
])
def test_malformed_nesting_raises(content, message):
    with pytest.raises(ValueError, match=message):
        locate_blocks(content)


def test_missing_top_block_returns_none():
    _, block = top_block("[Mesh]\n[]\n", "Outputs")
    assert block is None
    assert direct_children("[Mesh]\n[]\n", "Outputs") == []
