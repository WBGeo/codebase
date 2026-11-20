"""Checks the colour code and sidebar order of the Workbench components.

The rules are described in ``core/workbench_palette.py`` and in
``docs/developers/components.md`` (section "Colours, groups and order").
The decorators are read from the source (AST), so no module is imported.
"""
import ast
import collections
from pathlib import Path

import pytest

from core import workbench_palette

CORE = Path(__file__).resolve().parents[1] / "core"
DECORATORS = {"wbgeo_component", "wbgeo_type", "AnnotatedScriptType"}
COLOUR_KWARGS = ("color", "border_color")
PALETTE = {name: value for name, value in vars(workbench_palette).items()
           if name.isupper() and isinstance(value, str)}
MIN_CONTRAST = 4.5


def _decorator_calls():
    """Yield (location, decorator name, {kwarg: value node}) for every WBGeo decorator in core/."""
    for path in sorted(CORE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name in DECORATORS:
                location = f"{path.relative_to(CORE.parent).as_posix()}:{node.lineno}"
                yield location, name, {k.arg: k.value for k in node.keywords if k.arg}


CALLS = list(_decorator_calls())
COMPONENTS_IN_GROUPS = [(loc, kws) for loc, name, kws in CALLS
                        if name == "wbgeo_component" and "group" in kws]


def _relative_luminance(hex_colour):
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(a, b):
    high, low = sorted((_relative_luminance(a), _relative_luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def test_decorators_found():
    # guards against the scan silently finding nothing (e.g. after moving core/)
    assert len(COMPONENTS_IN_GROUPS) >= 20


def test_colours_come_from_palette():
    offenders = []
    for location, _, kws in CALLS:
        for key in COLOUR_KWARGS:
            value = kws.get(key)
            if value is None:
                continue
            is_palette_ref = (isinstance(value, ast.Attribute)
                              and isinstance(value.value, ast.Name)
                              and value.value.id == "palette"
                              and value.attr in PALETTE)
            if not is_palette_ref:
                offenders.append(f"{location}: {key}={ast.unparse(value)}")
    assert not offenders, ("Use the constants from core/workbench_palette.py "
                           "(e.g. color=palette.MESH):\n" + "\n".join(offenders))


def test_components_in_groups_have_colour_and_position():
    offenders = []
    for location, kws in COMPONENTS_IN_GROUPS:
        missing = [k for k in ("color", "border_color", "positionInGroup") if k not in kws]
        if missing:
            offenders.append(f"{location}: missing {', '.join(missing)}")
        elif not (isinstance(kws["positionInGroup"], ast.Constant)
                  and isinstance(kws["positionInGroup"].value, int)):
            offenders.append(f"{location}: positionInGroup must be an int literal")
    assert not offenders, "\n".join(offenders)


def test_positions_unique_within_group():
    by_slot = collections.defaultdict(list)
    for location, kws in COMPONENTS_IN_GROUPS:
        group, position = kws["group"], kws.get("positionInGroup")
        if isinstance(group, ast.Constant) and isinstance(position, ast.Constant):
            by_slot[(group.value, position.value)].append(location)
    duplicates = {slot: locs for slot, locs in by_slot.items() if len(locs) > 1}
    assert not duplicates, "\n".join(f"group {g!r}, position {p}: {', '.join(locs)}"
                                     for (g, p), locs in duplicates.items())


@pytest.mark.parametrize("name", sorted(set(PALETTE) - {"BORDER"}))
def test_palette_contrast_against_border(name):
    contrast = _contrast(PALETTE[name], workbench_palette.BORDER)
    assert contrast >= MIN_CONTRAST, f"{name} = {PALETTE[name]}: contrast {contrast:.2f} < {MIN_CONTRAST}"


def test_palette_colours_are_distinct():
    values = collections.Counter(v.lower() for v in PALETTE.values())
    assert all(count == 1 for count in values.values()), values
