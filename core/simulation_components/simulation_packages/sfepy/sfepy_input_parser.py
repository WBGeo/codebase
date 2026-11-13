"""Static reading of custom SfePy input files (no execution).

Used by the material preview to show the coefficients a CustomSfepyBuilder
file assigns to each mesh group, without running user code.
"""
import ast
import re
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple

import numpy as np


@dataclass
class CustomSfepyMaterials:
    """Literal material coefficients of a custom SfePy file, per mesh group.

    group_values: SfePy group id (``cells of group N``) -> coefficient -> value.
    conflicts: (group id, coefficient) pairs that two materials linked to the
        same group define with different values; left out of group_values.
    unlinked: coefficients defined in ``materials`` whose material is not
        linked to any ``cells of group N`` region through ``equations``.
    """
    group_values: Dict[int, Dict[str, float]] = field(default_factory=dict)
    conflicts: Set[Tuple[int, str]] = field(default_factory=set)
    unlinked: Set[str] = field(default_factory=set)


def custom_sfepy_group_properties(content: str) -> CustomSfepyMaterials:
    """Read the literal coefficients a custom SfePy file assigns to each mesh group.

    Only direct numeric constants in the module-level ``materials`` mapping
    are read (arithmetic of literals and of module-level scalar constants is
    allowed). A material is linked to a group through any equation term
    ``dw_*.<integral>.<region>(<material>.<coefficient>, ...)`` whose region
    is ``cells of group N``; the coefficient used in the equation does not
    need to be the one read, so metadata such as porosity is picked up too.
    """
    tree = ast.parse(content)
    material_defs: Dict[str, Dict[str, float]] = {}
    region_groups: Dict[str, int] = {}
    equations = {}
    constants: Dict[str, float] = {}

    def scalar(node):
        """Evaluate only arithmetic of numeric literals and known scalar constants."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in constants:
            return constants[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = scalar(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            a, b = scalar(node.left), scalar(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            return a ** b
        raise ValueError("not a static scalar")

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name not in ("materials", "regions", "equations"):
                try:
                    result = scalar(value)
                    if np.isfinite(result):
                        constants[name] = result
                except (ValueError, TypeError, ZeroDivisionError, OverflowError):
                    pass
            if name == "materials" and isinstance(value, ast.Dict):
                # Parse material entries independently: a single expression or
                # callback must not make the entire dictionary unreadable.
                material_defs = {}
                for key, definition in zip(value.keys, value.values):
                    try:
                        mat_name = ast.literal_eval(key)
                    except (ValueError, TypeError, SyntaxError, MemoryError):
                        continue
                    if not isinstance(definition, (ast.Tuple, ast.List)) or not definition.elts:
                        continue
                    fields_node = definition.elts[0]
                    if not isinstance(fields_node, ast.Dict):
                        continue
                    fields = {}
                    for field_key, field_value in zip(fields_node.keys, fields_node.values):
                        try:
                            field_name = ast.literal_eval(field_key)
                            number = scalar(field_value)
                            if np.isfinite(number):
                                fields[field_name] = number
                        except (ValueError, TypeError, SyntaxError, MemoryError,
                                ZeroDivisionError, OverflowError):
                            continue
                    material_defs[mat_name] = fields
            elif name == "regions" and isinstance(value, ast.Dict):
                for key_node, definition_node in zip(value.keys, value.values):
                    try:
                        region_name = ast.literal_eval(key_node)
                        selector_node = (definition_node.elts[0]
                                         if isinstance(definition_node, (ast.Tuple, ast.List))
                                         and definition_node.elts else definition_node)
                        selector = ast.literal_eval(selector_node)
                    except (ValueError, TypeError, SyntaxError, MemoryError):
                        continue
                    match = re.fullmatch(r"cells of group\s+(\d+)", str(selector).strip())
                    if match:
                        region_groups[region_name] = int(match.group(1))
            elif name == "equations":
                try:
                    equations = ast.literal_eval(value)
                except (ValueError, TypeError, SyntaxError, MemoryError):
                    pass

    material_groups: Dict[str, Set[int]] = {}
    for equation in equations.values():
        if not isinstance(equation, str):
            continue
        # term.integral.region(material.coefficient, ...), also across line breaks.
        for region, material in re.findall(r"\b(?:dw_\w+|ev_\w+)\.\w+\.(\w+)\s*\(\s*(\w+)\.\w+", equation):
            if region in region_groups and material in material_defs:
                material_groups.setdefault(material, set()).add(region_groups[region])

    result = CustomSfepyMaterials()
    for material, fields in material_defs.items():
        if material not in material_groups:
            result.unlinked.update(fields)
            continue
        for group_id in material_groups[material]:
            values = result.group_values.setdefault(group_id, {})
            for coefficient, value in fields.items():
                if (group_id, coefficient) in result.conflicts:
                    continue
                if coefficient in values and values[coefficient] != value:
                    del values[coefficient]
                    result.conflicts.add((group_id, coefficient))
                    continue
                values[coefficient] = value
    linked = {c for values in result.group_values.values() for c in values}
    result.unlinked -= linked | {c for _, c in result.conflicts}
    return result
