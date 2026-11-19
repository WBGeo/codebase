"""Small, line-oriented MOOSE/HIT block locator shared by GOLEM components.

Recognises structural bracket lines, not arbitrary bracket text in values.
This is not a complete HIT grammar; MOOSE remains the final syntax authority.
"""
from dataclasses import dataclass
import re
import shlex

_OPEN = re.compile(r"^\[(?:\./)?([A-Za-z_][A-Za-z0-9_./-]*)\]$")

@dataclass(frozen=True)
class Block:
    name: str
    start: int
    end: int
    depth: int
    parent: int | None


def locate_blocks(content: str) -> tuple[list[str], list[Block]]:
    lines = content.splitlines(keepends=True)
    stack: list[int] = []
    blocks: list[Block] = []
    quote = None
    for line_no, line in enumerate(lines):
        # Ignore brackets in comments and within single/multi-line quoted values.
        # A quote is escaped when preceded by an odd number of backslashes.
        visible = []
        j = 0
        while j < len(line):
            c = line[j]
            if quote is not None:
                if c == quote and (j == 0 or line[j - 1] != '\\'):
                    quote = None
                j += 1
                continue
            if c == '#':
                break
            if c in ("'", '"'):
                quote = c
                j += 1
                continue
            visible.append(c)
            j += 1
        token = ''.join(visible).strip()
        if token in ('[]', '[../]'):
            if not stack:
                raise ValueError(f"Unexpected MOOSE block closing tag on line {line_no + 1}")
            idx = stack.pop()
            b = blocks[idx]
            blocks[idx] = Block(b.name, b.start, line_no, b.depth, b.parent)
        else:
            m = _OPEN.fullmatch(token)
            if m:
                idx = len(blocks)
                blocks.append(Block(m.group(1), line_no, -1, len(stack), stack[-1] if stack else None))
                stack.append(idx)
    if stack:
        b = blocks[stack[-1]]
        raise ValueError(f"Unterminated MOOSE block [{b.name}] starting on line {b.start + 1}")
    return lines, blocks


def top_block(content: str, name: str):
    lines, blocks = locate_blocks(content)
    return lines, next((b for b in blocks if b.depth == 0 and b.name == name), None)


def direct_children(content: str, name: str) -> list[str]:
    _, blocks = locate_blocks(content)
    parents = [i for i, b in enumerate(blocks) if b.depth == 0 and b.name == name]
    return [b.name for b in blocks if b.parent in parents]


def block_body(content: str, name: str) -> str:
    lines, block = top_block(content, name)
    return ''.join(lines[block.start + 1:block.end]) if block else ''


# ----------------------------------------------------------------------
# Literal material values of a custom input file (no MOOSE evaluation)
# ----------------------------------------------------------------------

_NUMERIC_LITERAL = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")


@dataclass(frozen=True)
class CustomMaterial:
    """One [Materials] subsection of a custom GOLEM input file.

    values holds the parameters given as plain numeric literals, plus
    UserObject references (``<name>_uo``) whose UserObject declares a
    literal ``value`` (stored as ``<name>``). Expressions and functions
    are left out rather than guessed.
    """
    name: str
    values: dict[str, float]


def section_parameters(lines: list[str], block: Block) -> dict[str, str]:
    """Scalar ``key = value`` assignments directly inside a block, unevaluated."""
    result = {}
    for line in lines[block.start + 1:block.end]:
        match = re.match(r"^\s*([A-Za-z_][\w]*)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue
        value = match.group(2).split("#", 1)[0].strip()
        try:
            tokens = shlex.split(value)
        except ValueError:
            tokens = [value]
        result[match.group(1)] = " ".join(tokens)
    return result


def numeric_literal(value: str | None) -> float | None:
    """The value as a float if it is a plain numeric literal, else None."""
    if value is None or not _NUMERIC_LITERAL.fullmatch(value):
        return None
    return float(value)


def custom_material_blocks(content: str, block_count: int) -> dict[int, CustomMaterial]:
    """Map each mesh block (zero-based) to the [Materials] subsection assigned to it.

    Block references may be Exodus ids (1-based) or names given to them by a
    RenameBlockGenerator. Raises ValueError when a material has no explicit
    ``block``, references a block outside 1..block_count or an unknown name,
    or when two materials claim the same block.
    """
    lines, blocks = locate_blocks(content)
    material_parents = {i for i, b in enumerate(blocks) if b.depth == 0 and b.name == "Materials"}
    material_blocks = [b for b in blocks if b.parent in material_parents]
    if not material_blocks:
        raise ValueError("Custom GOLEM input has no [Materials] subsections to preview.")

    user_parents = {i for i, b in enumerate(blocks) if b.depth == 0 and b.name == "UserObjects"}
    user_objects = {b.name: section_parameters(lines, b) for b in blocks if b.parent in user_parents}

    renamed = {}
    for block in blocks:
        if block.name == "Materials" or block.parent in material_parents:
            continue
        params = section_parameters(lines, block)
        if params.get("type") != "RenameBlockGenerator":
            continue
        old_ids = params.get("old_block", "").split()
        new_names = params.get("new_block", "").split()
        if len(old_ids) != len(new_names):
            raise ValueError("RenameBlockGenerator old_block/new_block counts differ.")
        for number, label in zip(old_ids, new_names):
            renamed[label] = int(number)

    result: dict[int, CustomMaterial] = {}
    for block in material_blocks:
        params = section_parameters(lines, block)
        if "block" not in params:
            raise ValueError(f"Material [{block.name}] has no explicit block assignment; cannot preview safely.")
        values = {}
        for key, raw in params.items():
            if key in ("block", "type"):
                continue
            number = numeric_literal(raw)
            if number is not None:
                values[key] = number
            elif key.endswith("_uo") and raw in user_objects:
                number = numeric_literal(user_objects[raw].get("value"))
                if number is not None:
                    values[key[:-3]] = number
        material = CustomMaterial(name=block.name, values=values)
        for token in params["block"].split():
            try:
                block_id = int(token)
            except ValueError:
                if token not in renamed:
                    raise ValueError(f"Material [{block.name}] references unresolved block {token!r}.")
                block_id = renamed[token]
            if not 1 <= block_id <= block_count:
                raise ValueError(f"Material [{block.name}] references block {block_id}, outside 1..{block_count}.")
            previous = result.get(block_id - 1)
            if previous is not None and previous.name != block.name:
                raise ValueError(f"Block {block_id} is assigned to both [{previous.name}] and [{block.name}].")
            result[block_id - 1] = material
    if not result:
        raise ValueError("No explicit material block assignments found in custom GOLEM input.")
    return result
