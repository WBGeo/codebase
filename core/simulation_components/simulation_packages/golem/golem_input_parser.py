"""Small, line-oriented MOOSE/HIT block locator shared by GOLEM components.

Recognises structural bracket lines, not arbitrary bracket text in values.
This is not a complete HIT grammar; MOOSE remains the final syntax authority.
"""
from dataclasses import dataclass
import re

_OPEN = re.compile(r"^\[(?:\./)?([A-Za-z_][A-Za-z0-9_]*)\]$")

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
