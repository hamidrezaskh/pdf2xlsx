"""Geometry-based table layout detection.

Nothing in this module is document- or domain-specific: column edges,
row bands, headers and text directions are always inferred from the
actual word positions of the page.

Two complementary detectors are provided:

* :func:`detect_grids`   — vector ruling lines (table borders / grid lines
  drawn in the PDF) are connected into grids;
* :func:`detect_whitespace_candidates` — borderless tables are found by
  detecting vertical gaps (column separators) that are consistent across
  several consecutive text lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ...pdf import Segment, WordBox
from ...textutil import first_strong_direction, has_alpha, is_rtl_char  # re-exported


# ---------------------------------------------------------------------------
# text lines
# ---------------------------------------------------------------------------


@dataclass
class TextLine:
    """Words that sit on the same visual line."""

    words: List[WordBox] = field(default_factory=list)
    direction: str = "ltr"

    @property
    def x0(self) -> float:
        return min(w.x0 for w in self.words)

    @property
    def x1(self) -> float:
        return max(w.x1 for w in self.words)

    @property
    def y0(self) -> float:
        return min(w.y0 for w in self.words)

    @property
    def y1(self) -> float:
        return max(w.y1 for w in self.words)

    @property
    def height(self) -> float:
        return max(1.0, self.y1 - self.y0)

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2.0

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2.0

    @property
    def text(self) -> str:
        ordered = sorted(self.words, key=lambda w: w.x0, reverse=(self.direction == "rtl"))
        return " ".join(w.text for w in ordered)


def build_lines(words: Sequence[WordBox]) -> List[TextLine]:
    """Cluster words into visual lines by vertical overlap."""
    lines: List[TextLine] = []
    for word in sorted(words, key=lambda w: (w.y0, w.x0)):
        best: Optional[TextLine] = None
        best_overlap = 0.0
        for line in lines:
            overlap = min(line.y1, word.y1) - max(line.y0, word.y0)
            if overlap <= 0:
                continue
            ratio = overlap / max(1e-6, min(line.height, word.height))
            if ratio > best_overlap:
                best_overlap, best = ratio, line
        if best is not None and best_overlap >= 0.5:
            best.words.append(word)
        else:
            lines.append(TextLine(words=[word]))

    lines.sort(key=lambda ln: (ln.y0, ln.x0))
    for line in lines:
        line.direction = _line_direction(line)
    return lines


def _line_direction(line: TextLine) -> str:
    rtl = ltr = 0
    for word in line.words:
        if first_strong_direction(word.text) == "rtl":
            rtl += 1
        else:
            ltr += 1
    return "rtl" if rtl > ltr else "ltr"


def page_direction(lines: Sequence[TextLine]) -> str:
    """Dominant writing direction of the page (weighted by word count)."""
    rtl = sum(len(ln.words) for ln in lines if ln.direction == "rtl")
    ltr = sum(len(ln.words) for ln in lines if ln.direction == "ltr")
    return "rtl" if rtl > ltr else "ltr"


# ---------------------------------------------------------------------------
# grids from ruling lines
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """A detected rectangular table grid defined by its edge coordinates."""

    x_edges: List[float]
    y_edges: List[float]
    source: str = "ruling_lines"

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x_edges[0], self.y_edges[0], self.x_edges[-1], self.y_edges[-1])


def _lines_intersect(h: Segment, v: Segment, tol: float) -> bool:
    return (
        h.start - tol <= v.position <= h.end + tol
        and v.start - tol <= h.position <= v.end + tol
    )


def detect_grids(
    h_segments: Sequence[Segment],
    v_segments: Sequence[Segment],
    *,
    tol: float = 3.0,
    min_cols: int = 2,
    min_rows: int = 2,
    min_width: float = 50.0,
    min_height: float = 20.0,
) -> List[Grid]:
    """Connect intersecting ruling lines into table grids.

    A valid grid needs at least ``min_cols`` columns and ``min_rows`` rows,
    which means at least ``min_cols + 1`` distinct vertical lines and
    ``min_rows + 1`` distinct horizontal lines connected together.
    """
    h_lines = [s for s in h_segments if s.end - s.start > tol]
    v_lines = [s for s in v_segments if s.end - s.start > tol]
    if len(h_lines) < min_rows + 1 or len(v_lines) < min_cols + 1:
        return []

    n = len(h_lines) + len(v_lines)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    offset = len(h_lines)
    for hi, h in enumerate(h_lines):
        for vi, v in enumerate(v_lines):
            if _lines_intersect(h, v, tol):
                union(hi, offset + vi)

    components: Dict[int, Tuple[List[int], List[int]]] = {}
    for hi in range(len(h_lines)):
        components.setdefault(find(hi), ([], []))[0].append(hi)
    for vi in range(len(v_lines)):
        components.setdefault(find(offset + vi), ([], []))[1].append(vi)

    grids: List[Grid] = []
    for h_idx, v_idx in components.values():
        if len(h_idx) < min_rows + 1 or len(v_idx) < min_cols + 1:
            continue
        y_edges = sorted({round(h_lines[i].position, 2) for i in h_idx})
        x_edges = sorted({round(v_lines[i].position, 2) for i in v_idx})
        # drop near-duplicate edges (within tolerance)
        x_edges = _dedupe_edges(x_edges, tol)
        y_edges = _dedupe_edges(y_edges, tol)
        if len(x_edges) < min_cols + 1 or len(y_edges) < min_rows + 1:
            continue
        if x_edges[-1] - x_edges[0] < min_width or y_edges[-1] - y_edges[0] < min_height:
            continue
        grids.append(Grid(x_edges=x_edges, y_edges=y_edges, source="ruling_lines"))

    grids.sort(key=lambda g: (round(g.bbox[1], 1), round(g.bbox[0], 1)))
    return grids


def _dedupe_edges(edges: List[float], tol: float) -> List[float]:
    result: List[float] = []
    for value in edges:
        if not result or value - result[-1] > tol:
            result.append(value)
        # else: keep the earlier edge (within tolerance)
    return result


# ---------------------------------------------------------------------------
# borderless (whitespace) detection
# ---------------------------------------------------------------------------


def _median(values: Sequence[float], default: float = 10.0) -> float:
    if not values:
        return default
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def detect_blocks(lines: Sequence[TextLine], *, gap_factor: float = 2.5) -> List[List[TextLine]]:
    """Group lines into blocks separated by large vertical gaps."""
    if not lines:
        return []
    heights = [ln.height for ln in lines]
    median_h = _median(heights)
    threshold = max(median_h * gap_factor, median_h + 3.0)

    blocks: List[List[TextLine]] = []
    current: List[TextLine] = []
    for line in sorted(lines, key=lambda ln: ln.y0):
        if current and line.y0 - max(ln.y1 for ln in current) > threshold:
            blocks.append(current)
            current = []
        current.append(line)
    if current:
        blocks.append(current)
    return blocks


def find_column_separators(
    block: Sequence[TextLine], *, tolerance: float, min_gap: float, coverage: float = 0.75
) -> List[float]:
    """Find x positions where a vertical gap is consistent across lines.

    These positions become column boundaries of borderless tables.  A
    candidate needs votes from ``max(2, ceil(coverage * n_lines))``
    *distinct* lines of the block, which prose almost never satisfies
    but table columns do.  Coverage counts the lines that actually fell
    into the cluster (not lines merely within tolerance afterwards),
    which prevents two near-duplicate separators.
    """
    n_lines = len(block)
    if n_lines < 2:
        return []
    import math

    required = max(2, math.ceil(coverage * n_lines))

    candidates: List[Tuple[float, int]] = []  # (gap midpoint, line index)
    for line_index, line in enumerate(block):
        ordered = sorted(line.words, key=lambda w: w.x0)
        for left, right in zip(ordered, ordered[1:]):
            if right.x0 - left.x1 >= min_gap:
                candidates.append(((left.x1 + right.x0) / 2.0, line_index))
    if not candidates:
        return []

    clusters: List[List[Tuple[float, int]]] = []
    for candidate in sorted(candidates, key=lambda c: c[0]):
        value = candidate[0]
        if clusters:
            mean = sum(v for v, _ in clusters[-1]) / len(clusters[-1])
            if value - mean <= tolerance:
                clusters[-1].append(candidate)
                continue
        clusters.append([candidate])

    separators: List[float] = []
    for members in clusters:
        voters = {line_index for _, line_index in members}
        if len(voters) >= required:
            separators.append(sum(v for v, _ in members) / len(members))
    return sorted(separators)


@dataclass
class WhitespaceCandidate:
    """A borderless-table candidate found from consistent column gaps."""

    x_bands: List[Tuple[float, float]]
    lines: List[TextLine]
    bbox: Tuple[float, float, float, float]


def detect_whitespace_candidates(
    lines: Sequence[TextLine],
    *,
    gap_factor: float = 2.5,
    coverage: float = 0.75,
    min_band: float = 8.0,
) -> List[WhitespaceCandidate]:
    """Detect borderless tables by consistent vertical gaps between columns."""
    if not lines:
        return []
    median_h = _median([ln.height for ln in lines])
    tolerance = max(2.0, median_h * 0.6)
    min_gap = max(2.0, median_h * 0.5)

    candidates: List[WhitespaceCandidate] = []
    for block in detect_blocks(lines, gap_factor=gap_factor):
        if len(block) < 2:
            continue
        separators = find_column_separators(
            block, tolerance=tolerance, min_gap=min_gap, coverage=coverage
        )
        if not separators:
            continue
        x0 = min(ln.x0 for ln in block)
        x1 = max(ln.x1 for ln in block)

        kept: List[float] = []
        for sep in sorted(separators):
            left_edge = kept[-1] if kept else x0
            if sep - left_edge >= min_band and x1 - sep >= min_band:
                kept.append(sep)
        if not kept:
            continue
        edges = [x0] + kept + [x1]
        bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        if len(bands) < 2:
            continue

        y0 = min(ln.y0 for ln in block)
        y1 = max(ln.y1 for ln in block)
        candidates.append(
            WhitespaceCandidate(
                x_bands=bands,
                lines=sorted(block, key=lambda ln: ln.y0),
                bbox=(x0, y0, x1, y1),
            )
        )
    return candidates


__all__ = [
    "is_rtl_char",
    "first_strong_direction",
    "has_alpha",
    "TextLine",
    "build_lines",
    "page_direction",
    "Grid",
    "detect_grids",
    "detect_blocks",
    "find_column_separators",
    "WhitespaceCandidate",
    "detect_whitespace_candidates",
]


