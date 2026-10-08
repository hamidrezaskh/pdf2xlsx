"""Text utilities shared by PDF processing and layout detection.

Provides direction analysis and a *non-destructive* RTL word assembly:

* glyph positions in a PDF are always visual, so sorting a word's
  characters by x yields the visual order;
* for RTL words the visual order is reversed to obtain the logical
  reading order, while embedded LTR runs (Persian/Latin digits, Latin
  fragments) are restored;
* Arabic presentation forms (a PDF shaping artifact) are mapped back to
  their canonical base letters via NFKC — applied ONLY to those code
  point ranges, so no other text is ever normalized.
"""

from __future__ import annotations

import unicodedata
from typing import Sequence

#: Blocks that contain right-to-left scripts (Hebrew / Arabic / ...).
RTL_RANGES = (
    (0x0590, 0x05FF),  # Hebrew
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended
    (0xFB1D, 0xFDFF),  # Alphabetic Presentation Forms
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms
)

#: Shaped glyphs produced by PDF writers; safe to map back to base letters.
SHAPED_RANGES = (
    (0xFB1D, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def is_rtl_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in RTL_RANGES)


def _is_shaped_char(ch: str) -> bool:
    code = ord(ch)
    return any(start <= code <= end for start, end in SHAPED_RANGES)


def is_strong_rtl_char(ch: str) -> bool:
    """RTL *letter* (digits and punctuation are direction-weak)."""
    return is_rtl_char(ch) and unicodedata.category(ch).startswith("L")


def first_strong_direction(text: str) -> str:
    """'rtl' when the first strong directional *letter* is RTL, else 'ltr'.

    Digits, spaces and punctuation are direction-weak and ignored.
    """
    for ch in text:
        if not ch.isalpha():
            continue
        return "rtl" if is_rtl_char(ch) else "ltr"
    return "ltr"


def has_alpha(value: str) -> bool:
    return any(ch.isalpha() for ch in value)


def normalize_shaping(text: str) -> str:
    """Map Arabic presentation forms to canonical letters (only those)."""
    if not any(_is_shaped_char(ch) for ch in text):
        return text
    return "".join(
        unicodedata.normalize("NFKC", ch) if _is_shaped_char(ch) else ch for ch in text
    )


def line_base_direction(chars: Sequence[str]) -> str:
    """Majority letter direction of a line (shaping-normalized).

    Paragraph/line direction is the reliable context for word-level bidi:
    it does not depend on any single word's visual order.
    """
    text = normalize_shaping("".join(chars))
    rtl = sum(1 for ch in text if is_strong_rtl_char(ch))
    ltr = sum(1 for ch in text if ch.isalpha() and not is_strong_rtl_char(ch))
    return "rtl" if rtl > ltr else "ltr"


def assemble_word_chars(chars: Sequence[str], base_dir: str | None = None) -> str:
    """Build the logical text of one word from its visual (x-ordered) chars.

    ``chars`` must be sorted left to right (visual order).  Shaped
    presentation forms are first mapped back to canonical letters.
    Words without strong RTL letters (Latin, digits) are returned as-is.
    Otherwise the sequence is reversed to logical order and embedded
    weak/LTR runs (digits, Latin fragments) are restored to their own
    order.  ``base_dir`` (the line's majority direction, when known)
    resolves mixed-script words whose first *visible* letter is Latin.
    """
    seq = [c for c in chars if c and not c.isspace()]
    if not seq:
        return ""
    visual = normalize_shaping("".join(seq))  # presentation forms -> base letters
    if not any(is_strong_rtl_char(ch) for ch in visual):
        return visual  # pure LTR / digits / punctuation
    if base_dir == "ltr" and first_strong_direction(visual) != "rtl":
        return visual  # LTR context and no visible RTL run start
    seq = list(visual)
    seq.reverse()
    i = 0
    while i < len(seq):
        if is_strong_rtl_char(seq[i]):
            i += 1
            continue
        j = i
        while j < len(seq) and not is_strong_rtl_char(seq[j]):
            j += 1
        seq[i:j] = reversed(seq[i:j])
        i = j
    return "".join(seq)


__all__ = [
    "RTL_RANGES",
    "SHAPED_RANGES",
    "is_rtl_char",
    "is_strong_rtl_char",
    "first_strong_direction",
    "has_alpha",
    "normalize_shaping",
    "line_base_direction",
    "assemble_word_chars",
]
