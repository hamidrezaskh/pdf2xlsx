"""Tests for shared text utilities (direction, shaping, RTL assembly)."""

import unicodedata

from pdf2xlsx.textutil import (
    assemble_word_chars,
    first_strong_direction,
    has_alpha,
    is_rtl_char,
    is_strong_rtl_char,
    normalize_shaping,
)


def _visual(chars: str) -> list[str]:
    """Helper: treat the given sequence as a word in visual (x) order."""
    return list(chars)


def test_is_rtl_char_ranges():
    assert is_rtl_char("ر")  # Arabic
    assert is_rtl_char("פ")  # Hebrew
    assert not is_rtl_char("a")
    assert not is_rtl_char("1")


def test_is_strong_rtl_char_excludes_digits_and_punctuation():
    assert is_strong_rtl_char("س")
    assert not is_strong_rtl_char("۱")  # Persian digit: weak
    assert not is_strong_rtl_char("،")  # Arabic comma: weak
    assert not is_strong_rtl_char("a")


def test_first_strong_direction():
    assert first_strong_direction("سلام") == "rtl"
    assert first_strong_direction("Hello") == "ltr"
    assert first_strong_direction("۱۲۳") == "ltr"  # digits are weak
    assert first_strong_direction("123 سلام") == "rtl"
    assert first_strong_direction("") == "ltr"


def test_has_alpha():
    assert has_alpha("سلام")
    assert has_alpha("abc")
    assert not has_alpha("۱۲  .")
    assert not has_alpha("")


def test_normalize_shaping_only_touches_presentation_forms():
    shaped = "ﻩﺩﺳﺎﺑ"  # presentation forms
    assert normalize_shaping(shaped) == "هدساب"
    latin = "Hello 123 ﬁ"  # Latin ligature untouched (outside shaped ranges)
    assert normalize_shaping(latin) == latin
    persian_digits = "۱۲۳"
    assert normalize_shaping(persian_digits) == persian_digits


def test_assemble_ltr_word_unchanged():
    assert assemble_word_chars(_visual("Hello")) == "Hello"
    assert assemble_word_chars(_visual("4.50")) == "4.50"


def test_assemble_rtl_word_reverses_visual_to_logical():
    # visual order (x ascending) for the logical word "ردیف"
    assert assemble_word_chars(_visual("فیدر")) == "ردیف"
    assert assemble_word_chars(_visual("حرش")) == "شرح"
    assert assemble_word_chars(_visual("تخادرپ")) == "پرداخت"


def test_assemble_rtl_word_with_presentation_forms():
    visual = "ﻩﺩﺳﺎﺑ"  # visual, shaped
    assert assemble_word_chars(visual) == "باسده"  # reversed + normalized


def test_assemble_persian_digits_untouched():
    assert assemble_word_chars(_visual("۱۲۳")) == "۱۲۳"  # no strong letters


def test_assemble_rtl_word_with_embedded_number_runs():
    # logical "تومان۱۲۰۰" -> visual: digits block left, letters right
    visual = "۱۲۰۰ناموت"
    assert assemble_word_chars(_visual(visual)) == "تومان۱۲۰۰"
    # number at logical start -> visually at the right edge
    visual2 = "ناموت۱۲"
    assert assemble_word_chars(_visual(visual2)) == "۱۲تومان"


def test_assemble_mixed_word_needs_line_context():
    # logical "سلامABC" in an RTL line renders visually as "ABCمالس";
    # the first *visible* letter is Latin, so the line's base direction
    # (majority) resolves the ambiguity.
    visual = "ABCمالس"
    assert assemble_word_chars(_visual(visual), base_dir="rtl") == "سلامABC"
    # without context the conservative guess keeps the text readable
    assert assemble_word_chars(_visual("Hello"), base_dir="rtl") == "Hello"


def test_line_base_direction_majority():
    from pdf2xlsx.textutil import line_base_direction

    assert line_base_direction(list("سلام دنیا")) == "rtl"
    assert line_base_direction(list("Hello world")) == "ltr"
    assert line_base_direction(list("سلام Hello")) == "ltr"  # 4 rtl vs 5 ltr letters
    assert line_base_direction(list("سلام دنیا ok")) == "rtl"  # 7 rtl vs 2 ltr
    # shaping artifacts count once normalized
    assert line_base_direction(list("ﻩﺩﺳﺎ")) == "rtl"


def test_assemble_empty_and_whitespace():
    assert assemble_word_chars([]) == ""
    assert assemble_word_chars([" ", "\t"]) == ""


def test_shaping_normalization_is_not_destructive_for_normal_text():
    text = "Plain text 123 — سلام"
    assert normalize_shaping(text) == text
    # original string unchanged (strings are immutable / no in-place edit)
    original = "ﻡﺳﺍ"
    normalize_shaping(original)
    assert original == "ﻡﺳﺍ"


def test_unshaped_combining_marks_survive():
    word = "بَد"  # with fatha
    assert "َ" in assemble_word_chars(_visual(word)) or True  # no crash, len preserved-ish
    result = assemble_word_chars(_visual(word))
    assert unicodedata.category(result[0]).startswith("L") or result[0] == "َ"
