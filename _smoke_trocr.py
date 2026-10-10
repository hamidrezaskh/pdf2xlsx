"""Smoke test: load hezarai/trocr-base-fa-v2 and recognize a Persian line.

Run with the .venv interpreter AFTER torch + transformers are installed.
First run downloads the weights (~935 MB) from Hugging Face.
Result goes to stdout (captured in trocr_smoke.log by the waiter script).
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, r"D:\pdf2xlsx")

from PIL import Image  # noqa: E402

FONT = r"C:\Windows\Fonts\tahoma.ttf"
TEXT = "سلام دنیا ۱۲۳"


def render_line() -> Image.Image:
    """Render the Persian line with PyMuPDF's HTML engine (proper bidi)."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=420, height=80)
    css = (
        f"@font-face {{font-family: fa; src: url({Path(FONT).as_posix()});}} "
        "* {font-family: fa; font-size: 24pt; margin: 0; padding: 0;}"
    )
    spare, _ = page.insert_htmlbox(pymupdf.Rect(15, 15, 405, 70), TEXT, css=css)
    if spare < 0:  # pragma: no cover
        raise RuntimeError("could not render the probe line")
    png = page.get_pixmap(dpi=150).tobytes("png")
    doc.close()
    return Image.open(io.BytesIO(png))


def main() -> int:
    image = render_line()

    from pdf2xlsx.engines.opensource.trocr_ocr import get_recognizer  # noqa: E402

    print("loading model (first run downloads ~935 MB)...", flush=True)
    recognizer = get_recognizer(None, "cpu")
    print("recognizer type:", type(recognizer).__name__, flush=True)
    text, confidence = recognizer.recognize(image)
    print("RECOGNIZED:", repr(text), flush=True)
    print("CONFIDENCE:", confidence, flush=True)
    print("SMOKE_OK" if "سلام" in text else "SMOKE_MISMATCH", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
