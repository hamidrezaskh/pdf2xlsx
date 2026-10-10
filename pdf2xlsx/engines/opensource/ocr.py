"""Pluggable OCR backends for the open-source processor.

Selected with the ``OCR_MODEL`` environment variable:

* ``easyocr``   (default) — supports Persian ``fa`` and English ``en``
* ``tesseract`` — requires the Tesseract binary on PATH
* ``native``    — OCR disabled; only embedded PDF text is used

All backends return the same ``WordBox`` list.  Models are loaded lazily
and cached at module level so they are **not** reloaded on every request.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

from ...config import Settings
from ...exceptions import ConfigurationError, OcrError
from ...pdf import WordBox

logger = logging.getLogger(__name__)

SUPPORTED_OCR_MODELS = ("easyocr", "tesseract", "trocr", "trocr+gemma", "paddleocr", "paddle", "native")

# module-level caches: one model instance per (languages, device) per process
_EASYOCR_READERS: Dict[Tuple[str, ...], Any] = {}


def resolve_device(device: str) -> str:
    """Resolve the DEVICE setting ('auto' -> cuda when available)."""
    value = (device or "auto").strip().lower()
    if value in ("cpu", ""):
        return "cpu"
    if value == "auto":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return value  # "cuda" or anything else is passed through


class OcrBackend:
    """Interface every OCR backend implements."""

    name = "ocr"

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:
        raise NotImplementedError


class NativeTextBackend(OcrBackend):
    """OCR disabled: the embedded PDF text layer is used instead."""

    name = "native"

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:  # pragma: no cover
        raise OcrError("OCR is disabled (OCR_MODEL=native).")


class EasyOcrBackend(OcrBackend):
    name = "easyocr"

    def __init__(self, languages: Sequence[str], device: str) -> None:
        self.languages = tuple(languages)
        self.device = resolve_device(device)
        self._reader: Any | None = None

    def _ensure(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            import easyocr  # type: ignore
        except ImportError as exc:
            raise OcrError(
                "EasyOCR is not installed. Install the OCR extras with: "
                "pip install -r requirements-ocr.txt"
            ) from exc

        key = (self.languages, self.device)
        reader = _EASYOCR_READERS.get(key)
        if reader is None:
            logger.info(
                "Loading EasyOCR model (languages=%s, gpu=%s)...",
                ",".join(self.languages),
                self.device == "cuda",
            )
            try:
                reader = easyocr.Reader(
                    list(self.languages), gpu=(self.device == "cuda"), verbose=False
                )
            except Exception as exc:
                raise OcrError(
                    "The OCR model could not be loaded. Check OCR_MODEL / OCR_LANGUAGES / DEVICE.",
                    detail=f"{type(exc).__name__}: {exc}",
                ) from exc
            _EASYOCR_READERS[key] = reader
        self._reader = reader
        return reader

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:
        import numpy as np  # hard dependency of easyocr

        reader = self._ensure()
        try:
            results = reader.readtext(np.asarray(image.convert("RGB")))
        except Exception as exc:
            raise OcrError(
                f"OCR failed while processing page {page_number}.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        words: List[WordBox] = []
        for item in results:
            if not isinstance(item, (tuple, list)) or len(item) < 3:
                continue
            box, text, conf = item[0], item[1], item[2]
            if not text or not str(text).strip():
                continue
            xs = [float(p[0]) for p in box]
            ys = [float(p[1]) for p in box]
            words.append(
                WordBox(
                    text=str(text),
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                    confidence=float(conf),
                )
            )
        return words


class TesseractBackend(OcrBackend):
    name = "tesseract"

    _LANG_MAP = {"fa": "fas", "en": "eng", "ar": "ara", "de": "deu", "fr": "fra"}

    def __init__(self, languages: Sequence[str], device: str) -> None:
        codes = [self._LANG_MAP.get(lang.strip(), lang.strip()) for lang in languages]
        self.lang = "+".join(codes) or "eng"
        self.device = resolve_device(device)

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:
        try:
            import pytesseract  # type: ignore
        except ImportError as exc:
            raise OcrError(
                "pytesseract is not installed. Install it with: pip install pytesseract "
                "(and make sure the Tesseract binary is on PATH)."
            ) from exc

        try:
            data = pytesseract.image_to_data(
                image.convert("RGB"), lang=self.lang, output_type=pytesseract.Output.DICT
            )
        except Exception as exc:
            raise OcrError(
                "Tesseract failed while processing a page. Is the Tesseract binary installed?",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc

        words: List[WordBox] = []
        n = len(data.get("text", []))
        for i in range(n):
            text = (data["text"][i] or "").strip()
            if not text:
                continue
            try:
                conf = float(data["conf"][i])
            except (KeyError, TypeError, ValueError):
                conf = -1.0
            if conf < 0:  # tesseract marks junk with -1
                continue
            x, y = float(data["left"][i]), float(data["top"][i])
            w, h = float(data["width"][i]), float(data["height"][i])
            if w <= 0 or h <= 0:
                continue
            words.append(
                WordBox(text=text, x0=x, y0=y, x1=x + w, y1=y + h, confidence=conf / 100.0)
            )
        return words


class PaddleOcrBackend(OcrBackend):
    """PaddleOCR backend (multilingual incl. Persian/Arabic + Latin).

    Lazy-loaded and cached per (languages, device). Returns one
    :class:`WordBox` per detected word/line token with polygon-derived
    bounding boxes and mean confidence.
    """

    name = "paddleocr"

    # PaddleOCR lang codes: fa is supported in PP-OCRv4 multilingual models;
    # fall back to 'ar' + 'en' when 'fa' weights are unavailable.
    _LANG_MAP = {"fa": "fa", "en": "en", "ar": "ar", "de": "german", "fr": "fr"}

    def __init__(self, languages: Sequence[str], device: str) -> None:
        langs = [self._LANG_MAP.get(l.strip().lower(), l.strip()) for l in languages if l.strip()]
        self.languages = tuple(langs) or ("fa", "en")
        # PaddleOCR uses 'cpu'/'gpu' naming.
        resolved = resolve_device(device)
        self.device = "gpu" if resolved == "cuda" else "cpu"
        self._ocr: Any | None = None

    def _ensure(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError as exc:
            raise OcrError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr paddlepaddle"
            ) from exc
        # Try requested langs in order; PaddleOCR takes a single `lang`.
        last_exc: Exception | None = None
        for lang in list(self.languages) + ["ar", "en"]:
            try:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=lang, device=self.device)
                self.languages = (lang,)
                logger.info("PaddleOCR loaded (lang=%s, device=%s)", lang, self.device)
                return self._ocr
            except Exception as exc:  # pragma: no cover - model download issues
                last_exc = exc
                continue
        raise OcrError(
            "PaddleOCR could not be initialised for languages "
            f"{','.join(self.languages)}.",
            detail=f"{type(last_exc).__name__}: {last_exc}" if last_exc else None,
        ) from last_exc

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:
        import numpy as np

        ocr = self._ensure()
        try:
            results = ocr.ocr(np.asarray(image.convert("RGB")))
        except Exception as exc:
            raise OcrError(
                f"OCR failed while processing page {page_number}.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        words: List[WordBox] = []
        # PaddleOCR returns [[ [box, (text, conf)], ... ]] per page.
        pages = results if isinstance(results, list) else []
        lines = pages[0] if pages and isinstance(pages[0], list) else pages
        for item in lines or []:
            try:
                box, (text, conf) = item[0], item[1]
            except (IndexError, TypeError, ValueError):
                continue
            if not text or not str(text).strip():
                continue
            # Split line text into word tokens; distribute box width evenly.
            tokens = str(text).split()
            try:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
            except (TypeError, ValueError):
                continue
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            if not tokens:
                continue
            w = (x1 - x0) / len(tokens)
            for i, tok in enumerate(tokens):
                words.append(
                    WordBox(
                        text=tok,
                        x0=x0 + i * w,
                        y0=y0,
                        x1=x0 + (i + 1) * w,
                        y1=y1,
                        confidence=float(conf),
                    )
                )
        return words


def create_ocr_backend(settings: Settings) -> OcrBackend | None:
    """Create the configured OCR backend (None when OCR is disabled)."""
    model = (settings.ocr_model or "").strip().lower()
    if model in ("native", "none", "off", "disabled"):
        return None
    if model == "easyocr":
        return EasyOcrBackend(settings.ocr_languages, settings.device)
    if model == "tesseract":
        return TesseractBackend(settings.ocr_languages, settings.device)
    if model in ("paddleocr", "paddle", "paddle-ocr"):
        paddle_langs = getattr(settings, "paddleocr_languages", settings.ocr_languages)
        return PaddleOcrBackend(paddle_langs, settings.device)
    if model in ("trocr", "trocr+gemma", "trocr_gemma", "trocr-gemma"):
        # Persian TrOCR line model (hezarai/trocr-base-fa-v2 by default),
        # optionally corrected by Gemma 2B (OCR_POSTPROCESSOR=gemma or
        # OCR_MODEL=trocr+gemma as a shorthand).
        from .gemma_postprocessor import create_corrector  # keep heavy imports lazy
        from .trocr_ocr import TrocrBackend  # keep heavy imports lazy

        postprocessor = getattr(settings, "ocr_postprocessor", "none") or "none"
        if model != "trocr":  # shorthand OCR_MODEL=trocr+gemma implies correction
            postprocessor = "gemma" if postprocessor in ("none", "trocr") else postprocessor
        corrector = create_corrector(
            postprocessor,
            model_id=getattr(settings, "gemma_model", "google/gemma-2-2b-it"),
            device=resolve_device(settings.device),
            max_new_tokens=int(getattr(settings, "gemma_max_new_tokens", 256)),
        )
        # NoOpCorrector -> plain TrOCR (no extra dependency loaded).
        from .gemma_postprocessor import NoOpCorrector

        return TrocrBackend(
            settings.trocr_model,
            resolve_device(settings.device),
            corrector=None if isinstance(corrector, NoOpCorrector) else corrector,
        )
    raise ConfigurationError(
        f"Unknown OCR_MODEL '{settings.ocr_model}'. "
        f"Supported values: {', '.join(SUPPORTED_OCR_MODELS)}."
    )


__all__ = [
    "OcrBackend",
    "NativeTextBackend",
    "EasyOcrBackend",
    "TesseractBackend",
    "PaddleOcrBackend",
    "create_ocr_backend",
    "resolve_device",
    "SUPPORTED_OCR_MODELS",
]

