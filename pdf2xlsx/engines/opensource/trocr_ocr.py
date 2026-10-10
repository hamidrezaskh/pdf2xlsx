"""OCR backend for the Persian TrOCR line model (hezarai/trocr-base-fa-v2).

The model is a *line-level* recognizer (image → text), so this backend
adds the missing page → lines → words pipeline:

1. binarize the page (Otsu, pure numpy — no OpenCV);
2. mask long ruling lines so table borders don't merge text rows;
3. horizontal projection → text-line bands;
4. per band: crop → TrOCR → tokens; word boxes come from the column
   projection of the band, aligned to the tokens (RTL-aware).

The model loads lazily (cached per process) through the official
``hezar`` API or a built-in ``transformers`` loader (the repo ships
``model.pt`` + ``tokenizer.json``, not standard transformers files).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ...config import DEFAULT_TROCR_MODEL
from ...exceptions import OcrError
from ...pdf import WordBox
from ...textutil import first_strong_direction
from .ocr import OcrBackend

logger = logging.getLogger(__name__)

_IMAGE_SIZE = (224, 224)  # matches preprocessor/image_processor_config.yaml

# module-level cache: one recogniser per (model id, device)
_RECOGNIZERS: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class Band:
    """A horizontal text-line band (y1 exclusive)."""

    y0: int
    y1: int

    @property
    def height(self) -> int:
        return max(1, self.y1 - self.y0)


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu threshold on an 8-bit grayscale image."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    sum_all = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0.0
    best_t, best_var = 127, -1.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return int(best_t)


def binarize(gray: np.ndarray) -> np.ndarray:
    """True where ink is present (dark pixels).

    Otsu's class boundary is inclusive (``<=``) so perfectly binary
    images with threshold 0 still keep their black pixels.
    """
    return gray <= otsu_threshold(gray)


def _max_run_lengths(mask: np.ndarray, axis: int) -> np.ndarray:
    """Longest True run per row (axis=1) or per column (axis=0)."""
    rows = mask if axis == 1 else mask.T
    out = np.zeros(rows.shape[0], dtype=np.int32)
    for i, row in enumerate(rows):
        idx = np.flatnonzero(row)
        if idx.size == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        parts = np.split(idx, splits + 1)
        out[i] = max(part.size for part in parts)
    return out


def _thin_strips(candidates: np.ndarray, max_thickness: int) -> List[Tuple[int, int]]:
    """Group candidate indices into strips no thicker than ``max_thickness``."""
    strips: List[Tuple[int, int]] = []
    start: Optional[int] = None
    last: Optional[int] = None
    gap = 0
    for i in range(len(candidates) + 1):
        present = i < len(candidates) and bool(candidates[i])
        if present:
            if start is None:
                start = i
            last = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 1 and last is not None:  # strip ended
                if (last - start + 1) <= max_thickness:
                    strips.append((start, last + 1))
                start, last = None, None
                gap = 0
    return strips


def mask_long_lines(
    ink: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Remove long table ruling lines from the ink mask.

    Returns ``(clean_mask, row_strips, col_strips)`` — thin strips that
    span most of the page width/height are borders, not text (headings
    are long but thick, so they survive).
    """
    height, width = ink.shape
    clean = ink.copy()

    min_h_run = max(20, int(round(0.8 * width)))
    max_h_thickness = max(3, int(round(0.005 * height)))
    cand_rows = _max_run_lengths(ink, axis=1) >= min_h_run
    row_strips = _thin_strips(cand_rows, max_h_thickness)
    for y0, y1 in row_strips:
        clean[y0:y1, :] = False

    min_v_run = max(20, int(round(0.8 * height)))
    max_v_thickness = max(3, int(round(0.005 * width)))
    cand_cols = _max_run_lengths(ink, axis=0) >= min_v_run
    col_strips = _thin_strips(cand_cols, max_v_thickness)
    for x0, x1 in col_strips:
        clean[:, x0:x1] = False

    return clean, row_strips, col_strips


def segment_bands(clean: np.ndarray, *, min_ink: int = 6, gap_tol: int = 2) -> List[Band]:
    """Horizontal projection → text-line bands."""
    projection = clean.sum(axis=1)
    bands: List[Band] = []
    start: Optional[int] = None
    end: Optional[int] = None
    ink_total = 0
    gap = 0

    def close() -> None:
        nonlocal start, end, ink_total, gap
        if start is not None and end is not None and ink_total >= min_ink:
            bands.append(Band(y0=start, y1=end + 1))
        start, end, ink_total, gap = None, None, 0, 0

    for y, count in enumerate(projection):
        if count > 0:
            if start is None:
                start = y
            end = y
            ink_total += int(count)
            gap = 0
        elif start is not None:
            gap += 1
            if gap > gap_tol:
                close()
    close()
    return bands


def band_word_boxes(
    clean: np.ndarray, band: Band, *, gap_factor: float = 0.30
) -> List[Tuple[int, int]]:
    """Word boxes of one band from its column projection."""
    strip = clean[band.y0 : band.y1]
    columns = strip.any(axis=0)
    idx = np.flatnonzero(columns)
    if idx.size == 0:
        return []
    min_gap = max(2, int(round(band.height * gap_factor)))
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > min_gap) + 1)
    return [(int(part[0]), int(part[-1]) + 1) for part in parts if part.size]


# ---------------------------------------------------------------------------
# token ↔ box alignment
# ---------------------------------------------------------------------------


def align_tokens_to_boxes(
    text: str, boxes: Sequence[Tuple[int, int]]
) -> List[Tuple[str, Tuple[int, int]]]:
    """Pair OCR tokens with image word boxes (RTL-aware).

    ``boxes`` are in visual order (x ascending); tokens are in logical
    reading order.  Mismatched counts are reconciled by merging the
    closest boxes or splitting the widest ones.
    """
    tokens = text.split()
    if not tokens or not boxes:
        return []
    boxes = sorted(boxes)

    while len(boxes) > len(tokens):  # merge neighbours with the smallest gap
        gaps = [(boxes[i + 1][0] - boxes[i][1], i) for i in range(len(boxes) - 1)]
        if not gaps:
            break
        _, i = min(gaps)
        boxes = boxes[:i] + [(boxes[i][0], boxes[i + 1][1])] + boxes[i + 2 :]

    while len(boxes) < len(tokens):  # split the widest box in two
        widths = [(boxes[i][1] - boxes[i][0], i) for i in range(len(boxes))]
        _, i = max(widths)
        x0, x1 = boxes[i]
        mid = (x0 + x1) // 2
        if mid <= x0 or mid >= x1:  # cannot split further
            boxes = boxes + [boxes[-1]]
            continue
        boxes = boxes[:i] + [(x0, mid), (mid, x1)] + boxes[i + 1 :]

    if first_strong_direction(text) == "rtl":
        boxes = list(reversed(boxes))  # visual x-ascending → logical order
    return list(zip(tokens, boxes))


# ---------------------------------------------------------------------------
# model recognisers (lazy, cached)
# ---------------------------------------------------------------------------


class _HezarRecogniser:
    """Official loader: ``from hezar.models import Model; Model.load(...)``."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    def _extract_text(raw: Any) -> str:
        item = raw[0] if isinstance(raw, (list, tuple)) and raw else raw
        if isinstance(item, dict):
            item = item.get("generated_text") or item.get("text") or ""
        return str(item)

    def recognize(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            path = handle.name
        try:
            image.save(path)
            raw = self.model.predict([path])
        except Exception as exc:
            raise OcrError(
                "TrOCR inference failed.", detail=f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            try:
                import os

                os.unlink(path)
            except OSError:  # pragma: no cover
                pass
        return self._extract_text(raw), None


class _TransformersRecogniser:
    """Fallback loader for the standard TrOCR architecture of this model."""

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def recognize(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        import torch

        resized = image.convert("RGB").resize(_IMAGE_SIZE, resample=Image.BILINEAR)
        array = np.asarray(resized).astype(np.float32) / 255.0
        array = (array - 0.5) / 0.5  # mean/std 0.5 per image_processor_config
        values = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        if self.device != "cpu":
            values = values.to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                values,
                max_length=64,
                num_beams=4,
                length_penalty=2.0,
                no_repeat_ngram_size=3,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
        text = self.tokenizer.decode(output.sequences[0], skip_special_tokens=True)
        confidence: Optional[float] = None
        scores = getattr(output, "scores", None)
        if scores:
            token_probs = [torch.softmax(step, dim=-1).max(dim=-1).values for step in scores]
            confidence = float(torch.stack([p[0] for p in token_probs]).mean())
        return text.strip(), confidence


# ---------------------------------------------------------------------------
# model recognisers (lazy, cached)
# ---------------------------------------------------------------------------

# Architecture of hezarai/trocr-base-fa-v2 from its model_config.yaml
# (used only by the built-in transformers fallback loader).
_ENCODER_KW = dict(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.0,
    image_size=224,
    patch_size=16,
    num_channels=3,
    qkv_bias=True,
)
_DECODER_KW = dict(
    vocab_size=42000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    max_position_embeddings=514,
    type_vocab_size=1,
    is_decoder=True,
    add_cross_attention=True,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    position_embedding_type="absolute",
    use_cache=True,
)


def _load_with_transformers(model_id: str, device: str) -> "_TransformersRecogniser":
    """Built-in loader: model.pt + tokenizer.json via huggingface_hub."""
    try:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import (
            PreTrainedTokenizerFast,
            RobertaConfig,
            ViTConfig,
            VisionEncoderDecoderConfig,
            VisionEncoderDecoderModel,
        )
    except ImportError as exc:
        raise OcrError(
            "TrOCR dependencies are not installed. Install the OCR extras with: "
            "pip install -r requirements-ocr.txt (torch + transformers).",
            detail=str(exc),
        ) from exc

    logger.info("Loading TrOCR weights via transformers: %s", model_id)
    try:
        weights_path = hf_hub_download(model_id, "model.pt")
        tokenizer_path = hf_hub_download(model_id, "preprocessor/tokenizer.json")
    except Exception as exc:
        raise OcrError(
            f"Could not download the model '{model_id}' from Hugging Face. "
            "Check your connection (or set a HF mirror via HF_ENDPOINT).",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    config = VisionEncoderDecoderConfig(
        encoder=ViTConfig(**_ENCODER_KW).to_dict(),
        decoder=RobertaConfig(**_DECODER_KW).to_dict(),
        decoder_start_token_id=0,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        max_length=64,
        num_beams=4,
        length_penalty=2.0,
        no_repeat_ngram_size=3,
        early_stopping=True,
        tie_encoder_decoder=False,
    )
    model = VisionEncoderDecoderModel(config)

    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except Exception:  # older torch checkpoints
        state = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if not isinstance(state, dict):  # pragma: no cover - unexpected format
        raise OcrError("Unsupported checkpoint format in model.pt.")

    model_keys = set(model.state_dict())
    candidates = [state] + [
        {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}
        for prefix in ("model.", "module.", "trocr.")
    ]
    best = max(candidates, key=lambda sd: len(model_keys & set(sd)))
    matched = len(model_keys & set(best))
    if matched < 0.5 * len(model_keys):
        raise OcrError(
            "The checkpoint weights do not match the TrOCR architecture "
            f"(matched {matched}/{len(model_keys)} tensors).",
        )
    model.load_state_dict(best, strict=False)
    model.eval()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        model_max_length=512,
    )
    return _TransformersRecogniser(model, tokenizer, device)


def _load_recognizer(model_id: str, device: str) -> Any:
    """Official hezar loader first, built-in transformers loader second."""
    try:
        from hezar.models import Model as HezarModel  # type: ignore
    except ImportError:
        HezarModel = None  # type: ignore

    if HezarModel is not None:
        try:
            logger.info("Loading TrOCR via hezar: %s", model_id)
            model = HezarModel.load(model_id)
            if device not in ("cpu", ""):
                for method_name in ("to", "move_to_device"):
                    method = getattr(model, method_name, None)
                    if callable(method):
                        try:
                            method(device)
                        except Exception:  # pragma: no cover - best effort
                            pass
                        break
            return _HezarRecogniser(model)
        except Exception as exc:
            logger.warning(
                "hezar loader failed (%s: %s); trying built-in transformers loader",
                type(exc).__name__,
                exc,
            )
    else:
        logger.info("hezar not installed; using the built-in transformers loader")
    return _load_with_transformers(model_id, device)


def get_recognizer(model_id: Optional[str] = None, device: str = "cpu") -> Any:
    """Return a cached recogniser for ``(model_id, device)``."""
    model_id = model_id or DEFAULT_TROCR_MODEL
    device = (device or "cpu").strip().lower() or "cpu"
    key = (model_id, device)
    if key not in _RECOGNIZERS:
        _RECOGNIZERS[key] = _load_recognizer(model_id, device)
    return _RECOGNIZERS[key]


# ---------------------------------------------------------------------------
# OCR backend
# ---------------------------------------------------------------------------


def _crop_line(
    image: Image.Image,
    band: Band,
    boxes: Sequence[Tuple[int, int]],
    col_strips: Sequence[Tuple[int, int]],
) -> Image.Image:
    """Crop one text line; blank out vertical ruling-line columns."""
    width, height = image.size
    x0 = max(0, boxes[0][0] - 4)
    x1 = min(width, boxes[-1][1] + 4)
    y0 = max(0, band.y0 - 2)
    y1 = min(height, band.y1 + 2)
    crop = image.crop((x0, y0, x1, y1))
    if col_strips:
        array = np.asarray(crop).copy()
        for strip_x0, strip_x1 in col_strips:
            a = max(0, strip_x0 - x0)
            b = min(array.shape[1], strip_x1 - x0)
            if b > a:
                array[:, a:b] = 255
        crop = Image.fromarray(array)
    return crop


class TrocrBackend(OcrBackend):
    """Page OCR through the Persian TrOCR line model + numpy segmentation.

    When ``corrector`` is set (e.g. :class:`GemmaCorrector`), raw TrOCR
    line texts are corrected *before* token↔box alignment, so geometry
    stays from the image while wording benefits from the language model.
    """

    name = "trocr"

    def __init__(
        self,
        model_id: Optional[str] = None,
        device: str = "cpu",
        recognizer: Any | None = None,
        corrector: Any | None = None,
    ) -> None:
        self.model_id = model_id or DEFAULT_TROCR_MODEL
        self.device = device
        self._recognizer = recognizer  # injectable for tests
        self._corrector = corrector  # LineCorrector-like; None = no correction

    def _ensure(self) -> Any:
        if self._recognizer is None:
            self._recognizer = get_recognizer(self.model_id, self.device)
        return self._recognizer

    def run(self, image: Image.Image, page_number: int) -> List[WordBox]:
        recognizer = self._ensure()
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        ink = binarize(gray)
        clean, _row_strips, col_strips = mask_long_lines(ink)
        bands = segment_bands(clean)

        # 1) TrOCR proposals per band (geometry + raw text kept separate so
        #    the corrector can rewrite words without touching boxes).
        proposals: List[Tuple[Band, List[Tuple[int, int]], str, Optional[float]]] = []
        for band in bands:
            boxes = band_word_boxes(clean, band)
            if not boxes:
                continue
            crop = _crop_line(image, band, boxes, col_strips)
            text, confidence = recognizer.recognize(crop)
            if not text or not text.strip():
                continue
            proposals.append((band, boxes, text, confidence))

        # 2) Optional Gemma 2B correction (line-level, count-preserving).
        texts: List[str] = [text for _, _, text, _ in proposals]
        if self._corrector is not None and texts:
            try:
                fixed = self._corrector.correct_lines(texts)
                if len(fixed) == len(texts):
                    texts = [f if f and f.strip() else o for f, o in zip(fixed, texts)]
                else:
                    logger.warning(
                        "corrector returned %d line(s), expected %d; keeping TrOCR text",
                        len(fixed),
                        len(texts),
                    )
            except Exception as exc:
                logger.warning(
                    "line correction failed (%s: %s); keeping TrOCR text",
                    type(exc).__name__,
                    exc,
                )

        # 3) Align (possibly corrected) tokens back onto image boxes.
        words: List[WordBox] = []
        for (band, boxes, _, confidence), text in zip(proposals, texts):
            for token, (x0, x1) in align_tokens_to_boxes(text, boxes):
                words.append(
                    WordBox(
                        text=token,
                        x0=float(x0),
                        y0=float(band.y0),
                        x1=float(x1),
                        y1=float(band.y1),
                        confidence=confidence,
                    )
                )
        logger.info(
            "trocr page %d: %d band(s) -> %d word(s)", page_number, len(bands), len(words)
        )
        return words


__all__ = [
    "Band",
    "TrocrBackend",
    "align_tokens_to_boxes",
    "band_word_boxes",
    "binarize",
    "get_recognizer",
    "mask_long_lines",
    "otsu_threshold",
    "segment_bands",
]




