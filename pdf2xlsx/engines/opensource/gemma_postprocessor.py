"""Gemma 2B post-corrector for TrOCR line proposals.

Role split (why this hybrid):

* **TrOCR** (``hezarai/trocr-base-fa-v2``) is the *eyes*: image -> raw
  Persian line text. It sees glyphs well but has a weak language model,
  so digits, half-spaces, and ي/ی ـ ك/ک confusions slip through.
* **Gemma 2B IT** (``google/gemma-2-2b-it``) is the *editor*: raw lines
  -> corrected lines. It is text-only (NOT vision), so it never sees the
  image — it only fixes what TrOCR proposed, constrained to return the
  same number of lines and never invent new content.

Usage::

    OCR_MODEL=trocr OCR_POSTPROCESSOR=gemma GEMMA_MODEL=google/gemma-2-2b-it

The corrector loads lazily (cached per process) via ``transformers``.
When dependencies / weights are missing it raises a friendly
:class:`~pdf2xlsx.exceptions.OcrError`; the :class:`TrocrBackend`
falls back to raw TrOCR text per-line so one bad batch never kills a page.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Protocol, Sequence

logger = logging.getLogger(__name__)

_CORRECTION_INSTRUCTIONS = (
    "You are a Persian OCR post-corrector. Fix spelling, spacing "
    "(including Persian half-space U+200C), Arabic/Persian letter variants "
    "(ي->ی, ك->ک, ة->ه), and Persian/Arabic/Latin digit consistency in each "
    "line. Do NOT translate, reorder, add, or remove lines. Return ONLY a "
    "JSON array of corrected strings, same length and order as the input."
)


def build_correction_prompt(lines: Sequence[str]) -> str:
    """Build the Gemma prompt for one batch of TrOCR line proposals."""
    payload = json.dumps(list(lines), ensure_ascii=False)
    return (
        f"{_CORRECTION_INSTRUCTIONS}\n"
        f"Input lines ({len(list(lines))}):\n{payload}\n"
        "Corrected JSON array:"
    )


def parse_corrected_lines(raw: str, expected_count: int, fallback: Sequence[str]) -> List[str]:
    """Parse Gemma output back into exactly ``expected_count`` lines.

    Strategy: find the first ``[...]`` JSON block and decode it; on any
    failure (or length mismatch) fall back to line-splitting, and finally
    to the original TrOCR proposals (never drop content).
    """
    text = (raw or "").strip()
    if not text:
        return list(fallback)
    start, end = text.find("["), text.rfind("]")
    candidate = text[start : end + 1] if 0 <= start < end else text
    try:
        data = json.loads(candidate)
        if isinstance(data, list) and len(data) == expected_count:
            return [str(item) for item in data]
    except (json.JSONDecodeError, ValueError):
        pass
    # lenient fallback: one non-empty line per input line
    stripped = [ln.strip().strip("\"'") for ln in text.splitlines() if ln.strip()]
    cleaned = [ln for ln in stripped if ln not in ("[", "]", ",", "```")]
    if len(cleaned) == expected_count:
        return cleaned
    logger.warning(
        "gemma correction parse failed (got %d lines, expected %d); keeping TrOCR text",
        len(cleaned),
        expected_count,
    )
    return list(fallback)


class LineCorrector(Protocol):
    """Interface for TrOCR line post-correctors."""

    def correct_lines(self, lines: List[str]) -> List[str]:
        ...


class NoOpCorrector:
    """Default: no correction (``OCR_POSTPROCESSOR=none``)."""

    def correct_lines(self, lines: List[str]) -> List[str]:
        return list(lines)


@dataclass
class GemmaCorrector:
    """Persian OCR correction with a local Gemma 2B instruction model.

    Heavy objects (tokenizer/model) load lazily on first use and are
    cached on the instance. ``model``/``tokenizer`` may be injected
    (tests) to avoid downloading weights.
    """

    model_id: str = "google/gemma-2-2b-it"
    device: str = "cpu"
    max_new_tokens: int = 256
    batch_size: int = 8
    model: Any = None
    tokenizer: Any = None

    # -- loading ------------------------------------------------------
    def _ensure(self) -> tuple[Any, Any]:
        if self.model is not None and self.tokenizer is not None:
            return self.model, self.tokenizer
        try:
            import torch  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            from ...exceptions import OcrError

            raise OcrError(
                "Gemma dependencies are not installed. Install them with: "
                "pip install -r requirements-ocr.txt "
                "(torch + transformers + accelerate) and accept the "
                "model licence at https://huggingface.co/google/gemma-2-2b-it",
                detail=str(exc),
            ) from exc
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            kwargs: dict[str, Any] = {"trust_remote_code": False}
            device = (self.device or "cpu").strip().lower()
            if device == "cuda" and torch.cuda.is_available():
                # 4-bit when bitsandbytes is present keeps 2B VRAM-friendly.
                try:
                    import bitsandbytes  # noqa: F401  # type: ignore

                    kwargs.update({"load_in_4bit": True, "device_map": "auto"})
                except ImportError:
                    kwargs.update({"device_map": "auto"})
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            model.eval()
        except Exception as exc:
            from ...exceptions import OcrError

            raise OcrError(
                f"Could not load Gemma model '{self.model_id}'. "
                "Check GEMMA_MODEL, your HF token (gated repo), and connection.",
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        self.model, self.tokenizer = model, tokenizer
        logger.info("Gemma corrector loaded: %s (device=%s)", self.model_id, self.device)
        return model, tokenizer

    # -- inference ----------------------------------------------------
    def _generate(self, prompt: str) -> str:
        import torch  # type: ignore

        model, tokenizer = self._ensure()
        inputs = tokenizer(prompt, return_tensors="pt")
        device = getattr(model, "device", None)
        if device is not None:
            try:
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:  # pragma: no cover - best effort
                pass
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # deterministic correction, no creativity
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        new_tokens = output[0][inputs["input_ids"].shape[1] :]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def correct_lines(self, lines: List[str]) -> List[str]:
        """Correct one page worth of TrOCR lines (order + count preserved)."""
        if not lines:
            return []
        self._ensure()  # fail fast with a friendly error when unusable
        corrected: List[str] = []
        for i in range(0, len(lines), max(1, self.batch_size)):
            batch = lines[i : i + self.batch_size]
            try:
                raw = self._generate(build_correction_prompt(batch))
                corrected.extend(parse_corrected_lines(raw, len(batch), batch))
            except Exception as exc:  # one bad batch must not kill the page
                logger.warning(
                    "gemma correction failed for batch (%s: %s); keeping TrOCR text",
                    type(exc).__name__,
                    exc,
                )
                corrected.extend(batch)
        return corrected


def create_corrector(
    postprocessor: str,
    *,
    model_id: str = "google/gemma-2-2b-it",
    device: str = "cpu",
    max_new_tokens: int = 256,
) -> LineCorrector:
    """Factory from the ``OCR_POSTPROCESSOR`` setting.

    * ``none`` (default) -> :class:`NoOpCorrector`
    * ``gemma`` / ``trocr+gemma`` / ``trocr_gemma`` -> :class:`GemmaCorrector`
    """
    key = (postprocessor or "none").strip().lower()
    if key in ("none", "off", "disabled", "", "trocr"):
        return NoOpCorrector()
    if key in ("gemma", "trocr+gemma", "trocr_gemma", "trocr-gemma", "gemma2b"):
        return GemmaCorrector(model_id=model_id, device=device, max_new_tokens=max_new_tokens)
    from ...exceptions import ConfigurationError

    raise ConfigurationError(
        f"Unknown OCR_POSTPROCESSOR '{postprocessor}'. "
        "Supported values: none, gemma (or trocr+gemma)."
    )


__all__ = [
    "LineCorrector",
    "NoOpCorrector",
    "GemmaCorrector",
    "build_correction_prompt",
    "parse_corrected_lines",
    "create_corrector",
]
