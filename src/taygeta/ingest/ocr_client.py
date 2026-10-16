"""Pass 1: local visual extraction via OCR/Vision-LLM (section 4.2).

The client communicates with a local service that exposes an OpenAI-compatible
API, configured exclusively through the ``OCR_BASE_URL``, ``OCR_API_KEY``, and
``OCR_MODEL`` variables. Its responsibility is to *faithfully transcribe* the
visible content to raw Markdown with TeX math (``$...$`` and ``$$...$$``),
marking doubtful segments with ``[ILEGIBLE]``.

It does not classify curricularly, does not assign difficulty, does not infer
correct answers, does not solve exercises, and does not produce the final
``QuestionRaw`` object.

The interface is kept independent of the engine: :class:`OCRBackend` defines the
contract and :class:`OCRClient` is the implementation on top of the official
``openai`` SDK. Other engines (for example GOT-OCR2.0) can expose the same
contract through a lightweight adapter.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from openai import OpenAI, OpenAIError

from taygeta.core.schema import ILLEGIBLE_MARKER
from taygeta.core.settings import Settings, get_settings

logger = logging.getLogger(__name__)

__all__ = [
    "ILLEGIBLE_MARKER",
    "OCRClient",
    "OCRError",
    "OCRImage",
    "OCRResult",
    "OCR_SYSTEM_PROMPT",
    "build_ocr_client",
    "sniff_mime_type",
    "to_data_uri",
]

#: Explicit marker for segments the OCR cannot read with confidence.
ILLEGIBLE = ILLEGIBLE_MARKER

#: Default bounded retries for transient service errors.
DEFAULT_MAX_RETRIES = 2

#: Maximum number of tokens in the response.
DEFAULT_MAX_TOKENS = 4096

#: URL markers accepted directly as image input.
_URL_PREFIXES = ("http://", "https://", "data:")

OCR_SYSTEM_PROMPT = """Eres un transcriptor OCR de documentos científicos.

Tu única tarea es transcribir FIELMENTE el contenido visible de las imágenes a \
Markdown, sin interpretar ni resolver nada. Reglas obligatorias:

1. Conserva la estructura de lectura, la numeración, el texto del enunciado y \
las alternativas exactamente como aparecen.
2. Escribe las expresiones matemáticas con delimitadores TeX estándar \
OBLIGATORIOS: $...$ para inline y $$...$$ para display. PROHIBIDO emitir \
\\[...\\], \\], \\(...\\), \\( o \\), así como sus variantes escapadas; si \
aparecen en la imagen, transcríbelos siempre como $$...$$ (bloque) o $...$ \
(inline) respectivamente.
3. Conserva funciones por partes, matrices, sumatorias, subíndices, \
superíndices, fracciones, límites, integrales y operadores definidos por el autor.
4. Usa los entornos TeX adecuados cuando correspondan a la estructura visible: \
cases, pmatrix, aligned, substack.
5. Mantén los placeholders de figura {{fig:N}} EXACTAMENTE en su posición dentro \
del texto; no los muevas ni los reescribas.
6. No inventes texto ilegible: escribe [ILEGIBLE] en su lugar y añade una línea \
final "ADVERTENCIAS: ..." indicando qué no pudiste leer.
7. NO resuelvas el ejercicio, NO añadas soluciones, respuestas correctas, \
claves de respuesta, explicaciones ni interpretaciones matemáticas.
8. Devuelve únicamente la transcripción en Markdown/TeX, sin comentarios sobre \
tu proceso.
"""


class OCRError(RuntimeError):
    """Controlled error from the local OCR client."""


# ---------------------------------------------------------------------------
# Input/output types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class OCRImage:
    """OCR input image: raw bytes or an accessible URL/data URI."""

    data: bytes | None = None
    url: str | None = None
    mime_type: str = "image/png"
    page_index: int | None = None

    def __post_init__(self) -> None:
        if (self.data is None) == (self.url is None):
            raise OCRError("OCRImage requires exactly one of: data or url")


@dataclass(slots=True)
class OCRResult:
    """Raw transcription returned by the local OCR."""

    text: str
    model: str
    pages: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    usage: dict[str, Any] | None = None

    @property
    def has_illegible(self) -> bool:
        """Whether the transcription contains segments marked as illegible."""
        return ILLEGIBLE in self.text

    @property
    def illegible_count(self) -> int:
        """Number of ``[ILEGIBLE]`` marks present in the transcription."""
        return self.text.count(ILLEGIBLE)


@runtime_checkable
class OCRBackend(Protocol):
    """Minimal, engine-independent OCR contract."""

    def transcribe(
        self,
        images: Sequence[OCRImage | bytes | str | Path],
        *,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe the images to Markdown/TeX."""
        ...

    def health(self) -> bool:
        """Check that the OCR service is available."""
        ...


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


def sniff_mime_type(data: bytes) -> str:
    """Detect the MIME type from the image's magic bytes."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    return "application/octet-stream"


def to_data_uri(data: bytes, mime_type: str | None = None) -> str:
    """Encode bytes as a data URI, supported by engines that do not download URLs."""
    resolved = mime_type or sniff_mime_type(data)
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{resolved};base64,{encoded}"


# ---------------------------------------------------------------------------
# OCR client (OpenAI-compatible)
# ---------------------------------------------------------------------------


class OCRClient:
    """Client for the local OCR service compatible with the OpenAI API."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Create the OCR client.

        Args:
            settings: Configuration to use; the global one by default.
            client: Client compatible with the OpenAI SDK (injectable in unit
                tests). If omitted, a real one is built.
            model / base_url / api_key / timeout: Override the configuration
                values, useful for selecting the engine without touching code.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._base_url = (base_url or self.settings.OCR_BASE_URL).rstrip("/")
        self._api_key = api_key if api_key is not None else self.settings.OCR_API_KEY
        self._timeout = timeout if timeout is not None else float(self.settings.OCR_TIMEOUT_SECONDS)
        self._max_retries = max(0, max_retries)
        self._max_tokens = max_tokens

    # -- Configuration ------------------------------------------------------
    @property
    def model(self) -> str:
        """Identifier of the configured OCR model."""
        return self._model or self.settings.OCR_MODEL

    @property
    def base_url(self) -> str:
        """OpenAI-compatible endpoint of the OCR service."""
        return self._base_url

    @property
    def client(self) -> Any:
        """Lazily built ``openai`` client."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> OpenAI:
        logger.info("OCR client -> base_url=%s model=%s", self._base_url, self.model)
        return OpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )

    # -- Health -------------------------------------------------------------
    def health(self) -> bool:
        """Check the OCR service health endpoint (list of models)."""
        try:
            self.client.models.list()
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means unhealthy
            logger.warning("the OCR service is not available at %s: %s", self._base_url, exc)
            return False

    is_healthy = health

    # -- Transcription ------------------------------------------------------
    def transcribe(
        self,
        images: Sequence[OCRImage | bytes | str | Path],
        *,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe one or more images to Markdown with TeX math.

        Args:
            images: Input images (bytes, paths, URLs, or :class:`OCRImage`).
            hints: Optional corrective context provided by the user.
            source_reference: Source reference for traceability.

        Raises:
            OCRError: if there are no images, if the provider fails, or the
                response cannot be interpreted.
        """
        if not images:
            raise OCRError("at least one image is required to transcribe")

        content: list[dict[str, Any]] = [
            {"type": "text", "text": self._build_instruction(hints, source_reference, images)}
        ]
        content.extend(self._to_image_part(image) for image in images)

        messages = [
            {"role": "system", "content": OCR_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=self._max_tokens,
            )
        except OpenAIError as exc:
            raise OCRError(f"OCR service failure ({self.base_url}): {exc}") from exc
        except Exception as exc:  # injected client or other engines
            raise OCRError(f"unexpected OCR service error: {exc}") from exc

        text = self._extract_text(response)
        pages = self._collect_pages(images)
        return OCRResult(
            text=text,
            model=self.model,
            pages=pages,
            warnings=self._collect_warnings(text),
            usage=self._extract_usage(response),
        )

    def transcribe_page(
        self,
        image: OCRImage | bytes | str | Path,
        *,
        page_index: int | None = None,
        hints: str | None = None,
        source_reference: str | None = None,
    ) -> OCRResult:
        """Transcribe a single page or crop."""
        entry: OCRImage | bytes | str | Path
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
            entry = OCRImage(
                data=payload, mime_type=sniff_mime_type(payload), page_index=page_index
            )
        elif isinstance(image, OCRImage) and page_index is not None:
            entry = OCRImage(
                data=image.data, url=image.url, mime_type=image.mime_type, page_index=page_index
            )
        else:
            entry = image
        return self.transcribe([entry], hints=hints, source_reference=source_reference)

    # -- Internals ----------------------------------------------------------
    def _build_instruction(
        self,
        hints: str | None,
        source_reference: str | None,
        images: Sequence[Any],
    ) -> str:
        lines = ["Transcribe fielmente a Markdown el contenido visible de la(s) imagen(es)."]
        pages = self._collect_pages(images)
        if source_reference:
            lines.append(f"Origen: {source_reference}")
        if pages:
            lines.append("Páginas (índice físico basado en cero): " + ", ".join(map(str, pages)))
        if hints:
            lines.append(
                "Contexto adicional del revisor (úsalo solo para desambiguar la lectura, "
                f"nunca para inventar contenido): {hints}"
            )
        lines.append(
            "Recuerda: no resuelvas el ejercicio, no añadas soluciones ni respuestas correctas."
        )
        return "\n".join(lines)

    def _coerce_image(self, image: OCRImage | bytes | str | Path) -> OCRImage:
        if isinstance(image, OCRImage):
            return image
        if isinstance(image, (bytes, bytearray)):
            payload = bytes(image)
            return OCRImage(data=payload, mime_type=sniff_mime_type(payload))
        if isinstance(image, Path):
            return self._image_from_path(image)
        if isinstance(image, str):
            if image.startswith(_URL_PREFIXES):
                return OCRImage(url=image)
            return self._image_from_path(Path(image))
        raise OCRError(f"unsupported image type: {type(image).__name__}")

    @staticmethod
    def _image_from_path(path: Path) -> OCRImage:
        if not path.is_file():
            raise OCRError(f"image file does not exist: {path}")
        payload = path.read_bytes()
        return OCRImage(data=payload, mime_type=sniff_mime_type(payload))

    def _to_image_part(self, image: OCRImage | bytes | str | Path) -> dict[str, Any]:
        coerced = self._coerce_image(image)
        url = coerced.url
        if url is None:
            if coerced.data is None:
                raise OCRError("the image contains neither data nor URL")
            url = to_data_uri(coerced.data, coerced.mime_type)
        return {"type": "image_url", "image_url": {"url": url}}

    @staticmethod
    def _collect_pages(images: Iterable[Any]) -> list[int]:
        pages: list[int] = []
        for image in images:
            page_index = getattr(image, "page_index", None)
            if isinstance(page_index, int) and page_index not in pages:
                pages.append(page_index)
        return pages

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            choices = response.choices
        except AttributeError as exc:
            raise OCRError("OCR response has no 'choices' field") from exc
        if not choices:
            raise OCRError("the OCR response contains no choices")
        content = getattr(choices[0].message, "content", None)
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                if text:
                    parts.append(str(text))
            content = "".join(parts)
        return (content or "").strip()

    @staticmethod
    def _extract_usage(response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return dict(usage)
        return None

    @staticmethod
    def _collect_warnings(text: str) -> list[str]:
        warnings: list[str] = []
        count = text.count(ILLEGIBLE)
        if count:
            warnings.append(
                f"detected {count} segment(s) marked as {ILLEGIBLE}; they require human review"
            )
        if not text.strip():
            warnings.append("the OCR transcription is empty")
        return warnings


def build_ocr_client(settings: Settings | None = None) -> OCRClient:
    """Build the OCR client with the global ``OCR_*`` configuration."""
    return OCRClient(settings)
