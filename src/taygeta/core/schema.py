"""Canonical Pydantic v2 schemas for the Taygeta project.

This module defines the data models used by the hybrid ingestion, the curricular
classification and the question storage.

Cross-cutting rules:

* ``extra="forbid"`` on every model: any undeclared field, in particular any
  attempt to add solutions, answer keys or correct answers, raises a validation
  error.
* Curricular slugs (``course``, ``topic``, ``subtopic`` and prerequisites) are
  validated with the regular expression ``^[a-z0-9_]+$`` (no spaces).
* Statements use Markdown with TeX math ``$...$`` and ``$$...$$`` and figure
  placeholders ``{{fig:N}}``. Non-canonical LaTeX delimiters (``\\[...\\]`` and
  ``\\(...\\)``) are normalized automatically with :func:`sanitize_math_delimiters`
  when the schemas are validated.

CRITICAL NOTE: No model accepts or stores solutions, answer keys, correct answers
or reasoning that solves the exercise. Extraction transcribes only the visible
content.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

__all__ = [
    "CURRICULAR_SLUG_PATTERN",
    "ILLEGIBLE_MARKER",
    "PLACEHOLDER_PATTERN",
    "RELATIVE_ASSET_PREFIX",
    "Choice",
    "EnrichedQuestion",
    "FigureReference",
    "QuestionRaw",
    "Slug",
    "TranscriptionIntermediate",
    "compute_canonical_hash",
    "deterministic_question_id",
    "get_public_url",
    "is_relative_asset_path",
    "new_question_id",
    "normalize_slug",
    "sanitize_math_delimiters",
    "sanitize_norm_bars",
    "slugify",
    "utc_now_iso",
]


# ---------------------------------------------------------------------------
# Constants and helper types
# ---------------------------------------------------------------------------

#: Regular expression for curricular slugs: lowercase letters, digits and underscore.
CURRICULAR_SLUG_PATTERN = r"^[a-z0-9_]+$"

#: Explicit convention to mark illegible fragments during OCR. Kept as a stable
#: data marker (it is produced and consumed verbatim by the OCR pipeline).
ILLEGIBLE_MARKER = "[ILEGIBLE]"

#: Placeholders that interleave figures inside the statement, e.g. ``{{fig:1}}``.
PLACEHOLDER_PATTERN = r"\{\{fig:(\d+)\}\}"

#: Canonical prefix of relative asset paths, e.g.
#: ``/buckets/exams/img.png`` (equivalent to ``s3://exams/img.png``).
RELATIVE_ASSET_PREFIX = "/buckets/"

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)
_FIGURE_ID_PATTERN = r"^fig:\d+$"

#: Non-standard TeX delimiters that some OCR/LLM models emit and that break
#: KaTeX/Typst rendering: ``\[...\]`` denotes display and ``\(...\)`` denotes
#: inline. Over-escaped backslashes (``\\[``, ``\\]``) are tolerated.
_DISPLAY_MATH_RE = re.compile(r"\\+\[(.*?)\\+\]", re.DOTALL)
_INLINE_MATH_RE = re.compile(r"\\+\((.*?)\\+\)", re.DOTALL)

#: Validated curricular slug (lowercase letters, digits and underscore, no spaces).
Slug = Annotated[
    str,
    StringConstraints(pattern=CURRICULAR_SLUG_PATTERN, strip_whitespace=False),
]

#: Stable namespace for deterministic UUID5 question identifiers.
_QUESTION_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


def slugify(value: str | None) -> str:
    """Turn free text into a valid slug ``^[a-z0-9_]+$`` (or ``""``).

    - strips accents/diacritics (an accented letter becomes its base letter);
    - lowercases;
    - replaces spaces and special characters with ``_``;
    - collapses repeated ``_`` and trims leading/trailing ones.
    """
    if value is None:
        return ""
    decomposed = unicodedata.normalize("NFKD", str(value))
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    normalized = re.sub(r"[^a-z0-9_]+", "_", without_marks.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def normalize_slug(value: str | None, default: str = "general") -> str:
    """Normalize a slug and return ``default`` when it ends up empty."""
    return slugify(value) or default


def sanitize_math_delimiters(text: str) -> str:
    """Normalize TeX math delimiters to ``$...$`` / ``$$...$$``.

    Some OCR/LLM models emit LaTeX-style delimiters ``\\[...\\]`` (display) and
    ``\\(...\\)`` (inline), which KaTeX and Typst do not interpret the same way as
    the project's canonical delimiters. This function rewrites them to ``$$...$$``
    and ``$...$`` respectively and tolerates over-escaped backslashes (for example
    ``\\\\[...\\\\]``). Statements without those patterns are returned intact.
    """
    if not text:
        return text
    sanitized = _DISPLAY_MATH_RE.sub(lambda match: f"$${match.group(1)}$$", text)
    sanitized = _INLINE_MATH_RE.sub(lambda match: f"${match.group(1)}$", sanitized)
    return sanitized


def sanitize_norm_bars(text: str) -> str:
    """Rewrite LaTeX norm bars ``\\|`` as ``||`` (the Typst norm).

    ``\\|`` is the norm bar in LaTeX and ``||`` its Typst equivalent; the
    transpiler does not recognize the ``\\|`` command and would fail to compile.
    This function replaces every ``\\|`` that is not preceded by another backslash
    (that is, one that is not part of a ``\\\\|``, where ``\\\\`` is a line break)
    with ``||``. A manual scan is used, without regular expressions.
    """
    parts: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if (
            char == "\\"
            and index + 1 < length
            and text[index + 1] == "|"
            and (index == 0 or text[index - 1] != "\\")
        ):
            parts.append("||")
            index += 2
            continue
        parts.append(char)
        index += 1
    return "".join(parts)


def is_relative_asset_path(value: str | None) -> bool:
    """Whether ``value`` is a relative asset path ``/buckets/{bucket}/{key}``.

    Rejects empty segments, ``.`` and ``..`` to prevent traversal, but accepts the
    ``/buckets/`` prefix (which, unlike ``/app/...`` or ``file://``, is not an
    absolute local path but a path relative to the public storage).
    """
    if not value or not value.startswith(RELATIVE_ASSET_PREFIX):
        return False
    rest = value[len(RELATIVE_ASSET_PREFIX) :]
    if not rest:
        return False
    return all(segment not in ("", ".", "..") for segment in rest.split("/"))


def get_public_url(relative_path: str) -> str:
    """Compose the public URL by concatenating ``S3_PUBLIC_BASE_URL`` with the relative path.

    Images are stored in the database as relative paths ``/buckets/{bucket}/{key}``;
    this function composes the full URL right at the output boundary. If
    ``relative_path`` is empty, returns ``""``.
    """
    if not relative_path:
        return ""
    from taygeta.core.settings import get_settings

    base = (get_settings().S3_PUBLIC_BASE_URL or "").rstrip("/")
    path = relative_path if relative_path.startswith("/") else "/" + relative_path
    return base + path


def compute_canonical_hash(tex_string: str) -> str:
    """Compute a stable canonical hash for question deduplication.

    Cleans the whitespace and line breaks from the statement (collapsing any run of
    whitespace into a single space) and returns the hexadecimal SHA-256 hash of the
    normalized content. It uses no solutions or correct answers: only the visible
    content of the statement.
    """
    normalized = " ".join(str(tex_string).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def new_question_id() -> str:
    """Generate a random UUID (version 4) as a question identifier."""
    return str(uuid.uuid4())


def deterministic_question_id(*parts: str) -> str:
    """Generate a deterministic UUID5 from the given parts.

    Allows the same question to always obtain the same identifier while its source
    content does not change.
    """
    if not parts:
        raise ValueError("at least one part is required for the deterministic UUID")
    return str(uuid.uuid5(_QUESTION_ID_NAMESPACE, "|".join(parts)))


def utc_now_iso() -> str:
    """Return the current date/time in ISO 8601 with UTC timezone."""
    return datetime.now(tz=UTC).isoformat()


# ---------------------------------------------------------------------------
# 3.1. FigureReference
# ---------------------------------------------------------------------------


class FigureReference(BaseModel):
    """Geometric reference to a figure extracted from a document.

    ``bbox`` uses the format ``[ymin, xmin, ymax, xmax]`` with coordinates
    normalized to the ``0-1000`` range. ``asset_path`` is a stable object
    identifier:

    * a public relative path ``/buckets/{bucket}/{key}`` (recommended canonical
      form, e.g. ``/buckets/exams/img.png``);
    * an ``s3://bucket/key`` URI;
    * an object key relative to the bucket (without a leading ``/``).

    Never an absolute local path such as ``/app/assets/...``.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        ...,
        pattern=_FIGURE_ID_PATTERN,
        description='Stable figure identifier, for example "fig:1".',
    )
    page: int = Field(
        ...,
        ge=0,
        description="Zero-based physical page index.",
    )
    bbox: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [ymin, xmin, ymax, xmax] normalized to 0-1000.",
    )
    caption: str | None = Field(
        default=None,
        description="Visible figure caption, if any.",
    )
    asset_path: str | None = Field(
        default=None,
        description=(
            "Stable object identifier: relative path /buckets/{bucket}/{key}, "
            "s3://bucket/key or a relative key. Never an absolute local path."
        ),
    )

    @field_validator("bbox")
    @classmethod
    def _validate_bbox(cls, value: list[int]) -> list[int]:
        ymin, xmin, ymax, xmax = value
        if any(coord < 0 or coord > 1000 for coord in value):
            raise ValueError("bbox coordinates must be normalized between 0 and 1000")
        if ymin >= ymax or xmin >= xmax:
            raise ValueError(
                "bbox must satisfy ymin < ymax and xmin < xmax "
                f"(received [ymin={ymin}, xmin={xmin}, ymax={ymax}, xmax={xmax}])"
            )
        return value

    @field_validator("asset_path")
    @classmethod
    def _validate_asset_path(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.startswith(RELATIVE_ASSET_PREFIX):
            if not is_relative_asset_path(value):
                raise ValueError(
                    "the relative asset path must have the form /buckets/{bucket}/{key} "
                    "without empty segments, '.' or '..'"
                )
            return value
        if value.startswith("/") or value.startswith("file://"):
            raise ValueError(
                "asset_path cannot be an absolute local path; use /buckets/{bucket}/{key}, "
                "s3://bucket/key or an object key relative to the bucket"
            )
        if re.match(r"^[a-zA-Z]:[\\/]", value):
            raise ValueError(
                "asset_path cannot be an absolute Windows local path; use "
                "/buckets/{bucket}/{key}, s3://bucket/key or an object key relative to the bucket"
            )
        return value


# ---------------------------------------------------------------------------
# 3.2. Choice
# ---------------------------------------------------------------------------


class Choice(BaseModel):
    """Choice of a multiple-choice question."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        min_length=1,
        pattern=r"^[A-Za-z0-9]+$",
        description='Choice identifier, for example "A", "B", "C" or "D".',
    )
    content_raw: str = Field(
        ...,
        min_length=1,
        description="Markdown content of the choice with TeX math expressions.",
    )

    @field_validator("content_raw", mode="before")
    @classmethod
    def _normalize_content_math(cls, value: Any) -> Any:
        """Normalize the math delimiters before validating the choice."""
        return sanitize_math_delimiters(value) if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# 3.3. QuestionRaw
# ---------------------------------------------------------------------------


class QuestionRaw(BaseModel):
    """Transcribed and structured question, with no solution or correct answer.

    The statement uses Markdown with ``$...$`` and ``$$...$$`` for math and
    ``{{fig:N}}`` placeholders to interleave figures at the exact point in the
    reading flow. Placeholders and declared figures must match one to one.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ...,
        description="Deterministic or generated UUID that identifies the question.",
    )
    statement_raw: str = Field(
        ...,
        min_length=1,
        description=(
            "Statement in Markdown with $...$ / $$...$$ math and {{fig:N}} "
            "placeholders for figures."
        ),
    )
    question_type: Literal["multiple_choice", "open_ended", "true_false", "numerical"] = Field(
        ...,
        description="Question type according to its visible format.",
    )
    choices: list[Choice] | None = Field(
        default=None,
        description="Choices; required (>= 2) for multiple_choice.",
    )
    figures: list[FigureReference] = Field(
        default_factory=list,
        description="Figures referenced by the statement through {{fig:N}}.",
    )
    notation_notes: list[str] = Field(
        default_factory=list,
        description="Notes about detected notation ambiguities or uncertainties.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        try:
            uuid.UUID(value)
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("id must be a valid UUID") from exc
        return value

    @field_validator("statement_raw", mode="before")
    @classmethod
    def _normalize_statement_math(cls, value: Any) -> Any:
        """Sanitize the math delimiters right when validating/ingesting.

        It runs before model validation and, therefore, before any persistence:
        every ``statement_raw`` that enters the system (ingestion, refinement, UI
        or database) ends up with ``$...$`` and ``$$...$$``.
        """
        return sanitize_math_delimiters(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_structure(self) -> QuestionRaw:
        figure_ids = [figure.id for figure in self.figures]
        if len(figure_ids) != len(set(figure_ids)):
            raise ValueError("figure identifiers must be unique")

        declared = set(figure_ids)
        referenced = {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(self.statement_raw)}
        missing = sorted(referenced - declared)
        unused = sorted(declared - referenced)
        if missing:
            raise ValueError(
                "the statement references placeholders without a declared figure: "
                + ", ".join(missing)
            )
        if unused:
            raise ValueError(
                "there are declared figures without a placeholder in the statement: "
                + ", ".join(unused)
            )

        if self.question_type == "multiple_choice":
            choices = self.choices or []
            if len(choices) < 2:
                raise ValueError("multiple_choice questions require at least two choices")
        if self.choices is not None:
            choice_ids = [choice.id for choice in self.choices]
            if len(choice_ids) != len(set(choice_ids)):
                raise ValueError("choice identifiers must be unique")
        return self


# ---------------------------------------------------------------------------
# 3.4. EnrichedQuestion
# ---------------------------------------------------------------------------


class EnrichedQuestion(QuestionRaw):
    """Raw question enriched with curricular classification and metadata."""

    model_config = ConfigDict(extra="forbid")

    course: Slug = Field(
        ...,
        description="Course slug, without spaces (regex ^[a-z0-9_]+$).",
    )
    topic: Slug = Field(
        ...,
        description="Topic slug, without spaces (regex ^[a-z0-9_]+$).",
    )
    subtopic: Slug | None = Field(
        default=None,
        description="Subtopic slug, without spaces, or None (regex ^[a-z0-9_]+$).",
    )
    difficulty: int = Field(
        ...,
        ge=1,
        le=5,
        description="Curricular difficulty in the 1-5 range.",
    )
    prerequisites: list[Slug] = Field(
        default_factory=list,
        description="Prerequisite slugs (format ^[a-z0-9_]+$).",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Source document metadata (pages, file, etc.).",
    )
    created_at: str = Field(
        ...,
        description="Creation date/time in ISO 8601, preferably with a timezone.",
    )
    canonical_hash: str | None = Field(
        default=None,
        description=(
            "Canonical hash (hex SHA-256) of the normalized statement, for future "
            "deduplication. Computed with ``compute_canonical_hash`` and it never "
            "includes solutions or correct answers."
        ),
    )

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise ValueError(
                f"created_at must be a valid ISO 8601 date/time (received {value!r})"
            ) from exc
        return value

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _normalize_prerequisites(cls, value: Any) -> list[str]:
        """Normalize and deduplicate the prerequisites before validating them.

        Tolerates a bare string (comma- or semicolon-separated) and applies
        ``slugify`` to each element, discarding empties and duplicates. Semantic
        canonicalization (reuse of slugs already present in the database) is done
        by the LLM during structuring through the dynamic catalog injected into the
        prompt; here only the basic ``^[a-z0-9_]+$`` format imposed by :data:`Slug`
        is guaranteed.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            slug = slugify(item)
            if slug and slug not in result:
                result.append(slug)
        return result


# ---------------------------------------------------------------------------
# 3.5. Internal transcription schema
# ---------------------------------------------------------------------------


class TranscriptionIntermediate(BaseModel):
    """Raw output of the local OCR before the remote structuring pass.

    Keeps the raw Markdown/TeX text, the source reference, the pages involved, the
    figure information and the reading warnings. It does NOT replace
    :class:`QuestionRaw`: it performs no curricular classification or logical
    validation.
    """

    model_config = ConfigDict(extra="forbid")

    source_reference: str = Field(
        ...,
        min_length=1,
        description="Stable source reference (file name or identifier).",
    )
    raw_markdown: str = Field(
        ...,
        description="Raw Markdown/TeX text transcribed by the local OCR.",
    )
    pages: list[int] = Field(
        default_factory=list,
        description="Zero-based physical page indices involved.",
    )
    figures: list[FigureReference] = Field(
        default_factory=list,
        description="Detected figures with their geometry and source reference.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Reading warnings, including segments marked as illegible.",
    )

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, value: list[int]) -> list[int]:
        if any(page < 0 for page in value):
            raise ValueError("page indices must be greater than or equal to zero")
        if len(value) != len(set(value)):
            raise ValueError("page indices must be unique")
        return value
