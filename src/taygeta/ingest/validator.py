"""Two-layer validation of the ingestion pipeline (section 4.4).

* **Layer 1 — local heuristic** (:func:`validate_local`): cheap checks (target <
  1 ms on small documents) that identify likely errors without rewriting
  statements: minimum length, parity of math delimiters, brace balance,
  placeholder/figure agreement, validity of figures and pages, empty required
  fields, and duplicate choices.
* **Layer 2 — remote semantic validation** (:class:`SemanticValidator`): uses
  the remote LLM configured with ``LLM_*`` (independent of the OCR) to validate
  legibility, sufficiency, truncations, incomplete choices, and obvious
  contradictions, *without solving* the exercise. Returns
  :class:`LogicalValidationResult`.

Objects that fail any validation stay in staging with status
``NEEDS_ATTENTION`` and a technical reason; those that pass may be marked
``VALID`` but are not saved definitively until human confirmation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from taygeta.core.schema import PLACEHOLDER_PATTERN, QuestionRaw
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.translation import QuestionTranslation

if TYPE_CHECKING:
    import instructor  # noqa: F401

__all__ = [
    "SEVERITY_ERROR",
    "SEVERITY_WARNING",
    "STATUS_VALID",
    "STATUS_NEEDS_ATTENTION",
    "MIN_STATEMENT_LENGTH",
    "LocalIssue",
    "LocalValidationResult",
    "LogicalValidationResult",
    "RemoteValidationError",
    "SemanticValidator",
    "validate_local",
    "validate_translation",
]

#: Staging statuses allowed for questions.
STATUS_VALID = "VALID"
STATUS_NEEDS_ATTENTION = "NEEDS_ATTENTION"

#: Severity levels of a local issue.
SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"

#: Default minimum statement length (section 4.4).
MIN_STATEMENT_LENGTH = 20

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)
_ALLOWED_TYPES = {"multiple_choice", "open_ended", "true_false", "numerical"}


@dataclass(frozen=True, slots=True)
class LocalIssue:
    """Issue detected by the local heuristic."""

    code: str
    message: str
    severity: Literal["error", "warning"] = SEVERITY_ERROR


@dataclass(slots=True)
class LocalValidationResult:
    """Result of the local validation (Layer 1)."""

    is_valid: bool
    issues: list[LocalIssue] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def errors(self) -> list[LocalIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_ERROR]

    @property
    def warnings(self) -> list[LocalIssue]:
        return [issue for issue in self.issues if issue.severity == SEVERITY_WARNING]

    def format_reasons(self) -> str:
        """Join all the issues into a readable technical reason."""
        if not self.issues:
            return ""
        return "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)


# ---------------------------------------------------------------------------
# Layer 1 — Local heuristic
# ---------------------------------------------------------------------------


def _check_min_length(statement: str, min_length: int) -> LocalIssue | None:
    if len(statement.strip()) < min_length:
        return LocalIssue(
            code="statement_too_short",
            message=(
                f"statement of {len(statement.strip())} characters (< {min_length}); "
                "it may be valid if it is a justified short expression"
            ),
            severity=SEVERITY_WARNING,
        )
    return None


def _scan_math_delimiters(text: str) -> str:
    """Return a description of the math delimiter imbalance, or ''."""
    display_open = False
    inline_open = False
    index = 0
    length = len(text)
    while index < length:
        if text[index : index + 2] == "$$":
            display_open = not display_open
            index += 2
            continue
        if text[index] == "$":
            if index > 0 and text[index - 1] == "\\":
                index += 1
                continue
            inline_open = not inline_open
            index += 1
            continue
        index += 1
    problems: list[str] = []
    if display_open:
        problems.append("unclosed display delimiter $$")
    if inline_open:
        problems.append("unclosed inline delimiter $")
    return "; ".join(problems)


def _check_math_delimiters(statement: str, choices_text: str) -> LocalIssue | None:
    combined = statement + "\n" + choices_text
    problem = _scan_math_delimiters(combined)
    if problem:
        return LocalIssue(code="unbalanced_math_delimiters", message=problem)
    return None


def _check_braces(statement: str, choices_text: str) -> LocalIssue | None:
    depth = 0
    text = statement + "\n" + choices_text
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\\" and index + 1 < length:
            # Skip the escaped character (\{ \} \frac ...) so as not to confuse braces.
            index += 2
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    if depth != 0:
        return LocalIssue(
            code="unbalanced_braces",
            message=f"unbalanced braces {{}} (balance {depth:+d})",
        )
    return None


def _check_placeholder_figure_match(
    statement: str, choices_text: str, figure_ids: set[str]
) -> LocalIssue | None:
    referenced = {
        f"fig:{match}" for match in _PLACEHOLDER_RE.findall(statement + "\n" + choices_text)
    }
    missing = sorted(referenced - figure_ids)
    unused = sorted(figure_ids - referenced)
    if missing or unused:
        message = []
        if missing:
            message.append("placeholders without a declared figure: " + ", ".join(missing))
        if unused:
            message.append("figures declared without a placeholder: " + ", ".join(unused))
        return LocalIssue(code="placeholder_figure_mismatch", message="; ".join(message))
    return None


def _check_figure_references(question: QuestionRaw, max_pages: int | None) -> list[LocalIssue]:
    issues: list[LocalIssue] = []
    seen: set[str] = set()
    for figure in question.figures:
        if figure.id in seen:
            issues.append(
                LocalIssue(code="duplicate_figure_id", message=f"duplicate figure {figure.id}")
            )
        seen.add(figure.id)
        if figure.page < 0:
            issues.append(
                LocalIssue(code="invalid_figure_page", message=f"negative page in {figure.id}")
            )
        if max_pages is not None and figure.page >= max_pages:
            issues.append(
                LocalIssue(
                    code="figure_page_out_of_range",
                    message=f"page {figure.page} of {figure.id} out of range (< {max_pages})",
                )
            )
        ymin, xmin, ymax, xmax = figure.bbox
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            issues.append(
                LocalIssue(code="invalid_figure_bbox", message=f"invalid BBox in {figure.id}")
            )
    return issues


def _check_required_fields(question: QuestionRaw) -> list[LocalIssue]:
    issues: list[LocalIssue] = []
    if not question.statement_raw.strip():
        issues.append(LocalIssue(code="empty_statement", message="empty statement"))
    if question.question_type not in _ALLOWED_TYPES:
        issues.append(
            LocalIssue(
                code="invalid_question_type",
                message=f"invalid question type: {question.question_type!r}",
            )
        )
    if question.question_type == "multiple_choice":
        if not question.choices or len(question.choices) < 2:
            issues.append(
                LocalIssue(
                    code="missing_choices",
                    message="multiple_choice requires at least two choices",
                )
            )
    if question.choices is not None:
        for choice in question.choices:
            if not choice.content_raw.strip():
                issues.append(
                    LocalIssue(
                        code="empty_choice_content",
                        message=f"empty content in choice {choice.id}",
                    )
                )
    return issues


def _check_duplicate_choices(question: QuestionRaw) -> LocalIssue | None:
    if question.choices is None:
        return None
    ids = [choice.id for choice in question.choices]
    duplicates = sorted({choice_id for choice_id in ids if ids.count(choice_id) > 1})
    if duplicates:
        return LocalIssue(
            code="duplicate_choices", message="duplicate choices: " + ", ".join(duplicates)
        )
    return None


def validate_local(
    question: QuestionRaw,
    *,
    min_statement_length: int = MIN_STATEMENT_LENGTH,
    max_pages: int | None = None,
) -> LocalValidationResult:
    """Run Layer 1 (local heuristic) on a question.

    Args:
        question: Question already validated by the Pydantic schema.
        min_statement_length: Minimum statement length (0 to disable).
        max_pages: Total number of document pages to validate figures.
    """
    issues: list[LocalIssue] = []
    checks: dict[str, bool] = {}

    choices_text = "\n".join(choice.content_raw for choice in (question.choices or []))
    figure_ids = {figure.id for figure in question.figures}

    def add(code: str, issue: LocalIssue | None) -> None:
        checks[code] = issue is None
        if issue is not None:
            issues.append(issue)

    if min_statement_length > 0:
        add("min_length", _check_min_length(question.statement_raw, min_statement_length))
    add("math_delimiters", _check_math_delimiters(question.statement_raw, choices_text))
    add("braces", _check_braces(question.statement_raw, choices_text))
    add(
        "placeholder_figure_match",
        _check_placeholder_figure_match(question.statement_raw, choices_text, figure_ids),
    )
    issues.extend(_check_figure_references(question, max_pages))
    checks["figure_references"] = not any(
        issue.code in {"duplicate_figure_id", "invalid_figure_page", "invalid_figure_bbox"}
        for issue in issues
    )
    issues.extend(_check_required_fields(question))
    checks["required_fields"] = not any(issue.code.endswith("empty") for issue in issues)
    add("duplicate_choices", _check_duplicate_choices(question))

    is_valid = not any(issue.severity == SEVERITY_ERROR for issue in issues)
    return LocalValidationResult(is_valid=is_valid, issues=issues, checks=checks)


def _extract_math_spans(text: str) -> list[str]:
    """Extract the content of all math spans ``$...$`` and ``$$...$$``."""
    spans: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        if text[index : index + 2] == "$$":
            end = text.find("$$", index + 2)
            if end == -1:
                break
            spans.append(text[index + 2 : end].strip())
            index = end + 2
            continue
        if text[index] == "$" and not (index > 0 and text[index - 1] == "\\"):
            end = text.find("$", index + 1)
            if end == -1:
                break
            spans.append(text[index + 1 : end].strip())
            index = end + 1
            continue
        index += 1
    return spans


def _check_math_preserved(source_text: str, translated_text: str) -> LocalIssue | None:
    """Check that the translation keeps every TeX expression intact.

    The translation should only affect the prose: every math expression
    ``$...$`` / ``$$...$$`` must remain byte-for-byte identical (the order may
    vary). A mismatch indicates that the model altered or translated the formulas.
    """
    if Counter(_extract_math_spans(source_text)) == Counter(_extract_math_spans(translated_text)):
        return None
    return LocalIssue(
        code="math_not_preserved",
        message="the math expressions ($...$ / $$...$$) must be preserved identically",
    )


def _check_translation_choices(
    source: QuestionRaw, translation: QuestionTranslation
) -> LocalIssue | None:
    """Check that the translated choices keep the ids and order of the original."""
    source_ids = [choice.id for choice in (source.choices or [])]
    translated_ids = [choice.id for choice in (translation.choices or [])]
    if source_ids == translated_ids:
        return None
    return LocalIssue(
        code="translation_choices_mismatch",
        message=(
            "the translated choices must keep the ids and order of the "
            f"original ({source_ids}); received {translated_ids}"
        ),
    )


def validate_translation(
    source: QuestionRaw,
    translation: QuestionTranslation,
    *,
    min_statement_length: int = MIN_STATEMENT_LENGTH,
) -> LocalValidationResult:
    """Validate a translated copy with the same local heuristic as the original.

    It checks balanced delimiters and braces, placeholder/figure agreement
    (against the original figures), exact preservation of the TeX expressions,
    and choice alignment. It does not detect solutions: the schema already
    forbids them.
    """
    issues: list[LocalIssue] = []
    checks: dict[str, bool] = {}

    translated_text = (
        translation.statement_raw
        + "\n"
        + "\n".join(choice.content_raw for choice in (translation.choices or []))
    )
    source_text = (
        source.statement_raw
        + "\n"
        + "\n".join(choice.content_raw for choice in (source.choices or []))
    )
    choices_text = "\n".join(choice.content_raw for choice in (translation.choices or []))
    figure_ids = {figure.id for figure in source.figures}

    def add(code: str, issue: LocalIssue | None) -> None:
        checks[code] = issue is None
        if issue is not None:
            issues.append(issue)

    if min_statement_length > 0:
        add("min_length", _check_min_length(translation.statement_raw, min_statement_length))
    add("math_delimiters", _check_math_delimiters(translation.statement_raw, choices_text))
    add("braces", _check_braces(translation.statement_raw, choices_text))
    add(
        "placeholder_figure_match",
        _check_placeholder_figure_match(translation.statement_raw, choices_text, figure_ids),
    )
    add("math_preserved", _check_math_preserved(source_text, translated_text))
    add("choices_aligned", _check_translation_choices(source, translation))

    is_valid = not any(issue.severity == SEVERITY_ERROR for issue in issues)
    return LocalValidationResult(is_valid=is_valid, issues=issues, checks=checks)


# ---------------------------------------------------------------------------
# Layer 2 — Remote semantic validation
# ---------------------------------------------------------------------------


class LogicalValidationResult(BaseModel):
    """Result of the remote semantic validation (Layer 2)."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    rejection_reason: str = Field(default="")


_SEMANTIC_SYSTEM_PROMPT = """You are a quality reviewer of scientific exam \
question statements. Evaluate ONLY the legibility and sufficiency of the \
statement. Do NOT solve the exercise, do NOT compute its result and do NOT \
evaluate whether the answer is correct.

Consider the statement INVALID (is_valid=false) if:
- it is truncated or incomplete;
- it contains incomplete or poorly delimited choices;
- it presents obvious textual contradictions;
- it does not contain enough information to understand what is being asked;
- it has structural problems that prevent reading it.

If it is invalid, write a brief, concrete rejection_reason in English indicating \
the problem. If it is valid, leave rejection_reason empty.
"""


class RemoteValidationError(RuntimeError):
    """Controlled error from the remote semantic validation."""


class SemanticValidator:
    """Layer 2: semantic validation through the remote LLM (independent of the OCR)."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
    ) -> None:
        """Create the semantic validator.

        Args:
            settings: Configuration to use; the global one by default.
            client: Already-built ``instructor`` client (injectable in tests).
            model: Overrides ``LLM_MODEL``.
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self._max_retries = (
            max_retries if max_retries is not None else self.settings.LLM_MAX_RETRIES
        )

    @property
    def model(self) -> str:
        return self._model or self.settings.LLM_MODEL

    @property
    def client(self) -> Any:
        if self._client is None:
            import instructor

            self._client = instructor.from_openai(self._openai_client())
        return self._client

    def _openai_client(self) -> OpenAI:
        return OpenAI(
            base_url=self.settings.LLM_BASE_URL,
            api_key=self.settings.LLM_API_KEY,
            timeout=float(self.settings.LLM_TIMEOUT_SECONDS),
            max_retries=self._max_retries,
        )

    def validate(
        self, question: QuestionRaw, *, hints: str | None = None
    ) -> LogicalValidationResult:
        """Semantically evaluate a question.

        Raises:
            RemoteValidationError: if the provider fails or the response cannot
                be validated.
        """
        user_text = "Statement to review (Markdown/TeX):\n\n" + question.statement_raw
        if question.choices:
            alternatives = "\n".join(
                f"{choice.id}: {choice.content_raw}" for choice in question.choices
            )
            user_text += f"\n\nChoices:\n{alternatives}"
        user_text += f"\n\nType: {question.question_type}"
        if hints:
            user_text += f"\n\nAdditional context from the reviewer: {hints}"
        user_text += "\n\nEvaluate and return is_valid and rejection_reason."

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                response_model=LogicalValidationResult,
                messages=[
                    {"role": "system", "content": _SEMANTIC_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - provider and schema errors
            raise RemoteValidationError(f"remote semantic validation failure: {exc}") from exc

        if not result.is_valid and not result.rejection_reason.strip():
            result.rejection_reason = "reason not specified by the remote validator"
        if not result.is_valid:
            result.rejection_reason = result.rejection_reason.strip()
        return result
