"""Single-pass structuring (section 4.3).

Consolidates the OCR transcription, the TeX formatting and the curricular
classification into **ONE single call** to the remote LLM (DeepSeek or another
provider compatible with the OpenAI API via ``instructor``). The model returns,
per question, in a single JSON response:

* ``statement_raw`` in Markdown/TeX (``$...$``, ``$$...$$``, ``{{fig:N}}`` placeholders);
* ``question_type`` and ``choices``;
* ``course``, ``topic`` and ``subtopic`` (snake_case slugs);
* ``difficulty`` (integer 1-5);
* ``prerequisites`` (list of slugs).

The slugs are declared with the :data:`~taygeta.core.schema.Slug` type, which
imposes the regex ``^[a-z0-9_]+$``, and are normalized (``mode="before"``) to
tolerate accents, uppercase and separators from the LLM without breaking
validation. The ``id`` and the figure association are assigned locally in a
deterministic way. It does not solve the exercise or add solutions or correct
answers.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from taygeta.core.schema import (
    PLACEHOLDER_PATTERN,
    Choice,
    EnrichedQuestion,
    Slug,
    TranscriptionIntermediate,
    compute_canonical_hash,
    deterministic_question_id,
    normalize_slug,
    sanitize_math_delimiters,
    slugify,
    utc_now_iso,
)
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.taxonomy import TaxonomyKind, TaxonomyNames
from taygeta.i18n import SUPPORTED_LOCALES

logger = logging.getLogger(__name__)

__all__ = [
    "LocalizedName",
    "PrerequisiteName",
    "QuestionDraft",
    "StructuredQuestion",
    "Structurer",
    "StructuringError",
]

_PLACEHOLDER_RE = re.compile(PLACEHOLDER_PATTERN)


def _names_suffix(
    names: dict[tuple[str, str, str], dict[str, str]] | None,
    kind: str,
    parent_slug: str,
    slug: str,
) -> str:
    """Format the existing localized names as a grounding suffix."""
    if not names:
        return ""
    entry = names.get((kind, parent_slug, slug))
    if not entry:
        return ""
    parts = [f'{locale}: "{label}"' for locale, label in entry.items() if label]
    return f" ({', '.join(parts)})" if parts else ""


def _build_dynamic_curriculum_prompt(
    taxonomy_tree: dict[str, dict[str, list[str]]],
    names: dict[tuple[str, str, str], dict[str, str]] | None = None,
) -> str:
    """Format the live curricular taxonomy as grounding text for the LLM.

    Receives the ``course -> topic -> subtopics`` tree rebuilt from the database
    and turns it into a readable catalog that the LLM must strictly reuse to
    canonicalize ``course``, ``topic`` and ``subtopic``. If ``names`` is provided
    (``(kind, parent_slug, slug) -> {locale: label}``), each slug is annotated
    with its already existing localized names so the LLM reuses them instead of
    inventing variants.
    """
    header = (
        "CATALOG OF COURSES AND TOPICS THAT ALREADY EXIST IN THE DATABASE:\n"
        "Here is the catalog of courses and topics that ALREADY EXIST in the database. "
        "If the content of the question matches one of them, you MUST use the exact slug "
        "listed. Only coin a new slug if the subject or topic is genuinely new and does "
        "not exist in the catalog."
    )
    if not taxonomy_tree:
        return (
            header + "\n(The catalog is empty: there are no confirmed courses or topics in "
            "the database yet; you may propose new snake_case slugs.)"
        )
    lines: list[str] = [header]
    for course in sorted(taxonomy_tree):
        topics = taxonomy_tree[course]
        course_suffix = _names_suffix(names, "course", "", course)
        topic_desc = ", ".join(
            f"`{topic}`{_names_suffix(names, 'topic', course, topic)}" for topic in sorted(topics)
        )
        lines.append(f"- Course: `{course}`{course_suffix} -> Topics: {topic_desc}")
        for topic in sorted(topics):
            subtopics = sorted(topics[topic])
            if subtopics:
                subtopic_desc = ", ".join(
                    f"`{subtopic}`{_names_suffix(names, 'subtopic', course, subtopic)}"
                    for subtopic in subtopics
                )
                lines.append(f"    - `{course}` / `{topic}` -> Subtopics: {subtopic_desc}")
    return "\n".join(lines)


def _build_dynamic_prerequisites_prompt(
    prerequisites: list[str],
    names: dict[tuple[str, str, str], dict[str, str]] | None = None,
) -> str:
    """Format the live prerequisite catalog as grounding text.

    The slugs come from the database (``prerequisites`` table), so that the LLM
    reuses the already existing vocabulary before coining new concepts. If the
    catalog is empty, it says so explicitly. When ``names`` is provided, each slug
    is annotated with its existing localized names.
    """
    header = (
        "CATALOG OF PREREQUISITES THAT ALREADY EXIST IN THE DATABASE:\n"
        "Reuse these exact slugs when a prior concept of the exercise is equivalent or "
        "akin to one of them (ignoring singular/plural, articles or common mathematical "
        "synonyms)."
    )
    if not prerequisites:
        return (
            header + "\n(The catalog is empty: there are no confirmed prerequisites in the "
            "database yet; you may propose new snake_case slugs.)"
        )
    formatted_slugs = "\n".join(
        f"- `{slug}`{_names_suffix(names, 'prerequisite', '', slug)}"
        for slug in sorted(set(prerequisites))
    )
    return header + "\n" + formatted_slugs


_STRUCTURING_SYSTEM_PROMPT = """You are a structurer of scientific exam questions. \
From the text transcribed by OCR, return ONE list of questions and, in the SAME \
response, infer for each one both the formatted statement and its curricular \
classification. Rules:

1. statement_raw: statement in Markdown/TeX, keeping the {{fig:N}} placeholders \
exactly in their position and associating them only with the declared figures. \
Insert {{fig:N}} ONLY if the statement explicitly depends on a graph, diagram or \
illustration present in that region; if there is no explicit figure, do NOT \
declare fig:N and do not insert any placeholder. \
MANDATORY math delimiters: use EXCLUSIVELY $...$ for inline math and $$...$$ \
for display math; PROHIBITED \\[...\\], \\], \\(...\\), \\( or \\), as well as \
their escaped variants (always rewrite them as $$...$$ or $...$ respectively).
2. question_type: multiple_choice (if there are A/B/C/D... choices), open_ended, \
true_false or numerical.
3. choices: choices (minimum 2 for multiple_choice).
4. notation_notes: reading uncertainties; do not resolve them by inventing text.
5. course: course slug in snake_case (lowercase, digits and underscore; no \
spaces or accents). You MUST reuse the exact slug from the course and topic \
catalog attached below if the subject matches an already existing course; only \
coin a new slug if the subject is genuinely new and does not appear in the \
catalog.
6. topic: topic slug within the course (snake_case), reusing the exact slug from \
the attached catalog if the topic already exists.
7. subtopic: subtopic slug (snake_case) or null if it does not apply; reuse the \
exact slug from the catalog if the subtopic already exists.
8. difficulty: integer 1-5 based on the MATHEMATICAL COMPLEXITY; do NOT fix it at 3.
9. prerequisites: list of prior mathematical concepts in snake_case. For each \
identified concept:
   a. MANDATORY DYNAMIC MAPPING: compare the concept semantically against the \
prerequisite catalog attached below (read from the database). If the concept is \
equivalent to an existing term (ignoring singular/plural number, mathematical \
synonyms or slight wording differences), you MUST use exactly the catalog slug.
   b. DEDUPLICATION: do not return duplicate or redundant slugs.
   c. NEW CONCEPTS: only if the mathematical concept has no equivalence in the \
catalog, coin a new concise snake_case slug (preferably singular, without \
unnecessary articles).
10. localization: for every taxonomy entry you emit, also provide human-readable \
DISPLAY NAMES in Spanish and English — course_name, topic_name, subtopic_name \
(null when there is no subtopic) and prerequisite_names (one entry per \
prerequisite, each with its slug, es and en). Names are presentation only: keep \
the slugs unchanged, NEVER include solutions or answers, and reuse the existing \
localized names shown in the catalog below whenever a slug already has one.

Do not solve the exercise, do not add solutions or correct answers, and do not \
invent data that cannot be deduced from the statement.
"""

_REFINING_SYSTEM_PROMPT = """You are a reviewer of scientific exam questions. \
You receive an already structured question and a corrective context from the human \
reviewer. Fix the statement by applying the corrective context and, in the SAME \
response, re-infer the full curricular classification. Rules:

1. statement_raw: fix the statement in Markdown/TeX by applying the corrective \
context, keeping the {{fig:N}} placeholders exactly in their position and \
associating them only with the declared figures. Insert {{fig:N}} ONLY if the \
statement explicitly depends on a graph, diagram or illustration present in that \
region; if there is no explicit figure, do NOT declare fig:N and do not insert \
any placeholder. MANDATORY math delimiters: use EXCLUSIVELY $...$ for inline math \
and $$...$$ for display math; PROHIBITED \\[...\\], \\], \\(...\\), \\( or \\), as \
well as their escaped variants (always rewrite them as $$...$$ or $...$ \
respectively).
2. question_type: multiple_choice (if there are A/B/C/D... choices), open_ended, \
true_false or numerical.
3. choices: corrected choices (minimum 2 for multiple_choice).
4. notation_notes: reading uncertainties; do not resolve them by inventing text.
5. course: course slug in snake_case (lowercase, digits and underscore; no spaces \
or accents), re-inferred from the new context. You MUST reuse the exact slug from \
the course and topic catalog attached below if the subject matches an already \
existing course; only coin a new slug if the subject is genuinely new and does not \
appear in the catalog.
6. topic: topic slug within the course (snake_case), reusing the exact slug from \
the attached catalog if the topic already exists.
7. subtopic: subtopic slug (snake_case) or null if it does not apply; reuse the \
exact slug from the catalog if the subtopic already exists.
8. difficulty: integer 1-5 based on the MATHEMATICAL COMPLEXITY; do NOT fix it at 3.
9. prerequisites: list of prior concepts as snake_case slugs. For each identified \
concept:
   a. MANDATORY DYNAMIC MAPPING: compare the concept semantically against the \
prerequisite catalog attached below (read from the database). If the concept is \
equivalent to an existing term (ignoring singular/plural number, mathematical \
synonyms or slight wording differences), you MUST use exactly the catalog slug.
   b. DEDUPLICATION: do not return duplicate or redundant slugs.
   c. NEW CONCEPTS: only if the mathematical concept has no equivalence in the \
catalog, coin a new concise snake_case slug (preferably singular, without \
unnecessary articles).
10. localization: for every taxonomy entry you emit, also provide human-readable \
DISPLAY NAMES in Spanish and English — course_name, topic_name, subtopic_name \
(null when there is no subtopic) and prerequisite_names (one entry per \
prerequisite, each with its slug, es and en). Names are presentation only: keep \
the slugs unchanged, NEVER include solutions or answers, and reuse the existing \
localized names shown in the catalog below whenever a slug already has one.

Do not solve the exercise, do not add solutions or correct answers, and do not \
invent data that cannot be deduced from the statement.
"""


class LocalizedName(BaseModel):
    """Human-readable name (es/en) of a taxonomy entry emitted by the LLM."""

    model_config = ConfigDict(extra="forbid")

    es: str = Field(..., min_length=1, description="Human-readable name in Spanish.")
    en: str = Field(..., min_length=1, description="Human-readable name in English.")


class PrerequisiteName(LocalizedName):
    """Human-readable name of a prerequisite, linked to its slug."""

    slug: str = Field(..., min_length=1, description="Slug of the named prerequisite.")


class QuestionDraft(BaseModel):
    """The LLM's single response per question: structure + curricular classification.

    The ``id`` and the figure association are assigned locally, in a deterministic
    way, so as not to depend on the model generating correct UUIDs.
    """

    model_config = ConfigDict(extra="forbid")

    statement_raw: str = Field(..., min_length=1)
    question_type: Literal["multiple_choice", "open_ended", "true_false", "numerical"]
    choices: list[Choice] | None = Field(default=None)
    notation_notes: list[str] = Field(default_factory=list)
    course: Slug = Field(..., description="Course slug (regex ^[a-z0-9_]+$).")
    topic: Slug = Field(..., description="Topic slug (regex ^[a-z0-9_]+$).")
    subtopic: Slug | None = Field(default=None, description="Subtopic slug or null.")
    difficulty: int = Field(..., ge=1, le=5, description="Estimated difficulty 1-5.")
    prerequisites: list[Slug] = Field(default_factory=list, description="Slugs of prior concepts.")
    course_name: LocalizedName | None = Field(
        default=None, description="Human-readable course name (es/en)."
    )
    topic_name: LocalizedName | None = Field(
        default=None, description="Human-readable topic name (es/en)."
    )
    subtopic_name: LocalizedName | None = Field(
        default=None, description="Human-readable subtopic name (es/en), or null."
    )
    prerequisite_names: list[PrerequisiteName] = Field(
        default_factory=list, description="Human-readable names (es/en) of the prerequisites."
    )

    @field_validator("course", "topic", mode="before")
    @classmethod
    def _normalize_required(cls, value: Any) -> str:
        return normalize_slug(value if isinstance(value, str) else None)

    @field_validator("subtopic", mode="before")
    @classmethod
    def _normalize_subtopic(cls, value: Any) -> str | None:
        return slugify(value if isinstance(value, str) else None) or None

    @field_validator("prerequisites", mode="before")
    @classmethod
    def _normalize_prerequisites(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = value.replace(";", ",").split(",")
        if not isinstance(value, (list, tuple, set)):
            return []
        result: list[str] = []
        for item in value:
            slug = slugify(str(item))
            if slug and slug not in result:
                result.append(slug)
        return result

    @field_validator("difficulty", mode="before")
    @classmethod
    def _coerce_difficulty(cls, value: Any) -> int:
        try:
            difficulty = int(float(value))
        except (TypeError, ValueError):
            return 3
        return max(1, min(5, difficulty))


@dataclass(slots=True)
class StructuredQuestion:
    """Per-question result of the single structuring pass."""

    question: EnrichedQuestion | None
    error: str | None = None
    taxonomy_names: list[TaxonomyNames] = field(default_factory=list)


class StructuringError(RuntimeError):
    """Controlled error during remote structuring."""


class Structurer:
    """Structures the OCR output into ``EnrichedQuestion`` in a single pass."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        client: Any | None = None,
        model: str | None = None,
        max_retries: int | None = None,
        database: Any | None = None,
    ) -> None:
        """Create the structurer.

        Args:
            settings: Configuration to use; the global one by default.
            client: An already built ``instructor`` client (injectable in tests).
            model: Overrides ``LLM_MODEL``.
            database: Database access to read the live taxonomy
                (course/topic/subtopic and prerequisites) and assemble the dynamic
                grounding prompt. Injectable in tests; if ``None``, the prompt is
                built without a catalog (empty database).
        """
        self.settings = settings or get_settings()
        self._client = client
        self._model = model
        self.database = database
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

    def _live_taxonomy_tree(self) -> dict[str, dict[str, list[str]]]:
        """Read the live curricular taxonomy from the database, or ``{}`` if none."""
        if self.database is None or not hasattr(self.database, "get_live_taxonomy_tree"):
            return {}
        try:
            return self.database.get_live_taxonomy_tree()
        except Exception:  # noqa: BLE001 - grounding is optional, never blocks
            logger.warning("could not read the live taxonomy from the database", exc_info=True)
            return {}

    def _live_prerequisites(self) -> list[str]:
        """Read the live prerequisite catalog from the database, or ``[]`` if none."""
        if self.database is None or not hasattr(self.database, "list_prerequisites"):
            return []
        try:
            rows = self.database.list_prerequisites()
        except Exception:  # noqa: BLE001 - grounding is optional, never blocks
            logger.warning(
                "could not read the prerequisite catalog from the database",
                exc_info=True,
            )
            return []
        return [str(row["slug"]) for row in rows if row.get("slug")]

    def _live_taxonomy_names(self) -> dict[tuple[str, str, str], dict[str, str]]:
        """Read the already existing localized names from the database.

        Returns a map ``(kind, parent_slug, slug) -> {locale: label}`` so the LLM
        reuses the known names instead of inventing variants.
        """
        if self.database is None or not hasattr(self.database, "taxonomy_label_map"):
            return {}
        names: dict[tuple[str, str, str], dict[str, str]] = {}
        try:
            for locale in SUPPORTED_LOCALES:
                for key, label in self.database.taxonomy_label_map(locale=locale).items():
                    names.setdefault(key, {})[locale] = label
        except Exception:  # noqa: BLE001 - grounding is optional, never blocks
            logger.warning(
                "could not read the taxonomy names from the database",
                exc_info=True,
            )
            return {}
        return names

    def _build_system_prompt(self, *, refining: bool) -> str:
        """Assemble the system prompt dynamically with the live DB taxonomy.

        Starts from the static formatting rules and appends the course/topic/
        subtopic and prerequisite catalog read live, so that the LLM reuses the
        slugs and names already in existence instead of inventing variants.
        """
        rules = _REFINING_SYSTEM_PROMPT if refining else _STRUCTURING_SYSTEM_PROMPT
        names = self._live_taxonomy_names()
        return (
            rules
            + "\n\n"
            + _build_dynamic_curriculum_prompt(self._live_taxonomy_tree(), names)
            + "\n\n"
            + _build_dynamic_prerequisites_prompt(self._live_prerequisites(), names)
        )

    def structure(
        self,
        intermediate: TranscriptionIntermediate,
        *,
        hints: str | None = None,
    ) -> list[StructuredQuestion]:
        """Turn the intermediate transcription into enriched questions.

        A single LLM call returns, per question, the formatted TeX and the
        curricular classification (course, topic, subtopic, difficulty and
        prerequisites).

        Raises:
            StructuringError: if there is no text, the provider fails or returns
                no question.
        """
        text = intermediate.raw_markdown.strip()
        if not text:
            raise StructuringError("there is no transcribed text to structure")

        messages = [
            {"role": "system", "content": self._build_system_prompt(refining=False)},
            {"role": "user", "content": self._build_user_message(intermediate, hints)},
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                response_model=list[QuestionDraft],
                messages=messages,
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - provider and schema errors
            raise StructuringError(f"remote structuring failed: {exc}") from exc

        if raw is None:
            drafts: list[QuestionDraft] = []
        elif isinstance(raw, list):
            drafts = raw
        else:
            drafts = list(raw)
        if not drafts:
            raise StructuringError("the remote LLM produced no question")

        return [self._assemble(draft, index, intermediate) for index, draft in enumerate(drafts)]

    def refine(
        self,
        intermediate: TranscriptionIntermediate,
        current: EnrichedQuestion,
        *,
        hints: str | None = None,
    ) -> StructuredQuestion:
        """Reanalyze ONE question applying corrective context.

        A single LLM call returns, in the SAME JSON response, the corrected TeX
        and the re-inferred curricular classification (``course``, ``topic``,
        ``subtopic``, ``difficulty`` and ``prerequisites``) using the same
        :class:`QuestionDraft` schema. It preserves the ``id`` and ``created_at``
        of the original question so the object updates the database. It does not
        solve the exercise.

        Raises:
            StructuringError: if the provider fails or returns no question.
        """
        messages = [
            {"role": "system", "content": self._build_system_prompt(refining=True)},
            {
                "role": "user",
                "content": self._build_refine_user_message(intermediate, current, hints),
            },
        ]
        try:
            raw = self.client.chat.completions.create(
                model=self.model,
                response_model=QuestionDraft,
                messages=messages,
                temperature=0.0,
                max_retries=self._max_retries,
            )
        except Exception as exc:  # noqa: BLE001 - provider and schema errors
            raise StructuringError(f"remote refinement failed: {exc}") from exc

        if raw is None:
            raise StructuringError("the remote LLM produced no refined question")

        return self._assemble(
            raw,
            0,
            intermediate,
            question_id=current.id,
            created_at=current.created_at,
        )

    def _build_user_message(
        self, intermediate: TranscriptionIntermediate, hints: str | None
    ) -> str:
        lines = ["Text transcribed by OCR (Markdown/TeX):", "", intermediate.raw_markdown]
        lines += [
            "",
            "Metadata:",
            f"- Source: {intermediate.source_reference}",
            f"- Pages: {', '.join(str(p) for p in intermediate.pages) or '(none)'}",
        ]
        if intermediate.figures:
            lines.append("- Available figures:")
            for figure in intermediate.figures:
                ymin, xmin, ymax, xmax = figure.bbox
                lines.append(
                    f"  - {figure.id} (page {figure.page}, bbox [ymin={ymin}, "
                    f"xmin={xmin}, ymax={ymax}, xmax={xmax}])"
                )
        if hints:
            lines.append(f"\nAdditional context from the reviewer: {hints}")
        lines.append(
            "\nReturn the list of structured questions, one per question, "
            "including statement_raw (TeX), question_type, choices, notation_notes, "
            "course, topic, subtopic, difficulty and prerequisites."
        )
        return "\n".join(lines)

    def _build_refine_user_message(
        self,
        intermediate: TranscriptionIntermediate,
        current: EnrichedQuestion,
        hints: str | None,
    ) -> str:
        lines = ["Current question to fix (Markdown/TeX):", "", current.statement_raw]
        lines += [
            "",
            "Current curricular classification:",
            f"- course: {current.course}",
            f"- topic: {current.topic}",
            f"- subtopic: {current.subtopic}",
            f"- difficulty: {current.difficulty}",
            f"- prerequisites: {', '.join(current.prerequisites) or '(none)'}",
        ]
        if current.choices:
            lines += ["", "Current choices:"]
            lines += [f"- {choice.id}: {choice.content_raw}" for choice in current.choices]
        if intermediate.figures:
            lines += ["", "Available figures:"]
            for figure in intermediate.figures:
                ymin, xmin, ymax, xmax = figure.bbox
                lines.append(
                    f"  - {figure.id} (page {figure.page}, bbox [ymin={ymin}, "
                    f"xmin={xmin}, ymax={ymax}, xmax={xmax}])"
                )
        if hints:
            lines += ["", f"Reviewer corrective context: {hints}"]
        lines.append(
            "\nFix the statement by applying the corrective context and RE-INFER the "
            "full curricular classification. Return ONE single question with "
            "statement_raw (TeX), question_type, choices, notation_notes, course, topic, "
            "subtopic, difficulty and prerequisites."
        )
        return "\n".join(lines)

    def _assemble(
        self,
        draft: QuestionDraft,
        index: int,
        intermediate: TranscriptionIntermediate,
        *,
        question_id: str | None = None,
        created_at: str | None = None,
    ) -> StructuredQuestion:
        resolved_id = question_id or deterministic_question_id(
            intermediate.source_reference, *[str(page) for page in intermediate.pages], str(index)
        )
        referenced = self._referenced_figure_ids(draft)
        figures = [figure for figure in intermediate.figures if figure.id in referenced]
        # Normalize the math delimiters before computing the canonical hash so that
        # the statement and deduplication stay consistent (idempotent with the
        # ``QuestionRaw`` validator).
        statement_raw = sanitize_math_delimiters(draft.statement_raw)
        try:
            question = EnrichedQuestion(
                id=resolved_id,
                statement_raw=statement_raw,
                question_type=draft.question_type,
                choices=draft.choices,
                figures=figures,
                notation_notes=draft.notation_notes,
                course=draft.course,
                topic=draft.topic,
                subtopic=draft.subtopic,
                difficulty=draft.difficulty,
                prerequisites=draft.prerequisites,
                source_metadata={"source_reference": intermediate.source_reference},
                created_at=created_at or utc_now_iso(),
                canonical_hash=compute_canonical_hash(statement_raw),
            )
        except ValidationError as exc:
            return StructuredQuestion(
                question=None,
                error=f"question {index + 1} did not pass the schema: {exc}",
            )
        return StructuredQuestion(
            question=question,
            taxonomy_names=self._collect_taxonomy_names(draft),
        )

    @staticmethod
    def _localized(entry: LocalizedName) -> dict[str, str] | None:
        """Normalize an LLM name to ``{es, en}``, or ``None`` if it ends up empty."""
        es = entry.es.strip()
        en = entry.en.strip()
        if not es or not en:
            return None
        return {"es": es, "en": en}

    @classmethod
    def _collect_taxonomy_names(cls, draft: QuestionDraft) -> list[TaxonomyNames]:
        """Project the LLM names into valid taxonomy entries.

        Only the names that correspond to the classification actually emitted are
        kept (course, topic, subtopic and prerequisites present); a named
        prerequisite that does not appear in the list is dropped.
        """
        entries: list[TaxonomyNames] = []

        def add(
            kind: TaxonomyKind, slug: str, parent_slug: str, source: LocalizedName | None
        ) -> None:
            if source is None:
                return
            names = cls._localized(source)
            if names is None:
                return
            entries.append(TaxonomyNames(kind=kind, slug=slug, parent_slug=parent_slug, **names))

        add("course", draft.course, "", draft.course_name)
        add("topic", draft.topic, draft.course, draft.topic_name)
        if draft.subtopic:
            add("subtopic", draft.subtopic, draft.course, draft.subtopic_name)
        prerequisites = set(draft.prerequisites)
        seen: set[str] = set()
        for item in draft.prerequisite_names:
            slug = slugify(item.slug)
            if not slug or slug not in prerequisites or slug in seen:
                continue
            names = cls._localized(item)
            if names is None:
                continue
            seen.add(slug)
            entries.append(TaxonomyNames(kind="prerequisite", slug=slug, **names))
        return entries

    @staticmethod
    def _referenced_figure_ids(draft: QuestionDraft) -> set[str]:
        text = draft.statement_raw
        if draft.choices:
            text += "\n" + "\n".join(choice.content_raw for choice in draft.choices)
        return {f"fig:{match}" for match in _PLACEHOLDER_RE.findall(text)}
