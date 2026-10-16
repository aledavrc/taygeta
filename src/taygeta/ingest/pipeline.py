"""Orchestration of the hybrid ingestion pipeline (sections 4.1-4.4).

Flow:

1. **Preparation** (:mod:`image_extractor`): rasterization at 300 DPI, figure
   detection, and cropping/storage in SeaweedFS.
2. **Pass 1 — local OCR** (:mod:`ocr_client`): faithful transcription to
   Markdown/TeX.
3. **Single-pass structuring** (:mod:`taygeta.agents.structurer`): a single LLM
   call converts the text into
   :class:`~taygeta.core.schema.EnrichedQuestion` objects (formatted TeX +
   ``course``, ``topic``, ``subtopic``, ``difficulty``, and ``prerequisites``)
   using ``instructor`` with Pydantic v2, without solving the exercise.
4. **Two-layer validation** (:mod:`validator`): local heuristic + remote
   semantic. The result is strictly classified as ``VALID`` or
   ``NEEDS_ATTENTION`` and stays in staging (it is never saved to the
   definitive database without human confirmation).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from taygeta.agents.structurer import Structurer, StructuringError
from taygeta.core.schema import EnrichedQuestion, FigureReference, TranscriptionIntermediate
from taygeta.core.settings import Settings, get_settings
from taygeta.domain.taxonomy import TaxonomyNames
from taygeta.ingest.image_extractor import (
    DEFAULT_DPI,
    assign_figure_ids,
    detect_figures,
    extract_and_save_figures,
    open_document,
    rasterize_document,
    upload_crops,
    working_directory,
)
from taygeta.ingest.ocr_client import OCRClient
from taygeta.ingest.validator import (
    STATUS_NEEDS_ATTENTION,
    STATUS_VALID,
    LocalValidationResult,
    LogicalValidationResult,
    RemoteValidationError,
    SemanticValidator,
    validate_local,
)

logger = logging.getLogger(__name__)

__all__ = [
    "STATUS_VALID",
    "STATUS_NEEDS_ATTENTION",
    "IngestPipeline",
    "StagedQuestion",
    "StagingResult",
    "main",
]


# ---------------------------------------------------------------------------
# Staging results
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class StagedQuestion:
    """Enriched and classified question after two-layer validation."""

    question: EnrichedQuestion | None
    status: Literal["VALID", "NEEDS_ATTENTION"]
    reasons: list[str] = field(default_factory=list)
    local: LocalValidationResult | None = None
    logical: LogicalValidationResult | None = None
    taxonomy_names: list[TaxonomyNames] = field(default_factory=list)


@dataclass(slots=True)
class StagingResult:
    """Full result of an ingestion, ready for human review."""

    source_reference: str
    status: Literal["VALID", "NEEDS_ATTENTION"]
    questions: list[StagedQuestion] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    session_id: str | None = None
    intermediate: TranscriptionIntermediate | None = None

    @property
    def valid_count(self) -> int:
        return sum(1 for item in self.questions if item.status == STATUS_VALID)

    @property
    def attention_count(self) -> int:
        return sum(1 for item in self.questions if item.status == STATUS_NEEDS_ATTENTION)

    def summary(self) -> dict[str, Any]:
        """Human-readable summary of the ingestion result."""
        return {
            "source_reference": self.source_reference,
            "status": self.status,
            "session_id": self.session_id,
            "total": len(self.questions),
            "valid": self.valid_count,
            "needs_attention": self.attention_count,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestPipeline:
    """Orchestrates the two-pass ingestion and the two-layer validation."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        ocr: Any | None = None,
        structurer: Structurer | None = None,
        semantic_validator: SemanticValidator | None = None,
        storage: Any | None = None,
        database: Any | None = None,
    ) -> None:
        """Create the pipeline.

        ``ocr``, ``structurer``, ``semantic_validator``, ``storage``, and
        ``database`` are injectable for unit tests; by default they are built
        from the global configuration.
        """
        self.settings = settings or get_settings()
        self.database = database
        self.ocr = ocr if ocr is not None else OCRClient(self.settings)
        self.structurer = (
            structurer if structurer is not None else Structurer(self.settings, database=database)
        )
        self.semantic = (
            semantic_validator
            if semantic_validator is not None
            else SemanticValidator(self.settings)
        )
        self.storage = storage

    def ingest(
        self,
        source_path: str | Path,
        *,
        hints: str | None = None,
        persist_staging: bool = True,
        session_id: str | None = None,
        upload_figures: bool = True,
    ) -> StagingResult:
        """Run the full ingestion and return a staging result.

        OCR, structuring, or validation failures become an explicable
        ``NEEDS_ATTENTION`` result; they are never lost silently.
        """
        source = Path(source_path)
        source_reference = source.name
        warnings: list[str] = []

        try:
            rasters = rasterize_document(source, dpi=DEFAULT_DPI)
        except Exception as exc:  # noqa: BLE001
            return self._finish(
                StagingResult(
                    source_reference=source_reference,
                    status=STATUS_NEEDS_ATTENTION,
                    questions=[
                        StagedQuestion(
                            question=None,
                            status=STATUS_NEEDS_ATTENTION,
                            reasons=[f"could not rasterize the document: {exc}"],
                        )
                    ],
                    warnings=[f"rasterization failed: {exc}"],
                ),
                persist_staging,
                session_id,
            )

        figures = self._detect_and_store_figures(source, upload_figures, warnings)

        raw_parts: list[str] = []
        for raster in rasters:
            figure_hint = self._figure_hint(figures, raster.page_index)
            combined_hint = "\n".join(part for part in (figure_hint, hints) if part) or None
            try:
                ocr_result = self.ocr.transcribe_page(
                    raster.png_bytes,
                    page_index=raster.page_index,
                    hints=combined_hint,
                    source_reference=source_reference,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"OCR of page {raster.page_index} failed: {exc}")
                raw_parts.append(f"<!-- page {raster.page_index} -->\n[ILEGIBLE]")
                continue
            warnings.extend(ocr_result.warnings)
            raw_parts.append(f"<!-- page {raster.page_index} -->\n{ocr_result.text}")

        intermediate = TranscriptionIntermediate(
            source_reference=source_reference,
            raw_markdown="\n\n".join(raw_parts),
            pages=[raster.page_index for raster in rasters],
            figures=figures,
            warnings=warnings,
        )

        try:
            structured = self.structurer.structure(intermediate, hints=hints)
        except StructuringError as exc:
            return self._finish(
                StagingResult(
                    source_reference=source_reference,
                    status=STATUS_NEEDS_ATTENTION,
                    questions=[
                        StagedQuestion(
                            question=None, status=STATUS_NEEDS_ATTENTION, reasons=[str(exc)]
                        )
                    ],
                    warnings=[*warnings, str(exc)],
                ),
                persist_staging,
                session_id,
            )

        staged: list[StagedQuestion] = []
        for item in structured:
            if item.question is None:
                staged.append(
                    StagedQuestion(
                        question=None,
                        status=STATUS_NEEDS_ATTENTION,
                        reasons=[item.error or "structuring produced no question"],
                    )
                )
                continue
            staged.append(
                self.validate_question(
                    item.question, max_pages=len(rasters), taxonomy_names=item.taxonomy_names
                )
            )

        overall = (
            STATUS_VALID
            if staged and all(item.status == STATUS_VALID for item in staged)
            else STATUS_NEEDS_ATTENTION
        )
        result = StagingResult(
            source_reference=source_reference,
            status=overall,
            questions=staged,
            warnings=warnings,
            intermediate=intermediate,
        )
        return self._finish(result, persist_staging, session_id)

    def validate_question(
        self,
        question: EnrichedQuestion,
        *,
        max_pages: int | None = None,
        taxonomy_names: Sequence[TaxonomyNames] | None = None,
    ) -> StagedQuestion:
        """Run the two validation layers on a question and classify it.

        ``taxonomy_names`` carries the human-readable names (es/en) generated by
        the LLM so that they survive until confirmation and are persisted then.
        """
        local = validate_local(question, max_pages=max_pages)
        reasons: list[str] = []

        logical: LogicalValidationResult | None = None
        try:
            logical = self.semantic.validate(question)
        except RemoteValidationError as exc:
            reasons.append(f"semantic validation unavailable: {exc}")
        else:
            if not logical.is_valid:
                reasons.append(logical.rejection_reason or "reason not specified")

        if not local.is_valid:
            reasons.append(
                "local validation: " + "; ".join(issue.message for issue in local.errors)
            )

        valid = local.is_valid and logical is not None and logical.is_valid
        return StagedQuestion(
            question=question,
            status=STATUS_VALID if valid else STATUS_NEEDS_ATTENTION,
            reasons=reasons,
            local=local,
            logical=logical,
            taxonomy_names=list(taxonomy_names or []),
        )

    def refine_question(
        self,
        intermediate: TranscriptionIntermediate,
        question: EnrichedQuestion,
        *,
        hints: str | None = None,
    ) -> StagedQuestion:
        """Re-analyze a question with corrective context and revalidate it.

        Re-runs the structuring pass on a SINGLE question using the same
        ``QuestionDraft`` schema: the LLM returns, in the same JSON response,
        the corrected TeX and the full re-inferred taxonomy (``course``,
        ``topic``, ``subtopic``, ``difficulty``, and ``prerequisites``). Then it
        applies the two validation layers. It does not solve the exercise.
        """
        refined = self.structurer.refine(intermediate, question, hints=hints)
        if refined.question is None:
            return StagedQuestion(
                question=None,
                status=STATUS_NEEDS_ATTENTION,
                reasons=[refined.error or "the re-analysis did not produce a valid question"],
            )
        max_pages = len(intermediate.pages) or None
        return self.validate_question(
            refined.question, max_pages=max_pages, taxonomy_names=refined.taxonomy_names
        )

    # -- Internals ----------------------------------------------------------
    def _detect_and_store_figures(
        self, source: Path, upload: bool, warnings: list[str]
    ) -> list[FigureReference]:
        detected: list[FigureReference] = []
        try:
            with open_document(source) as document:
                for page in document:
                    detected.extend(detect_figures(page))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not detect figures: {exc}")
            return []

        if not detected:
            return []

        detected = assign_figure_ids(detected)
        if self.storage is None or not upload:
            return detected

        figures: list[FigureReference] = []
        try:
            with working_directory() as tmpdir:
                by_page: dict[int, list[FigureReference]] = {}
                for figure in detected:
                    by_page.setdefault(figure.page, []).append(figure)
                for page_index, page_figures in sorted(by_page.items()):
                    crops = extract_and_save_figures(
                        source, page_index, page_figures, tmpdir, dpi=DEFAULT_DPI
                    )
                    crops = upload_crops(crops, self.storage, prefix="figures")
                    for crop in crops:
                        figures.append(
                            FigureReference(
                                id=crop.figure_id,
                                page=crop.page_index,
                                bbox=crop.normalized_bbox,
                                caption=crop.caption,
                                asset_path=crop.asset_uri,
                            )
                        )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"could not store the final figures: {exc}")
            return detected
        return figures

    @staticmethod
    def _figure_hint(figures: list[FigureReference], page_index: int) -> str:
        page_figures = [figure for figure in figures if figure.page == page_index]
        if not page_figures:
            return ""
        lines = ["Figuras detectadas por geometría en esta página:"]
        for figure in page_figures:
            ymin, xmin, ymax, xmax = figure.bbox
            lines.append(
                f"- {figure.id}: región normalizada [ymin={ymin}, xmin={xmin}, "
                f"ymax={ymax}, xmax={xmax}]"
            )
        lines.append(
            "No todas estas regiones son necesariamente diagramas útiles: puede "
            "haber fondos, logotipos o páginas escaneadas. Inserta el placeholder "
            "{{fig:N}} SOLO si el enunciado depende explícitamente de un gráfico, "
            "diagrama o ilustración dentro de esa región; si no hay tal figura, no "
            "insertes ningún placeholder ni declares fig:N."
        )
        return "\n".join(lines)

    def _staging_data(self, result: StagingResult) -> dict[str, Any]:
        return {
            "source_reference": result.source_reference,
            "status": result.status,
            "warnings": result.warnings,
            "intermediate": (
                result.intermediate.model_dump() if result.intermediate is not None else None
            ),
            "questions": [
                {
                    "status": item.status,
                    "reasons": item.reasons,
                    "question": item.question.model_dump() if item.question else None,
                    "taxonomy_names": [names.model_dump() for names in item.taxonomy_names],
                }
                for item in result.questions
            ],
        }

    def _finish(
        self,
        result: StagingResult,
        persist_staging: bool,
        session_id: str | None,
    ) -> StagingResult:
        if persist_staging and self.database is not None:
            result.session_id = self.database.create_staging_session(
                self._staging_data(result), status=result.status, session_id=session_id
            )
        return result


def main(argv: list[str] | None = None) -> None:
    """Minimal ingestion CLI (``taygeta-ingest``).

    Processes a document with the hybrid pipeline (two passes + two-layer
    validation), persists the staging session, and prints a JSON summary.
    """
    parser = argparse.ArgumentParser(
        prog="taygeta-ingest",
        description="Hybrid ingestion of a PDF/PNG/JPG document and staging summary.",
    )
    parser.add_argument("source", help="Path of the document to process")
    parser.add_argument("--hint", default=None, help="Corrective context for the OCR/LLM analysis")
    parser.add_argument(
        "--no-persist",
        action="store_true",
        help="Do not create the staging session in the database",
    )
    args = parser.parse_args(argv)

    pipeline = IngestPipeline(get_settings())
    result = pipeline.ingest(args.source, hints=args.hint, persist_staging=not args.no_persist)
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
