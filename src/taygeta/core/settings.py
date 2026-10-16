"""Typed configuration and environment validation for Taygeta.

Loads the configuration from environment variables and, optionally, from a
``.env`` file (never versioned). It keeps three families of credentials separate,
which must not be mixed:

* Remote LLM (``LLM_*``) for structuring and logical validation.
* Local OpenAI-compatible OCR (``OCR_*``) for visual extraction.
* S3/SeaweedFS object storage (``S3_*``).

Keys are never logged: use :meth:`Settings.safe_summary`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from taygeta.i18n import SUPPORTED_LOCALES

__all__ = ["Settings", "get_settings", "mask_secret"]


def mask_secret(value: str | None) -> str:
    """Return a safe representation of a secret for logs or summaries."""
    if not value:
        return "<empty>"
    if len(value) <= 4:
        return "****"
    return f"{value[:2]}****{value[-2:]}"


class Settings(BaseSettings):
    """Taygeta deployment configuration loaded from the environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -- Remote LLM (structuring and logical validation) --------------------
    LLM_API_KEY: str = Field(
        default="",
        description="API key of the remote provider. The credential only, not the endpoint.",
    )
    LLM_BASE_URL: str = Field(
        default="",
        description="Base URL of the remote endpoint, OpenAI API compatible.",
    )
    LLM_MODEL: str = Field(
        default="",
        description="Identifier of the remote model chosen by the administrator.",
    )
    LLM_TIMEOUT_SECONDS: int = Field(default=120, ge=1, description="Timeout per LLM request.")
    LLM_MAX_RETRIES: int = Field(default=2, ge=0, le=10, description="Bounded LLM retries.")
    LLM_EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description=(
            "sentence-transformers model (Hugging Face id or local path) to generate "
            "deduplication embeddings locally, without an external API. A multilingual "
            "model is recommended for Spanish text."
        ),
    )

    # -- Semantic deduplication ---------------------------------------------
    DUPLICATE_SIMILARITY_THRESHOLD: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine similarity threshold to mark a question as a duplicate candidate "
            "during ingestion."
        ),
    )

    # -- Local OCR (OpenAI-compatible) --------------------------------------
    OCR_API_KEY: str = Field(
        default="local-ocr",
        description="Credential required by the client for the local OCR service.",
    )
    OCR_BASE_URL: str = Field(
        default="http://ocr:8080/v1",
        description="OpenAI-compatible endpoint of the local OCR server.",
    )
    OCR_MODEL: str = Field(
        default="ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M",
        description="Model identifier accepted by the local OCR server.",
    )
    OCR_MODEL_PATH: str = Field(
        default="",
        description="Path of the model/weights inside the OCR persistent volume.",
    )
    OCR_TIMEOUT_SECONDS: int = Field(default=180, ge=1, description="Timeout for an OCR request.")
    OCR_CONTEXT_SIZE: int = Field(
        default=8192, ge=512, description="Context length of the local OCR model."
    )

    # -- Runtime environment and database -----------------------------------
    APP_ENV: Literal["local", "production"] = Field(
        default="local",
        description="Execution environment: 'local' or 'production'.",
    )
    DATABASE_URL: str = Field(
        default="",
        description=(
            "Database URL (e.g. sqlite:////app/data/taygeta.db). If empty, DB_PATH is used."
        ),
    )

    # -- SQLite and runtime directories -------------------------------------
    DB_PATH: str = Field(default="/app/data/taygeta.db", description="SQLite database path.")
    DOWNLOADS_DIR: str = Field(
        default="/app/downloads", description="Directory of temporary deliverables."
    )
    UPLOADS_DIR: str = Field(default="/app/uploads", description="Directory of received documents.")
    DOWNLOADS_TTL_SECONDS: int = Field(
        default=3600,
        ge=0,
        description="Cleanup TTL of /app/downloads (60 minutes by default).",
    )

    # -- S3/SeaweedFS object storage ----------------------------------------
    S3_ENDPOINT_URL: str = Field(
        default="http://seaweedfs-s3:8333", description="Internal S3 endpoint."
    )
    S3_PUBLIC_ENDPOINT: str = Field(
        default="http://localhost:8333",
        description=(
            "Public/external S3 endpoint for the UI (rewrites the internal host in links "
            "and images so they are reachable from the browser)."
        ),
    )
    S3_PUBLIC_BASE_URL: str = Field(
        default="http://localhost:8333",
        description=(
            "Public base for composing asset URLs from relative paths (e.g. "
            "https://assets.example.com). It is concatenated with the relative path "
            "/buckets/{bucket}/{key} when sending the information to the frontend or the MCP."
        ),
    )
    S3_BUCKET: str = Field(default="taygeta-assets", description="Figures bucket.")
    S3_ACCESS_KEY_ID: str = Field(default="change-me", description="S3 access credential.")
    S3_SECRET_ACCESS_KEY: str = Field(default="change-me", description="S3 secret key.")
    S3_REGION: str = Field(default="us-east-1", description="Region compatible with the S3 client.")
    ASSET_PUBLIC_BASE_URL: str = Field(
        default="", description="Optional public base or proxy endpoint for the UI."
    )
    ASSET_URL_TTL_SECONDS: int = Field(
        default=300, ge=1, description="Lifetime of the presigned asset URLs."
    )

    # -- Public service addresses -------------------------------------------
    MCP_HOST: str = Field(default="0.0.0.0", description="MCP server listen host.")
    MCP_PORT: int = Field(default=8000, ge=1, le=65535, description="MCP server port.")
    UI_PUBLIC_URL: str = Field(
        default="http://localhost:8501",
        validation_alias=AliasChoices("UI_PUBLIC_URL", "STREAMLIT_PUBLIC_URL"),
        description="Public URL of the HITL interface (NiceGUI).",
    )
    DOWNLOAD_PUBLIC_URL: str = Field(
        default="http://localhost:8000", description="Public URL of the downloads server."
    )

    # -- HITL interface (NiceGUI) -------------------------------------------
    UI_LANGUAGE: str = Field(
        default="es",
        description=(
            "Default language of the HITL interface ('es' or 'en'). Each session can "
            "switch it at runtime from the UI itself."
        ),
    )

    # -- Validators ---------------------------------------------------------
    @field_validator("APP_ENV", mode="before")
    @classmethod
    def _normalize_app_env(cls, value: object) -> str:
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized not in ("local", "production"):
            raise ValueError("APP_ENV must be 'local' or 'production'")
        return normalized

    @field_validator("UI_LANGUAGE", mode="before")
    @classmethod
    def _normalize_ui_language(cls, value: object) -> str:
        normalized = str(value).strip().lower() if value is not None else ""
        if normalized not in SUPPORTED_LOCALES:
            raise ValueError("UI_LANGUAGE must be one of: " + ", ".join(SUPPORTED_LOCALES))
        return normalized

    @field_validator("DB_PATH", "DOWNLOADS_DIR", "UPLOADS_DIR")
    @classmethod
    def _validate_runtime_paths(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("the path cannot be empty")
        return value

    @field_validator(
        "LLM_BASE_URL",
        "OCR_BASE_URL",
        "S3_ENDPOINT_URL",
        "S3_PUBLIC_ENDPOINT",
        "S3_PUBLIC_BASE_URL",
        "ASSET_PUBLIC_BASE_URL",
    )
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        if value and value != "/":
            return value.rstrip("/")
        return value

    # -- Properties and utilities -------------------------------------------
    @property
    def llm_configured(self) -> bool:
        """Whether the remote LLM has a complete credential, endpoint and model."""
        return bool(self.LLM_API_KEY and self.LLM_BASE_URL and self.LLM_MODEL)

    @property
    def ocr_configured(self) -> bool:
        """Whether the local OCR has an endpoint and model defined."""
        return bool(self.OCR_BASE_URL and self.OCR_MODEL)

    @property
    def s3_configured(self) -> bool:
        """Whether S3 storage has a bucket and credentials defined."""
        return bool(
            self.S3_ENDPOINT_URL
            and self.S3_BUCKET
            and self.S3_ACCESS_KEY_ID
            and self.S3_SECRET_ACCESS_KEY
        )

    @property
    def is_production(self) -> bool:
        """Whether the application runs in the production environment."""
        return self.APP_ENV == "production"

    @property
    def database_path(self) -> str:
        """Effective database path.

        If ``DATABASE_URL`` is set to a ``sqlite://`` URL, the path is extracted
        from it; otherwise ``DB_PATH`` is used. Non-SQLite URLs (e.g.
        ``postgresql://``) are not supported by the current SQLite layer and raise
        an explicit ``ValueError``.
        """
        url = (self.DATABASE_URL or "").strip()
        if not url:
            return self.DB_PATH
        if url == "sqlite://":
            return ":memory:"
        if url.startswith("sqlite:///"):
            # sqlite:///rel.db  -> "rel.db"        (relative path)
            # sqlite:////abs.db -> "/abs.db"       (absolute path)
            # sqlite:///:memory: -> ":memory:"
            return url[len("sqlite://") :][1:]
        raise ValueError(
            f"unsupported DATABASE_URL: {url!r}; the current layer only supports sqlite://"
        )

    def require_llm(self) -> None:
        """Validate that the remote LLM configuration is complete.

        Raises:
            RuntimeError: if any of the required variables is missing.
        """
        missing = [
            name for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL") if not getattr(self, name)
        ]
        if missing:
            raise RuntimeError("incomplete remote LLM configuration; define: " + ", ".join(missing))

    def ensure_runtime_directories(self) -> list[Path]:
        """Create the required runtime directories and return their paths.

        It does not create the database directory if its path is relative to the
        repository; in that case it still returns the paths involved.
        """
        directories = [
            Path(self.DB_PATH).expanduser().parent,
            Path(self.DOWNLOADS_DIR).expanduser(),
            Path(self.UPLOADS_DIR).expanduser(),
        ]
        created: list[Path] = []
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
        return created

    def safe_summary(self) -> dict[str, object]:
        """Log-friendly summary, with every credential masked."""
        return {
            "app_env": self.APP_ENV,
            "database_url": mask_secret(self.DATABASE_URL),
            "llm_configured": self.llm_configured,
            "llm_base_url": self.LLM_BASE_URL,
            "llm_model": self.LLM_MODEL,
            "llm_timeout_seconds": self.LLM_TIMEOUT_SECONDS,
            "llm_max_retries": self.LLM_MAX_RETRIES,
            "llm_embedding_model": self.LLM_EMBEDDING_MODEL,
            "duplicate_similarity_threshold": self.DUPLICATE_SIMILARITY_THRESHOLD,
            "ocr_base_url": self.OCR_BASE_URL,
            "ocr_model": self.OCR_MODEL,
            "ocr_timeout_seconds": self.OCR_TIMEOUT_SECONDS,
            "ocr_context_size": self.OCR_CONTEXT_SIZE,
            "db_path": self.DB_PATH,
            "downloads_dir": self.DOWNLOADS_DIR,
            "uploads_dir": self.UPLOADS_DIR,
            "downloads_ttl_seconds": self.DOWNLOADS_TTL_SECONDS,
            "s3_endpoint_url": self.S3_ENDPOINT_URL,
            "s3_public_endpoint": self.S3_PUBLIC_ENDPOINT,
            "s3_public_base_url": self.S3_PUBLIC_BASE_URL,
            "s3_bucket": self.S3_BUCKET,
            "s3_region": self.S3_REGION,
            "s3_access_key_id": mask_secret(self.S3_ACCESS_KEY_ID),
            "s3_secret_access_key": mask_secret(self.S3_SECRET_ACCESS_KEY),
            "llm_api_key": mask_secret(self.LLM_API_KEY),
            "ocr_api_key": mask_secret(self.OCR_API_KEY),
            "mcp_host": self.MCP_HOST,
            "mcp_port": self.MCP_PORT,
            "ui_public_url": self.UI_PUBLIC_URL,
            "download_public_url": self.DOWNLOAD_PUBLIC_URL,
            "ui_language": self.UI_LANGUAGE,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a single, cached instance of :class:`Settings`."""
    return Settings()
