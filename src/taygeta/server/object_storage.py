"""S3 object storage (SeaweedFS) for Taygeta figures and assets.

Section 7 of the specification. Centralizes the object lifecycle:

* create the bucket when needed, upload, download, query metadata, generate
  presigned URLs and delete;
* store in the database a stable key or an ``s3://bucket/key`` URI, never a
  temporary presigned URL nor an absolute local path;
* validate identifiers to prevent traversal in the HTTP figure proxy;
* let the UI display figures without knowing internal paths.

The client is built with boto3 and *path-style* addressing, suitable for the
SeaweedFS S3 gateway.
"""

from __future__ import annotations

import mimetypes
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO
from urllib.parse import quote

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError

from taygeta.core.settings import Settings, get_settings
from taygeta.core.urls import build_public_url, build_relative_path, split_asset_reference

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]

__all__ = [
    "ObjectStorage",
    "ObjectStorageError",
    "build_s3_uri",
    "get_object_storage",
    "guessed_content_type",
    "parse_s3_uri",
    "sanitize_asset_id",
    "validate_object_key",
]

_S3_SCHEME = "s3://"
_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


class ObjectStorageError(RuntimeError):
    """Controlled error of the S3 object storage."""


# ---------------------------------------------------------------------------
# Key and identifier validation
# ---------------------------------------------------------------------------


def validate_object_key(key: str) -> str:
    """Validates an object key relative to the bucket.

    Rejects absolute paths, backslashes, control characters and ``.`` / ``..``
    segments to prevent traversal. Returns the validated key.

    Raises:
        ObjectStorageError: if the key is not safe.
    """
    if not isinstance(key, str) or not key.strip():
        raise ObjectStorageError("the object key cannot be empty")
    if key != key.strip():
        raise ObjectStorageError("the object key cannot have leading or trailing whitespace")
    if key.startswith("/") or key.startswith("\\"):
        raise ObjectStorageError("the object key cannot be an absolute path")
    if "\\" in key:
        raise ObjectStorageError("the object key cannot contain backslashes")
    if _CONTROL_CHARS_RE.search(key):
        raise ObjectStorageError("the object key cannot contain control characters")
    segments = key.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise ObjectStorageError("the object key cannot contain empty, '.' or '..' segments")
    return key


def sanitize_asset_id(asset_id: str) -> str:
    """Validates a single-segment asset identifier (for the HTTP proxy)."""
    if not isinstance(asset_id, str):
        raise ObjectStorageError("the asset identifier must be text")
    candidate = asset_id.strip()
    if not _ASSET_ID_RE.match(candidate):
        raise ObjectStorageError(
            "invalid asset identifier; only letters, digits, '.', '_' and '-' are allowed"
        )
    return candidate


def build_s3_uri(bucket: str, key: str) -> str:
    """Builds a stable ``s3://bucket/key`` URI."""
    if not bucket:
        raise ObjectStorageError("the bucket cannot be empty")
    return f"{_S3_SCHEME}{bucket}/{validate_object_key(key)}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Splits ``s3://bucket/key`` into ``(bucket, key)`` validating the key."""
    if not uri.startswith(_S3_SCHEME):
        raise ObjectStorageError(f"invalid object URI (expected s3://): {uri!r}")
    remainder = uri[len(_S3_SCHEME) :]
    bucket, _, key = remainder.partition("/")
    if not bucket or not key:
        raise ObjectStorageError(f"invalid object URI: {uri!r}")
    return bucket, validate_object_key(key)


def guessed_content_type(key: str) -> str:
    """Infers the MIME type from the key extension."""
    content_type, _ = mimetypes.guess_type(key)
    return content_type or "application/octet-stream"


# ---------------------------------------------------------------------------
# Storage client
# ---------------------------------------------------------------------------


class ObjectStorage:
    """High-level S3 client for the Taygeta figures bucket."""

    def __init__(self, settings: Settings | None = None, *, client: Any | None = None) -> None:
        """Creates the client.

        Args:
            settings: Configuration to use; defaults to the global one.
            client: An already-built boto3 client (injected for unit tests). If
                omitted, a real one is built lazily.
        """
        self.settings = settings or get_settings()
        self.bucket = self.settings.S3_BUCKET
        self._client = client

    # -- Client -------------------------------------------------------------
    @property
    def client(self) -> Any:
        """boto3 S3 client, built lazily on first use."""
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        return boto3.session.Session().client(
            "s3",
            endpoint_url=self.settings.S3_ENDPOINT_URL,
            aws_access_key_id=self.settings.S3_ACCESS_KEY_ID,
            aws_secret_access_key=self.settings.S3_SECRET_ACCESS_KEY,
            region_name=self.settings.S3_REGION,
            config=config,
        )

    # -- Bucket -------------------------------------------------------------
    def ensure_bucket(self, bucket: str | None = None) -> bool:
        """Creates the bucket if it does not exist. Returns ``True`` if it had to create it."""
        target = bucket or self.bucket
        if not target:
            raise ObjectStorageError("no bucket was configured")
        try:
            self.client.head_bucket(Bucket=target)
            return False
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code not in ("404", "NoSuchBucket", "NotFound") and status != 404:
                raise self._wrap_error("head_bucket", target, exc) from exc

        params: dict[str, Any] = {"Bucket": target}
        if self.settings.S3_REGION and self.settings.S3_REGION != "us-east-1":
            params["CreateBucketConfiguration"] = {"LocationConstraint": self.settings.S3_REGION}
        try:
            self.client.create_bucket(**params)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                raise self._wrap_error("create_bucket", target, exc) from exc
        return True

    # -- Writing ------------------------------------------------------------
    def put_bytes(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Uploads ``data`` to ``key`` and returns the stable ``s3://bucket/key`` URI."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.put_object(
                Bucket=target,
                Key=key,
                Body=data,
                ContentType=content_type or guessed_content_type(key),
                Metadata=metadata or {},
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("put_object", key, exc) from exc
        return build_s3_uri(target, key)

    def put_file(
        self,
        key: str,
        path: str | Path,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Uploads the file ``path`` to ``key`` and returns the stable URI."""
        source = Path(path)
        if not source.is_file():
            raise ObjectStorageError(f"the source file does not exist: {source}")
        return self.put_bytes(
            key,
            source.read_bytes(),
            content_type=content_type,
            metadata=metadata,
            bucket=bucket,
        )

    def put_stream(
        self,
        key: str,
        stream: BinaryIO,
        *,
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
        bucket: str | None = None,
    ) -> str:
        """Uploads a binary stream to ``key`` and returns the stable URI."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.upload_fileobj(
                stream,
                target,
                key,
                ExtraArgs={
                    "ContentType": content_type or guessed_content_type(key),
                    "Metadata": metadata or {},
                },
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("upload_fileobj", key, exc) from exc
        return build_s3_uri(target, key)

    # -- Reading ------------------------------------------------------------
    def get_bytes(self, key: str, *, bucket: str | None = None) -> bytes:
        """Downloads and returns the bytes of object ``key``."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            response = self.client.get_object(Bucket=target, Key=key)
            return response["Body"].read()
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("get_object", key, exc) from exc

    def download_to_file(
        self,
        key: str,
        destination: str | Path,
        *,
        bucket: str | None = None,
    ) -> Path:
        """Downloads object ``key`` to ``destination`` and returns the path."""
        target = bucket or self.bucket
        validate_object_key(key)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(target, key, str(destination_path))
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("download_file", key, exc) from exc
        return destination_path

    def exists(self, key: str, *, bucket: str | None = None) -> bool:
        """Indicates whether the object exists."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.head_object(Bucket=target, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("404", "NoSuchKey", "NotFound") or status == 404:
                return False
            raise self._wrap_error("head_object", key, exc) from exc

    def head(self, key: str, *, bucket: str | None = None) -> dict[str, Any]:
        """Returns the object metadata (size, type, ETag, etc.)."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            response = self.client.head_object(Bucket=target, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("head_object", key, exc) from exc
        return {
            "bucket": target,
            "key": key,
            "size": response.get("ContentLength"),
            "content_type": response.get("ContentType"),
            "etag": response.get("ETag"),
            "last_modified": response.get("LastModified"),
            "metadata": response.get("Metadata", {}),
        }

    # -- Deletion -----------------------------------------------------------
    def delete(self, key: str, *, bucket: str | None = None) -> None:
        """Deletes object ``key`` idempotently (does not fail if it does not exist)."""
        target = bucket or self.bucket
        validate_object_key(key)
        try:
            self.client.delete_object(Bucket=target, Key=key)
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("delete_object", key, exc) from exc

    def delete_uri(self, uri: str) -> None:
        """Deletes the object referenced by an ``s3://bucket/key`` URI."""
        bucket, key = parse_s3_uri(uri)
        self.delete(key, bucket=bucket)

    # -- URLs ---------------------------------------------------------------
    def presigned_url(
        self,
        key: str,
        *,
        expires_in: int | None = None,
        bucket: str | None = None,
    ) -> str:
        """Generates a time-limited presigned URL to read the object.

        It must not be persisted in the database; it is generated on demand. The
        host of the URL is not rewritten after signing it, because that would
        invalidate the signature.
        """
        target = bucket or self.bucket
        validate_object_key(key)
        ttl = expires_in or self.settings.ASSET_URL_TTL_SECONDS
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": target, "Key": key},
                ExpiresIn=ttl,
            )
        except (ClientError, BotoCoreError) as exc:
            raise self._wrap_error("generate_presigned_url", key, exc) from exc

    def display_url(self, reference: str) -> str:
        """Returns a URL suitable for the UI from a key or stable URI.

        If ``ASSET_PUBLIC_BASE_URL`` is configured, it composes a public URL on
        top of the key; otherwise it generates a temporary presigned URL. The UI
        does not need to know internal storage paths.
        """
        bucket, key = self._split_reference(reference)
        public_base = self.settings.ASSET_PUBLIC_BASE_URL
        if public_base:
            return f"{public_base.rstrip('/')}/{quote(key)}"
        return self.presigned_url(key, bucket=bucket)

    # -- Assets -------------------------------------------------------------
    def generate_asset_key(
        self,
        prefix: str = "figures",
        *,
        asset_id: str | None = None,
        extension: str = ".png",
    ) -> str:
        """Generates a stable and safe object key for an asset."""
        normalized_prefix = prefix.strip("/")
        if normalized_prefix:
            validate_object_key(normalized_prefix)
        suffix = extension if extension.startswith(".") else f".{extension}"
        suffix = re.sub(r"[^A-Za-z0-9.]", "", suffix) or ".png"
        name = sanitize_asset_id(asset_id) if asset_id else uuid.uuid4().hex
        key = f"{normalized_prefix}/{name}{suffix}" if normalized_prefix else f"{name}{suffix}"
        return validate_object_key(key)

    def store_asset_bytes(
        self,
        data: bytes,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Uploads bytes as an asset with a stable key and returns the ``s3://...`` URI."""
        key = self.generate_asset_key(prefix, asset_id=asset_id, extension=extension)
        return self.put_bytes(key, data, content_type=content_type, metadata=metadata)

    def store_asset_file(
        self,
        path: str | Path,
        *,
        prefix: str = "figures",
        asset_id: str | None = None,
        extension: str = ".png",
        content_type: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Uploads a file as an asset with a stable key and returns the URI."""
        source = Path(path)
        return self.store_asset_bytes(
            source.read_bytes(),
            prefix=prefix,
            asset_id=asset_id,
            extension=extension or source.suffix or ".png",
            content_type=content_type,
            metadata=metadata,
        )

    def resolve(self, reference: str) -> tuple[str, str]:
        """Resolves a stable reference to a validated ``(bucket, key)``.

        Accepts ``s3://bucket/key``, ``/buckets/{bucket}/{key}`` or a key
        relative to the configured bucket. Rejects traversal and absolute local
        paths, so it is suitable for the ``/assets/{asset_id}`` proxy endpoint.
        """
        return self._split_reference(reference)

    def relative_path(self, reference: str) -> str:
        """Returns the canonical relative path ``/buckets/{bucket}/{key}`` of a reference."""
        bucket, key = self._split_reference(reference)
        return build_relative_path(bucket, key)

    def public_url(self, reference: str) -> str:
        """Returns the full public URL of an asset reference.

        Concatenates ``S3_PUBLIC_BASE_URL`` with the canonical relative path, to
        send the information to the frontend or the MCP client.
        """
        return build_public_url(self.relative_path(reference), self.settings)

    def _split_reference(self, reference: str) -> tuple[str, str]:
        value = (reference or "").strip()
        if not value:
            raise ObjectStorageError("the asset reference cannot be empty")
        try:
            bucket, key = split_asset_reference(value, default_bucket=self.bucket)
        except ValueError as exc:
            raise ObjectStorageError(str(exc)) from exc
        return bucket or self.bucket, validate_object_key(key)

    @staticmethod
    def _wrap_error(operation: str, key: str, exc: Exception) -> ObjectStorageError:
        return ObjectStorageError(f"failure in {operation} on object {key!r}: {exc}")


def get_object_storage() -> ObjectStorage:
    """Returns an object storage client with the global configuration."""
    return ObjectStorage(get_settings())
