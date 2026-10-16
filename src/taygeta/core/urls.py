"""Helpers to turn relative asset paths into public URLs.

Supports the dual Local/Production execution: the models store images as relative
paths (e.g. ``/buckets/exams/img.png``) and, right before sending the information
to the frontend or the MCP client, the full URLs are composed by concatenating
``S3_PUBLIC_BASE_URL``.
"""

from __future__ import annotations

from taygeta.core.schema import RELATIVE_ASSET_PREFIX, is_relative_asset_path
from taygeta.core.settings import Settings, get_settings

__all__ = [
    "build_public_url",
    "build_relative_path",
    "parse_relative_path",
    "relative_path_from_s3_uri",
    "split_asset_reference",
]

_S3_SCHEME = "s3://"


def build_relative_path(bucket: str, key: str) -> str:
    """Build the canonical relative path ``/buckets/{bucket}/{key}``.

    Raises:
        ValueError: if ``bucket`` or ``key`` end up empty.
    """
    normalized_bucket = (bucket or "").strip("/")
    normalized_key = (key or "").strip("/")
    if not normalized_bucket or not normalized_key:
        raise ValueError("bucket and key cannot be empty")
    return f"{RELATIVE_ASSET_PREFIX}{normalized_bucket}/{normalized_key}"


def parse_relative_path(path: str) -> tuple[str, str]:
    """Split ``/buckets/{bucket}/{key}`` into ``(bucket, key)``.

    Raises:
        ValueError: if ``path`` is not a valid relative asset path.
    """
    if not is_relative_asset_path(path):
        raise ValueError(f"invalid relative asset path: {path!r}")
    rest = path[len(RELATIVE_ASSET_PREFIX) :]
    bucket, _, key = rest.partition("/")
    return bucket, key


def relative_path_from_s3_uri(uri: str) -> str:
    """Convert ``s3://bucket/key`` into ``/buckets/{bucket}/{key}``.

    Raises:
        ValueError: if ``uri`` is not a valid ``s3://`` URI.
    """
    if not uri.startswith(_S3_SCHEME):
        raise ValueError(f"invalid object URI (expected s3://): {uri!r}")
    rest = uri[len(_S3_SCHEME) :]
    bucket, _, key = rest.partition("/")
    if not bucket or not key:
        raise ValueError(f"invalid object URI: {uri!r}")
    return build_relative_path(bucket, key)


def split_asset_reference(
    reference: str, *, default_bucket: str | None = None
) -> tuple[str | None, str]:
    """Split an asset reference into ``(bucket, key)``.

    Accepts ``s3://bucket/key``, ``/buckets/{bucket}/{key}`` or a key relative to
    the default bucket. For a relative key without ``default_bucket`` it returns
    ``(None, key)`` and the caller decides the bucket.

    Raises:
        ValueError: if the reference is empty or cannot be split.
    """
    value = (reference or "").strip()
    if not value:
        raise ValueError("the asset reference cannot be empty")
    if value.startswith(_S3_SCHEME):
        rest = value[len(_S3_SCHEME) :]
        bucket, _, key = rest.partition("/")
        if not bucket or not key:
            raise ValueError(f"invalid object URI: {reference!r}")
        return bucket, key
    if is_relative_asset_path(value):
        return parse_relative_path(value)
    if default_bucket:
        return default_bucket, value
    return None, value


def build_public_url(relative_path: str | None, settings: Settings | None = None) -> str:
    """Return the full public URL of a relative asset path.

    Concatenates ``S3_PUBLIC_BASE_URL`` (without a trailing slash) with the
    relative path, so ``/buckets/exams/img.png`` becomes, for example,
    ``https://cdn.example.com/buckets/exams/img.png``. If no public base is
    configured, returns the relative path as is (normalized with ``/``).
    """
    if not relative_path:
        return ""
    resolved = settings or get_settings()
    base = (resolved.S3_PUBLIC_BASE_URL or "").rstrip("/")
    path = relative_path if relative_path.startswith("/") else "/" + relative_path
    return base + path if base else path
