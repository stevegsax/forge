"""S3 blob storage — shared by the platform and its consumer apps.

Imperative shell: all S3 blob I/O is encapsulated in the :class:`S3Blobs` client
so callers keep their byte-in / byte-out signatures. Blob I/O cannot be mediated
cross-queue (Temporal payload limits), so both the platform and consumer apps
import :class:`S3Blobs` and address blobs by key carried in contract messages.

The bucket and an optional key prefix are supplied at construction (a composition
root resolves them from ``FORGE_OCR_S3_BUCKET`` / ``FORGE_OCR_S3_PREFIX`` via
:class:`~sax_platform.config.BlobSettings` and injects the client). Credentials
come from the default AWS chain — no static keys in code or env.

S3 is the only blob store: an empty bucket raises at construction and any S3
error raises, failing the calling task. There is no inline-in-DB fallback and no
runtime failover.

NOTE: the ``FORGE_OCR_S3_*`` env names and the flat key scheme are retained from
the pre-split layout. Generalizing the names and adding a per-kind key namespace
(separate TTLs for reapable request/result blobs vs durable image/file blobs) is
a tracked follow-up — see development-plans/separate-ocr-into-its-own-repo.md.
"""

from __future__ import annotations

from typing import Any


class S3ConfigError(RuntimeError):
    """The blob store is misconfigured (an empty/unset bucket)."""


def _client() -> Any:
    # boto3 is heavy; import lazily so non-blob code paths don't pay for it. A
    # fresh client per call reuses the default session's cached service model and
    # keeps this module free of import-time / module-level I/O state.
    #
    # Typed ``Any``: boto3 ships no py.typed marker / stubs, so its client is
    # unavoidably untyped here (no boto3-stubs dependency per task scope).
    import boto3

    return boto3.client("s3")


class S3Blobs:
    """Blob client bound to an explicit bucket + prefix (T3.6).

    The bucket and key prefix are supplied at construction (resolved by a
    composition root from ``FORGE_OCR_S3_BUCKET`` / ``FORGE_OCR_S3_PREFIX`` and
    injected), never read from the environment per call. A lazy per-call boto3
    client keeps ``sax_platform.contracts`` sandbox-light.

    Frozen after construction (``__slots__`` plus a ``__setattr__`` guard); a
    bucket/prefix that changed after construction would be a bug, not a feature.

    Fail-fast: an empty ``bucket`` raises :class:`S3ConfigError` at construction
    — blob storage requires an S3 bucket.
    """

    __slots__ = ("_bucket", "_prefix")

    _bucket: str
    _prefix: str

    def __init__(self, bucket: str, prefix: str = "") -> None:
        if not bucket:
            raise S3ConfigError(
                "S3Blobs requires a non-empty bucket; blob storage requires an S3 bucket."
            )
        object.__setattr__(self, "_bucket", bucket)
        object.__setattr__(self, "_prefix", prefix)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is frozen; cannot set {name!r}")

    def build_key(self, blob_id: str) -> str:
        """Build the S3 object key for a blob id, applying the instance prefix."""
        return f"{self._prefix}{blob_id}"

    def put(self, key: str, data: bytes, content_type: str) -> None:
        """Upload bytes to ``s3://{bucket}/{key}``."""
        _client().put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes:
        """Fetch bytes from ``s3://{bucket}/{key}``."""
        response = _client().get_object(Bucket=self._bucket, Key=key)
        body: bytes = response["Body"].read()
        return body

    def delete(self, key: str) -> None:
        """Delete ``s3://{bucket}/{key}``."""
        _client().delete_object(Bucket=self._bucket, Key=key)
