"""S3 blob storage — shared by the platform and its consumer apps.

Imperative shell: all S3 blob I/O is encapsulated here so callers keep their
byte-in / byte-out signatures. Blob I/O cannot be mediated cross-queue (Temporal
payload limits), so both the platform and consumer apps import this module from
``forge-contracts`` and address blobs by key carried in contract messages.

The bucket is configured by ``FORGE_OCR_S3_BUCKET``; an optional
``FORGE_OCR_S3_PREFIX`` namespaces keys. Credentials come from the default AWS
chain (env / instance role).

To target an S3-COMPATIBLE backend (e.g. Supabase Storage, MinIO, Cloudflare R2)
set ``FORGE_OCR_S3_ENDPOINT_URL`` to the service's S3 endpoint. When set, the
client uses path-style addressing (required by most non-AWS S3 services, and by
bucket names that aren't DNS-safe) and the region from ``AWS_DEFAULT_REGION`` /
``AWS_REGION`` — which for these services must match the backend's region, and the
credentials must be that backend's S3 keys, not AWS keys. Unset → plain AWS S3.

S3 is the only blob store: an unset bucket or an S3 error raises, which fails the
calling task. There is no inline-in-DB fallback and no runtime failover.

NOTE: the ``FORGE_OCR_S3_*`` env names and the flat key scheme are retained from
the pre-split layout. Generalizing the names and adding a per-kind key namespace
(separate TTLs for reapable request/result blobs vs durable image/file blobs) is
a tracked follow-up — see development-plans/separate-ocr-into-its-own-repo.md.
"""

from __future__ import annotations

import os


class S3ConfigError(RuntimeError):
    """``FORGE_OCR_S3_BUCKET`` is not configured."""


def get_bucket() -> str:
    """Return the configured blob bucket, or raise if unset."""
    bucket = os.environ.get("FORGE_OCR_S3_BUCKET")
    if not bucket:
        raise S3ConfigError(
            "FORGE_OCR_S3_BUCKET is not set; blob storage requires an S3 bucket."
        )
    return bucket


def build_key(blob_id: str) -> str:
    """Build the S3 object key for a blob id, applying ``FORGE_OCR_S3_PREFIX``."""
    return f"{os.environ.get('FORGE_OCR_S3_PREFIX', '')}{blob_id}"


def _client():
    # boto3 is heavy; import lazily so non-blob code paths don't pay for it. A
    # fresh client per call reuses the default session's cached service model and
    # keeps this module free of import-time / module-level I/O state.
    import boto3

    endpoint = os.environ.get("FORGE_OCR_S3_ENDPOINT_URL") or None
    region = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or None
    if endpoint is None:
        # Plain AWS S3: default endpoint + default credential/region chain.
        return boto3.client("s3", region_name=region)

    # S3-compatible backend: path-style addressing is required by most non-AWS
    # services (and by non-DNS-safe bucket names).
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        config=Config(s3={"addressing_style": "path"}),
    )


def put(key: str, data: bytes, content_type: str) -> None:
    """Upload bytes to ``s3://{bucket}/{key}``."""
    _client().put_object(Bucket=get_bucket(), Key=key, Body=data, ContentType=content_type)


def get(key: str) -> bytes:
    """Fetch bytes from ``s3://{bucket}/{key}``."""
    response = _client().get_object(Bucket=get_bucket(), Key=key)
    return response["Body"].read()


def delete(key: str) -> None:
    """Delete ``s3://{bucket}/{key}``."""
    _client().delete_object(Bucket=get_bucket(), Key=key)
