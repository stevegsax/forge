# S3 lifecycle policy

Closes the G34 finding (versioning on with no expiry — blobs and
noncurrent versions accumulate forever; see T0.7). Apply once, with the
real bucket creds in the environment:

```bash
aws s3api put-bucket-lifecycle-configuration \
  --bucket "$FORGE_OCR_S3_BUCKET" \
  --lifecycle-configuration file://deploy/s3/lifecycle.json
```

Verify with `aws s3api get-bucket-lifecycle-configuration --bucket …`.

## What it does — and deliberately does not — expire

- **Aborts incomplete multipart uploads** after 7 days (pure garbage).
- **Expires noncurrent versions** after 30 days (the bucket is
  versioned; old versions of overwritten blobs are recoverable for a
  month, then go).
- **Does not expire current objects.** Blob keys are flat ids under one
  optional global `FORGE_OCR_S3_PREFIX` (`forge_contracts.s3_blobs`), so
  a lifecycle rule cannot distinguish ephemeral batch payloads (dead
  after the ~25h batch window) from OCR results worth keeping. If batch
  blobs ever get their own key prefix, add a short-TTL expiry rule
  scoped to that prefix.

Once applied, the June review's §8 rationale (which cites a "bucket
TTL") becomes true.
