# Mistral

## API Documentation

- API docs: <https://docs.mistral.ai/api/>
- Batch API: <https://docs.mistral.ai/capabilities/batch>
- OCR batch cookbook: <https://docs.mistral.ai/cookbooks/mistral-ocr-batch_ocr>
- API keys: <https://console.mistral.ai/api-keys/>

## Authentication

Set the `MISTRAL_API_KEY` environment variable. The `make_mistral_client()` factory in `sax_platform.ocr` (`libs/sax-platform/src/sax_platform/ocr.py`) reads it and raises if it is unset.

## SDK

The project uses the `mistralai` Python SDK. As of v1.12.4, batch job creation uses `requests=` (not `input_data=`) — the parameter was renamed.

```python
from mistralai import Mistral

client = Mistral(api_key=api_key)
```

## Batch API

### Endpoints

| Operation | SDK method | REST endpoint |
| ----------- | ----------- | --------------- |
| Create job | `client.batch.jobs.create_async()` | `POST /v1/batch/jobs` |
| Get job | `client.batch.jobs.get_async(job_id=)` | `GET /v1/batch/jobs/{id}` |
| List jobs | `client.batch.jobs.list_async()` | `GET /v1/batch/jobs` |
| Download output | `client.files.download_async(file_id=)` | `GET /v1/files/{id}/content` |

### Job statuses

- `QUEUED` — waiting to start
- `RUNNING` — in progress
- `SUCCESS` — completed
- `FAILED` — failed
- `TIMEOUT_EXCEEDED` — exceeded timeout (default 24h)
- `CANCELLATION_REQUESTED` — cancel pending
- `CANCELLED` — cancelled

### Batch request format

Each request in the batch is a dict with `custom_id` and `body`:

```json
{"custom_id": "request-1", "body": {"model": "...", "messages": [...]}}
```

### Batch result format

Output is JSONL. Each line:

```json
{"custom_id": "request-1", "response": {"body": {"choices": [...], "usage": {...}}}}
```

## Supported endpoints for batch

The `endpoint` parameter on `create_async` accepts:

- `/v1/chat/completions`
- `/v1/embeddings`
- `/v1/fim/completions`
- `/v1/moderations`
- `/v1/chat/moderations`
- `/v1/ocr`
- `/v1/classifications`
- `/v1/chat/classifications`
- `/v1/conversations`
- `/v1/audio/transcriptions`

## OCR

OCR uses model `mistral-ocr-latest`. Two execution paths are available:

- **Synchronous** (`OcrSyncWorkflow`): Calls `client.ocr.process_async()` directly. Results return in seconds. Best for small documents.
- **Batch** (`OcrSubmitWorkflow`): Submits to the `/v1/ocr` batch endpoint. Results arrive via polling. Batch OCR offers a 50% cost reduction over synchronous calls.

Both paths use `include_image_base64: true` and share the same parse/store logic.

### Image extraction

The response returns images in `pages[].images[]` alongside markdown text. Each image has an `id` (e.g. `img-0.jpeg`), `image_base64` data, and optional bounding box coordinates.

Image IDs are sequential within a single API call but **not unique across chunks** of the same document. The Forge pipeline assigns a UUID to each image (during batch polling or inline in the sync path), strips `image_base64` from the response JSON (to stay under Temporal's 2MB signal limit), and rewrites markdown references from `![alt](img-0.jpeg)` to `![alt](ocr-image://{uuid})`.

## curl: list batch jobs

```bash
curl -s \
    -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
    -H "Accept: application/json" \
    "https://api.mistral.ai/v1/batch/jobs" | python -m json.tool
```

Filter by status:

```bash
curl -s \
    -H "Authorization: Bearer ${MISTRAL_API_KEY}" \
    "https://api.mistral.ai/v1/batch/jobs?status=RUNNING" | python -m json.tool
```

See also `scripts/mistral-batch-jobs.sh`.

## Forge integration

- Provider: `sax_platform.ocr.MistralOcr` (`libs/sax-platform/src/sax_platform/ocr.py`)
- Unsupported features (silently skipped per D63 degradation policy):

  - Prompt caching
  - Extended thinking
  - Cache control headers
