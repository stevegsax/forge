# Mistral

## API Documentation

- API docs: <https://docs.mistral.ai/api/>
- Batch API: <https://docs.mistral.ai/capabilities/batch>
- OCR batch cookbook: <https://docs.mistral.ai/cookbooks/mistral-ocr-batch_ocr>
- API keys: <https://console.mistral.ai/api-keys/>

## Authentication

The composition root resolves `MISTRAL_API_KEY` via `LlmSettings` and passes it explicitly to `make_mistral_client(api_key)` (`sax_platform.ocr`, `libs/sax-platform/src/sax_platform/ocr.py`). The factory itself reads no environment variable — it raises only when the passed `api_key` is empty.

## SDK

The project uses the `mistralai` Python SDK. OCR batch submission uploads a JSONL file (`client.files.upload_async(file=..., purpose="batch")`) and creates the job from that file (`client.batch.jobs.create_async(input_files=[<file_id>], endpoint="/v1/ocr")`) — batch job creation for OCR does not take inline `requests=`/`input_data=`.

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

Each request in the batch JSONL is a dict with `custom_id` and `body`. For `/v1/ocr`, `body` is `{"document": ..., "include_image_base64": true}`, where `document` is `{"type": "image_url", "image_url": <data-uri>}` for images or `{"type": "document_url", "document_url": <data-uri>}` for PDFs:

```json
{"custom_id": "request-1", "body": {"document": {"type": "document_url", "document_url": "data:application/pdf;base64,..."}, "include_image_base64": true}}
```

### Batch result format

Output is JSONL. Each line's `response.body` is the OCR response shape — `pages[]` (each with `markdown` and `images[]`) plus `usage_info`:

```json
{"custom_id": "request-1", "response": {"body": {"model": "...", "pages": [{"markdown": "...", "images": [...]}], "usage_info": {"pages_processed": 1, "doc_size_bytes": 12345}}}}
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

OCR uses model `mistral-ocr-latest`. There is a single batch execution path: `OcrSubmitWorkflow` submits one `/v1/ocr` batch per chunk (start-only — it does not wait on the provider) and starts a parent-awaited, self-polling `OcrStoreWorkflow` child per chunk; each child polls its batch's status on a timer until it ends, then fetches and stores the result. (The library also retains a synchronous `MistralOcr.process`, a thin wrapper around `client.ocr.process_async` — no app currently calls it.) Batch OCR offers a 50% cost reduction over synchronous calls.

Both the batch path and `MistralOcr.process` use `include_image_base64: true` and share the same parse/store logic.

### Image extraction

The response returns images in `pages[].images[]` alongside markdown text. Each image has an `id` (e.g. `img-0.jpeg`), `image_base64` data, and optional bounding box coordinates.

Image IDs are sequential within a single API call and not unique across chunks of the same document, so the pipeline assigns each image a stable UUID (`uuid5` of the request id, source image id, and page index — `ocr_image_id`), which makes re-stores idempotent. Extraction and storage both happen inside one activity, `fetch_and_store_ocr_result`: `extract_images` pulls each image's base64 out of the response body (deleting `image_base64` from the JSON in place), the decoded bytes are written to blob storage as `ocr_images` rows, and the page markdown is rewritten from `![alt](img-0.jpeg)` to `![alt](ocr-image://{uuid})`. Because that all runs in a single activity, only a small `OcrStoreResult` summary returns to the workflow — the image bytes never enter Temporal workflow history or any activity payload. (The D88/T4.2 transport polls batch status on a timer and downloads results only in this one activity; the former "stay under Temporal's 2 MB signal limit" rationale no longer applies — there are no signals.)

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
