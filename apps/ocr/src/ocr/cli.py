"""CLI for the OCR app: ``ocr worker``, ``ocr submit``, ``ocr list``, ``ocr export``,
``ocr mark``, ``ocr unmark``.

Workflows are started on ``ocr-task-queue`` (the OCR worker's queue) so they hit the
OCR-side activities; the platform worker on ``forge-task-queue`` services the batch
submit SPI and the poller.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid

import click
from sax_platform.contracts.constants import OCR_TASK_QUEUE
from sax_platform.temporal.client import connect_temporal

DEFAULT_TEMPORAL_ADDRESS = "localhost:7233"
EXIT_INFRASTRUCTURE_ERROR = 2


async def _start_and_wait(
    workflow_name: str, arg: object, *, workflow_id: str, address: str, timeout_hours: float
) -> object:
    from datetime import timedelta

    client = await connect_temporal(address)
    handle = await client.start_workflow(
        workflow_name,
        arg,
        id=workflow_id,
        task_queue=OCR_TASK_QUEUE,
    )
    return await handle.result(rpc_timeout=timedelta(hours=timeout_hours))


def _auto_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@click.group()
def main() -> None:
    """OCR app commands."""


@main.command("worker")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def worker_cmd(temporal_address: str) -> None:
    """Run the OCR Temporal worker (ocr-task-queue)."""
    from ocr.worker import run_worker

    asyncio.run(run_worker(temporal_address))


@main.command("submit")
@click.argument("file_path")
@click.option("--model", default="mistral:mistral-ocr-latest", show_default=True)
@click.option("--skip-duplicate-detection", is_flag=True)
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def submit_cmd(
    file_path: str, model: str, skip_duplicate_detection: bool, temporal_address: str
) -> None:
    """Submit a document for OCR via the platform batch service."""
    from ocr.models import OcrSubmitInput

    wf_input = OcrSubmitInput(
        file_path=file_path,
        model_name=model,
        skip_duplicate_detection=skip_duplicate_detection,
    )
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrSubmitWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-submit"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("list")
@click.option("--limit", default=50, show_default=True, type=int)
@click.option("--status", "status_filter", default="", help="processing|succeeded|errored")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def list_cmd(limit: int, status_filter: str, temporal_address: str) -> None:
    """List OCR submissions with status (ocr_job_status ⋈ batch_jobs)."""
    from ocr.models import OcrListJobsInput

    wf_input = OcrListJobsInput(limit=limit, status_filter=status_filter)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrListJobsWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-list"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("export")
@click.argument("document_id")
@click.option("--output-dir", default="", help="Defaults to $XDG_DATA_HOME/ocr/export/<id>.")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def export_cmd(document_id: str, output_dir: str, temporal_address: str) -> None:
    """Export OCR text + images for a document to the filesystem."""
    from ocr.models import OcrExportInput

    wf_input = OcrExportInput(document_id=document_id, output_dir=output_dir)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrExportWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-export"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("mark")
@click.argument("document_id")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def mark_cmd(document_id: str, temporal_address: str) -> None:
    """Mark a document for removal (soft-delete; a periodic workflow deletes it)."""
    from ocr.models import OcrMarkInput

    wf_input = OcrMarkInput(document_id=document_id)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrMarkForRemovalWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-mark"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


@main.command("unmark")
@click.argument("document_id")
@click.option(
    "--temporal-address",
    envvar="FORGE_TEMPORAL_ADDRESS",
    default=DEFAULT_TEMPORAL_ADDRESS,
    show_default=True,
)
def unmark_cmd(document_id: str, temporal_address: str) -> None:
    """Clear the removal mark on a document."""
    from ocr.models import OcrMarkInput

    wf_input = OcrMarkInput(document_id=document_id)
    try:
        result = asyncio.run(
            _start_and_wait(
                "OcrClearRemovalMarkWorkflow",
                wf_input,
                workflow_id=_auto_id("ocr-unmark"),
                address=temporal_address,
                timeout_hours=1.0,
            )
        )
        _echo(result)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(EXIT_INFRASTRUCTURE_ERROR)


def _echo(result: object) -> None:
    if hasattr(result, "model_dump_json"):
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
