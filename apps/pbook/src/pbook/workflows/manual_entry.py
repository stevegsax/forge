"""Temporal workflow for manually submitted playbook entries.

Validates, reviews via LLM, and saves a manually submitted playbook
entry. The LLM call goes through the generic ``llm_chat`` step;
``apply_suggestions`` runs in workflow body to produce the final entry.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.llm import ReviewResult
    from pbook.models import (
        CapabilityTier,
        ModelConfig,
        PlaybookEntry,
        resolve_model,
    )
    from pbook.prompts.review import (
        apply_suggestions,
        build_review_system_prompt,
        build_review_user_prompt,
    )
    from pbook.workflow_steps.llm import LLMChatInput, LLMChatResult
    from pbook.workflow_steps.retry import LLM_RETRY_POLICY

_VALIDATE_TIMEOUT = timedelta(seconds=30)
_FETCH_TIMEOUT = timedelta(seconds=30)
_REVIEW_TIMEOUT = timedelta(minutes=2)
_REVIEW_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)
_EMBED_TIMEOUT = timedelta(seconds=60)


@workflow.defn
class ManualEntryWorkflow:
    """Validate, review, and save a manually submitted playbook entry."""

    @workflow.run
    async def run(self, raw_json: str) -> dict[str, Any]:
        # Step 1: Validate raw JSON
        validation_json = await workflow.execute_activity(
            "validate_entry",
            raw_json,
            start_to_close_timeout=_VALIDATE_TIMEOUT,
            result_type=str,
        )
        validation = json.loads(validation_json)

        if not validation["valid"]:
            return {
                "approved": False,
                "validation_error": validation["error"],
            }

        entry_dict = validation["entry"]

        # Step 2: Compute embedding for the proposed entry (base64 string).
        text_to_embed = f"{entry_dict['title']}\n{entry_dict['content']}"
        entry_embedding = await workflow.execute_activity(
            "llm_embed",
            text_to_embed,
            start_to_close_timeout=_EMBED_TIMEOUT,
            result_type=str,
            retry_policy=LLM_RETRY_POLICY,
        )
        entry_dict["embedding"] = entry_embedding

        # Step 3: Find semantic duplicates to prevent context collapse (ACE).
        duplicates = await workflow.execute_activity(
            "find_duplicates",
            json.dumps({"embedding": entry_embedding, "threshold": 0.85}),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        # Step 4: Fetch recent entries for broader context.
        existing = await workflow.execute_activity(
            "fetch_existing_entries",
            50,
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        # Combine duplicates and existing entries for the reviewer.
        # Prioritize duplicates in the review context.
        context_entries = duplicates + [
            e for e in existing if e["id"] not in {d["id"] for d in duplicates}
        ]

        # Step 5: LLM review via the generic chat step. The proposed entry
        # is the Pydantic class; the embedding lives separately as a
        # base64 string and is re-attached after apply_suggestions.
        proposed = PlaybookEntry.model_validate(
            {
                **entry_dict,
                "embedding": None,  # PlaybookEntry expects bytes; keep base64 separate
            }
        )
        model = resolve_model(CapabilityTier.CLASSIFICATION, ModelConfig())

        chat_result = await workflow.execute_activity(
            "llm_chat",
            LLMChatInput(
                system_prompt=build_review_system_prompt(context_entries[:50]),
                user_prompt=build_review_user_prompt(proposed),
                output_type_name="ReviewResult",
                model=model,
                max_tokens=1024,
            ),
            start_to_close_timeout=_REVIEW_TIMEOUT,
            heartbeat_timeout=_REVIEW_HEARTBEAT,
            result_type=LLMChatResult,
            retry_policy=LLM_RETRY_POLICY,
        )
        review = ReviewResult.model_validate(chat_result.tool_input)

        if not review.approved:
            return {
                "approved": False,
                "rejection_reason": review.rejection_reason,
            }

        # Step 6: Apply suggestions, restore the precomputed embedding, save.
        final = apply_suggestions(proposed, review)
        final_dict = final.model_dump()
        # base64; matches save_extracted_entries' expectation
        final_dict["embedding"] = entry_embedding

        save_input = json.dumps(
            {
                "entries": [final_dict],
                "project": final_dict.get("source_project", ""),
            }
        )
        count = await workflow.execute_activity(
            "save_extracted_entries",
            save_input,
            start_to_close_timeout=_SAVE_TIMEOUT,
            result_type=int,
        )

        return {
            "approved": True,
            "entry": final_dict,
            "entries_saved": count,
        }
