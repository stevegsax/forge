"""Temporal workflow for playbook maintenance (Grow-and-Refine).

Orchestrates pruning of stale or harmful entries and consolidation
of semantically similar entries to prevent context collapse.
"""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from pbook.activities.maintenance import identify_prune_candidates
    from pbook.llm import ConsolidationResult
    from pbook.models import CapabilityTier, ModelConfig, resolve_model
    from pbook.prompts.consolidation import (
        build_consolidation_system_prompt,
        build_consolidation_user_prompt,
    )
    from pbook.workflow_steps.llm import LLMChatInput, LLMChatResult
    from pbook.workflow_steps.retry import LLM_RETRY_POLICY

_FETCH_TIMEOUT = timedelta(seconds=60)
_PRUNE_TIMEOUT = timedelta(seconds=60)
_CONSOLIDATE_TIMEOUT = timedelta(minutes=5)
_CONSOLIDATE_HEARTBEAT = timedelta(seconds=60)
_SAVE_TIMEOUT = timedelta(seconds=30)
_EMBEDDING_TIMEOUT = timedelta(seconds=60)


@workflow.defn
class MaintenanceWorkflow:
    """Orchestrate pruning and consolidation of playbook entries."""

    @workflow.run
    async def run(self) -> dict[str, Any]:
        # Step 1: Fetch all entries
        all_entries = await workflow.execute_activity(
            "fetch_all_entries_for_maintenance",
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )

        if not all_entries:
            return {"pruned": 0, "consolidated": 0}

        # Step 2: Identify and prune stale/harmful entries
        # ACE: Pruning mainly removes stale or harmful fragments
        prune_candidates = identify_prune_candidates(all_entries)
        prune_ids = [e["id"] for e in prune_candidates]

        pruned_count = 0
        if prune_ids:
            pruned_count = await workflow.execute_activity(
                "prune_entries",
                prune_ids,
                start_to_close_timeout=_PRUNE_TIMEOUT,
                result_type=int,
            )

        # Step 3: Identify clusters of semantically similar entries for
        # consolidation. Clustering runs server-side (embeddings never
        # cross the workflow boundary); the activity returns id-clusters
        # which we map back to the embedding-free entry dicts.
        # ACE: Grow-and-refine mechanism balances expansion with redundancy control
        cluster_id_lists = await workflow.execute_activity(
            "cluster_similar_entries",
            json.dumps({"threshold": 0.85, "exclude_ids": prune_ids}),
            start_to_close_timeout=_FETCH_TIMEOUT,
            result_type=list,
        )
        by_id = {e["id"]: e for e in all_entries}
        clusters = [[by_id[i] for i in ids if i in by_id] for ids in cluster_id_lists]
        clusters = [c for c in clusters if len(c) > 1]

        model = resolve_model(CapabilityTier.REASONING, ModelConfig())

        consolidated_count = 0
        for cluster in clusters:
            # Consolidation logic:
            # 1. Ask LLM to merge cluster into a single entry
            # 2. Compute embedding for the new entry
            # 3. Save the new entry (re-parents source rows)
            # 4. Prune the original entries in the cluster

            chat_result = await workflow.execute_activity(
                "llm_chat",
                LLMChatInput(
                    system_prompt=build_consolidation_system_prompt(),
                    user_prompt=build_consolidation_user_prompt(cluster),
                    output_type_name="ConsolidationResult",
                    model=model,
                    max_tokens=2048,
                ),
                start_to_close_timeout=_CONSOLIDATE_TIMEOUT,
                heartbeat_timeout=_CONSOLIDATE_HEARTBEAT,
                result_type=LLMChatResult,
                retry_policy=LLM_RETRY_POLICY,
            )
            result = ConsolidationResult.model_validate(chat_result.tool_input)

            if not result.merged_title or not result.merged_content:
                continue

            # Compute embedding for the new entry
            text_to_embed = f"{result.merged_title}\n{result.merged_content}"
            embedding = await workflow.execute_activity(
                "llm_embed",
                text_to_embed,
                start_to_close_timeout=_EMBEDDING_TIMEOUT,
                result_type=str,
                retry_policy=LLM_RETRY_POLICY,
            )

            # Save the consolidated entry directly (bypassing match-or-
            # attach) and re-parent the cluster's entry_sources rows
            # to the survivor before pruning the originals — otherwise
            # the cascade would drop them.
            cluster_ids = [e["id"] for e in cluster]
            await workflow.execute_activity(
                "save_consolidated_entry",
                json.dumps(
                    {
                        "merged_entry": {
                            "title": result.merged_title,
                            "content": result.merged_content,
                            "tags": result.merged_tags,
                            "embedding": embedding,
                        },
                        "cluster_ids": cluster_ids,
                    }
                ),
                start_to_close_timeout=_SAVE_TIMEOUT,
                result_type=int,
            )

            # Prune the originals (entry_sources already re-parented above).
            await workflow.execute_activity(
                "prune_entries",
                cluster_ids,
                start_to_close_timeout=_PRUNE_TIMEOUT,
            )

            consolidated_count += 1

        return {
            "pruned": pruned_count,
            "consolidated": consolidated_count,
            "clusters_found": len(clusters),
        }
