+++
title = "How-to Guides"
weight = 30
description = "Task-oriented recipes for accomplishing specific goals with Forge."
+++

How-to guides assume you already know the basics and are looking for a direct answer to a specific question. Each guide is scoped to one task. When you need background on why the system works this way, follow the links to the matching explanation; when you need field-level details, follow the links to reference.

- [How to Control Context Assembly](control-context-assembly/) — Include or exclude files, limit exploration, and tune token budgets.
- [How to Submit Tasks](submit-tasks/) — Run single-step and planned tasks from the CLI, with and without fan-out.
- [How to Configure Validation](configure-validation/) — Enable or disable validation checks, add test execution, and configure retry limits.
- [How to Configure LLM Dispatch](configure-llm-dispatch/) — Switch between sync and batch modes, override models, and tune the poll interval.
- [How to Debug a Workflow](debug-workflow/) — Inspect workflow history, find the prompt for a step, and diagnose validation failures.
- [How to Manage Playbooks](manage-playbooks/) — Run extraction on completed runs and list, inspect, or add entries to the forge playbook store.
- [How to Ingest Transcripts](ingest-transcripts/) — Ingest Claude Code session files individually or in bulk from `~/.claude/projects/`.
- [How to Add a Domain](add-domain/) — Register a new task domain with its role prompt, output requirements, and validation defaults.
- [How to Run OCR](run-ocr/) — Run synchronous OCR or submit a document to the batch path, then fetch results.
- [How to Run Evaluations](run-evaluations/) — Run planner evaluations against the eval corpus with deterministic and LLM-as-judge scoring.
- [How to Use the Nushell OCR Module](use-nushell-module/) — Load and call the Nushell OCR pipeline functions against a running Temporal worker.
