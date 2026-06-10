# Grill Session: forge-diataxis

Started: 2026-03-31
Last updated: 2026-03-31
Status: complete
Domain: Technical documentation (Diataxis framework) for an LLM task orchestrator

## Summary

Create a full Diataxis documentation set for Forge — a batch-first LLM task orchestrator. The documentation is a human-friendly entry point that sits alongside the existing design docs (which may be LLM-optimized). Organized by concept (not implementation phase), with a golden path tutorial, full quadrant coverage, and heavy cross-linking to avoid duplication.

## Decision Log

### DECIDED: Audience

- **Decision**: Internal developers who create or debug Forge workflows
- **Rationale**: These are the primary consumers who need to understand how the system works end-to-end
- **Date**: 2026-03-31

### DECIDED: Prerequisites

- **Decision**: Assume basic understanding of Python, Temporal, LLM prompting, templates, batch processing, and OCR. Link to official external docs for background on these topics rather than explaining them.
- **Rationale**: Internal developers already have this foundation; documenting it would be redundant
- **Date**: 2026-03-31

### DECIDED: Relationship to existing docs

- **Decision**: Sits alongside existing docs as a more accessible, human-facing entry point. Existing docs (ARCHITECTURE.md, DESIGN.md, phase specs) remain as-is — they may be optimized for LLM consumption.
- **Rationale**: Different audiences (human vs LLM) benefit from different documentation styles. The Diataxis set prioritizes explanation, step-by-step examples, and thorough discussion.
- **Date**: 2026-03-31

### DECIDED: Topic organization

- **Decision**: Organize by concept (context assembly, prompt construction, execution modes, etc.), not by implementation phase (Phase 1-14)
- **Rationale**: A developer debugging a workflow needs "how does context assembly work" — not "what was Phase 4"
- **Date**: 2026-03-31

### DECIDED: Quadrant coverage

- **Decision**: Full Diataxis quadrant set (tutorials, how-to guides, explanations, reference)
- **Rationale**: Different readers need different documentation types
- **Date**: 2026-03-31

### DECIDED: Scope

- **Decision**: Include everything: core orchestration pipeline, OCR pipeline, eval framework, observability store, knowledge extraction
- **Rationale**: Complete coverage of the system
- **Date**: 2026-03-31

### DECIDED: Depth balance and cross-linking

- **Decision**: Core orchestration gets full depth. Extensions (OCR, eval, observability, knowledge extraction) describe how they use the core and what they add — not re-describe core functionality. Heavy cross-linking ("See X for a detailed explanation") to avoid duplication.
- **Rationale**: Don't describe the same thing in multiple places. Extensions build on the core; document the delta.
- **Date**: 2026-03-31

### DECIDED: Tutorial approach

- **Decision**: Start with one golden path tutorial (end-to-end trace of a planned, multi-step task). Iterate from there to add mode-specific examples if needed.
- **Rationale**: The golden path covers the most concepts in one pass. Additional examples can be added incrementally.
- **Date**: 2026-03-31

## Open Threads

None — all branches resolved.

## Parking Lot

None.
