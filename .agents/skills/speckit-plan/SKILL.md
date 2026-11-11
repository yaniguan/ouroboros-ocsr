---
name: speckit-plan
description: Generate a Spec-Kit implementation plan (plan.md, research.md, data-model.md) for a feature in specs/###-feature/.
---

# Spec-Kit Plan Skill (Antigravity Integration)

1. Verify `specs/<###-feature-name>/spec.md` and `research.md` against `.specify/memory/constitution.md`.
2. Produce `plan.md` and `data-model.md` detailing:
   - Technical Context & Dual-Track Backend Design (CPU deterministic default + GPU/DFT/VLM pluggable backends),
   - Constitution Gate Checklist,
   - Concrete module paths under `ouroboros/`, CLI entrypoints under `scripts/`, and tests under `tests/`.
