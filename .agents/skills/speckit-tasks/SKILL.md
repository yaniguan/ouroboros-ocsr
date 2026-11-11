---
name: speckit-tasks
description: Derive an ordered, parallelizable task breakdown (tasks.md) and execute verification gates (pytest, ruff, check_no_data).
---

# Spec-Kit Tasks & Convergence Skill (Antigravity Integration)

1. Read `spec.md`, `plan.md`, and `data-model.md` inside `specs/<###-feature-name>/`.
2. Write `tasks.md` grouped into Setup, Foundational, User Story phases (`US1`..`USk`), and Verification Gates, marking file-disjoint tasks with `[P]`.
3. Verify convergence via `pytest -q`, `ruff check .`, and `python scripts/check_no_data.py`.
