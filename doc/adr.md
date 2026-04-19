# Architecture Decision Records

## ADR 1: Project Structure and Dependency Management

**Date:** 2026-04-19

**Status:** Accepted

**Context:**
The goal is to convert a complex Colab notebook into a maintainable Python project. The project involves heavy dependencies like `inference-gpu` and `supervision`.

**Decision:**
- Use `uv` for lightning-fast dependency management and project isolation.
- Separate logic into modules (`config`, `models`, `tracking`, `visualization`, `pipeline`) to ensure clean separation of concerns and ease of testing.
- Use a `src/` layout for the package.

**Consequences:**
- Easier to maintain and extend compared to a single script.
- Consistent environment setup using `uv.lock`.
- Clear boundaries between model inference and domain logic (basketball tracking).
