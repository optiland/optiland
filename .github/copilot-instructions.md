# Copilot Instructions for Optiland

Follow `AGENTS.md` at the repository root. It is the single source of guidance for every
AI coding agent in this repository: setup, the definition of done, hard rules and where
code lives. Human-facing details are in `CONTRIBUTING.md`.

Copilot-specific notes:

- Set up the environment with `uv` (`pip install uv`, then `uv sync --group dev`).
- Use the pinned tools through `uv run` (for example `uv run ruff check .`) so local results
  match CI.
