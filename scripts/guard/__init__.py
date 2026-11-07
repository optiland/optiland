"""Repository guardrails: API snapshot, import contracts, quality ratchet, PR report.

Run ``python -m scripts.guard --help`` from the repository root. Only the API
snapshot and import-contract tests (under ``tests/``) can fail CI; everything
else is reported for review and never blocks a pull request.
"""
