"""BFCL benchmark integration.

Bridges the official Berkeley Function Calling Leaderboard (vendored at
``third_party/bfcl-official``) into the repository's shared benchmark
architecture:

    Runner -> Adapter -> Runtime -> Agent -> Action -> Bridge -> Official evaluator

All BFCL-specific logic lives in this package. The official repo is treated
as a read-only dependency: datasets, checkers, category constants, and decode
conventions are reused from ``bfcl_eval`` — never copied or re-implemented.
See BFCL.md (project root) for analysis notes on the vendored repo.
"""
