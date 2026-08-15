"""
Decision procedure: given (L, lambda, batching, f1_tolerance) → which architecture.

See paper/BACKBONE.md §4 and issue #64 for the full specification.
"""

from models.decision.procedure import DecisionResult, decide, CONSTANTS, VALIDITY_DOMAIN
