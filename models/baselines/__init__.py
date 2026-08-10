"""
Baselines package: rule-based and Drain-only log extractors.

These represent the classical stack that the FlowFill paper does NOT
benchmark against — but should.
"""

from .rule_based import extract as rule_based_extract, RuleBasedResult
from .drain_only import extract as drain_extract, DrainResult
