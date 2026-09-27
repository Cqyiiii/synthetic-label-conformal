"""V11 experiment-plan utilities.

The completed neural fitting runner is provided by ``synrcp.v11``.
This package retains the original frozen design registry and its validation.
"""
from .registry import MAIN_METHODS, ABLATION_METHODS, COUNTERPARTS
from .plan import load_plan, validate_plan

__all__ = ["MAIN_METHODS", "ABLATION_METHODS", "COUNTERPARTS", "load_plan", "validate_plan"]
