"""Test framework utilities.

This module provides:
- Mermaid diagram generation from IR
- IR statistics collection
- Markdown report generation
- Test runner with verification
"""

from .mermaid import generate_op_distribution_pie, ir_to_mermaid
from .report import ReportGenerator, TestResult
from .runner import TestRunner
from .statistics import IRStatistics

__all__ = [
    "ir_to_mermaid",
    "generate_op_distribution_pie",
    "IRStatistics",
    "TestResult",
    "ReportGenerator",
    "TestRunner",
]
