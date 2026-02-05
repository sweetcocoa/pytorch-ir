"""Test framework utilities.

This module provides:
- Mermaid diagram generation from IR
- IR statistics collection
- Markdown report generation
- Test runner with verification
"""

from .mermaid import ir_to_mermaid, generate_op_distribution_pie
from .statistics import IRStatistics
from .report import TestResult, ReportGenerator
from .runner import TestRunner

__all__ = [
    "ir_to_mermaid",
    "generate_op_distribution_pie",
    "IRStatistics",
    "TestResult",
    "ReportGenerator",
    "TestRunner",
]
