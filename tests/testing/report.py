"""Markdown report generation for test results."""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from torch_ir import IR, VerificationReport

from .mermaid import generate_op_distribution_pie, ir_to_mermaid
from .statistics import IRStatistics


@dataclass
class TestResult:
    """Result of a single model test.

    Attributes:
        model_name: Name of the tested model.
        passed: Whether the test passed.
        ir: Extracted IR (if successful).
        verification_report: Verification report from comparison.
        statistics: IR statistics (if IR was extracted).
        error_message: Error message (if failed).
        duration_seconds: Test duration in seconds.
    """

    model_name: str
    passed: bool
    ir: Optional[IR] = None
    verification_report: Optional[VerificationReport] = None
    statistics: Optional[IRStatistics] = None
    error_message: Optional[str] = None
    duration_seconds: float = 0.0


class ReportGenerator:
    """Generates markdown reports from test results."""

    def __init__(self, output_dir: Path):
        """Initialize report generator.

        Args:
            output_dir: Directory to save reports.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_model_report(self, result: TestResult) -> str:
        """Generate detailed markdown report for a single model.

        Args:
            result: Test result to report.

        Returns:
            Markdown content as string.
        """
        lines = []

        # Header
        status = "PASSED" if result.passed else "FAILED"
        status_emoji = "✅" if result.passed else "❌"
        lines.append(f"# {result.model_name}")
        lines.append("")
        lines.append(f"**Status:** {status_emoji} {status}")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append(f"**Duration:** {result.duration_seconds:.3f}s")
        lines.append("")

        # Error message if failed
        if result.error_message:
            lines.append("## Error")
            lines.append("")
            lines.append("```")
            lines.append(result.error_message)
            lines.append("```")
            lines.append("")

        # Statistics summary
        if result.statistics:
            stats = result.statistics
            lines.append("## Summary")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            lines.append(f"| Nodes | {stats.num_nodes} |")
            lines.append(f"| Edges | {stats.num_edges} |")
            lines.append(f"| Inputs | {stats.num_inputs} |")
            lines.append(f"| Outputs | {stats.num_outputs} |")
            lines.append(f"| Weights | {stats.num_weights} |")
            lines.append(f"| Total Params | {stats.total_weight_params:,} |")
            lines.append("")

        # Verification report
        if result.verification_report:
            report = result.verification_report
            lines.append("## Numerical Verification")
            lines.append("")
            status = "PASSED" if report.is_valid else "FAILED"
            lines.append(f"**Status:** {status}")
            lines.append(f"**Max Difference:** {report.max_diff:.2e}")
            lines.append(f"**Mean Difference:** {report.mean_diff:.2e}")
            lines.append(f"**Outputs Compared:** {report.num_outputs}")
            lines.append("")

        # DAG visualization
        if result.ir:
            lines.append("## DAG Visualization")
            lines.append("")
            lines.append("```mermaid")
            lines.append(ir_to_mermaid(result.ir, max_nodes=30))
            lines.append("```")
            lines.append("")

        # Operator distribution
        if result.ir and result.ir.nodes:
            lines.append("## Operator Distribution")
            lines.append("")
            lines.append("```mermaid")
            lines.append(generate_op_distribution_pie(result.ir))
            lines.append("```")
            lines.append("")

        # Node details (collapsible)
        if result.statistics and result.statistics.node_shapes:
            lines.append("## Node Details")
            lines.append("")
            lines.append("<details>")
            lines.append("<summary>Click to expand</summary>")
            lines.append("")
            lines.append("| Name | Op Type | Input Shapes | Output Shapes |")
            lines.append("|------|---------|--------------|---------------|")

            for node in result.statistics.node_shapes:
                in_shapes = ", ".join(str(s) for s in node["input_shapes"])
                out_shapes = ", ".join(str(s) for s in node["output_shapes"])
                op_type = node["op_type"]
                # Shorten op_type for readability
                if op_type.startswith("aten."):
                    op_type = op_type[5:]
                if op_type.endswith(".default"):
                    op_type = op_type[:-8]
                lines.append(f"| {node['name']} | {op_type} | {in_shapes} | {out_shapes} |")

            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Weight metadata
        if result.statistics and result.statistics.weight_metadata:
            lines.append("## Weight Metadata")
            lines.append("")
            lines.append("| Name | Shape | Dtype | Parameters |")
            lines.append("|------|-------|-------|------------|")

            for w in result.statistics.weight_metadata:
                shape_str = "x".join(str(d) for d in w["shape"])
                lines.append(f"| {w['name']} | {shape_str} | {w['dtype']} | {w['num_params']:,} |")

            lines.append("")

        return "\n".join(lines)

    def generate_summary_report(self, results: List[TestResult]) -> str:
        """Generate summary markdown report for all tests.

        Args:
            results: List of test results.

        Returns:
            Markdown content as string.
        """
        lines = []

        # Header
        lines.append("# Test Summary Report")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().isoformat()}")
        lines.append("")

        # Overall statistics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        failed = total - passed

        lines.append("## Overall Results")
        lines.append("")
        lines.append(f"- **Total Tests:** {total}")
        lines.append(f"- **Passed:** {passed} ✅")
        lines.append(f"- **Failed:** {failed} ❌")
        lines.append(f"- **Pass Rate:** {100 * passed / total:.1f}%")
        lines.append("")

        # Results table
        lines.append("## Test Results")
        lines.append("")
        lines.append("| Model | Status | Nodes | Params | Max Diff | Duration |")
        lines.append("|-------|--------|-------|--------|----------|----------|")

        for result in results:
            status = "✅" if result.passed else "❌"
            nodes = result.statistics.num_nodes if result.statistics else "-"
            params = f"{result.statistics.total_weight_params:,}" if result.statistics else "-"

            if result.verification_report:
                max_diff = f"{result.verification_report.max_diff:.2e}"
            else:
                max_diff = "-"

            duration = f"{result.duration_seconds:.3f}s"

            # Link to detailed report
            model_link = f"[{result.model_name}](./{result.model_name}.md)"

            lines.append(f"| {model_link} | {status} | {nodes} | {params} | {max_diff} | {duration} |")

        lines.append("")

        # Category breakdown
        lines.append("## Results by Category")
        lines.append("")

        # Group by categories (requires importing models)
        try:
            from tests.models import MODEL_REGISTRY

            category_results: dict = {}
            for result in results:
                spec = MODEL_REGISTRY.get(result.model_name)
                if spec:
                    for cat in spec.categories:
                        if cat not in category_results:
                            category_results[cat] = {"passed": 0, "failed": 0}
                        if result.passed:
                            category_results[cat]["passed"] += 1
                        else:
                            category_results[cat]["failed"] += 1

            lines.append("| Category | Passed | Failed | Total |")
            lines.append("|----------|--------|--------|-------|")

            for cat, counts in sorted(category_results.items()):
                total_cat = counts["passed"] + counts["failed"]
                lines.append(f"| {cat} | {counts['passed']} | {counts['failed']} | {total_cat} |")

            lines.append("")
        except ImportError:
            pass

        # Failed tests details
        failed_results = [r for r in results if not r.passed]
        if failed_results:
            lines.append("## Failed Tests")
            lines.append("")

            for result in failed_results:
                lines.append(f"### {result.model_name}")
                lines.append("")
                if result.error_message:
                    lines.append("```")
                    lines.append(result.error_message)
                    lines.append("```")
                lines.append("")

        return "\n".join(lines)

    def save_report(self, result: TestResult) -> Path:
        """Save model report to file.

        Args:
            result: Test result to save.

        Returns:
            Path to saved file.
        """
        content = self.generate_model_report(result)
        path = self.output_dir / f"{result.model_name}.md"
        path.write_text(content)
        return path

    def save_summary(self, results: List[TestResult]) -> Path:
        """Save summary report to file.

        Args:
            results: List of test results.

        Returns:
            Path to saved file.
        """
        content = self.generate_summary_report(results)
        path = self.output_dir / "SUMMARY.md"
        path.write_text(content)
        return path
