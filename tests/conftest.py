"""Pytest configuration and fixtures for comprehensive testing."""

from pathlib import Path
from typing import List

import pytest

from .testing import ReportGenerator, TestResult


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--output",
        action="store",
        default="reports",
        help="Output directory for reports (default: reports)",
    )
    parser.addoption(
        "--generate-reports",
        action="store_true",
        default=False,
        help="Generate markdown reports after tests",
    )
    parser.addoption(
        "--category",
        action="store",
        default=None,
        help="Filter tests by category",
    )


class ReportCollector:
    """Collects test results for report generation."""

    def __init__(self):
        self.results: List[TestResult] = []
        self._seen_models: set = set()

    def add_result(self, result: TestResult):
        # Avoid duplicate entries for same model
        if result.model_name not in self._seen_models:
            self.results.append(result)
            self._seen_models.add(result.model_name)


# Global collector instance
_report_collector: "ReportCollector | None" = None


@pytest.fixture(scope="session")
def report_collector(request) -> ReportCollector:
    """Fixture to collect test results for reporting."""
    global _report_collector
    # Reset for new session
    _report_collector = ReportCollector()
    return _report_collector


@pytest.fixture(scope="session")
def generate_reports(request) -> bool:
    """Fixture to check if reports should be generated."""
    return request.config.getoption("--generate-reports")


def pytest_sessionfinish(session, exitstatus):
    """Generate summary report after all tests complete."""
    global _report_collector

    if _report_collector is None or not _report_collector.results:
        return

    # Check if reports should be generated
    if not session.config.getoption("--generate-reports"):
        return

    output_dir = Path(session.config.getoption("--output"))
    generator = ReportGenerator(output_dir)

    # Save individual reports
    for result in _report_collector.results:
        generator.save_report(result)

    # Save summary
    summary_path = generator.save_summary(_report_collector.results)

    # Print summary location
    print(f"\n\nTest reports generated in: {output_dir}")
    print(f"Summary: {summary_path}")


# Common fixtures for model testing


@pytest.fixture
def rtol():
    """Default relative tolerance for numerical comparison."""
    return 1e-4


@pytest.fixture
def atol():
    """Default absolute tolerance for numerical comparison."""
    return 1e-4
