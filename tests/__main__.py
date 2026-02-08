"""CLI entry point for comprehensive test runner.

Usage:
    # Run all tests
    python -m tests --output reports/

    # Filter by category
    python -m tests --category attention

    # Run single model
    python -m tests --model SelfAttention

    # List models
    python -m tests --list-models

    # List categories
    python -m tests --list-categories
"""

import argparse
import sys
from pathlib import Path

from .models import MODEL_REGISTRY, list_categories, list_models
from .testing import TestRunner


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test runner for IR extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports"),
        help="Output directory for reports (default: reports)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Filter tests by category",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this specific model",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-4,
        help="Relative tolerance for verification (default: 1e-4)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for verification (default: 1e-4)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all registered models and exit",
    )
    parser.add_argument(
        "--list-categories",
        action="store_true",
        help="List all categories and exit",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_models:
        print("Registered Models:")
        print("-" * 60)
        for spec in list_models():
            categories = ", ".join(spec.categories)
            print(f"  {spec.name}")
            print(f"    Categories: {categories}")
            print(f"    Inputs: {spec.input_shapes}")
            print(f"    Description: {spec.description}")
            print()
        return 0

    if args.list_categories:
        print("Available Categories:")
        print("-" * 40)
        for cat in list_categories():
            models = list_models(cat)
            print(f"  {cat} ({len(models)} models)")
            for spec in models:
                print(f"    - {spec.name}")
        return 0

    # Validate model name if provided
    if args.model and args.model not in MODEL_REGISTRY:
        print(f"Error: Unknown model '{args.model}'")
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}")
        return 1

    # Run tests
    runner = TestRunner(
        output_dir=args.output,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("=" * 60)
    print("IR Extraction Comprehensive Test Runner")
    print("=" * 60)
    print()

    if args.model:
        print(f"Running test for model: {args.model}")
    elif args.category:
        print(f"Running tests for category: {args.category}")
    else:
        print("Running all tests")

    print(f"Output directory: {args.output}")
    print(f"Tolerances: rtol={args.rtol}, atol={args.atol}")
    print()

    results, summary_path = runner.run_and_report(
        category_filter=args.category,
        model_filter=args.model,
    )

    # Print results
    print("-" * 60)
    print("Results:")
    print("-" * 60)

    passed = 0
    failed = 0

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        status_mark = "✓" if result.passed else "✗"

        if result.passed:
            passed += 1
        else:
            failed += 1

        if args.verbose or not result.passed:
            print(f"  [{status_mark}] {result.model_name}: {status}")
            if result.verification_report:
                print(f"      Max diff: {result.verification_report.max_diff:.2e}")
            if result.error_message and not result.passed:
                # Truncate long error messages
                error = result.error_message
                if len(error) > 100:
                    error = error[:100] + "..."
                print(f"      Error: {error}")
            print(f"      Duration: {result.duration_seconds:.3f}s")
        else:
            print(f"  [{status_mark}] {result.model_name}: {status} ({result.duration_seconds:.3f}s)")

    print()
    print("-" * 60)
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed}")
    print(f"Pass rate: {100 * passed / len(results):.1f}%")
    print("-" * 60)
    print()
    print(f"Reports saved to: {args.output}/")
    print(f"Summary: {summary_path}")

    # Return exit code based on results
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
