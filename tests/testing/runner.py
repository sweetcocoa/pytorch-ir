"""Test runner for comprehensive model testing."""

import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from torch_ir import extract_ir, verify_ir_with_state_dict

from ..models import MODEL_REGISTRY, TestModelSpec, list_models
from .report import ReportGenerator, TestResult
from .statistics import IRStatistics


class TestRunner:
    """Runner for comprehensive model testing."""

    def __init__(
        self,
        output_dir: Path = Path("reports"),
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ):
        """Initialize test runner.

        Args:
            output_dir: Directory for reports.
            rtol: Relative tolerance for verification.
            atol: Absolute tolerance for verification.
        """
        self.output_dir = Path(output_dir)
        self.rtol = rtol
        self.atol = atol
        self.report_generator = ReportGenerator(output_dir)

    def _create_test_inputs(
        self,
        spec: TestModelSpec,
        device: str = "cpu",
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, ...]:
        """Create test input tensors.

        Args:
            spec: Model specification.
            device: Device for tensors.
            batch_size: Batch size to prepend.

        Returns:
            Tuple of input tensors.
        """
        inputs = []
        for shape in spec.input_shapes:
            # Check if this is a token input (for embedding models)
            if spec.name == "WeightTying":
                # Integer inputs for embedding
                full_shape = (batch_size,) + shape
                tensor = torch.randint(0, 100, full_shape, device=device)
            else:
                full_shape = (batch_size,) + shape
                tensor = torch.randn(full_shape, device=device)
            inputs.append(tensor)
        return tuple(inputs)

    def _create_meta_inputs(
        self,
        spec: TestModelSpec,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, ...]:
        """Create meta device inputs for IR extraction.

        Args:
            spec: Model specification.
            batch_size: Batch size to prepend.

        Returns:
            Tuple of meta tensors.
        """
        inputs = []
        for shape in spec.input_shapes:
            if spec.name == "WeightTying":
                # Integer inputs for embedding - use regular tensor for tracing
                full_shape = (batch_size,) + shape
                tensor = torch.randint(0, 100, full_shape, device="meta")
            else:
                full_shape = (batch_size,) + shape
                tensor = torch.randn(full_shape, device="meta")
            inputs.append(tensor)
        return tuple(inputs)

    def run_single_test(self, spec: TestModelSpec) -> TestResult:
        """Run a single model test.

        Args:
            spec: Model specification to test.

        Returns:
            TestResult with all information.
        """
        start_time = time.time()
        model_name = spec.name

        try:
            # Create model with weights on CPU
            original_model = spec.model_class()
            original_model.eval()
            state_dict = original_model.state_dict()

            # Create model on meta device for IR extraction
            with torch.device("meta"):
                meta_model = spec.model_class()
            meta_model.eval()

            # Create inputs
            meta_inputs = self._create_meta_inputs(spec)
            test_inputs = self._create_test_inputs(spec)

            # Extract IR
            ir = extract_ir(meta_model, meta_inputs, model_name=model_name, strict=False)

            # Collect statistics
            statistics = IRStatistics.from_ir(ir)

            # Verify IR against original model
            is_valid, verification_report = verify_ir_with_state_dict(
                ir=ir,
                state_dict=state_dict,
                original_model=original_model,
                test_inputs=test_inputs,
                rtol=self.rtol,
                atol=self.atol,
            )

            duration = time.time() - start_time

            return TestResult(
                model_name=model_name,
                passed=is_valid,
                ir=ir,
                verification_report=verification_report,
                statistics=statistics,
                error_message=None if is_valid else verification_report.error_message,
                duration_seconds=duration,
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestResult(
                model_name=model_name,
                passed=False,
                ir=None,
                verification_report=None,
                statistics=None,
                error_message=str(e),
                duration_seconds=duration,
            )

    def run_all_tests(
        self,
        category_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> List[TestResult]:
        """Run all registered model tests.

        Args:
            category_filter: Only run models in this category.
            model_filter: Only run this specific model.

        Returns:
            List of TestResult objects.
        """
        results = []

        if model_filter:
            # Run single model
            spec = MODEL_REGISTRY.get(model_filter)
            if spec:
                result = self.run_single_test(spec)
                results.append(result)
        else:
            # Run all models (optionally filtered by category)
            models = list_models(category_filter)
            for spec in models:
                result = self.run_single_test(spec)
                results.append(result)

        return results

    def run_and_report(
        self,
        category_filter: Optional[str] = None,
        model_filter: Optional[str] = None,
    ) -> Tuple[List[TestResult], Path]:
        """Run all tests and generate reports.

        Args:
            category_filter: Only run models in this category.
            model_filter: Only run this specific model.

        Returns:
            Tuple of (results list, path to summary report).
        """
        results = self.run_all_tests(category_filter, model_filter)

        # Generate individual reports
        for result in results:
            self.report_generator.save_report(result)

        # Generate summary
        summary_path = self.report_generator.save_summary(results)

        return results, summary_path
