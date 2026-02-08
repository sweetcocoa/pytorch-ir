"""Tests for the IR executor."""

import pytest
import torch
import torch.nn as nn

from torch_ir import extract_ir
from torch_ir.executor import ExecutionError, IRExecutor, TensorRegistry, execute_ir


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestTensorRegistry:
    """Tests for TensorRegistry."""

    def test_register_and_get(self):
        """Test registering and getting tensors."""
        registry = TensorRegistry()
        t = torch.randn(2, 3)

        registry.register("test", t)

        assert registry.has("test")
        result = registry.get("test")
        assert result is not None
        assert torch.equal(result, t)

    def test_getitem(self):
        """Test __getitem__ access."""
        registry = TensorRegistry()
        t = torch.randn(2, 3)
        registry["test"] = t

        assert torch.equal(registry["test"], t)

    def test_contains(self):
        """Test __contains__."""
        registry = TensorRegistry()
        registry.register("test", torch.randn(1))

        assert "test" in registry
        assert "nonexistent" not in registry

    def test_clear(self):
        """Test clearing registry."""
        registry = TensorRegistry()
        registry.register("test", torch.randn(1))

        registry.clear()

        assert not registry.has("test")


class TestIRExecutor:
    """Tests for IRExecutor."""

    @pytest.fixture
    def simple_model_and_ir(self):
        """Create a simple model and its IR."""
        model = SimpleModel()
        model.eval()

        # Extract IR from meta model
        with torch.device("meta"):
            meta_model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(meta_model, inputs, strict=False)

        return model, ir

    def test_executor_without_weights_raises(self, simple_model_and_ir):
        """Test that executing without weights raises error."""
        model, ir = simple_model_and_ir

        executor = IRExecutor(ir)
        inputs = (torch.randn(1, 10),)

        with pytest.raises(ExecutionError, match="Weights not loaded"):
            executor.execute(inputs)

    def test_executor_with_state_dict(self, simple_model_and_ir):
        """Test executing with state dict."""
        model, ir = simple_model_and_ir

        executor = IRExecutor(ir)
        executor.load_weights_from_state_dict(model.state_dict())

        inputs = (torch.randn(1, 10),)
        outputs = executor.execute(inputs)

        assert len(outputs) >= 1
        assert outputs[0].shape == (1, 5)

    def test_executor_call(self, simple_model_and_ir):
        """Test calling executor directly."""
        model, ir = simple_model_and_ir

        executor = IRExecutor(ir, weights=model.state_dict())

        inputs = torch.randn(1, 10)
        outputs = executor(inputs)

        assert len(outputs) >= 1


class TestExecuteIR:
    """Tests for execute_ir function."""

    def test_execute_ir_with_weights(self):
        """Test execute_ir with weights dict."""
        model = SimpleModel()
        model.eval()

        with torch.device("meta"):
            meta_model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(meta_model, inputs, strict=False)

        test_input = (torch.randn(1, 10),)
        outputs = execute_ir(ir, test_input, weights=model.state_dict())

        assert len(outputs) >= 1
        assert outputs[0].shape == (1, 5)

    def test_execute_ir_missing_args(self):
        """Test that missing weights raises ValueError."""
        with torch.device("meta"):
            model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        ir = extract_ir(model, inputs, strict=False)

        with pytest.raises(ValueError, match="Either weights or weights_path"):
            execute_ir(ir, inputs)
