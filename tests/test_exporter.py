"""Tests for the model exporter."""

import pytest
import torch
import torch.nn as nn

from npu_ir.exporter import (
    export_model,
    is_meta_tensor,
    is_meta_module,
    validate_meta_device,
    validate_inputs_meta,
    ExportError,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestMetaDevice:
    """Tests for meta device validation."""

    def test_is_meta_tensor_true(self):
        """Test that meta tensors are detected."""
        t = torch.randn(2, 3, device="meta")
        assert is_meta_tensor(t)

    def test_is_meta_tensor_false(self):
        """Test that CPU tensors are not meta."""
        t = torch.randn(2, 3)
        assert not is_meta_tensor(t)

    def test_is_meta_module_true(self):
        """Test that meta modules are detected."""
        with torch.device("meta"):
            model = SimpleModel()
        assert is_meta_module(model)

    def test_is_meta_module_false(self):
        """Test that CPU modules are not meta."""
        model = SimpleModel()
        assert not is_meta_module(model)

    def test_validate_meta_device_raises(self):
        """Test that validation raises for non-meta models."""
        model = SimpleModel()
        with pytest.raises(ExportError, match="must be on meta device"):
            validate_meta_device(model)

    def test_validate_inputs_meta_raises(self):
        """Test that validation raises for non-meta inputs."""
        inputs = (torch.randn(1, 10),)
        with pytest.raises(ExportError, match="must be on meta device"):
            validate_inputs_meta(inputs)


class TestExportModel:
    """Tests for model export."""

    def test_export_simple_model(self):
        """Test exporting a simple model on meta device."""
        with torch.device("meta"):
            model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)

        exported = export_model(model, inputs, strict=True)

        assert exported is not None
        assert hasattr(exported, "graph_module")
        assert hasattr(exported, "graph_signature")

    def test_export_non_strict_mode(self):
        """Test that non-strict mode allows CPU tensors."""
        model = SimpleModel()
        inputs = (torch.randn(1, 10),)

        exported = export_model(model, inputs, strict=False)

        assert exported is not None

    def test_export_raises_for_cpu_model_strict(self):
        """Test that strict mode raises for CPU model."""
        model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)

        with pytest.raises(ExportError, match="must be on meta device"):
            export_model(model, inputs, strict=True)


class TestConvModel:
    """Tests for conv model export."""

    def test_export_conv2d(self):
        """Test exporting a conv2d model."""

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)

            def forward(self, x):
                return self.conv(x)

        with torch.device("meta"):
            model = ConvModel()
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)

        exported = export_model(model, inputs)

        assert exported is not None
