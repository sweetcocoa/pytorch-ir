"""Tests for the IR converter."""

import pytest
import torch
import torch.nn as nn

from torch_ir.converter import convert_exported_program
from torch_ir.exporter import export_model


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestConvertExportedProgram:
    """Tests for convert_exported_program."""

    @pytest.fixture
    def simple_exported(self):
        """Create a simple exported model."""
        with torch.device("meta"):
            model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        return export_model(model, inputs)

    def test_basic_conversion(self, simple_exported):
        """Test basic IR conversion."""
        ir = convert_exported_program(simple_exported, model_name="SimpleModel")

        assert ir is not None
        assert ir.model_name == "SimpleModel"
        assert len(ir.nodes) >= 1
        assert len(ir.graph_inputs) >= 1
        assert len(ir.graph_outputs) >= 1
        assert len(ir.weights) >= 1

    def test_node_structure(self, simple_exported):
        """Test that converted nodes have correct structure."""
        ir = convert_exported_program(simple_exported)

        for node in ir.nodes:
            assert node.name is not None
            assert node.op_type is not None
            assert isinstance(node.inputs, list)
            assert isinstance(node.outputs, list)
            assert isinstance(node.attrs, dict)

    def test_tensor_meta_structure(self, simple_exported):
        """Test that tensor metadata has correct structure."""
        ir = convert_exported_program(simple_exported)

        for tensor_meta in ir.graph_inputs + ir.graph_outputs + ir.weights:
            assert tensor_meta.name is not None
            assert isinstance(tensor_meta.shape, tuple)
            assert tensor_meta.dtype is not None


class TestConvertWithOptions:
    """Tests for convert_exported_program with different options."""

    def test_convert_with_strict_false(self):
        """Test convert with strict=False."""
        with torch.device("meta"):
            model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        exported = export_model(model, inputs)

        ir = convert_exported_program(exported, model_name="TestModel", strict=False)

        assert ir.model_name == "TestModel"


class TestConvConversion:
    """Tests for conv model conversion."""

    def test_conv2d_conversion(self):
        """Test converting conv2d model."""

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, stride=2, padding=1)

            def forward(self, x):
                return self.conv(x)

        with torch.device("meta"):
            model = ConvModel()
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)
        exported = export_model(model, inputs)

        ir = convert_exported_program(exported)

        # Check that we have conv-related operations
        op_types = [n.op_type for n in ir.nodes]
        assert any("conv" in op.lower() for op in op_types)


class TestComplexModels:
    """Tests for more complex models."""

    def test_sequential_model(self):
        """Test converting a sequential model."""
        with torch.device("meta"):
            model = nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5),
            )
        inputs = (torch.randn(1, 10, device="meta"),)
        exported = export_model(model, inputs)

        ir = convert_exported_program(exported)

        assert len(ir.nodes) >= 2  # At least 2 linear ops
        assert ir.graph_inputs[0].shape == (1, 10)
        assert ir.graph_outputs[0].shape == (1, 5)
