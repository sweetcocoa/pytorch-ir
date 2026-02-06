"""Tests for the graph analyzer."""

import pytest
import torch
import torch.nn as nn

from npu_ir.analyzer import GraphAnalyzer, _dtype_to_str
from npu_ir.exporter import export_model


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestDtypeConversion:
    """Tests for dtype conversion."""

    def test_float32(self):
        assert _dtype_to_str(torch.float32) == "float32"

    def test_float16(self):
        assert _dtype_to_str(torch.float16) == "float16"

    def test_bfloat16(self):
        assert _dtype_to_str(torch.bfloat16) == "bfloat16"

    def test_int64(self):
        assert _dtype_to_str(torch.int64) == "int64"


class TestGraphAnalyzer:
    """Tests for GraphAnalyzer."""

    @pytest.fixture
    def simple_exported(self):
        """Create a simple exported model."""
        with torch.device("meta"):
            model = SimpleModel()
        inputs = (torch.randn(1, 10, device="meta"),)
        return export_model(model, inputs)

    def test_get_graph_inputs(self, simple_exported):
        """Test extracting graph inputs."""
        analyzer = GraphAnalyzer(simple_exported)
        inputs = analyzer.get_graph_inputs()

        assert len(inputs) >= 1
        # First input should be the user input
        user_input = inputs[0]
        assert user_input.shape == (1, 10)
        assert user_input.dtype == "float32"

    def test_get_graph_outputs(self, simple_exported):
        """Test extracting graph outputs."""
        analyzer = GraphAnalyzer(simple_exported)
        outputs = analyzer.get_graph_outputs()

        assert len(outputs) >= 1
        # Output should have shape (1, 5)
        assert outputs[0].shape == (1, 5)

    def test_get_weights(self, simple_exported):
        """Test extracting weight metadata."""
        analyzer = GraphAnalyzer(simple_exported)
        weights = analyzer.get_weights()

        # Should have weight and bias
        weight_names = [w.name for w in weights]
        assert any("weight" in name for name in weight_names)
        assert any("bias" in name for name in weight_names)

    def test_get_call_function_nodes(self, simple_exported):
        """Test extracting call_function nodes."""
        analyzer = GraphAnalyzer(simple_exported)
        nodes = analyzer.get_call_function_nodes()

        # Should have at least one operation node
        assert len(nodes) >= 1

        # Check that nodes have expected attributes
        for node in nodes:
            assert node.name is not None
            assert node.target is not None


class TestConvAnalyzer:
    """Tests for analyzing conv models."""

    @pytest.fixture
    def conv_exported(self):
        """Create a conv model exported."""

        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.relu(self.conv(x))

        with torch.device("meta"):
            model = ConvModel()
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)
        return export_model(model, inputs)

    def test_conv_shapes(self, conv_exported):
        """Test that conv shapes are extracted correctly."""
        analyzer = GraphAnalyzer(conv_exported)

        inputs = analyzer.get_graph_inputs()
        assert len(inputs) >= 1
        assert inputs[0].shape == (1, 3, 32, 32)

        outputs = analyzer.get_graph_outputs()
        assert len(outputs) >= 1
        assert outputs[0].shape == (1, 16, 32, 32)  # Same size due to padding=1
