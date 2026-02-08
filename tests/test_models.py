"""Integration tests with real models."""

import torch
import torch.nn as nn

from torch_ir import extract_ir, verify_ir_with_state_dict


class TestSimpleModels:
    """Tests with simple custom models."""

    def test_mlp_model(self):
        """Test with a multi-layer perceptron."""

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 256)
                self.fc2 = nn.Linear(256, 128)
                self.fc3 = nn.Linear(128, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)

        # Create and test
        model = MLP()
        model.eval()

        with torch.device("meta"):
            meta_model = MLP()
        inputs = (torch.randn(1, 784, device="meta"),)

        ir = extract_ir(meta_model, inputs, strict=False)

        assert len(ir.nodes) >= 3  # At least 3 linear ops
        assert ir.graph_inputs[0].shape == (1, 784)
        assert ir.graph_outputs[0].shape == (1, 10)

        # Verify
        test_inputs = (torch.randn(1, 784),)
        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
        )
        assert is_valid, f"Verification failed: {report}"

    def test_conv_model(self):
        """Test with a convolutional model."""

        class ConvNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.pool = nn.MaxPool2d(2)
                self.fc = nn.Linear(64 * 8 * 8, 10)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = x.flatten(1)
                return self.fc(x)

        model = ConvNet()
        model.eval()

        with torch.device("meta"):
            meta_model = ConvNet()
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)

        ir = extract_ir(meta_model, inputs, strict=False)

        assert ir.graph_inputs[0].shape == (1, 3, 32, 32)
        assert ir.graph_outputs[0].shape == (1, 10)

        # Verify
        test_inputs = (torch.randn(1, 3, 32, 32),)
        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
        )
        assert is_valid, f"Verification failed: {report}"


class TestBatchNorm:
    """Tests with batch normalization."""

    def test_batchnorm_model(self):
        """Test model with batch normalization."""

        class BNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.bn = nn.BatchNorm2d(16)
                self.relu = nn.ReLU()
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.relu(self.bn(self.conv(x)))
                x = self.pool(x)
                x = x.flatten(1)
                return self.fc(x)

        model = BNModel()
        model.eval()

        with torch.device("meta"):
            meta_model = BNModel()
        meta_model.eval()  # Important: must match eval mode of original model
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)

        ir = extract_ir(meta_model, inputs, strict=False)

        # Batch norm might be decomposed in export
        assert len(ir.nodes) >= 3

        # Verify
        test_inputs = (torch.randn(1, 3, 32, 32),)
        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
            rtol=1e-4,
            atol=1e-4,
        )
        assert is_valid, f"Verification failed: {report}"


class TestResidual:
    """Tests with residual connections."""

    def test_residual_block(self):
        """Test model with residual connections."""

        class ResBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.relu = nn.ReLU()

            def forward(self, x):
                residual = x
                x = self.relu(self.conv1(x))
                x = self.conv2(x)
                return self.relu(x + residual)

        class ResModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.block = ResBlock(16)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 10)

            def forward(self, x):
                x = self.conv(x)
                x = self.block(x)
                x = self.pool(x)
                x = x.flatten(1)
                return self.fc(x)

        model = ResModel()
        model.eval()

        with torch.device("meta"):
            meta_model = ResModel()
        inputs = (torch.randn(1, 3, 32, 32, device="meta"),)

        ir = extract_ir(meta_model, inputs, strict=False)

        # Verify
        test_inputs = (torch.randn(1, 3, 32, 32),)
        is_valid, report = verify_ir_with_state_dict(
            ir=ir,
            state_dict=model.state_dict(),
            original_model=model,
            test_inputs=test_inputs,
        )
        assert is_valid, f"Verification failed: {report}"


class TestSerializationRoundTrip:
    """Tests for serialization round-trip."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading IR."""

        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x):
                return self.fc(x)

        with torch.device("meta"):
            model = SimpleNet()
        inputs = (torch.randn(1, 10, device="meta"),)

        ir = extract_ir(model, inputs, strict=False)

        # Save
        ir_path = tmp_path / "test_ir.json"
        ir.save(str(ir_path))

        # Load
        from torch_ir import load_ir

        loaded_ir = load_ir(str(ir_path))

        # Compare
        assert loaded_ir.model_name == ir.model_name
        assert len(loaded_ir.nodes) == len(ir.nodes)
        assert len(loaded_ir.weights) == len(ir.weights)
        assert loaded_ir.graph_inputs[0].shape == ir.graph_inputs[0].shape
