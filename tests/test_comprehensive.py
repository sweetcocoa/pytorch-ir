"""Comprehensive parameterized tests for all registered models."""

import pytest
import torch

from npu_ir import extract_ir, verify_ir_with_state_dict

from .models import MODEL_REGISTRY, list_models, TestModelSpec
from .testing import IRStatistics, TestResult


def get_model_ids():
    """Get list of model IDs for parameterization."""
    return list(MODEL_REGISTRY.keys())


def create_test_inputs(spec: TestModelSpec, batch_size: int = 1):
    """Create test inputs for a model specification."""
    inputs = []
    for shape in spec.input_shapes:
        if spec.name == "WeightTying":
            # Integer inputs for embedding
            full_shape = (batch_size,) + shape
            tensor = torch.randint(0, 100, full_shape)
        else:
            full_shape = (batch_size,) + shape
            tensor = torch.randn(full_shape)
        inputs.append(tensor)
    return tuple(inputs)


def create_meta_inputs(spec: TestModelSpec, batch_size: int = 1):
    """Create meta device inputs for IR extraction."""
    inputs = []
    for shape in spec.input_shapes:
        if spec.name == "WeightTying":
            full_shape = (batch_size,) + shape
            tensor = torch.randint(0, 100, full_shape, device="meta")
        else:
            full_shape = (batch_size,) + shape
            tensor = torch.randn(full_shape, device="meta")
        inputs.append(tensor)
    return tuple(inputs)


class TestComprehensiveModels:
    """Parameterized tests for all registered models."""

    @pytest.mark.parametrize("model_name", get_model_ids())
    def test_ir_extraction(self, model_name: str):
        """Test that IR can be extracted from the model."""
        spec = MODEL_REGISTRY[model_name]

        # Create model on meta device
        with torch.device("meta"):
            model = spec.model_class()

        # Create meta inputs
        meta_inputs = create_meta_inputs(spec)

        # Extract IR
        ir = extract_ir(model, meta_inputs, model_name=model_name, strict=False)

        # Basic assertions
        assert ir is not None
        assert len(ir.nodes) > 0
        assert len(ir.graph_inputs) == len(spec.input_shapes)
        assert ir.model_name == model_name

    @pytest.mark.parametrize("model_name", get_model_ids())
    def test_ir_verification(
        self,
        model_name: str,
        report_collector,
        generate_reports,
        rtol,
        atol,
    ):
        """Test that IR execution matches original model output."""
        import time

        spec = MODEL_REGISTRY[model_name]
        start_time = time.time()

        try:
            # Create original model with weights
            original_model = spec.model_class()
            original_model.eval()
            state_dict = original_model.state_dict()

            # Create model on meta device for IR extraction
            with torch.device("meta"):
                meta_model = spec.model_class()
            meta_model.eval()

            # Create inputs
            meta_inputs = create_meta_inputs(spec)
            test_inputs = create_test_inputs(spec)

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
                rtol=rtol,
                atol=atol,
            )

            duration = time.time() - start_time

            # Create result for reporting
            result = TestResult(
                model_name=model_name,
                passed=is_valid,
                ir=ir,
                verification_report=verification_report,
                statistics=statistics,
                error_message=None if is_valid else verification_report.error_message,
                duration_seconds=duration,
            )

            # Add to collector if reporting is enabled
            if generate_reports:
                report_collector.add_result(result)

            # Assert verification passed
            assert is_valid, f"Verification failed: {verification_report.error_message}"

        except Exception as e:
            duration = time.time() - start_time

            # Create failed result for reporting
            result = TestResult(
                model_name=model_name,
                passed=False,
                ir=None,
                verification_report=None,
                statistics=None,
                error_message=str(e),
                duration_seconds=duration,
            )

            if generate_reports:
                report_collector.add_result(result)

            raise

    @pytest.mark.parametrize("model_name", get_model_ids())
    def test_ir_statistics(self, model_name: str):
        """Test that IR statistics can be collected."""
        spec = MODEL_REGISTRY[model_name]

        # Create model on meta device
        with torch.device("meta"):
            model = spec.model_class()

        # Create meta inputs
        meta_inputs = create_meta_inputs(spec)

        # Extract IR
        ir = extract_ir(model, meta_inputs, model_name=model_name, strict=False)

        # Collect statistics
        stats = IRStatistics.from_ir(ir)

        # Verify statistics
        assert stats.num_nodes == len(ir.nodes)
        assert stats.num_inputs == len(ir.graph_inputs)
        assert stats.num_outputs == len(ir.graph_outputs)
        assert stats.num_weights == len(ir.weights)
        assert len(stats.op_distribution) > 0
        assert len(stats.node_shapes) == stats.num_nodes


class TestMultiInputOutput:
    """Tests specific to multi-input/output models."""

    def test_siamese_encoder_two_outputs(self):
        """Test that SiameseEncoder produces two outputs."""
        from .models.multi_io import SiameseEncoder

        model = SiameseEncoder()
        model.eval()

        x1 = torch.randn(1, 3, 64, 64)
        x2 = torch.randn(1, 3, 64, 64)

        with torch.no_grad():
            out1, out2 = model(x1, x2)

        assert out1.shape == out2.shape
        assert out1.shape == (1, 128)

    def test_multitask_head_two_outputs(self):
        """Test that MultiTaskHead produces two outputs."""
        from .models.multi_io import MultiTaskHead

        model = MultiTaskHead(num_classes=10, aux_dim=4)
        model.eval()

        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            cls_out, aux_out = model(x)

        assert cls_out.shape == (1, 10)
        assert aux_out.shape == (1, 4)


class TestSkipConnections:
    """Tests specific to skip connection models."""

    def test_deep_resnet_residual(self):
        """Test that DeepResNet has residual connections."""
        from .models.skip_connections import DeepResNet

        model = DeepResNet()
        model.eval()

        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 10)

    def test_dense_block_concatenation(self):
        """Test that DenseBlock uses concatenation."""
        from .models.skip_connections import DenseBlock

        model = DenseBlock()
        model.eval()

        x = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            out = model(x)

        assert out.shape == (1, 10)


class TestSharedWeights:
    """Tests specific to shared weight models."""

    def test_recurrent_unroll_weight_sharing(self):
        """Test that RecurrentUnroll reuses the same weights."""
        from .models.shared_weights import RecurrentUnroll

        model = RecurrentUnroll(hidden_dim=32, num_steps=5)

        # Only one Linear layer should exist
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 1

    def test_weight_tying(self):
        """Test that WeightTying shares embedding weights."""
        from .models.shared_weights import WeightTying

        model = WeightTying(vocab_size=100, embed_dim=64)

        # Check embedding exists
        assert hasattr(model, "embedding")
        assert model.embedding.weight.shape == (100, 64)


class TestAttention:
    """Tests specific to attention models."""

    def test_self_attention_shape(self):
        """Test self-attention output shape."""
        from .models.attention import SelfAttention

        model = SelfAttention(d_model=64, num_heads=4)
        model.eval()

        x = torch.randn(1, 16, 64)

        with torch.no_grad():
            out = model(x)

        assert out.shape == x.shape

    def test_cross_attention_shape(self):
        """Test cross-attention output shape."""
        from .models.attention import CrossAttention

        model = CrossAttention(d_model=64, num_heads=4)
        model.eval()

        query = torch.randn(1, 16, 64)
        context = torch.randn(1, 32, 64)

        with torch.no_grad():
            out = model(query, context)

        assert out.shape == query.shape

    def test_transformer_block_shape(self):
        """Test transformer block output shape."""
        from .models.attention import TransformerBlock

        model = TransformerBlock(d_model=64, num_heads=4)
        model.eval()

        x = torch.randn(1, 16, 64)

        with torch.no_grad():
            out = model(x)

        assert out.shape == x.shape
