"""Basic usage example for NPU IR extraction framework."""

import torch
import torch.nn as nn

from npu_ir import extract_ir, verify_ir_with_state_dict


# Define a simple model
class SimpleConvNet(nn.Module):
    """A simple CNN for demonstration."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    print("=" * 60)
    print("NPU IR Extraction Framework - Basic Usage Example")
    print("=" * 60)

    # 1. Create original model with weights (for verification)
    print("\n1. Creating original model with weights...")
    original_model = SimpleConvNet(num_classes=10)
    original_model.eval()
    print(f"   Model: {original_model.__class__.__name__}")

    # 2. Create model on meta device for IR extraction
    print("\n2. Creating meta model for IR extraction...")
    with torch.device("meta"):
        meta_model = SimpleConvNet(num_classes=10)
    print("   Meta model created (no actual weights loaded)")

    # 3. Create example inputs on meta device
    print("\n3. Preparing example inputs...")
    batch_size = 1
    input_shape = (batch_size, 3, 32, 32)
    example_inputs = (torch.randn(*input_shape, device="meta"),)
    print(f"   Input shape: {input_shape}")

    # 4. Extract IR
    print("\n4. Extracting IR from meta model...")
    ir = extract_ir(meta_model, example_inputs, model_name="SimpleConvNet")
    print(f"   Extracted IR:")
    print(f"     - Model name: {ir.model_name}")
    print(f"     - PyTorch version: {ir.pytorch_version}")
    print(f"     - Number of nodes: {len(ir.nodes)}")
    print(f"     - Number of weights: {len(ir.weights)}")
    print(f"     - Graph inputs: {len(ir.graph_inputs)}")
    print(f"     - Graph outputs: {len(ir.graph_outputs)}")

    # 5. Inspect some nodes
    print("\n5. First 10 operations:")
    for i, node in enumerate(ir.nodes[:10]):
        in_shapes = [t.shape for t in node.inputs]
        out_shapes = [t.shape for t in node.outputs]
        print(f"   [{i}] {node.op_type}")
        print(f"       inputs: {in_shapes}")
        print(f"       outputs: {out_shapes}")

    # 6. Inspect weights metadata
    print("\n6. Weight metadata (first 5):")
    for weight in ir.weights[:5]:
        print(f"   - {weight.name}: shape={weight.shape}, dtype={weight.dtype}")

    # 7. Save IR to file
    print("\n7. Saving IR to file...")
    ir_path = "simple_convnet_ir.json"
    ir.save(ir_path)
    print(f"   Saved to: {ir_path}")

    # 8. Verify IR against original model
    print("\n8. Verifying IR execution matches original model...")
    test_input = torch.randn(*input_shape)
    is_valid, report = verify_ir_with_state_dict(
        ir=ir,
        state_dict=original_model.state_dict(),
        original_model=original_model,
        test_inputs=(test_input,),
        rtol=1e-5,
        atol=1e-5,
    )
    print(f"   {report}")

    # 9. Load IR from file
    print("\n9. Loading IR from file...")
    from npu_ir import load_ir

    loaded_ir = load_ir(ir_path)
    print(f"   Loaded IR: {loaded_ir}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
