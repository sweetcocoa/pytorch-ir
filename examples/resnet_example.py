"""ResNet example for NPU IR extraction framework.

This example shows how to extract IR from a torchvision ResNet model.
Requires: pip install torchvision
"""

import torch

try:
    import torchvision
    from torchvision.models import resnet18, ResNet18_Weights

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from npu_ir import extract_ir, verify_ir_with_state_dict


def main():
    if not HAS_TORCHVISION:
        print("This example requires torchvision.")
        print("Install with: pip install torchvision")
        return

    print("=" * 60)
    print("NPU IR Extraction Framework - ResNet Example")
    print("=" * 60)

    # 1. Load pretrained ResNet18
    print("\n1. Loading pretrained ResNet18...")
    original_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    original_model.eval()
    print("   Loaded ResNet18 with ImageNet weights")

    # 2. Create meta model
    print("\n2. Creating meta ResNet18 for IR extraction...")
    with torch.device("meta"):
        meta_model = resnet18()
    print("   Meta model created")

    # 3. Example inputs
    print("\n3. Preparing example inputs...")
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    example_inputs = (torch.randn(*input_shape, device="meta"),)
    print(f"   Input shape: {input_shape}")

    # 4. Extract IR
    print("\n4. Extracting IR (this may take a moment)...")
    ir = extract_ir(meta_model, example_inputs, model_name="ResNet18")
    print(f"   Extracted IR:")
    print(f"     - Model name: {ir.model_name}")
    print(f"     - Number of nodes: {len(ir.nodes)}")
    print(f"     - Number of weights: {len(ir.weights)}")
    print(f"     - Graph input shape: {ir.graph_inputs[0].shape}")
    print(f"     - Graph output shape: {ir.graph_outputs[0].shape}")

    # 5. Analyze operation types
    print("\n5. Operation type distribution:")
    op_types = {}
    for node in ir.nodes:
        op_type = node.op_type.split(".")[-1] if "." in node.op_type else node.op_type
        op_types[op_type] = op_types.get(op_type, 0) + 1

    for op_type, count in sorted(op_types.items(), key=lambda x: -x[1])[:15]:
        print(f"   {op_type}: {count}")

    # 6. Verify IR
    print("\n6. Verifying IR execution against original model...")
    test_input = torch.randn(*input_shape)
    is_valid, report = verify_ir_with_state_dict(
        ir=ir,
        state_dict=original_model.state_dict(),
        original_model=original_model,
        test_inputs=(test_input,),
        rtol=1e-4,
        atol=1e-4,
    )
    print(f"   {report}")

    # 7. Save IR
    print("\n7. Saving IR to file...")
    ir_path = "resnet18_ir.json"
    ir.save(ir_path)
    print(f"   Saved to: {ir_path}")

    # 8. Calculate IR size
    import os

    ir_size = os.path.getsize(ir_path) / 1024
    print(f"   File size: {ir_size:.1f} KB")

    print("\n" + "=" * 60)
    print("ResNet example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
