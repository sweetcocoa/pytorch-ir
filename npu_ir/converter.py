"""IR Builder/Converter - converts ExportedProgram to NPU_IR."""

import torch
from torch.export import ExportedProgram

from .analyzer import GraphAnalyzer, NodeInfo
from .ir import NPU_IR, OpNode
from .ops.aten_ops import get_op_type
from .ops.registry import get_conversion_fn


class ConversionError(Exception):
    """Raised when IR conversion fails."""

    pass


def _validate_static_shapes(exported: ExportedProgram) -> None:
    """Validate that all tensor shapes are static (no SymInt dimensions).

    Raises:
        ConversionError: If dynamic shapes are detected.
    """
    for node in exported.graph_module.graph.nodes:
        if "val" not in node.meta:
            continue
        val = node.meta["val"]
        tensors = [val] if isinstance(val, torch.Tensor) else []
        if isinstance(val, (list, tuple)):
            tensors = [v for v in val if isinstance(v, torch.Tensor)]
        for tensor in tensors:
            for dim_size in tensor.shape:
                if not isinstance(dim_size, int):
                    raise ConversionError(
                        f"Dynamic shape detected in node '{node.name}': "
                        f"shape={tuple(tensor.shape)}. "
                        f"NPU IR requires static shapes. "
                        f"Do not pass dynamic_shapes to torch.export.export()."
                    )


def _default_conversion(node_info: NodeInfo) -> OpNode:
    """Default conversion for unsupported operations."""
    op_type = get_op_type(node_info.target)

    return OpNode(
        name=node_info.name,
        op_type=op_type,
        inputs=node_info.input_metas,
        outputs=node_info.output_metas,
        attrs=node_info.attrs,
    )


def convert_node(node_info: NodeInfo) -> OpNode:
    """Convert a single FX node to an OpNode.

    Args:
        node_info: The analyzed node information.

    Returns:
        The converted OpNode.
    """
    op_type = get_op_type(node_info.target)

    # Try to find a registered conversion function
    conversion_fn = get_conversion_fn(op_type)

    if conversion_fn is not None:
        return conversion_fn(node_info)
    else:
        # Use default conversion for unregistered ops
        return _default_conversion(node_info)


def convert_exported_program(
    exported: ExportedProgram,
    model_name: str = "",
    strict: bool = False,
) -> NPU_IR:
    """Convert an ExportedProgram to NPU_IR.

    Args:
        exported: The exported program from torch.export.
        model_name: Optional name for the model.
        strict: If True, raise error for unsupported ops. If False, use default conversion.

    Returns:
        The converted NPU_IR.

    Raises:
        ConversionError: If strict mode and unsupported operation encountered.
    """
    _validate_static_shapes(exported)

    analyzer = GraphAnalyzer(exported)

    # Extract graph metadata
    graph_inputs = analyzer.get_graph_inputs()
    graph_outputs = analyzer.get_graph_outputs()
    weights = analyzer.get_weights()
    weight_name_mapping = analyzer.get_weight_name_mapping()

    # Convert all call_function nodes
    nodes = []

    for node_info in analyzer.get_call_function_nodes():
        try:
            op_node = convert_node(node_info)
            nodes.append(op_node)
        except Exception as e:
            if strict:
                op_type = get_op_type(node_info.target)
                raise ConversionError(f"Failed to convert node '{node_info.name}' with op '{op_type}': {e}") from e
            else:
                # Use default conversion as fallback
                op_node = _default_conversion(node_info)
                nodes.append(op_node)

    return NPU_IR(
        nodes=nodes,
        graph_inputs=graph_inputs,
        graph_outputs=graph_outputs,
        weights=weights,
        weight_name_mapping=weight_name_mapping,
        model_name=model_name,
        pytorch_version=torch.__version__,
    )


class IRConverter:
    """Class-based interface for IR conversion with customization options."""

    def __init__(self, strict: bool = False):
        """Initialize the converter.

        Args:
            strict: If True, raise errors for unsupported operations.
        """
        self.strict = strict

    def convert(
        self,
        exported: ExportedProgram,
        model_name: str = "",
    ) -> NPU_IR:
        """Convert an ExportedProgram to NPU_IR.

        Args:
            exported: The exported program from torch.export.
            model_name: Optional name for the model.

        Returns:
            The converted NPU_IR.
        """
        return convert_exported_program(
            exported,
            model_name=model_name,
            strict=self.strict,
        )
