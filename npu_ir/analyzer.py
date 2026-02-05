"""Graph analyzer for extracting metadata from ExportedProgram."""

import torch
from torch.export import ExportedProgram
from torch.fx import Node, Graph
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass

from .ir import TensorMeta


@dataclass
class NodeInfo:
    """Extracted information from an FX node."""

    name: str
    op: str  # "call_function", "placeholder", "output", "get_attr"
    target: Any  # The operation (e.g., torch.ops.aten.conv2d.default)
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    input_metas: List[TensorMeta]  # Metadata of input tensors
    output_metas: List[TensorMeta]  # Metadata of output tensors
    attrs: Dict[str, Any]  # Operation attributes (kernel_size, stride, etc.)


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert torch dtype to string representation."""
    dtype_map = {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.bool: "bool",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
    }
    return dtype_map.get(dtype, str(dtype).replace("torch.", ""))


def _extract_tensor_meta(
    value: Any, name_prefix: str, index: int = 0
) -> List[TensorMeta]:
    """Extract TensorMeta from a value (tensor or tuple of tensors)."""
    metas = []

    if value is None:
        return metas

    if isinstance(value, (torch.Tensor, torch._subclasses.FakeTensor)):
        # Handle single tensor
        name = f"{name_prefix}_{index}" if index > 0 else name_prefix
        metas.append(
            TensorMeta(
                name=name,
                shape=tuple(value.shape),
                dtype=_dtype_to_str(value.dtype),
            )
        )
    elif isinstance(value, (list, tuple)):
        # Handle tuple/list of tensors
        for i, v in enumerate(value):
            if isinstance(v, (torch.Tensor, torch._subclasses.FakeTensor)):
                metas.append(
                    TensorMeta(
                        name=f"{name_prefix}_{i}",
                        shape=tuple(v.shape),
                        dtype=_dtype_to_str(v.dtype),
                    )
                )

    return metas


def _extract_node_output_meta(node: Node) -> List[TensorMeta]:
    """Extract output tensor metadata from node.meta['val']."""
    if "val" not in node.meta:
        return []

    val = node.meta["val"]
    return _extract_tensor_meta(val, node.name)


def _get_input_names(node: Node) -> List[str]:
    """Get input tensor names from node args."""
    input_names = []
    for arg in node.args:
        if isinstance(arg, Node):
            input_names.append(arg.name)
        elif isinstance(arg, (list, tuple)):
            for a in arg:
                if isinstance(a, Node):
                    input_names.append(a.name)
    return input_names


def _extract_op_attrs(node: Node) -> Dict[str, Any]:
    """Extract operation attributes from node args/kwargs."""
    attrs = {}

    # Copy kwargs directly
    attrs.update(node.kwargs)

    # For specific ops, extract positional args as named attributes
    target_str = str(node.target) if node.target else ""

    # Common patterns for extracting attributes from positional args
    # This is operation-specific
    if "conv" in target_str.lower():
        # conv2d: input, weight, bias, stride, padding, dilation, groups
        if len(node.args) > 3 and not isinstance(node.args[3], Node):
            attrs["stride"] = node.args[3]
        if len(node.args) > 4 and not isinstance(node.args[4], Node):
            attrs["padding"] = node.args[4]
        if len(node.args) > 5 and not isinstance(node.args[5], Node):
            attrs["dilation"] = node.args[5]
        if len(node.args) > 6 and not isinstance(node.args[6], Node):
            attrs["groups"] = node.args[6]

    elif "pool" in target_str.lower():
        # Pooling ops: kernel_size, stride, padding
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["kernel_size"] = node.args[1]
        if len(node.args) > 2 and not isinstance(node.args[2], Node):
            attrs["stride"] = node.args[2]
        if len(node.args) > 3 and not isinstance(node.args[3], Node):
            attrs["padding"] = node.args[3]

    elif "linear" in target_str.lower() or "addmm" in target_str.lower():
        # Linear: input, weight, bias
        pass  # No special attrs needed

    elif "batch_norm" in target_str.lower():
        if len(node.args) > 5 and not isinstance(node.args[5], Node):
            attrs["training"] = node.args[5]
        if len(node.args) > 6 and not isinstance(node.args[6], Node):
            attrs["momentum"] = node.args[6]
        if len(node.args) > 7 and not isinstance(node.args[7], Node):
            attrs["eps"] = node.args[7]

    elif "layer_norm" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["normalized_shape"] = node.args[1]

    elif "softmax" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["dim"] = node.args[1]

    elif "view" in target_str.lower() or "reshape" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["shape"] = node.args[1]

    elif "permute" in target_str.lower() or "transpose" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["dims"] = node.args[1]

    elif "split" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["split_size"] = node.args[1]
        if len(node.args) > 2 and not isinstance(node.args[2], Node):
            attrs["dim"] = node.args[2]

    elif "cat" in target_str.lower():
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["dim"] = node.args[1]

    elif "flatten" in target_str.lower():
        # flatten: input, start_dim, end_dim
        if len(node.args) > 1 and not isinstance(node.args[1], Node):
            attrs["start_dim"] = node.args[1]
        if len(node.args) > 2 and not isinstance(node.args[2], Node):
            attrs["end_dim"] = node.args[2]

    return attrs


class GraphAnalyzer:
    """Analyzes ExportedProgram to extract metadata."""

    def __init__(self, exported: ExportedProgram):
        self.exported = exported
        self.graph_module = exported.graph_module
        self.graph = exported.graph_module.graph
        self.graph_signature = exported.graph_signature

        # Cache for node output metadata
        self._node_outputs: Dict[str, List[TensorMeta]] = {}
        self._build_node_output_cache()

    def _build_node_output_cache(self) -> None:
        """Build cache of node output metadata."""
        for node in self.graph.nodes:
            metas = _extract_node_output_meta(node)
            self._node_outputs[node.name] = metas

    def get_node_output_meta(self, node_name: str) -> List[TensorMeta]:
        """Get output metadata for a node by name."""
        return self._node_outputs.get(node_name, [])

    def get_node_input_meta(self, node: Node) -> List[TensorMeta]:
        """Get input metadata for a node by looking up its input nodes."""
        input_metas = []
        for arg in node.args:
            if isinstance(arg, Node):
                metas = self._node_outputs.get(arg.name, [])
                input_metas.extend(metas)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, Node):
                        metas = self._node_outputs.get(a.name, [])
                        input_metas.extend(metas)
        return input_metas

    def get_graph_inputs(self) -> List[TensorMeta]:
        """Extract graph input tensor metadata."""
        inputs = []
        user_inputs = self.graph_signature.user_inputs

        for node in self.graph.nodes:
            if node.op == "placeholder" and node.name in user_inputs:
                metas = self._node_outputs.get(node.name, [])
                inputs.extend(metas)

        return inputs

    def get_graph_outputs(self) -> List[TensorMeta]:
        """Extract graph output tensor metadata."""
        outputs = []
        user_outputs = self.graph_signature.user_outputs

        for node in self.graph.nodes:
            if node.op == "output":
                # Output node's args contain the actual output nodes
                for arg in node.args[0] if isinstance(node.args[0], (list, tuple)) else [node.args[0]]:
                    if isinstance(arg, Node) and arg.name in user_outputs:
                        metas = self._node_outputs.get(arg.name, [])
                        outputs.extend(metas)

        return outputs

    def get_weights(self) -> List[TensorMeta]:
        """Extract weight tensor metadata from graph signature."""
        weights = []

        # Get parameter names from signature
        params_dict = dict(self.graph_signature.inputs_to_parameters)
        buffers_dict = dict(self.graph_signature.inputs_to_buffers)

        for node in self.graph.nodes:
            if node.op == "placeholder":
                param_name = params_dict.get(node.name)
                buffer_name = buffers_dict.get(node.name)

                if param_name or buffer_name:
                    name = param_name or buffer_name
                    metas = self._node_outputs.get(node.name, [])
                    for meta in metas:
                        # Use the actual parameter name
                        weights.append(
                            TensorMeta(
                                name=name,
                                shape=meta.shape,
                                dtype=meta.dtype,
                            )
                        )

        return weights

    def get_weight_name_mapping(self) -> Dict[str, str]:
        """Get mapping from placeholder names to state_dict keys.

        Returns:
            Dict mapping placeholder names (e.g., 'p_linear_weight') to
            state_dict keys (e.g., 'linear.weight').
        """
        mapping = {}

        params_dict = dict(self.graph_signature.inputs_to_parameters)
        buffers_dict = dict(self.graph_signature.inputs_to_buffers)

        for node in self.graph.nodes:
            if node.op == "placeholder":
                param_name = params_dict.get(node.name)
                buffer_name = buffers_dict.get(node.name)

                if param_name:
                    mapping[node.name] = param_name
                elif buffer_name:
                    mapping[node.name] = buffer_name

        return mapping

    def get_call_function_nodes(self) -> List[NodeInfo]:
        """Extract all call_function nodes (actual operations)."""
        nodes = []

        for node in self.graph.nodes:
            if node.op == "call_function":
                target = node.target
                target_str = str(target)

                # Skip trivial operations if needed
                if target_str.startswith("operator."):
                    continue

                input_metas = self.get_node_input_meta(node)
                output_metas = self._node_outputs.get(node.name, [])
                attrs = _extract_op_attrs(node)

                nodes.append(
                    NodeInfo(
                        name=node.name,
                        op=node.op,
                        target=target,
                        args=node.args,
                        kwargs=dict(node.kwargs),
                        input_metas=input_metas,
                        output_metas=output_metas,
                        attrs=attrs,
                    )
                )

        return nodes

    def get_op_target_str(self, node: Node) -> str:
        """Get string representation of operation target."""
        target = node.target
        if hasattr(target, "__module__") and hasattr(target, "__name__"):
            return f"{target.__module__}.{target.__name__}"
        return str(target)
