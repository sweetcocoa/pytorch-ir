"""Graph analyzer for extracting metadata from ExportedProgram."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.export import ExportedProgram
from torch.fx import Node

from .ir import TensorMeta


@dataclass
class NodeInfo:
    """Extracted information from an FX node.

    Attributes:
        name: Node name from the FX graph.
        op: FX node operation type (``"call_function"``, ``"placeholder"``, ``"output"``, ``"get_attr"``).
        target: The operation target (e.g., ``torch.ops.aten.conv2d.default``).
        args: Raw positional arguments from the FX node.
        kwargs: Raw keyword arguments from the FX node.
        input_metas: Metadata of input tensors with producer tracking information.
        output_metas: Metadata of output tensors inferred from ``node.meta["val"]``.
        attrs: Operation attributes extracted via schema introspection (e.g., ``kernel_size``, ``stride``).
    """

    name: str
    op: str
    target: Any
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    input_metas: List[TensorMeta]
    output_metas: List[TensorMeta]
    attrs: Dict[str, Any]


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


def _extract_tensor_meta(value: Any, name_prefix: str, index: int = 0) -> List[TensorMeta]:
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
    """Extract operation attributes from node args/kwargs using schema introspection."""
    import operator as _operator

    attrs = {}
    attrs.update(node.kwargs)

    target = node.target

    # Special case: operator.getitem — extract index from second arg
    if target is _operator.getitem:
        if len(node.args) >= 2 and not isinstance(node.args[1], Node):
            attrs["index"] = node.args[1]
        return attrs

    if not hasattr(target, "_schema"):
        return attrs

    schema = target._schema

    # Record sizes of Tensor[] and Tensor?[] args so executor can re-group flat tensor list
    tensor_list_sizes = []
    tensor_list_none_masks = []
    for i, schema_arg in enumerate(schema.arguments):
        if i >= len(node.args):
            break
        value = node.args[i]
        arg_type_str = str(schema_arg.type)
        is_tensor_list = (
            "Tensor[]" in arg_type_str
            or "Tensor?[]" in arg_type_str
            or "List[Tensor]" in arg_type_str
            or "List[Optional[Tensor]]" in arg_type_str
        )
        if is_tensor_list and isinstance(value, (list, tuple)):
            # Count only actual tensors (non-None entries)
            actual_tensor_count = sum(1 for v in value if v is not None)
            tensor_list_sizes.append(actual_tensor_count)
            # Record None positions for Tensor?[] reconstruction
            none_mask = [v is None for v in value]
            if any(none_mask):
                tensor_list_none_masks.append(none_mask)

    if tensor_list_sizes:
        attrs["_tensor_list_sizes"] = tensor_list_sizes
    if tensor_list_none_masks:
        attrs["_tensor_list_none_masks"] = tensor_list_none_masks

    for i, schema_arg in enumerate(schema.arguments):
        if i >= len(node.args):
            break
        value = node.args[i]
        if isinstance(value, Node):
            continue
        if isinstance(value, (list, tuple)) and any(isinstance(v, Node) for v in value):
            continue
        if value is None and "Tensor" in str(schema_arg.type):
            continue
        attrs[schema_arg.name] = value

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
        """Get input metadata for a node by looking up its input nodes.

        Each returned TensorMeta includes producer_node and producer_output_idx
        so the executor can resolve inputs via explicit producer references
        instead of name-based lookup.  Weight/buffer placeholders are left with
        producer_node=None so the executor falls back to registry lookup.
        """
        # Weight/buffer placeholders should NOT get producer tracking —
        # they are external inputs resolved from the registry, not computed
        # by a graph node.
        weight_placeholders = set()
        weight_placeholders.update(dict(self.graph_signature.inputs_to_parameters).keys())
        weight_placeholders.update(dict(self.graph_signature.inputs_to_buffers).keys())
        if hasattr(self.graph_signature, "inputs_to_lifted_tensor_constants"):
            weight_placeholders.update(dict(self.graph_signature.inputs_to_lifted_tensor_constants).keys())

        input_metas = []
        for arg in node.args:
            if isinstance(arg, Node):
                metas = self._node_outputs.get(arg.name, [])
                is_weight = arg.name in weight_placeholders
                for output_idx, meta in enumerate(metas):
                    input_metas.append(
                        TensorMeta(
                            name=meta.name,
                            shape=meta.shape,
                            dtype=meta.dtype,
                            producer_node=None if is_weight else arg.name,
                            producer_output_idx=0 if is_weight else output_idx,
                        )
                    )
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, Node):
                        metas = self._node_outputs.get(a.name, [])
                        is_weight = a.name in weight_placeholders
                        for output_idx, meta in enumerate(metas):
                            input_metas.append(
                                TensorMeta(
                                    name=meta.name,
                                    shape=meta.shape,
                                    dtype=meta.dtype,
                                    producer_node=None if is_weight else a.name,
                                    producer_output_idx=0 if is_weight else output_idx,
                                )
                            )
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
        constants_dict = (
            dict(self.graph_signature.inputs_to_lifted_tensor_constants)
            if hasattr(self.graph_signature, "inputs_to_lifted_tensor_constants")
            else {}
        )

        for node in self.graph.nodes:
            if node.op == "placeholder":
                param_name = params_dict.get(node.name)
                buffer_name = buffers_dict.get(node.name)
                constant_name = constants_dict.get(node.name)

                name = param_name or buffer_name or constant_name
                if name:
                    metas = self._node_outputs.get(node.name, [])
                    for meta in metas:
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
            Also includes lifted tensor constants.
        """
        mapping = {}

        params_dict = dict(self.graph_signature.inputs_to_parameters)
        buffers_dict = dict(self.graph_signature.inputs_to_buffers)
        constants_dict = (
            dict(self.graph_signature.inputs_to_lifted_tensor_constants)
            if hasattr(self.graph_signature, "inputs_to_lifted_tensor_constants")
            else {}
        )

        for node in self.graph.nodes:
            if node.op == "placeholder":
                param_name = params_dict.get(node.name)
                buffer_name = buffers_dict.get(node.name)
                constant_name = constants_dict.get(node.name)

                if param_name:
                    mapping[node.name] = param_name
                elif buffer_name:
                    mapping[node.name] = buffer_name
                elif constant_name:
                    mapping[node.name] = constant_name

        return mapping

    def get_call_function_nodes(self) -> List[NodeInfo]:
        """Extract all call_function nodes (actual operations)."""
        nodes = []

        for node in self.graph.nodes:
            if node.op == "call_function":
                target = node.target

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
