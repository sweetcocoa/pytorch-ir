"""Graph analyzer for extracting metadata from ExportedProgram."""

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.export import ExportedProgram
from torch.fx import Node

from .ir import TensorMeta

_SCHEMA_ARG_SPEC_CACHE: dict[int, list[tuple[str, bool, bool]]] = {}


@dataclass
class NodeInfo:
    """Extracted information from an FX node.

    Attributes:
        name: Node name from the FX graph.
        target: The operation target (e.g., ``torch.ops.aten.conv2d.default``).
        input_metas: Metadata of input tensors with producer tracking information.
        output_metas: Metadata of output tensors inferred from ``node.meta["val"]``.
        attrs: Operation attributes extracted via schema introspection (e.g., ``kernel_size``, ``stride``).
    """

    name: str
    target: Any
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


def _get_schema_arg_specs(target: Any) -> list[tuple[str, bool, bool]]:
    """Cache schema-derived argument metadata per operator target."""
    cache_key = id(target)
    cached = _SCHEMA_ARG_SPEC_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not hasattr(target, "_schema"):
        cached = []
        _SCHEMA_ARG_SPEC_CACHE[cache_key] = cached
        return cached

    specs = []
    for schema_arg in target._schema.arguments:
        arg_type_str = str(schema_arg.type)
        is_tensor_list = (
            "Tensor[]" in arg_type_str
            or "Tensor?[]" in arg_type_str
            or "List[Tensor]" in arg_type_str
            or "List[Optional[Tensor]]" in arg_type_str
        )
        specs.append((schema_arg.name, is_tensor_list, "Tensor" in arg_type_str))

    _SCHEMA_ARG_SPEC_CACHE[cache_key] = specs
    return specs


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

    schema_arg_specs = _get_schema_arg_specs(target)

    # Record sizes of Tensor[] and Tensor?[] args so executor can re-group flat tensor list
    tensor_list_sizes = []
    tensor_list_none_masks = []
    for i, (_, is_tensor_list, _) in enumerate(schema_arg_specs):
        if i >= len(node.args):
            break
        value = node.args[i]
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

    for i, (arg_name, _, is_tensor_arg) in enumerate(schema_arg_specs):
        if i >= len(node.args):
            break
        value = node.args[i]
        if isinstance(value, Node):
            continue
        if isinstance(value, (list, tuple)) and any(isinstance(v, Node) for v in value):
            continue
        if value is None and is_tensor_arg:
            continue
        attrs[arg_name] = value

    return attrs


class GraphAnalyzer:
    """Analyzes ExportedProgram to extract metadata."""

    def __init__(self, exported: ExportedProgram):
        self.exported = exported
        self.graph_module = exported.graph_module
        self.graph = exported.graph_module.graph
        self.graph_signature = exported.graph_signature

        self._params_dict = dict(self.graph_signature.inputs_to_parameters)
        self._buffers_dict = dict(self.graph_signature.inputs_to_buffers)
        self._constants_dict = (
            dict(self.graph_signature.inputs_to_lifted_tensor_constants)
            if hasattr(self.graph_signature, "inputs_to_lifted_tensor_constants")
            else {}
        )
        self._weight_placeholders = frozenset(
            (*self._params_dict.keys(), *self._buffers_dict.keys(), *self._constants_dict.keys())
        )
        self._user_inputs = set(self.graph_signature.user_inputs)
        self._user_outputs = set(self.graph_signature.user_outputs)

        self._node_outputs: Dict[str, List[TensorMeta]] = {}
        self._graph_inputs: List[TensorMeta] = []
        self._graph_outputs: List[TensorMeta] = []
        self._weights: List[TensorMeta] = []
        self._weight_name_mapping: Dict[str, str] = {}
        self._call_function_nodes: List[NodeInfo] = []
        self._build_caches()

    def _build_caches(self) -> None:
        """Build cached metadata from a single graph traversal."""
        for node in self.graph.nodes:
            metas = _extract_node_output_meta(node)
            self._node_outputs[node.name] = metas

            if node.op == "placeholder":
                if node.name in self._user_inputs:
                    self._graph_inputs.extend(metas)

                name = (
                    self._params_dict.get(node.name)
                    or self._buffers_dict.get(node.name)
                    or self._constants_dict.get(node.name)
                )
                if name:
                    self._weight_name_mapping[node.name] = name
                    for meta in metas:
                        self._weights.append(
                            TensorMeta(
                                name=name,
                                shape=meta.shape,
                                dtype=meta.dtype,
                            )
                        )
                continue

            if node.op == "call_function":
                self._call_function_nodes.append(
                    NodeInfo(
                        name=node.name,
                        target=node.target,
                        input_metas=self.get_node_input_meta(node),
                        output_metas=metas,
                        attrs=_extract_op_attrs(node),
                    )
                )
                continue

            if node.op == "output":
                output_args = node.args[0] if isinstance(node.args[0], (list, tuple)) else [node.args[0]]
                for arg in output_args:
                    if isinstance(arg, Node) and arg.name in self._user_outputs:
                        self._graph_outputs.extend(self._node_outputs.get(arg.name, []))

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
        input_metas = []
        for arg in node.args:
            if isinstance(arg, Node):
                metas = self._node_outputs.get(arg.name, [])
                is_weight = arg.name in self._weight_placeholders
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
                        is_weight = a.name in self._weight_placeholders
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
        return list(self._graph_inputs)

    def get_graph_outputs(self) -> List[TensorMeta]:
        """Extract graph output tensor metadata."""
        return list(self._graph_outputs)

    def get_weights(self) -> List[TensorMeta]:
        """Extract weight tensor metadata from graph signature."""
        return list(self._weights)

    def get_weight_name_mapping(self) -> Dict[str, str]:
        """Get mapping from placeholder names to state_dict keys.

        Returns:
            Dict mapping placeholder names (e.g., 'p_linear_weight') to
            state_dict keys (e.g., 'linear.weight').
            Also includes lifted tensor constants.
        """
        return dict(self._weight_name_mapping)

    def get_call_function_nodes(self) -> List[NodeInfo]:
        """Extract all call_function nodes (actual operations)."""
        return list(self._call_function_nodes)
