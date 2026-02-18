"""IR Executor for running IR graphs with actual weights."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from .ir import IR, OpNode
from .ops.registry import get_execution_fn
from .weight_loader import load_weights


class ExecutionError(Exception):
    """Raised when IR execution fails."""

    pass


class TensorRegistry:
    """Name-to-tensor mapping used during IR execution.

    Stores weights, graph inputs, and intermediate results so that
    each node can retrieve its input tensors by name.
    """

    def __init__(self):
        self._tensors: Dict[str, torch.Tensor] = {}

    def register(self, name: str, tensor: torch.Tensor) -> None:
        """Register a tensor under the given name."""
        self._tensors[name] = tensor

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Return the tensor for *name*, or ``None`` if not found."""
        return self._tensors.get(name)

    def has(self, name: str) -> bool:
        """Return ``True`` if *name* is registered."""
        return name in self._tensors

    def clear(self) -> None:
        """Remove all registered tensors."""
        self._tensors.clear()

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._tensors:
            raise KeyError(f"Tensor '{name}' not found in registry")
        return self._tensors[name]

    def __setitem__(self, name: str, tensor: torch.Tensor) -> None:
        self._tensors[name] = tensor

    def __contains__(self, name: str) -> bool:
        return name in self._tensors


def _get_input_tensors(
    node: OpNode,
    registry: TensorRegistry,
    output_map: Dict[str, List[torch.Tensor]],
) -> List[torch.Tensor]:
    """Get input tensors for a node using explicit producer references.

    If an input has producer_node set, look up the producer's outputs directly
    via output_map. Otherwise fall back to registry lookup (for weights and
    graph inputs that have no producer node).
    """
    tensors = []
    for input_meta in node.inputs:
        if input_meta.producer_node is not None:
            producer_outputs = output_map.get(input_meta.producer_node)
            if producer_outputs is None:
                raise ExecutionError(
                    f"Producer node '{input_meta.producer_node}' not found in output_map. "
                    f"Check node ordering. (consumer: '{node.name}')"
                )
            if input_meta.producer_output_idx >= len(producer_outputs):
                raise ExecutionError(
                    f"Producer '{input_meta.producer_node}' has {len(producer_outputs)} outputs, "
                    f"but index {input_meta.producer_output_idx} requested by node '{node.name}'."
                )
            tensors.append(producer_outputs[input_meta.producer_output_idx])
        else:
            # Weight or graph input (no producer node)
            if input_meta.name not in registry:
                raise ExecutionError(f"Weight/input '{input_meta.name}' not found in registry for node '{node.name}'.")
            tensors.append(registry[input_meta.name])
    return tensors


def _resolve_aten_op(op_type: str):
    """Resolve an aten op string to its torch.ops.aten function.

    e.g. 'aten.conv3d.default' -> torch.ops.aten.conv3d.default
    """
    if not op_type.startswith("aten."):
        return None
    parts = op_type.split(".")
    # parts = ['aten', 'op_name', 'overload'] or ['aten', 'op_name']
    if len(parts) < 2:
        return None
    try:
        op = getattr(torch.ops.aten, parts[1])
        if len(parts) >= 3:
            overload = getattr(op, parts[2], None)
            if overload is not None:
                return overload
        return op.default
    except AttributeError:
        return None


def _is_tensor_type(arg_type_str: str) -> bool:
    """Check if a schema arg type is a single Tensor (not list, not optional)."""
    return arg_type_str in ("Tensor", "Tensor(a)")


def _is_tensor_list_type(arg_type_str: str) -> bool:
    """Check if a schema arg type is a list of tensors (Tensor[], Tensor?[], List[Optional[Tensor]])."""
    return (
        "Tensor[]" in arg_type_str
        or "Tensor?[]" in arg_type_str
        or "List[Tensor]" in arg_type_str
        or "List[Optional[Tensor]]" in arg_type_str
    )


def _is_optional_tensor_type(arg_type_str: str) -> bool:
    """Check if a schema arg type is an optional Tensor."""
    return "Tensor?" in arg_type_str or "Tensor(a)?" in arg_type_str or "Optional[Tensor]" in arg_type_str


def _sanitize_attr_value(name: str, value):
    """Sanitize attr values for execution (e.g., meta device -> cpu)."""
    if name == "device" and isinstance(value, torch.device) and value.type == "meta":
        return torch.device("cpu")
    return value


def _aten_fallback(
    node: OpNode,
    inputs: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Execute an ATen op by introspecting its schema and calling ``torch.ops.aten.*``.

    This is the core fallback strategy that enables automatic execution of any
    ATen operator without manual implementation.  The schema is used to
    reconstruct positional and keyword arguments from the flat tensor list
    and the node's ``attrs`` dictionary.

    Args:
        node: The IR node to execute.
        inputs: Flat list of input tensors resolved from producer references.

    Returns:
        List of output tensors produced by the operation.

    Raises:
        ExecutionError: If the op cannot be resolved or execution fails.
    """
    aten_fn = _resolve_aten_op(node.op_type)
    if aten_fn is None:
        raise ExecutionError(f"No execution function registered for op_type '{node.op_type}'\nNode: {node.name}")

    # Build args from schema, mapping flat tensor inputs + attrs to correct positions
    args = list(inputs)
    kwargs = {}

    if hasattr(aten_fn, "_schema"):
        schema = aten_fn._schema
        tensor_idx = 0
        positional_args = []

        # Use exact list sizes from analyzer if available
        tensor_list_sizes = node.attrs.get("_tensor_list_sizes", None)
        tensor_list_none_masks = node.attrs.get("_tensor_list_none_masks", None)
        list_size_idx = 0
        none_mask_idx = 0

        # Count single-Tensor and Tensor[] args for fallback fair-split
        single_tensor_count = 0
        tensor_list_count = 0
        for schema_arg in schema.arguments:
            arg_type_str = str(schema_arg.type)
            if _is_tensor_type(arg_type_str):
                single_tensor_count += 1
            elif _is_tensor_list_type(arg_type_str):
                tensor_list_count += 1

        remaining_for_lists = max(0, len(inputs) - single_tensor_count)

        for schema_arg in schema.arguments:
            arg_type_str = str(schema_arg.type)

            if _is_tensor_type(arg_type_str):
                if tensor_idx < len(inputs):
                    positional_args.append(inputs[tensor_idx])
                    tensor_idx += 1
                elif schema_arg.name in node.attrs:
                    # Scalar stored in attrs for binary ops (e.g., x / 2.0)
                    positional_args.append(node.attrs[schema_arg.name])
                else:
                    break
            elif _is_tensor_list_type(arg_type_str):
                # Use exact size from analyzer, or fall back to fair split
                if tensor_list_sizes and list_size_idx < len(tensor_list_sizes):
                    size = tensor_list_sizes[list_size_idx]
                    list_size_idx += 1
                elif tensor_list_count > 0:
                    size = remaining_for_lists // tensor_list_count
                else:
                    size = 0
                tensor_list = list(inputs[tensor_idx : tensor_idx + size])
                # Reconstruct None positions for Tensor?[] args
                if tensor_list_none_masks and none_mask_idx < len(tensor_list_none_masks):
                    none_mask = tensor_list_none_masks[none_mask_idx]
                    none_mask_idx += 1
                    full_list = []
                    t_idx = 0
                    for is_none in none_mask:
                        if is_none:
                            full_list.append(None)
                        else:
                            full_list.append(tensor_list[t_idx] if t_idx < len(tensor_list) else None)
                            t_idx += 1
                    tensor_list = full_list
                positional_args.append(tensor_list)
                tensor_idx += size
                remaining_for_lists -= size
                tensor_list_count = max(0, tensor_list_count - 1)
            elif _is_optional_tensor_type(arg_type_str):
                if tensor_idx < len(inputs):
                    positional_args.append(inputs[tensor_idx])
                    tensor_idx += 1
                elif schema_arg.name in node.attrs:
                    positional_args.append(node.attrs[schema_arg.name])
                else:
                    positional_args.append(None)
            elif schema_arg.name in node.attrs:
                val = _sanitize_attr_value(schema_arg.name, node.attrs[schema_arg.name])
                if schema_arg.kwarg_only:
                    kwargs[schema_arg.name] = val
                else:
                    positional_args.append(val)
            elif schema_arg.has_default_value():
                if schema_arg.kwarg_only:
                    pass  # Let the op use its default
                else:
                    break  # Stop adding positional args
            else:
                break  # Missing required arg — stop
        args = positional_args

    try:
        result = aten_fn(*args, **kwargs)
    except TypeError:
        # Fallback: try with just inputs and attrs as kwargs
        result = aten_fn(*inputs, **node.attrs)

    # Normalize output to list
    if isinstance(result, torch.Tensor):
        return [result]
    elif isinstance(result, (tuple, list)):
        return [r for r in result if isinstance(r, torch.Tensor)]
    else:
        return [result]


def _execute_node(
    node: OpNode,
    inputs: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Execute a single node and return outputs.

    Strategy:

    1. If a custom executor is registered, use it (for non-ATen ops like ``getitem``).
    2. Otherwise, call the original ATen op directly via ``_aten_fallback``.

    Args:
        node: The IR node to execute.
        inputs: Input tensors for this node.

    Returns:
        List of output tensors.

    Raises:
        ExecutionError: If execution fails.
    """
    # Custom executor takes priority (needed for non-ATen ops like getitem)
    execution_fn = get_execution_fn(node.op_type)
    if execution_fn is not None:
        try:
            return execution_fn(inputs, node.attrs)
        except Exception as e:
            raise ExecutionError(f"Failed to execute node '{node.name}' (op: {node.op_type}): {e}") from e

    # Default: call original ATen op directly
    try:
        return _aten_fallback(node, inputs)
    except Exception as e:
        raise ExecutionError(f"Failed to execute node '{node.name}' (op: {node.op_type}): {e}") from e


class IRExecutor:
    """Executes an IR graph with actual weight tensors.

    Typical usage::

        executor = IRExecutor(ir)
        executor.load_weights_from_state_dict(state_dict)
        outputs = executor.execute((input_tensor,))
    """

    def __init__(
        self,
        ir: IR,
        weights: Optional[Dict[str, torch.Tensor]] = None,
        constants: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize the executor.

        Args:
            ir: The IR graph to execute.
            weights: Optional pre-loaded weights. If None, must call load_weights().
            constants: Optional lifted tensor constants. When provided, these
                override ``ir.constants`` and are used for constant placeholders
                that are not part of the model's ``state_dict`` (e.g., index
                tensors assigned as plain attributes). This is needed when
                the IR was extracted on meta device, where constant values
                are unavailable.
        """
        self.ir = ir
        self.weights = weights
        self.constants = constants
        self.registry = TensorRegistry()

    def load_weights(self, path: Union[str, Path]) -> None:
        """Load weights from a file.

        Args:
            path: Path to the weight file (.pt or .safetensors).
        """
        self.weights = load_weights(path)

    def load_weights_from_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Use an existing state dict as weights.

        Args:
            state_dict: The state dict to use.
        """
        self.weights = state_dict

    def _prepare(self, inputs: Tuple[torch.Tensor, ...]) -> None:
        """Prepare for execution by registering weights and inputs."""
        self.registry.clear()

        # Register weights by their state_dict names
        if self.weights:
            for name, tensor in self.weights.items():
                self.registry.register(name, tensor)

        # Use the weight_name_mapping from IR to register weights by placeholder names
        # This maps placeholder names (e.g., 'p_linear_weight') to state_dict keys ('linear.weight')
        if self.weights and self.ir.weight_name_mapping:
            for placeholder_name, sd_key in self.ir.weight_name_mapping.items():
                if sd_key in self.weights:
                    self.registry.register(placeholder_name, self.weights[sd_key])

        # Register lifted tensor constants (e.g., index tensors in forward()).
        # User-supplied constants take precedence over ir.constants so that
        # meta-device-extracted IRs can be executed with externally provided values.
        effective_constants = self.ir.constants.copy()
        if self.constants:
            effective_constants.update(self.constants)
        if effective_constants:
            for const_name, const_tensor in effective_constants.items():
                self.registry.register(const_name, const_tensor)
                # Also register with placeholder prefix if mapped
                for placeholder_name, sd_key in self.ir.weight_name_mapping.items():
                    if sd_key == const_name:
                        self.registry.register(placeholder_name, const_tensor)

        # Register graph inputs
        if len(inputs) != len(self.ir.graph_inputs):
            raise ExecutionError(f"Expected {len(self.ir.graph_inputs)} inputs, got {len(inputs)}")

        for input_meta, tensor in zip(self.ir.graph_inputs, inputs):
            self.registry.register(input_meta.name, tensor)

    def execute(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        """Execute the IR graph.

        Args:
            inputs: Input tensors matching graph_inputs.

        Returns:
            Output tensors matching graph_outputs.

        Raises:
            ExecutionError: If execution fails.
        """
        if self.weights is None:
            raise ExecutionError("Weights not loaded. Call load_weights() or load_weights_from_state_dict() first.")

        self._prepare(inputs)

        # Track node outputs by node name for producer-based lookup
        output_map: Dict[str, List[torch.Tensor]] = {}

        # Register graph inputs in output_map so producer references work
        for input_meta, tensor in zip(self.ir.graph_inputs, inputs):
            output_map[input_meta.name] = [tensor]

        # Execute nodes in order
        for node in self.ir.nodes:
            # Gather inputs using producer references
            input_tensors = _get_input_tensors(node, self.registry, output_map)

            # Execute node
            output_tensors = _execute_node(node, input_tensors)

            # Store in output_map for downstream producer references
            output_map[node.name] = output_tensors

            # Also register in registry for backward compat and graph output lookup
            for output_meta, tensor in zip(node.outputs, output_tensors):
                self.registry.register(output_meta.name, tensor)

        # Gather graph outputs
        outputs = []
        for output_meta in self.ir.graph_outputs:
            if output_meta.name not in self.registry:
                raise ExecutionError(f"Graph output '{output_meta.name}' not found in registry")
            outputs.append(self.registry[output_meta.name])

        return tuple(outputs)


def execute_ir(
    ir: IR,
    inputs: Tuple[torch.Tensor, ...],
    weights: Optional[Dict[str, torch.Tensor]] = None,
    weights_path: Optional[Union[str, Path]] = None,
    constants: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, ...]:
    """Execute an IR graph (convenience function).

    Args:
        ir: The IR graph to execute.
        inputs: Input tensors.
        weights: Pre-loaded weights dict.
        weights_path: Path to weight file (alternative to weights).
        constants: Optional lifted tensor constants for meta-device-extracted IRs.
            See :class:`IRExecutor` for details.

    Returns:
        Output tensors.

    Raises:
        ExecutionError: If execution fails.
        ValueError: If neither weights nor weights_path is provided.
    """
    if weights is None and weights_path is None:
        raise ValueError("Either weights or weights_path must be provided")

    executor = IRExecutor(ir, weights, constants=constants)

    if weights_path is not None:
        executor.load_weights(weights_path)

    return executor.execute(inputs)
