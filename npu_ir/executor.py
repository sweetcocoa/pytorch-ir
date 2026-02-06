"""IR Executor for running IR graphs with actual weights."""

import torch
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

from .ir import NPU_IR, OpNode, TensorMeta
from .weight_loader import load_weights, WeightLoader
from .ops.registry import get_execution_fn


class ExecutionError(Exception):
    """Raised when IR execution fails."""

    pass


class TensorRegistry:
    """Manages tensor storage during IR execution."""

    def __init__(self):
        self._tensors: Dict[str, torch.Tensor] = {}

    def register(self, name: str, tensor: torch.Tensor) -> None:
        """Register a tensor with a name."""
        self._tensors[name] = tensor

    def get(self, name: str) -> Optional[torch.Tensor]:
        """Get a tensor by name."""
        return self._tensors.get(name)

    def has(self, name: str) -> bool:
        """Check if a tensor is registered."""
        return name in self._tensors

    def clear(self) -> None:
        """Clear all registered tensors."""
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
                raise ExecutionError(
                    f"Weight/input '{input_meta.name}' not found in registry "
                    f"for node '{node.name}'."
                )
            tensors.append(registry[input_meta.name])
    return tensors


def _execute_node(
    node: OpNode,
    inputs: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Execute a single node and return outputs."""
    execution_fn = get_execution_fn(node.op_type)

    if execution_fn is None:
        raise ExecutionError(
            f"No execution function registered for op_type '{node.op_type}'\n"
            f"Node: {node.name}"
        )

    try:
        outputs = execution_fn(inputs, node.attrs)
        return outputs
    except Exception as e:
        raise ExecutionError(
            f"Failed to execute node '{node.name}' (op: {node.op_type}): {e}"
        ) from e


class IRExecutor:
    """Executes an IR graph with weights."""

    def __init__(self, ir: NPU_IR, weights: Optional[Dict[str, torch.Tensor]] = None):
        """Initialize the executor.

        Args:
            ir: The IR graph to execute.
            weights: Optional pre-loaded weights. If None, must call load_weights().
        """
        self.ir = ir
        self.weights = weights
        self.registry = TensorRegistry()
        self._prepared = False

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

        # Register graph inputs
        if len(inputs) != len(self.ir.graph_inputs):
            raise ExecutionError(
                f"Expected {len(self.ir.graph_inputs)} inputs, got {len(inputs)}"
            )

        for input_meta, tensor in zip(self.ir.graph_inputs, inputs):
            self.registry.register(input_meta.name, tensor)

        self._prepared = True

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
            raise ExecutionError(
                "Weights not loaded. Call load_weights() or load_weights_from_state_dict() first."
            )

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
                raise ExecutionError(
                    f"Graph output '{output_meta.name}' not found in registry"
                )
            outputs.append(self.registry[output_meta.name])

        return tuple(outputs)

    def __call__(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Execute the IR graph (shorthand for execute())."""
        return self.execute(inputs)


def execute_ir(
    ir: NPU_IR,
    inputs: Tuple[torch.Tensor, ...],
    weights: Optional[Dict[str, torch.Tensor]] = None,
    weights_path: Optional[Union[str, Path]] = None,
) -> Tuple[torch.Tensor, ...]:
    """Execute an IR graph (convenience function).

    Args:
        ir: The IR graph to execute.
        inputs: Input tensors.
        weights: Pre-loaded weights dict.
        weights_path: Path to weight file (alternative to weights).

    Returns:
        Output tensors.

    Raises:
        ExecutionError: If execution fails.
        ValueError: If neither weights nor weights_path is provided.
    """
    if weights is None and weights_path is None:
        raise ValueError("Either weights or weights_path must be provided")

    executor = IRExecutor(ir, weights)

    if weights_path is not None:
        executor.load_weights(weights_path)

    return executor.execute(inputs)
