"""Statistics collection from NPU IR."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Set
from collections import defaultdict

from npu_ir import NPU_IR


@dataclass
class IRStatistics:
    """Statistics collected from an NPU IR graph.

    Attributes:
        num_nodes: Total number of operation nodes.
        num_edges: Total number of edges (tensor connections).
        num_inputs: Number of graph inputs.
        num_outputs: Number of graph outputs.
        num_weights: Number of weight tensors.
        op_distribution: Count of each operator type.
        node_shapes: List of node info dicts (name, op_type, input/output shapes).
        weight_metadata: List of weight info dicts (name, shape, dtype, num_params).
        total_weight_params: Total number of weight parameters.
    """
    num_nodes: int
    num_edges: int
    num_inputs: int
    num_outputs: int
    num_weights: int
    op_distribution: Dict[str, int]
    node_shapes: List[Dict[str, Any]]
    weight_metadata: List[Dict[str, Any]]
    total_weight_params: int

    @classmethod
    def from_ir(cls, ir: NPU_IR) -> "IRStatistics":
        """Collect statistics from an NPU IR.

        Args:
            ir: The NPU IR to analyze.

        Returns:
            IRStatistics object with collected data.
        """
        # Count nodes
        num_nodes = len(ir.nodes)
        num_inputs = len(ir.graph_inputs)
        num_outputs = len(ir.graph_outputs)
        num_weights = len(ir.weights)

        # Count edges (non-weight inputs to each node)
        weight_names: Set[str] = {w.name for w in ir.weights}
        num_edges = 0
        for node in ir.nodes:
            for inp in node.inputs:
                if inp.name not in weight_names:
                    num_edges += 1

        # Operator distribution
        op_distribution: Dict[str, int] = defaultdict(int)
        for node in ir.nodes:
            op_distribution[node.op_type] += 1

        # Node shapes
        node_shapes = []
        for node in ir.nodes:
            node_shapes.append({
                "name": node.name,
                "op_type": node.op_type,
                "input_shapes": [list(inp.shape) for inp in node.inputs],
                "output_shapes": [list(out.shape) for out in node.outputs],
            })

        # Weight metadata
        weight_metadata = []
        total_weight_params = 0
        for w in ir.weights:
            num_params = 1
            for dim in w.shape:
                num_params *= dim
            total_weight_params += num_params

            weight_metadata.append({
                "name": w.name,
                "shape": list(w.shape),
                "dtype": w.dtype,
                "num_params": num_params,
            })

        return cls(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            num_weights=num_weights,
            op_distribution=dict(op_distribution),
            node_shapes=node_shapes,
            weight_metadata=weight_metadata,
            total_weight_params=total_weight_params,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "num_weights": self.num_weights,
            "op_distribution": self.op_distribution,
            "node_shapes": self.node_shapes,
            "weight_metadata": self.weight_metadata,
            "total_weight_params": self.total_weight_params,
        }
