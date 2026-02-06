"""IR data structures for NPU compiler."""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import json


@dataclass
class TensorMeta:
    """Metadata for a tensor (shape and dtype only, no actual data)."""

    name: str
    shape: Tuple[int, ...]
    dtype: str  # "float32", "float16", "int8", "bfloat16", etc.
    # Producer tracking: which node produced this tensor
    producer_node: Optional[str] = None  # Name of the node that produced this tensor
    producer_output_idx: int = 0  # Index into producer node's output list

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
        if self.producer_node is not None:
            d["producer_node"] = self.producer_node
            d["producer_output_idx"] = self.producer_output_idx
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TensorMeta":
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            producer_node=data.get("producer_node"),
            producer_output_idx=data.get("producer_output_idx", 0),
        )


@dataclass
class OpNode:
    """A single operation node in the IR graph."""

    name: str  # Unique node name
    op_type: str  # e.g., "aten.conv2d.default", "aten.linear.default"
    inputs: List[TensorMeta]  # Input tensors with shape/dtype
    outputs: List[TensorMeta]  # Output tensors with shape/dtype
    attrs: Dict[str, Any] = field(default_factory=dict)  # kernel_size, stride, etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpNode":
        return cls(
            name=data["name"],
            op_type=data["op_type"],
            inputs=[TensorMeta.from_dict(t) for t in data["inputs"]],
            outputs=[TensorMeta.from_dict(t) for t in data["outputs"]],
            attrs=data.get("attrs", {}),
        )


@dataclass
class NPU_IR:
    """Complete IR representation of a model."""

    # Graph information
    nodes: List[OpNode]

    # Model inputs/outputs (graph entry/exit points)
    graph_inputs: List[TensorMeta]
    graph_outputs: List[TensorMeta]

    # Weight metadata (no values, only shape/dtype)
    weights: List[TensorMeta]

    # Mapping from placeholder names to state_dict keys
    # e.g., {"p_linear_weight": "linear.weight", "p_linear_bias": "linear.bias"}
    weight_name_mapping: Dict[str, str] = field(default_factory=dict)

    # Metadata
    model_name: str = ""
    pytorch_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pytorch_version": self.pytorch_version,
            "graph_inputs": [t.to_dict() for t in self.graph_inputs],
            "graph_outputs": [t.to_dict() for t in self.graph_outputs],
            "weights": [t.to_dict() for t in self.weights],
            "weight_name_mapping": self.weight_name_mapping,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPU_IR":
        return cls(
            model_name=data.get("model_name", ""),
            pytorch_version=data.get("pytorch_version", ""),
            graph_inputs=[TensorMeta.from_dict(t) for t in data["graph_inputs"]],
            graph_outputs=[TensorMeta.from_dict(t) for t in data["graph_outputs"]],
            weights=[TensorMeta.from_dict(t) for t in data["weights"]],
            weight_name_mapping=data.get("weight_name_mapping", {}),
            nodes=[OpNode.from_dict(n) for n in data["nodes"]],
        )

    def save(self, path: str) -> None:
        """Save IR to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NPU_IR":
        """Load IR from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        return (
            f"NPU_IR(model_name='{self.model_name}', "
            f"nodes={len(self.nodes)}, "
            f"inputs={len(self.graph_inputs)}, "
            f"outputs={len(self.graph_outputs)}, "
            f"weights={len(self.weights)})"
        )
