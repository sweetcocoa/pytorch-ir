"""IR data structures for NPU compiler."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TensorMeta:
    """Metadata for a tensor (shape and dtype only, no actual data).

    Attributes:
        name: Unique tensor name within the IR graph.
        shape: Static shape of the tensor (e.g., ``(1, 3, 224, 224)``).
        dtype: String representation of the data type (e.g., ``"float32"``).
        producer_node: Name of the node that produced this tensor.
            ``None`` for weights and graph inputs.
        producer_output_idx: Index into the producer node's output list.
            Used to resolve the correct tensor when a node produces multiple outputs.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: str
    producer_node: Optional[str] = None
    producer_output_idx: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary with ``name``, ``shape``, ``dtype`` keys.
            Includes ``producer_node`` and ``producer_output_idx`` when present.
        """
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
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            Reconstructed ``TensorMeta`` instance.
        """
        return cls(
            name=data["name"],
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            producer_node=data.get("producer_node"),
            producer_output_idx=data.get("producer_output_idx", 0),
        )


@dataclass
class OpNode:
    """A single operation node in the IR graph.

    Attributes:
        name: Unique node name (e.g., ``"conv2d"``).
        op_type: ATen operation type string (e.g., ``"aten.conv2d.default"``).
        inputs: Input tensor metadata list.
        outputs: Output tensor metadata list.
        attrs: Operation attributes such as ``kernel_size``, ``stride``, ``padding``, etc.
    """

    name: str
    op_type: str
    inputs: List[TensorMeta]
    outputs: List[TensorMeta]
    attrs: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary containing all node fields.
        """
        return {
            "name": self.name,
            "op_type": self.op_type,
            "inputs": [t.to_dict() for t in self.inputs],
            "outputs": [t.to_dict() for t in self.outputs],
            "attrs": self.attrs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpNode":
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            Reconstructed ``OpNode`` instance.
        """
        return cls(
            name=data["name"],
            op_type=data["op_type"],
            inputs=[TensorMeta.from_dict(t) for t in data["inputs"]],
            outputs=[TensorMeta.from_dict(t) for t in data["outputs"]],
            attrs=data.get("attrs", {}),
        )


@dataclass
class NPU_IR:
    """Complete IR representation of a model.

    Attributes:
        nodes: Ordered list of operation nodes in the graph.
        graph_inputs: Metadata for graph input tensors (user-provided activations).
        graph_outputs: Metadata for graph output tensors.
        weights: Metadata for weight/buffer tensors (shape and dtype only, no values).
        weight_name_mapping: Maps FX placeholder names (e.g., ``"p_linear_weight"``)
            to original ``state_dict`` keys (e.g., ``"linear.weight"``).
        constants: Lifted tensor constants from ``torch.export``
            (e.g., index tensors created in ``forward()`` via ``register_buffer``).
        model_name: Human-readable model name.
        pytorch_version: PyTorch version used during IR extraction.
    """

    nodes: List[OpNode]
    graph_inputs: List[TensorMeta]
    graph_outputs: List[TensorMeta]
    weights: List[TensorMeta]
    weight_name_mapping: Dict[str, str] = field(default_factory=dict)
    constants: Dict[str, Any] = field(default_factory=dict)
    model_name: str = ""
    pytorch_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary.

        Returns:
            Dictionary representation of the entire IR graph.
        """
        d = {
            "model_name": self.model_name,
            "pytorch_version": self.pytorch_version,
            "graph_inputs": [t.to_dict() for t in self.graph_inputs],
            "graph_outputs": [t.to_dict() for t in self.graph_outputs],
            "weights": [t.to_dict() for t in self.weights],
            "weight_name_mapping": self.weight_name_mapping,
            "nodes": [n.to_dict() for n in self.nodes],
        }
        if self.constants:
            d["constants"] = {
                k: {"data": v.tolist(), "dtype": str(v.dtype).replace("torch.", "")}
                for k, v in self.constants.items()
                if isinstance(v, torch.Tensor)
            }
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NPU_IR":
        """Deserialize from a dictionary.

        Args:
            data: Dictionary produced by ``to_dict()``.

        Returns:
            Reconstructed ``NPU_IR`` instance.
        """
        _dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16,
            "int32": torch.int32,
            "int64": torch.int64,
            "int8": torch.int8,
            "bool": torch.bool,
        }
        constants = {}
        for k, v in data.get("constants", {}).items():
            dt = _dtype_map.get(v["dtype"], torch.float32)
            constants[k] = torch.tensor(v["data"], dtype=dt)

        return cls(
            model_name=data.get("model_name", ""),
            pytorch_version=data.get("pytorch_version", ""),
            graph_inputs=[TensorMeta.from_dict(t) for t in data["graph_inputs"]],
            graph_outputs=[TensorMeta.from_dict(t) for t in data["graph_outputs"]],
            weights=[TensorMeta.from_dict(t) for t in data["weights"]],
            weight_name_mapping=data.get("weight_name_mapping", {}),
            constants=constants,
            nodes=[OpNode.from_dict(n) for n in data["nodes"]],
        )

    def save(self, path: str) -> None:
        """Save IR to a JSON file.

        Args:
            path: Output file path.
        """
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "NPU_IR":
        """Load IR from a JSON file.

        Args:
            path: Input file path.

        Returns:
            Deserialized ``NPU_IR`` instance.
        """
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
