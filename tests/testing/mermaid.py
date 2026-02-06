"""Mermaid diagram generation from NPU IR."""

from typing import Dict, List, Set, Tuple
from collections import defaultdict

from npu_ir import NPU_IR, OpNode


def _format_shape(shape: Tuple[int, ...]) -> str:
    """Format tensor shape for display."""
    return "x".join(str(d) for d in shape)


def _sanitize_label(text: str) -> str:
    """Sanitize text for Mermaid labels.

    Note: In Mermaid, <br/> is used for line breaks in labels,
    so we don't need to escape those. Mermaid uses HTML-like tags.
    """
    # Replace quotes with single quotes
    text = text.replace('"', "'")
    return text


def _get_short_op_name(op_type: str) -> str:
    """Get shortened operator name for display."""
    # Remove aten. prefix and .default suffix
    name = op_type
    if name.startswith("aten."):
        name = name[5:]
    if name.endswith(".default"):
        name = name[:-8]
    return name


def ir_to_mermaid(ir: NPU_IR, max_nodes: int = 30) -> str:
    """Convert NPU IR to Mermaid flowchart diagram.

    Args:
        ir: The NPU IR to visualize.
        max_nodes: Maximum number of nodes to include (for large graphs).

    Returns:
        Mermaid flowchart diagram as string.
    """
    lines = ["flowchart TD"]

    # Build tensor name to producing node mapping
    tensor_to_producer: Dict[str, str] = {}

    # Track input tensor names
    input_names: Set[str] = set()
    for inp in ir.graph_inputs:
        input_names.add(inp.name)
        tensor_to_producer[inp.name] = f"input_{inp.name}"

    # Track weight tensor names
    weight_names: Set[str] = set()
    for w in ir.weights:
        weight_names.add(w.name)

    # Track output tensor names
    output_names: Set[str] = set()
    for out in ir.graph_outputs:
        output_names.add(out.name)

    # Add input nodes
    for i, inp in enumerate(ir.graph_inputs):
        shape_str = _format_shape(inp.shape)
        label = _sanitize_label(f"Input: {inp.name}<br/>{shape_str}")
        lines.append(f'    input_{inp.name}[/"{label}"/]')

    # Process operation nodes
    nodes_to_show = ir.nodes[:max_nodes] if len(ir.nodes) > max_nodes else ir.nodes
    truncated = len(ir.nodes) > max_nodes

    for node in nodes_to_show:
        # Register outputs
        for out in node.outputs:
            tensor_to_producer[out.name] = f"op_{node.name}"

    # Add operation nodes and edges
    for node in nodes_to_show:
        op_name = _get_short_op_name(node.op_type)

        # Get output shape for label
        if node.outputs:
            out_shape = _format_shape(node.outputs[0].shape)
            label = _sanitize_label(f"{op_name}<br/>{out_shape}")
        else:
            label = _sanitize_label(op_name)

        lines.append(f'    op_{node.name}["{label}"]')

        # Add edges from inputs
        for inp in node.inputs:
            # Skip weight inputs for cleaner graph
            if inp.name in weight_names:
                continue

            producer = tensor_to_producer.get(inp.name)
            if producer:
                edge_label = _format_shape(inp.shape)
                lines.append(f'    {producer} -->|"{edge_label}"| op_{node.name}')

    # Add output nodes
    for i, out in enumerate(ir.graph_outputs):
        shape_str = _format_shape(out.shape)
        label = _sanitize_label(f"Output<br/>{shape_str}")
        lines.append(f'    output_{i}[\\"{label}"/]')

        # Find producer for this output
        producer = tensor_to_producer.get(out.name)
        if producer:
            lines.append(f'    {producer} --> output_{i}')

    # Add truncation note if needed
    if truncated:
        lines.append(f'    note["... {len(ir.nodes) - max_nodes} more nodes ..."]')

    return "\n".join(lines)


def generate_op_distribution_pie(ir: NPU_IR, top_n: int = 10) -> str:
    """Generate Mermaid pie chart showing operator distribution.

    Args:
        ir: The NPU IR to analyze.
        top_n: Maximum number of operator types to show.

    Returns:
        Mermaid pie chart as string.
    """
    # Count operators
    op_counts: Dict[str, int] = defaultdict(int)
    for node in ir.nodes:
        op_name = _get_short_op_name(node.op_type)
        op_counts[op_name] += 1

    # Sort by count
    sorted_ops = sorted(op_counts.items(), key=lambda x: -x[1])

    # Take top N
    top_ops = sorted_ops[:top_n]

    # Build pie chart
    lines = [
        "pie showData",
        '    title "Operator Distribution"',
    ]

    for op_name, count in top_ops:
        lines.append(f'    "{op_name}" : {count}')

    # Add "Others" if there are more
    if len(sorted_ops) > top_n:
        others_count = sum(count for _, count in sorted_ops[top_n:])
        lines.append(f'    "Others" : {others_count}')

    return "\n".join(lines)
