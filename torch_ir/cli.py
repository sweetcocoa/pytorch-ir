"""Command-line interface for pytorch-ir."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from torch_ir import __version__
from torch_ir.ir import IR
from torch_ir.serializer import SerializationError, load_ir
from torch_ir.visualize import ir_to_mermaid


def _load_ir_file(path: str) -> IR:
    """Load an IR file with user-friendly error handling."""
    try:
        return load_ir(path)
    except FileNotFoundError:
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except SerializationError as e:
        print(f"Error: invalid IR file: {e}", file=sys.stderr)
        sys.exit(1)


def _write_output(text: str, path: Optional[str]) -> None:
    """Write text to stdout or a file."""
    if path is None:
        print(text)
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text)


def _render_mermaid_image(mermaid_text: str, output_path: str) -> None:
    """Render Mermaid text to an image file using mmdc."""
    try:
        from mmdc import MermaidConverter  # ty: ignore[unresolved-import]
    except ImportError:
        print(
            "Error: image rendering requires the 'mmdc' package.\nInstall it with: pip install pytorch-ir[rendering]",
            file=sys.stderr,
        )
        sys.exit(1)

    converter = MermaidConverter()
    ext = Path(output_path).suffix.lower()
    if ext == ".png":
        data = converter.to_png(mermaid_text)
        if data is None:
            print("Error: failed to render PNG", file=sys.stderr)
            sys.exit(1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(data)
    elif ext == ".svg":
        data = converter.to_svg(mermaid_text)
        if data is None:
            print("Error: failed to render SVG", file=sys.stderr)
            sys.exit(1)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(data)


def cmd_visualize(args: argparse.Namespace) -> None:
    """Handle the 'visualize' subcommand."""
    ir = _load_ir_file(args.ir_file)
    mermaid_text = ir_to_mermaid(ir, max_nodes=args.max_nodes, include_weights=not args.no_weights)

    if args.output is not None:
        ext = Path(args.output).suffix.lower()
        if ext in (".png", ".svg"):
            _render_mermaid_image(mermaid_text, args.output)
            return

    _write_output(mermaid_text, args.output)


def cmd_info(args: argparse.Namespace) -> None:
    """Handle the 'info' subcommand."""
    ir = _load_ir_file(args.ir_file)

    # Compute op distribution
    op_counts: dict[str, int] = defaultdict(int)
    for node in ir.nodes:
        op_counts[node.op_type] += 1

    # Compute total weight parameters
    total_params = 0
    for w in ir.weights:
        p = 1
        for d in w.shape:
            p *= d
        total_params += p

    info = {
        "model_name": ir.model_name,
        "num_nodes": len(ir.nodes),
        "num_inputs": len(ir.graph_inputs),
        "num_outputs": len(ir.graph_outputs),
        "num_weights": len(ir.weights),
        "total_parameters": total_params,
        "input_shapes": {inp.name: list(inp.shape) for inp in ir.graph_inputs},
        "output_shapes": {out.name: list(out.shape) for out in ir.graph_outputs},
        "op_distribution": dict(sorted(op_counts.items(), key=lambda x: -x[1])),
    }

    if args.json:
        output = json.dumps(info, indent=2)
    else:
        lines = [
            f"Model: {info['model_name']}",
            f"Nodes: {info['num_nodes']}",
            f"Inputs: {info['num_inputs']}",
            f"Outputs: {info['num_outputs']}",
            f"Weights: {info['num_weights']}",
            f"Total parameters: {info['total_parameters']:,}",
            "",
            "Input shapes:",
        ]
        for name, shape in info["input_shapes"].items():
            lines.append(f"  {name}: {shape}")
        lines.append("")
        lines.append("Output shapes:")
        for name, shape in info["output_shapes"].items():
            lines.append(f"  {name}: {shape}")
        lines.append("")
        lines.append("Op distribution:")
        for op, count in info["op_distribution"].items():
            lines.append(f"  {op}: {count}")
        output = "\n".join(lines)

    _write_output(output, args.output)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for pytorch-ir CLI."""
    parser = argparse.ArgumentParser(prog="pytorch-ir", description="pytorch-ir CLI tools")
    parser.add_argument("--version", action="version", version=f"pytorch-ir {__version__}")

    subparsers = parser.add_subparsers(dest="command")

    # visualize subcommand
    vis_parser = subparsers.add_parser("visualize", help="Visualize IR graph as Mermaid diagram")
    vis_parser.add_argument("ir_file", help="Path to IR JSON file")
    vis_parser.add_argument(
        "--max-nodes",
        type=int,
        default=30,
        help="Maximum number of nodes to display (default: 30)",
    )
    vis_parser.add_argument(
        "--no-weights",
        action="store_true",
        help="Hide weight inputs from the diagram",
    )
    vis_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output file (.png/.svg for image, others for Mermaid text)",
    )

    # info subcommand
    info_parser = subparsers.add_parser("info", help="Show IR summary information")
    info_parser.add_argument("ir_file", help="Path to IR JSON file")
    info_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    info_parser.add_argument("-o", "--output", default=None, help="Output file")

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(2)

    if args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "info":
        cmd_info(args)
