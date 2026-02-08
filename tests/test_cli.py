"""Tests for the CLI."""

import json

import pytest
import torch

from torch_ir.cli import main
from torch_ir.ir import IR, OpNode, TensorMeta


@pytest.fixture
def sample_ir_file(tmp_path):
    """Create a sample IR JSON file for testing."""
    conv_out = TensorMeta(
        name="conv_out", shape=(1, 64, 112, 112), dtype="float32",
        producer_node="conv", producer_output_idx=0,
    )
    relu_out = TensorMeta(
        name="relu_out", shape=(1, 64, 112, 112), dtype="float32",
        producer_node="relu", producer_output_idx=0,
    )
    output_0 = TensorMeta(
        name="output_0", shape=(1, 10), dtype="float32",
        producer_node="linear", producer_output_idx=0,
    )
    ir = IR(
        model_name="TestModel",
        pytorch_version=torch.__version__,
        graph_inputs=[
            TensorMeta(name="input_0", shape=(1, 3, 224, 224), dtype="float32"),
        ],
        graph_outputs=[output_0],
        weights=[
            TensorMeta(name="weight_0", shape=(64, 3, 7, 7), dtype="float32"),
            TensorMeta(name="bias_0", shape=(64,), dtype="float32"),
        ],
        nodes=[
            OpNode(
                name="conv",
                op_type="aten.conv2d.default",
                inputs=[
                    TensorMeta(name="input_0", shape=(1, 3, 224, 224), dtype="float32"),
                    TensorMeta(name="weight_0", shape=(64, 3, 7, 7), dtype="float32"),
                ],
                outputs=[conv_out],
            ),
            OpNode(
                name="relu",
                op_type="aten.relu.default",
                inputs=[conv_out],
                outputs=[relu_out],
            ),
            OpNode(
                name="linear",
                op_type="aten.linear.default",
                inputs=[relu_out],
                outputs=[output_0],
            ),
        ],
    )
    path = tmp_path / "test_model.json"
    ir.save(str(path))
    return str(path)


class TestNoSubcommand:
    def test_no_subcommand_shows_help(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 2


class TestVersion:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "torch-ir" in captured.out


class TestInfo:
    def test_info_basic(self, sample_ir_file, capsys):
        main(["info", sample_ir_file])
        captured = capsys.readouterr()
        assert "TestModel" in captured.out
        assert "Nodes: 3" in captured.out
        assert "Inputs: 1" in captured.out
        assert "Outputs: 1" in captured.out

    def test_info_json_output(self, sample_ir_file, capsys):
        main(["info", sample_ir_file, "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["model_name"] == "TestModel"
        assert data["num_nodes"] == 3
        assert data["num_inputs"] == 1
        assert data["num_outputs"] == 1
        assert data["num_weights"] == 2
        assert data["total_parameters"] > 0
        assert "aten.conv2d.default" in data["op_distribution"]

    def test_info_file_not_found(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["info", "nonexistent.json"])
        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "file not found" in captured.err

    def test_info_output_to_file(self, sample_ir_file, tmp_path):
        out_path = str(tmp_path / "info.txt")
        main(["info", sample_ir_file, "-o", out_path])
        content = open(out_path).read()
        assert "TestModel" in content

    def test_info_json_output_to_file(self, sample_ir_file, tmp_path):
        out_path = str(tmp_path / "info.json")
        main(["info", sample_ir_file, "--json", "-o", out_path])
        data = json.loads(open(out_path).read())
        assert data["model_name"] == "TestModel"


class TestVisualize:
    def test_visualize_graph(self, sample_ir_file, capsys):
        main(["visualize", sample_ir_file])
        captured = capsys.readouterr()
        assert "flowchart TD" in captured.out

    def test_visualize_max_nodes(self, sample_ir_file, capsys):
        main(["visualize", sample_ir_file, "--max-nodes", "1"])
        captured = capsys.readouterr()
        assert "flowchart TD" in captured.out
        assert "more nodes" in captured.out

    def test_output_to_mmd_file(self, sample_ir_file, tmp_path):
        out_path = str(tmp_path / "graph.mmd")
        main(["visualize", sample_ir_file, "-o", out_path])
        content = open(out_path).read()
        assert "flowchart TD" in content

    def test_output_to_png(self, sample_ir_file, tmp_path):
        pytest.importorskip("mmdc")
        out_path = str(tmp_path / "graph.png")
        main(["visualize", sample_ir_file, "-o", out_path])
        import os

        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0

    def test_output_to_svg(self, sample_ir_file, tmp_path):
        pytest.importorskip("mmdc")
        out_path = str(tmp_path / "graph.svg")
        main(["visualize", sample_ir_file, "-o", out_path])
        content = open(out_path).read()
        assert len(content) > 0
