# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

IR Extraction Framework — extracts intermediate representation (IR) graphs from PyTorch models using `torch.export` **without loading actual weights**. Models run on PyTorch's meta device so only shape/dtype metadata is captured. The IR serializes to JSON for consumption by downstream compiler backends.

Documentation is bilingual (English + Korean via mkdocs-static-i18n). Each doc file has `.md` (English) and `.ko.md` (Korean) variants. README.md is English; README.ko.md is Korean. Match existing conventions when editing docs.

## Commands

```bash
# Install dependencies (--extra dev for test/lint tools)
uv sync --extra dev

# Run all tests
uv run pytest tests/ -v

# Run a single test file
uv run pytest tests/test_executor.py -v

# Run tests matching a keyword
uv run pytest tests/test_comprehensive.py -k "attention" -v

# Comprehensive tests with report generation
uv run pytest tests/test_comprehensive.py --generate-reports --output reports/

# CLI test runner (alternative to pytest)
uv run python -m tests --output reports/
uv run python -m tests --list-models
uv run python -m tests --category attention

# Lint
uv run ruff check torch_ir/ tests/

# Lint with auto-fix
uv run ruff check --fix torch_ir/ tests/

# CLI
uv run torch-ir info model.json          # Show IR summary
uv run torch-ir visualize model.json     # Mermaid diagram to stdout
uv run torch-ir visualize model.json -o graph.png  # Render to image

# Build package (version derived from git tags via hatch-vcs)
uv build
```

## Architecture

### Pipeline

```
Model (meta device) → export_model() → convert_exported_program() → IR
                                                                       ├── save/load (JSON)
                                                                       ├── execute_ir() (with real weights)
                                                                       └── verify_ir() (compare vs original)
```

The pipeline flows through these modules in order:
- **exporter.py** — wraps `torch.export.export()`, validates model/inputs are on meta device
- **analyzer.py** — inspects FX graph nodes, extracts op attributes via PyTorch op schema introspection
- **converter.py** — transforms `ExportedProgram` into `IR` (list of `OpNode`s with `TensorMeta`)
- **executor.py** — runs the IR graph with real tensors; uses schema-based ATen fallback for automatic op execution
- **verifier.py** — compares IR execution output against original model output (tolerances: rtol=1e-4, atol=1e-4)
- **serializer.py** — JSON serialization/deserialization of `IR`
- **visualize.py** — Mermaid flowchart generation (`ir_to_mermaid`) and op distribution pie charts
- **cli.py** — `torch-ir` CLI (subcommands: `info`, `visualize`)

### Key Design Decisions

**Schema-based ATen fallback**: Instead of manually implementing each ATen op, the executor introspects `torch.ops.aten.*` schemas to call ops automatically. Custom execution is only needed for non-ATen ops (e.g., `getitem`). New ATen ops are supported without framework changes.

**Producer-consumer tracking**: Each `TensorMeta` stores `producer_node` and `producer_output_idx` to explicitly track which node produced it. This prevents tensor lookup ambiguity and ensures correct execution order.

**Weight-name mapping**: `IR.weight_name_mapping` maps FX graph placeholder names to original `state_dict` keys, so weights can be loaded separately from IR extraction.

### Core Data Structures (ir.py)

- `TensorMeta` — name, shape, dtype, producer_node, producer_output_idx
- `OpNode` — name, op_type (e.g. `aten.conv2d.default`), inputs/outputs as `TensorMeta`, attrs dict
- `IR` — nodes, graph_inputs, graph_outputs, weights, weight_name_mapping, constants, model_name

### Extension Points (ops/)

- `@register_op(op_type)` — register custom conversion logic (rarely needed)
- `@register_executor(op_type)` — register custom execution logic for non-ATen ops
- **registry.py** — operator registry system
- **aten_ops.py** — op type normalization utilities
- **aten_impl.py** — non-ATen op implementations (e.g., `getitem`)

### Public API (__init__.py)

Main entry points: `extract_ir()`, `execute_ir()`, `verify_ir()`, `verify_ir_with_state_dict()`, `save_ir()`/`load_ir()`, `IR.save()`/`IR.load()`, `ir_to_mermaid()`, `generate_op_distribution_pie()`

### Test Structure

- `tests/models/` — test model definitions organized by category (attention, multi_io, skip_connections, shared_weights), registered via `TestModelSpec` in `base.py`
- `tests/testing/` — test utilities: mermaid DAG visualization, statistics collection, markdown report generation, test runner orchestration
- `tests/test_*.py` — unit tests per module plus `test_comprehensive.py` for full integration

## Constraints

- **Static shapes only** — no SymInt / dynamic dimensions
- **No data-dependent control flow** — if/for must be statically resolvable
- **Meta tensor constants** — `torch.tensor()` in `forward()` is unsupported; use `self.register_buffer()` instead
- Model and inputs must be on meta device during extraction

## Style

- Python 3.10+, type hints throughout, `@dataclass` for data structures
- Line length: 120 (ruff)
- Lint rules: E, F, I, W (errors, pyflakes, import sorting, warnings)
- Custom exceptions: `ExportError`, `ConversionError`, `ExecutionError`, `SerializationError`, `WeightLoadError`
