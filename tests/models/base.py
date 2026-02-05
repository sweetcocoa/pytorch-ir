"""Base classes and registry for test models."""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Optional
import torch.nn as nn


@dataclass
class TestModelSpec:
    """Specification for a test model.

    Attributes:
        name: Unique identifier for the model.
        model_class: Factory function that creates the model.
        input_shapes: List of input tensor shapes (without batch dimension will be added).
        categories: List of categories this model belongs to.
        description: Human-readable description of the model.
    """
    name: str
    model_class: Callable[[], nn.Module]
    input_shapes: List[Tuple[int, ...]]
    categories: List[str]
    description: str


# Global registry of test models
MODEL_REGISTRY: Dict[str, TestModelSpec] = {}


def register_model(
    name: str,
    input_shapes: List[Tuple[int, ...]],
    categories: List[str],
    description: str = "",
):
    """Decorator to register a model class in the registry.

    Args:
        name: Unique name for the model.
        input_shapes: List of input shapes (batch dimension will be prepended).
        categories: Categories this model belongs to.
        description: Human-readable description.

    Returns:
        Decorator function.

    Example:
        @register_model(
            name="SelfAttention",
            input_shapes=[(16, 64)],  # (seq_len, d_model)
            categories=["attention"],
            description="Basic self-attention mechanism"
        )
        class SelfAttention(nn.Module):
            ...
    """
    def decorator(cls):
        spec = TestModelSpec(
            name=name,
            model_class=cls,
            input_shapes=input_shapes,
            categories=categories,
            description=description,
        )
        MODEL_REGISTRY[name] = spec
        return cls
    return decorator


def get_model(name: str) -> Optional[TestModelSpec]:
    """Get a model spec by name."""
    return MODEL_REGISTRY.get(name)


def list_models(category: Optional[str] = None) -> List[TestModelSpec]:
    """List all registered models, optionally filtered by category.

    Args:
        category: If provided, only return models in this category.

    Returns:
        List of TestModelSpec objects.
    """
    models = list(MODEL_REGISTRY.values())
    if category:
        models = [m for m in models if category in m.categories]
    return models


def list_categories() -> List[str]:
    """List all unique categories from registered models."""
    categories = set()
    for spec in MODEL_REGISTRY.values():
        categories.update(spec.categories)
    return sorted(categories)
