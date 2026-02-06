"""Test model definitions for comprehensive testing.

This module provides test models that cover various computational graph patterns:
- Multi-Input/Output models
- Skip connections (residual, dense)
- Shared weights
- Attention mechanisms
"""

# Import model definitions to register them
from . import attention, multi_io, shared_weights, skip_connections
from .base import MODEL_REGISTRY, TestModelSpec, get_model, list_categories, list_models, register_model

__all__ = [
    "TestModelSpec",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "list_models",
    "list_categories",
    # Side-effect imports (model registration)
    "attention",
    "multi_io",
    "shared_weights",
    "skip_connections",
]
