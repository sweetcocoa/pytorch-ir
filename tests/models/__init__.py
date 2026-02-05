"""Test model definitions for comprehensive testing.

This module provides test models that cover various computational graph patterns:
- Multi-Input/Output models
- Skip connections (residual, dense)
- Shared weights
- Attention mechanisms
"""

from .base import TestModelSpec, MODEL_REGISTRY, register_model, get_model, list_models, list_categories

# Import model definitions to register them
from . import multi_io
from . import skip_connections
from . import shared_weights
from . import attention

__all__ = [
    "TestModelSpec",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "list_models",
    "list_categories",
]
