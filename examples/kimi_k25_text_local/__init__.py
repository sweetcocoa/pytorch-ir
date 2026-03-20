"""Local text-only Kimi-K2.5 model package for static IR extraction."""

from .model import CausalLMOutputWithPast, KimiK25TextForCausalLM

__all__ = [
    "CausalLMOutputWithPast",
    "KimiK25TextForCausalLM",
]
