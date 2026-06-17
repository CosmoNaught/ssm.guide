"""Minimal JAX ports of the NumPy reference code.

64-bit precision is enabled on import so the JAX results match the NumPy
reference (and the PyTorch port) to tight tolerance.
"""
import jax

jax.config.update("jax_enable_x64", True)
