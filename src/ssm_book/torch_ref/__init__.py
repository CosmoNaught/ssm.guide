"""Minimal PyTorch ports of the NumPy reference code (complex128 throughout)."""

import warnings

# PyTorch wheels built against a different NumPy ABI emit a cosmetic
# "Failed to initialize NumPy" warning; we never use that bridge (see
# ssm_book.common.to_numpy), so silence it to keep executed book cells clean.
warnings.filterwarnings("ignore", message="Failed to initialize NumPy")

