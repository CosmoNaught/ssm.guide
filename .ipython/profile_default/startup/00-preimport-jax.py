# Kernel startup hook for the book's executable cells.
#
# On this Windows setup, importing matplotlib or Pillow before jaxlib breaks
# jaxlib's DLL initialisation. Quarto loads matplotlib's inline backend at
# kernel start, so we preload JAX here first; once jaxlib has registered its
# DLL directories, matplotlib can load alongside it without conflict.
try:
    import jax  # noqa: F401
except Exception:
    pass
