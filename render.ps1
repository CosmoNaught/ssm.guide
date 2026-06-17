# Render the book with the executable-cell environment configured.
#
# Quarto loads matplotlib's inline backend at kernel startup; on Windows that
# breaks jaxlib's DLL init. The project IPython profile (.ipython) preloads JAX
# first, so all of NumPy / PyTorch / JAX execute. Pass any quarto args, e.g.:
#   ./render.ps1                     # whole site
#   ./render.ps1 chapters/02-foundations/04-recurrence-convolution.qmd
$env:IPYTHONDIR = "$PSScriptRoot\.ipython"
quarto render @args
