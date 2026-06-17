#!/usr/bin/env bash
# Render the book with the executable-cell environment configured.
#
# Quarto loads matplotlib's inline backend at kernel startup; on Windows that
# breaks jaxlib's DLL init. The project IPython profile (.ipython) preloads JAX
# first, so all of NumPy / PyTorch / JAX execute. Pass any quarto args, e.g.:
#   ./render.sh
#   ./render.sh chapters/02-foundations/04-recurrence-convolution.qmd
set -euo pipefail
export IPYTHONDIR="$(cd "$(dirname "$0")" && pwd)/.ipython"
quarto render "$@"
