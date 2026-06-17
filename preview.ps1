# Live-preview the book; forwards args to quarto preview.
# Sets IPYTHONDIR to the project profile (.ipython) so JAX is preloaded for executable cells.
$env:IPYTHONDIR = "$PSScriptRoot\.ipython"
quarto preview @args
