# Changelog

Notable changes to this book, newest first. Releases are tagged; the current
version is also in `CITATION.cff`.

## [0.1.0] — 2026-06-01

First public edition: the foundations of linear state space models and the
structured state space (S4) line, with companion reference code.

### Added
- Front matter: preface, acknowledgements, reader's guide, notation, and roadmap.
- Introduction (chapter 1) motivating the state space route to sequence
  modelling.
- Foundations (chapters 2–6): state as memory; continuous-time linear state
  spaces; discretisation; recurrence, convolution, and Toeplitz maps; and
  transfer functions and kernel generation.
- Structured state spaces and S4 (chapters 7–11): the memory problem; HiPPO;
  structured state matrices; the S4 kernel algorithm; and diagonal state spaces
  (DSS, S4D, S5).
- Companion reference code in `src/ssm_book` (NumPy, PyTorch, and JAX), checked
  by a test suite that cross-validates the three backends and the recurrence,
  convolution, and scan identities.
- Figures (matplotlib and TikZ), bibliography, and a Quarto site with shared
  maths macros and continuous deployment to GitHub Pages.
- Dual licensing: text, figures, and equations under CC BY 4.0; code under
  Apache 2.0.
