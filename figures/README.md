# Figures

Book-quality figures for *State Space Models*. They share one visual language
(the book's theme: blue-grey structure, orange accent, on white). SVG is the
production asset embedded in the chapters; a PNG is written alongside for review.

Two generators, one palette:

- **matplotlib** (`_src/`) — data plots, matrix/heatmap schematics, and
  complex-plane plots. Palette and helpers live in `_src/ssmstyle.py`.
- **TikZ** (`_tikz/`) — the pure node-and-arrow block diagrams, where TikZ gives
  cleaner arrows and self-loops. Palette mirrors `ssmstyle.py` in `_tikz/ssmtikz.tex`,
  so the two generators match.

## Layout

```
figures/
  fig-<chapter>-<n>-<slug>.svg   production asset (embedded in the .qmd)
  fig-<chapter>-<n>-<slug>.png   preview
  _src/                          matplotlib generators
    ssmstyle.py                  shared palette, typography, helpers
    fig_<...>.py                 one script per figure
  _tikz/                         TikZ generators (block diagrams)
    ssmtikz.tex                  shared preamble + palette (mirrors ssmstyle.py)
    fig_<...>.tex                one standalone file per figure
    build.ps1                    compile one .tex -> figures/<out>.svg + .png
  qa/                            contact sheets / review montages (gitignored)
```

Currently TikZ: `fig-2-1-bottleneck`, `fig-2-2-block`, `fig-3-2-convolution`,
`fig-5-1-recurrence-chain`, `fig-8-2-hippo-framework`.
Everything else is matplotlib.

## Numbering

`Figure <chapter>.<n>` — `n` is the figure's order within the chapter, matching
the book's manual numbering.

## Regenerate

matplotlib — each script writes its SVG + PNG into `figures/`:

```
python figures/_src/fig_6_1_roots.py
```

TikZ — needs a LaTeX engine. One was installed user-scoped with
`scoop install latex` (MiKTeX; bundles `pdflatex`, `dvisvgm`, `mgs`/ghostscript).
`build.ps1` adds MiKTeX to PATH itself, then renders SVG (text as paths) + PNG:

```
powershell -File figures/_tikz/build.ps1 -Name fig_2_2_block -Out fig-2-2-block
```

## Conventions

- Figures carry **no in-plot title** and **no explanatory sentences, pointer
  arrows, or em-dashes**; the teaching caption in the `.qmd` carries the words.
  Keep only labels that *are* the figure: axis labels, legends, symbols, a
  defining equation or transform label.
- Palette lives only in `ssmstyle.py` / `ssmtikz.tex` (mirrors `styles/theme.scss`);
  never hard-code off-brand colours.
- `ssmstyle.text_on(fill)` picks a legible label colour for any cell fill.
- matplotlib: schematic / matrix / complex-plane diagrams use `ssmstyle.blank_axes`;
  data plots use `ssmstyle.clean_axes`.
