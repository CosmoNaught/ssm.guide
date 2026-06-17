# Contributing

This book is developed openly. Corrections, clearer explanations, worked
examples, diagrams, and reference implementations are all welcome. If a
derivation is unclear or a step is missing, that is worth raising even if you
cannot fix it yourself.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- **Errors.** Typos, broken maths, wrong references, dead links, rendering
  glitches. Small fixes can go straight to a pull request.
- **Clarity.** A definition that does not land, a jump between steps, a missing
  piece of motivation. Say where you got lost; that is the useful part.
- **Examples and figures.** Worked numerical examples, or a diagram for a place
  where a formula needs a picture — open an issue to agree what is needed before
  drawing it.
- **Implementations.** Reference implementations in PyTorch or JAX that match a
  chapter's notation.

For anything larger than a fix — a new section, a reorganisation, a figure set —
please open an issue first so the approach can be agreed before you write it.
What is planned next is listed in the book's
[Roadmap](chapters/00-front/roadmap.qmd).

## Building the book

The rendered site *is* the book. Building it locally needs
[Quarto](https://quarto.org/docs/get-started/) (1.4 or newer):

```bash
quarto preview     # live site with hot reload (Quarto prints the local URL)
quarto render      # full build into _book/
```

No Python or other runtime is required: the chapters are prose and mathematics.
If you change anything, run `quarto render` once before opening a pull request
and confirm the build is clean.

> On Windows, a full `quarto render` can fail to clear `_book/` if a preview
> server or a browser tab still holds it open. Stop the preview first, or render
> the single file you changed with `quarto render <path>`.

## Repository layout

The source lives under `chapters/`; everything else is configuration and assets.
The parts you are most likely to touch:

```
chapters/
├── 00-front/            preface, acknowledgements, reader's guide, notation, roadmap
├── 01-introduction/     Chapter 1 (index.qmd)
├── 02-foundations/      Chapters 2–6 (one .qmd per chapter)
├── 03-s4/               Chapters 7–11 (one .qmd per chapter)
└── references.qmd
```

### Where new chapters go

Each thematic group is a folder `chapters/NN-slug/` holding one numbered
`.qmd` file per chapter. The directories planned next (see the roadmap) will be:

- `chapters/04-selective/` — Selective state spaces
- `chapters/05-duality/` — Matrix views and state space duality
- `chapters/06-frontier/` — Modern refinements

A chapter appears on the site only once it is listed under
`website.sidebar.contents` in [`_quarto.yml`](_quarto.yml); work in progress is
left out of that list until it is ready.

## Conventions

- **British English** throughout: *discretise*, *diagonalise*, *normalise*,
  *parameterise*, *colour*. The book's register is mathematical, calm, and
  cumulative — not hype or sales copy. State the problem first, then the object,
  then its name, then the acronym.
- **Sections** carry an explicit anchor, e.g. `## 2.1.1 Title {#sec-2-1-1}`, so
  the sidebar and cross-links can point at them. Keep the numbering consistent
  with the chapter.
- **Mathematics** renders through MathJax. Shared macros live in
  [`_mathjax-config.html`](_mathjax-config.html); the same definitions are kept
  in [`_macros.tex`](_macros.tex) for the LaTeX build. **Add any new macro to
  both files.**
- **Citations** use `@key` against [`references.bib`](references.bib). Cite the
  source of a model or a key mathematical result; keep incidental references
  out.

## Pull requests

- Keep each pull request focused on one thing; small reviews are faster reviews.
- Describe what changed and why. If it fixes an issue, link it.
- Confirm `quarto render` succeeds and the affected pages look right.
- Do not commit `_book/` or `.quarto/` — both are ignored build output.

## Licensing of contributions

The book is dual-licensed: prose, figures, and equations under
[CC BY 4.0](LICENSE), and code under [Apache 2.0](LICENSE-CODE). By contributing
you agree that your contribution is released under the same terms.
