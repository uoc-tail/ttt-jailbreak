# Project page

Static project page for *Test-Time Training Undermines Safety Guardrails*,
served from this folder via GitHub Pages.

## Layout

```
docs/
  index.html         single-page site, mirrors the NeurIPS poster
  style.css          poster-derived palette + responsive layout
  .nojekyll          tells GitHub Pages to skip Jekyll processing
  static/
    teaser.png       attack-overview diagram (paper/figures/teaser.pdf)
    cross_category.png
    defense_scatter.png
    poster.png       downscaled (2000px) poster preview
    poster.pdf       full-resolution poster
    paper.pdf        compiled paper PDF
    og_preview.jpg   social card (1200x630)
```

## Enable GitHub Pages

In the repo settings:

1. **Settings → Pages**
2. **Source**: *Deploy from a branch*
3. **Branch**: `main`, folder `/docs`

The page will be served at `https://uoc-tail.github.io/ttt-jailbreak/`.

## Updating

- **Bibliography / arXiv link.** `index.html` has two `TODO` markers (the arXiv
  badge `href`/text and the BibTeX `eprint` field). Replace once the arXiv ID
  is assigned.
- **Paper PDF.** Re-copy `paper/main.pdf` into `static/paper.pdf` after each
  paper rebuild.
- **Figures.** Re-render with `pdftocairo -png -r 200 -transp <pdf> <name>`.
- **Poster.** When the poster changes, regenerate `static/poster.png` from
  `poster.pdf`/`poster.png` at the repo root (target ~2000px wide).
