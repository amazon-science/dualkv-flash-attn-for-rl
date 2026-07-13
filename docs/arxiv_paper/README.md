# arXiv Paper Source

Exact LaTeX source for the DualKV paper as submitted to arXiv:
[arXiv:2605.15422](https://arxiv.org/abs/2605.15422).

`arxiv_submission.tar.gz` is the verbatim submission bundle; the extracted files here are its contents.

## Contents

- `main.tex` — paper source
- `main.bbl` — compiled bibliography (arXiv builds from this; no BibTeX pass needed)
- `neurips_2026.sty` — style file
- `figures/` — figure PDFs
- `main.pdf` — reference build (26 pages)
- `arxiv_submission.tar.gz` — the exact bytes uploaded to arXiv

## Rebuild

arXiv ships `main.bbl`, so no BibTeX step is required:

```bash
pdflatex main.tex
pdflatex main.tex   # second pass resolves cross-references
```
