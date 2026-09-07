# Paper v8 — editorial and evidence review

Start with [main.pdf](main.pdf). [REVIEW.md](REVIEW.md) explains the material corrections and remaining evidence gaps.

This revision is based on **v7 at repository commit `d4ffeac`**. All manuscript text, bibliography, numeric macros, and compiled figures are version-local. No earlier manuscript version, experiment implementation, or raw result was changed. The top-level `paper/Makefile` still targets the existing working manuscript; use this version's Makefile to build v8.

## Build

With Tectonic installed:

```bash
make -C paper/v8
```

Alternatively, with a standard LaTeX installation, run `pdflatex`, `bibtex`, then `pdflatex` twice from `paper/v8/`.

The figure PDFs are included. Regeneration requires Python, NumPy, Matplotlib, and this repository's committed `results/` directory:

```bash
make -C paper/v8 figures
make -C paper/v8
```

## Verify

```bash
python3 paper/v8/verify_snapshot.py
python3 paper/verify_numbers.py
```

The first command audits selected **v8** statistics, energy arithmetic, and local document dependencies. The existing repository verifier checks its original shared macros and provenance mappings; it does not audit v8's prose. Neither certifies scientific validity or reproduces GPU inference.

`numeric_bindings.json` maps newly introduced numeric macros to their saved sources. The PDF's code-generation results and long-output projections carry explicit limitations.

`SUBMISSION_DISCLOSURES.tex` is an author-facing draft, excluded from the compiled paper. Adapt disclosures, author metadata, anonymity, page limits, and formatting to the chosen venue before submission. The compiled paper includes a truthful statement of LLM-assisted editorial work.
