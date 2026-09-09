# Paper v9 — reader-focused presentation

Start with [main.pdf](main.pdf). This revision preserves v8's results and scientific scope while clarifying terminology, the opening narrative, and the motivation for timely log extraction. It is based on v8 at `50f9bc0`; earlier versions and result artifacts are unchanged.

[REVIEW.md](REVIEW.md) records the editorial changes and storyline checklist. [VALIDATION.md](VALIDATION.md) records checks performed on this snapshot. The evidence corrections documented in [v8's review](../v8/REVIEW.md) still apply.

## Build and verify

With a standard LaTeX installation (pdfLaTeX and BibTeX):

```bash
make -C paper/v9
python3 paper/v9/verify_snapshot.py
python3 paper/verify_numbers.py
```

The snapshot has its own text, bibliography, numeric macros, and figure PDFs. The v9 verifier reads its own `numeric_bindings.json` and checks selected claims against the shared committed results. The repository verifier checks its original shared macros; neither certifies scientific validity or reproduces inference.

The inherited figure filenames retain their v8 suffixes where their plotted data are unchanged. Figure 5 now emphasizes chain-latency ratios, and two conceptual diagrams explain decoding and deadlines. To regenerate all figures, install NumPy and Matplotlib and run `make -C paper/v9 figures`, then rebuild the paper. To regenerate only the chain plot and new diagrams, run `python3 paper/v9/make_reader_figures.py`.

`SUBMISSION_DISCLOSURES.tex` remains an author-facing draft excluded from the PDF. Venue formatting and author review remain separate from this editorial revision.
