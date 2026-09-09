# V9 validation

Validated on 9 September 2026.

- **Build:** `make -C paper/v9` completed using pdfLaTeX and BibTeX. The PDF has 22 pages. The final LaTeX log contains no warnings, undefined references, or overfull/underfull boxes.
- **Snapshot audit:** `python3 paper/v9/verify_snapshot.py` passed 133 selected numeric and document-integrity checks, using v9's own numeric bindings.
- **Repository audit:** `python3 paper/verify_numbers.py` completed with zero errors and zero precision violations. It reported 109 existing provenance warnings. Its prose checks target v7, so the separate v9 audit and editorial review are necessary.
- **Python syntax:** the original snapshot scripts and the new reader-figure builder passed `py_compile`.
- **Visual review:** inspected contact sheets of all revised pages and Figure 5 at larger resolution. Figure 1 appears directly below the abstract on page 1; the takeaways remain together on page 2. The new schematics and ratio annotations are legible, with no observed clipping. The abstract contains 150 whitespace-delimited words.
- **Figure data:** the new builder reproduced the median chain ratios from saved absolute timing summaries and checked agreement with the saved ratios. Tail ratios are plotted separately, without attaching median time labels to them. The two schematics are explicitly conceptual.
- **Preservation:** all 41 tracked v8 files were compared byte-for-byte with their committed versions and are unchanged. V9's numeric macros and bindings are identical to v8's; result artifacts are unchanged.
- **Scope:** no inference, new statistical experiments, or deployment measurements were performed. Quality matching, natural long-output validation, and end-to-end deadline success remain open evidence questions.
