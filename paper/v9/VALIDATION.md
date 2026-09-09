# V9 validation

Validated on 9 September 2026.

- **Build:** `make -C paper/v9` completed using pdfLaTeX and BibTeX. The PDF has 22 pages. The final LaTeX log contains no warnings, undefined references, or overfull/underfull boxes.
- **Snapshot audit:** `python3 paper/v9/verify_snapshot.py` passed 131 selected numeric and document-integrity checks, using v9's own numeric bindings.
- **Repository audit:** `python3 paper/verify_numbers.py` completed with zero errors and zero precision violations. It reported 109 existing provenance warnings. Its prose checks target v7, so the separate v9 audit and editorial review are necessary.
- **Python syntax:** all three snapshot scripts passed `py_compile`.
- **Visual review:** inspected contact sheets of all pages and the final opening pages at larger resolution. The three takeaways remain together, figures and tables are legible, and no clipping was observed. The abstract contains 158 whitespace-delimited words.
- **Preservation:** all 41 tracked v8 files were compared byte-for-byte with their committed versions and are unchanged. V9's numeric macros and bindings are identical to v8's; result artifacts are unchanged.
- **Scope:** no inference, new statistical experiments, or deployment measurements were performed. Quality matching, natural long-output validation, and end-to-end deadline success remain open evidence questions.
