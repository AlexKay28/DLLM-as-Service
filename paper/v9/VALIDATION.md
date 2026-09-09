# V9 validation

Validated on 9 September 2026 after the primary-pair revision.

- **Build:** pdfLaTeX/BibTeX compilation completed. The main text ends on page 7; the full PDF has 22 pages including references and appendices. The final LaTeX log has no warnings, unresolved references, or overfull/underfull boxes.
- **Snapshot audit:** 136 selected numeric and document-integrity checks passed.
- **Paired audit:** `audit_primary_pair.py --check` reproduced the 300-event quality table, paired bootstrap, source hashes, and generation-budget counts. It reproduces the archived AR seed-42 F1 to its recorded precision. All events remain in the budget denominators.
- **Repository audit:** zero errors and zero precision violations; 109 existing provenance warnings. Its prose checks target v7, so they do not validate v9's narrative or resolve the documented DG version conflict.
- **Python syntax:** the new audit and modified figure builder passed `py_compile`.
- **Visual review:** inspected contact sheets of all 22 pages. The main comparison, supporting experiments, and references occupy distinct parts of the PDF. The first-page summary, quality table, budget table, and annotated cumulative plot are legible without observed clipping.
- **Preservation:** v8 and original result artifacts are unchanged. Original numeric macros and bindings remain intact; new analysis products are confined to `paper/v9/data`.
- **Scope:** this revision adds CPU reanalysis of saved predictions and a paired bootstrap, not new inference or deployment experiments. It documents a small quality deficit and conflicting DG runtime metadata. Natural long-output generalization and end-to-end semantic success remain unvalidated.
