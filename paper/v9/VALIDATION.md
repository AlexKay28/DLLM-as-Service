# V9 validation

Validated on 9 September 2026 after appendix condensation and introduction spacing.

- **Build:** pdfLaTeX/BibTeX compilation completed. The main text ends on page 8, references occupy page 9, and appendices occupy pages 10–12. The final LaTeX log has no warnings, unresolved references, or overfull/underfull boxes.
- **Snapshot audit:** 104 selected numeric and document-integrity checks passed. The lower count reflects removed appendix inputs, figures, and references. Every remaining manuscript reference resolves, with no duplicate labels.
- **Paired audit:** `audit_primary_pair.py --check` reproduced the 300-event quality table, paired bootstrap, source hashes, and generation-budget counts. It reproduces the archived AR seed-42 F1 to its recorded precision. All events remain in the budget denominators.
- **Repository audit (preceding primary-pair revision):** zero errors and zero precision violations; 109 existing provenance warnings. Its prose checks target v7, so they do not validate v9's narrative or resolve the documented DG version conflict. Original inputs are unchanged by this editorial revision.
- **Python syntax:** the modified figure builder passed `py_compile`; the paired audit remains unchanged.
- **Visual review:** inspected the revised introduction and all three appendix pages at reading resolution. Original font size and margins are unchanged. The appendices have no spillover pages or observed clipping; the first-page and chain figures remain unchanged.
- **Chain evidence:** checked `run_chain_experiment.py`: prompts are precomputed and calls run serially, without answer-dependent prompt selection. The figure derives cumulative medians from the saved per-depth values and asserts agreement with saved median ratios. No tool time or successful-investigation rate is inferred from these timings.
- **Preservation:** v8 and original result artifacts are unchanged. Original numeric macros and bindings remain intact; new analysis products are confined to `paper/v9/data`.
- **Scope:** this revision adds CPU reanalysis of saved predictions and a paired bootstrap, not new inference or deployment experiments. It documents a small quality deficit and conflicting DG runtime metadata. Natural long-output generalization and end-to-end semantic success remain unvalidated.
