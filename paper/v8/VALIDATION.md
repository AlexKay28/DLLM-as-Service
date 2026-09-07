# Validation — v8

Validated on 7 September 2026.

- **PDF:** `main.pdf`, 20 pages, built with Tectonic 0.17.0. All pages have extractable text.
- **LaTeX / bibliography:** no overfull or underfull boxes, undefined references/citations, or font-substitution warnings in the final build; BibTeX completed successfully.
- **Visual review:** inspected a rendered contact sheet of all pages and reviewed the scientific figures. Final minor typography changes were rebuilt with no warnings.
- **Version-specific audit:** `python3 paper/v8/verify_snapshot.py` passes **131** selected numeric and document-integrity checks. These include saved-event descriptive statistics, corrected energy arithmetic, serial-sweep means, HumanEval baseline values, figure dependencies, labels, and citation keys.
- **Helper scripts:** all three Python scripts pass `py_compile`.
- **Existing repository audit:** `python3 paper/verify_numbers.py` passed with **0 errors and 109 pre-existing warnings** during the baseline review. That verifier checks shared files, not v8 prose; those shared files were not changed.
- **Isolation:** Git reports only the new `paper/v8/` directory. V7, the shared working manuscript, experiment code, and saved measurements are unchanged from the fetched repository revision.
- **Experiments:** no GPU inference, generated-code execution, or new data collection was performed. Statistical summaries were checked against stored event data; bootstrap/permutation experiments were not rerun.

The review documents unresolved scientific limitations; passing these checks is not proof of model quality equivalence, causal attribution, or venue compliance.

PDF SHA-256: `5d1cd3677d12c6d92983027de2bf0c46c97779bab4b97382b6c9809865ab1a0c`
