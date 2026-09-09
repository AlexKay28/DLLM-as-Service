# Paper v9 — primary-pair latency and quality

Start with [main.pdf](main.pdf). The PDF has **12 pages: eight main-text pages, one reference page, and three appendix pages**. The original font size and margins are retained.

The main comparison is Qwen3-30B-A3B versus DiffusionGemma-26B-A4B. V9 includes a direct reanalysis of 300 shared log events and generation-budget coverage from the saved timing runs. Section 5 defines the operational objective as correct resolution within a deadline, then explains accumulation of generation-time savings across dependent actions. A clearly labeled smaller-model timing illustration supports that explanation; its protocol and the other secondary experiments remain in the appendices. V8 and the original result artifacts remain unchanged.

[REVIEW.md](REVIEW.md) explains the evidence and editorial decisions; [VALIDATION.md](VALIDATION.md) records checks.

## Build and verify

Install a standard LaTeX distribution with pdfLaTeX/BibTeX, plus NumPy for the paired audit. Run:

```bash
make -C paper/v9
make -C paper/v9 verify
python3 paper/verify_numbers.py
```

The snapshot audit checks named numeric claims and document dependencies. The paired audit checks the frozen references, saved predictions, generated tables, and deterministic bootstrap output. The repository verifier checks its original macros and v7 prose, not the v9 rewrite. None of these checks certifies the experimental protocol.

## Reproduce the saved-output analysis

```bash
python3 paper/v9/audit_primary_pair.py --write
```

The tracked `data/bgl_reference_300.jsonl` contains the first 300 parser-derived records from the prepared test split. It was frozen using `--freeze-reference --write`, which requires the local prepared dataset and refuses to replace a different existing slice. Ordinary reproduction uses the tracked slice and does not need the full dataset.

`data/primary_pair_audit.json` records input hashes, per-field scores, bootstrap settings, and completion counts. The accompanying TeX files contain generated tables. The audit performs no inference, changes no metrics, and writes only version-local analysis products.

## Figures and scope

Install Matplotlib and run `make -C paper/v9 figures` to regenerate figures, then rebuild. The first-page figure is retained. The cumulative latency figure now labels the fractions completed by 1,000 ms and omits median-uncertainty shading. Older figure filenames are retained for stable artifact paths; numbered figures follow their new placement.

The quality analysis finds a small DG deficit, not demonstrated equality. Generation-budget coverage excludes queueing and downstream work. Appendix A preserves the protocol, paired analysis, and conflicting DG runtime metadata. Appendix B summarizes supporting measurements and ablations. Appendix C retains the code-quality limitation and protocol caveats. Removed historical detail remains accessible in Git history and the original result artifacts. All manuscript references point to retained sections.

`SUBMISSION_DISCLOSURES.tex` is an author-facing draft excluded from the PDF.
