# The Interrogation — 10 Questions

Answer each honestly. These choices shape every section of the paper, the
BACKBONE.md claim ledger, and the decision procedure's design. There are no
wrong answers — but there are answers that are inconsistent with each other,
and finding those inconsistencies is the point.

---

## Q1. What is the paper's primary contribution?

- **a)** The two cost laws (AR linear, DG staircase) and the crossover they produce
- **b)** The decision procedure that turns the laws into a deployable rule
- **c)** The five-defect checklist showing comparison harnesses fail asymmetrically
- **d)** The negative result: dLLMs have no surviving deployment advantage at the quality-matched pair

## Q2. The "shape of cost" distinction — what causes it?

- **a)** Diffusion vs autoregression (the decoding mechanism)
- **b)** Canvas-based vs token-by-token (the output structure, regardless of mechanism)
- **c)** Prefill-dominated vs decode-dominated (where the compute goes, regardless of either)
- **d)** We don't know — we measured the shape but did not isolate the cause (same logic as the batching retitle)

## Q3. The cost law is two-variable (T depends on L_in and L_out). How prominent should this be?

- **a)** It is THE central finding — it kills the flagship application claim and reframes the entire cost-law section
- **b)** It is a scope condition stated in §1 and §4, but the headline is still the output-length invariance at short input
- **c)** It is a threat to validity mentioned in §9, alongside single-corpus and single-GPU limitations
- **d)** It is a future-work item — the one-variable law is good enough for the decision procedure

## Q4. The decision procedure almost always recommends AR. Is that a feature or a bug?

- **a)** Feature — the paper's value is precisely that it tells you when NOT to use diffusion, and that's almost always
- **b)** Bug — we need to find the regime where DG wins and make that the headline, or the procedure is a negative result dressed as a tool
- **c)** Neither — the procedure is a *methodology contribution* (here is how to decide), not an *outcome* (here is what to decide)
- **d)** The procedure needs more inputs (e.g., cold-start budget, memory budget) to find the DG regime — currently it's incomplete

## Q5. The CHECKLIST.md (5 defects, 4 biased against AR) — where does it sit in the paper?

- **a)** It IS the paper — the cost laws and decision procedure are supporting evidence for the methodological finding
- **b)** It is a major section (§2, "how we got it wrong five times") on equal footing with the cost laws (§4)
- **c)** It is a methodology section that precedes the results, framing why the harness can be trusted
- **d)** It is an appendix — interesting but not load-bearing for the paper's thesis

## Q6. The quality-matched pair is AR-1.5B (3 GB) vs DG (52 GB). DG is 17× larger and still loses on quality. What does the paper say?

- **a)** State it plainly: "the quality-matched AR model is smaller, faster, and lower-energy. There is no deployment regime favouring diffusion." Full negative result.
- **b)** State it but caveat: "at the current canvas size and K; canvas=128 or future smaller dLLMs may change this"
- **c)** Don't headline it — the paper compares AR-3B vs DG (the size-matched pair), and the quality-matched comparison is a footnote
- **d)** Make it the motivation for the decision procedure: "given that the naive comparison is unfair in both directions, here is how to do it right"

## Q7. The DG base is 93% prefill, 6.5% denoising. The "denoising" part that makes DG special is nearly free. Does this change the paper's framing?

- **a)** Yes — the paper should say "diffusion's cost is dominated by the same prefill as AR; the denoising loop is a minor increment. The cost shape difference is structural (canvas) not algorithmic (diffusion)."
- **b)** No — the cost shape is still different (staircase vs linear), regardless of what the base is made of
- **c)** Partially — it weakens the "diffusion is fundamentally different" narrative but strengthens the "canvas is a tunable knob" angle (except we showed it's barely tunable)
- **d)** It's an interesting decomposition but doesn't change the framing — the decision procedure only cares about total T, not its components

## Q8. If the HDFS generalization (#67) shows the staircase does NOT hold on a second corpus, what happens?

- **a)** The cost-law claim is downgraded from "law" to "observation on BGL" — the paper still stands but is weaker
- **b)** The paper cannot be published in its current form — the "law" framing is load-bearing
- **c)** The paper drops the "law" language entirely and presents the cost shapes as empirical observations, which is fine
- **d)** We re-scope to "DiffusionGemma-26B on BGL" and state the validity domain narrowly — honest but narrow

## Q9. What kind of paper do you actually want to hand a reviewer?

- **a)** A systems paper: "Here is a measurement framework, a decision procedure, and a checklist. Apply it to your own architecture comparison."
- **b)** A negative result paper: "We tested every claimed advantage of dLLMs for structured extraction and found none survive a fair comparison."
- **c)** A cost-law paper: "AR and dLLM have different cost shapes; here are the laws, the crossover, and the deployment rule."
- **d)** A methodology paper: "Comparison harnesses fail asymmetrically. Here is the measured size of each failure, a checklist to prevent it, and a case study (AR vs dLLM) where it mattered."

## Q10. The title currently implies the batching cap is architectural. We retitled (EDR-004). What should the new title communicate?

- **a)** The cost shapes differ, and the shape determines deployment: "The Cost Shape of Diffusion: When It Wins and Why It Usually Doesn't"
- **b)** The methodology finding is the headline: "Five Ways to Bias an Architecture Comparison: A Case Study of AR vs Diffusion LLMs"
- **c)** The decision procedure is the headline: "A Decision Procedure for Choosing Between Autoregressive and Diffusion LLMs in Structured Extraction"
- **d)** The negative result is the headline: "No Regime for Diffusion: A Quality-Matched Cost Comparison of AR and dLLM Serving"

---

## How to answer

For each question, pick the letter that is *closest* to what you believe —
not the one that is perfectly right. If none feel right, say so. The goal is
to surface disagreements between what the data says and what you want it to
say, because those disagreements are where the paper's honesty is tested.
