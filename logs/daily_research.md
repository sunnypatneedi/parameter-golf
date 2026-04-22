# Parameter Golf Daily Research - 2026-04-19

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed)

valerio-oai comment (2026-03-27): "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." Train-then-score ordering is permanently disqualified. No appeal path.

---

## N-GRAM PR STATUS

| PR | Claimed BPB | Status | Notes |
|----|-------------|--------|-------|
| #727 | 0.9674 | **CLOSED (ILLEGAL)** | valerio-oai: target token in hash key = leaks eval tokens |
| #741 | — | **CLOSED (ILLEGAL)** | Same n-gram hash violation |
| #758 | 1.0465 | **OPEN (effectively dead)** | MatoTeziTanka (Apr 12): XOR hash key includes target token — same ruling as #727. Author hasn't responded. |
| #731 | 1.0400 | **OPEN — 1 seed only** | Reviewer "LOOKS CLEAN" (score-first-per-chunk, dense count + Laplace, no hashing). Seeds 1337 and 2024 still pending. No movement since Apr 17. |

---

## Leaderboard

| | Score | Author | Date |
|--|-------|--------|------|
| **Merged SOTA** | **1.0810** | bigbag (PR #1493) | 2026-04-09 |
| Best open (no CaseOps/SLOT) | **1.00995** | arsenis-cmd (PR #1698) — ⚠️ ARTIFACT BUG + BPB BUG: effectively dead | |
| Best open (CaseOps pending ruling) | **1.03540** | alertcat (PR #1738) — ⚠️ also builds on pre-quant TTT from PR #1735 (illegal) | |
| Best open (CaseOps, dexhunter) | **1.06549** | dexhunter (PR #1736) — clean, no legality flags | |
| Best open (CaseOps + casefold pending) | 1.05733 | dexhunter (PR #1693) — casefold ruling still pending | |
| Best open (SLOT risk) | 1.0616 | powerpratik (PR #1647) | |
| Best open (clean, no CaseOps) | **1.07139** | MarioPaerle (PR #1667) — Attn Output Gate + SmearGate | |
| Our PR #771 | 1.0705 | sunnypatneedi | CLOSED/ILLEGAL |

**Day 10 plateau** — no new merges since PR #1493 on Apr 9 (new record for longest plateau).

---

## What Changed (GitHub — Apr 17–19, 2026)

### New PRs filed since Apr 17

| PR | Author | BPB | Technique | Status | Notes |
|----|--------|-----|-----------|--------|-------|
| #1743 | OleStan | — | FreqGPTQ + GatedDeltaNet + Adaptive Quant | Open WIP | Watch |
| #1738 | alertcat | **1.03540** | CaseOps V15 + PR #1735 Pre-Quant TTT | Open | ⚠️ Builds on illegal PR #1735; no reviews yet |
| #1736 | dexhunter | **1.06549** | CaseOps + GatedAttn + QuantGate + SP8192 | Open | **CLEANEST new PR** — no legality flags, await CaseOps ruling |
| #1735 | AjAnubolu | 1.0429 | Pre-Quant AdamW TTT 21ep | Open | **⚠️ LIKELY ILLEGAL** — dexhunter flagged adapt-then-score, same pattern as PR #1351/#1416 |
| #1732 | Victory963 | 1.0785 | Hadamard Rotation + AWQ + Parallel Residuals | Open | New quant approach, no reviews |
| #1729 | romeerp | 1.0678 | CaseOps (bijective) + Tapered WD | Open | Bijective/reversible; no legality flags |
| #1727 | yahya010 | **1.07217** | MP-SGD TTT 4 phases + QK-Gain 5.25 | Open | **Appears legal** — score-first each phase, explicit compliance notes |
| #1739 | DevelopedByAnurag | 1.1497 | SP8192 + Depth Recurrence + Muon 0.99 | Open | Low interest |

### PR #1698 (GDN FLA, arsenis-cmd): Now effectively dead
- **BPB bug confirmed** by @dexhunter: `build_sentencepiece_luts` double-count inflates byte denominator ~17.7%. Corrected actual score: ~**1.189 BPB**, not 1.00995.
- **Artifact size violation confirmed**: Seeds range 16,474,250–16,600,916 bytes vs 16,000,000-byte decimal limit.
- No organizer response. Author lost GPU access. Do NOT track this PR.

### Key new technique: CaseOps Tokenizer
Three PRs (#1729 romeerp, #1736 dexhunter, #1738 alertcat) use a **bijective case-factoring transform** distinct from casefold:
- Encodes capitalization as control tokens: `TITLE`, `ALLCAPS`, `CAPNEXT`, `ESC`
- Fully reversible — original text reconstructs exactly
- BPB is scored on **original UTF-8 bytes** via a byte sidecar file, not on transformed tokens
- Stronger legality argument than casefold (which destroys information)
- **Issue #1604 ruling still pending** from @valerio-oai — do NOT implement until ruled legal

### PR #1698 successor: PR #1736 (dexhunter) is the cleanest GDN-free path
At 1.06549 with CaseOps + GatedAttn + QuantGate, if CaseOps gets ruled legal, dexhunter's stack is the primary implementation target.

### Existing priority PRs — status unchanged since Apr 17
| PR | Author | BPB | Status |
|----|--------|-----|--------|
| #1586 | dexhunter | 1.07493 | **OPEN, no reviews** — IMPLEMENT IMMEDIATELY |
| #1667 | MarioPaerle | 1.07139 | **OPEN, no reviews** — stack on #1586 |
| #1693 | dexhunter | 1.05733 | **OPEN** — casefold ruling pending (Issue #1604) |
| #1647 | powerpratik | 1.0616 | **OPEN** — SLOT-4, high risk |

---

## New Research Papers

Nothing transformative found for April 2026. Best candidates:

### arXiv:2604.13552 — Training-Free TTT Contrastive Learning (Apr 2026)
Adapts LLMs at test time via contrastive loss (positive/negative token pairs from context), no labeled data. Not directly applicable to score-first per-token BPB — our TTT is tightly constrained by the score-first rule. Low priority.

### arXiv:2604.09624 — SECL: Self-Calibrating LMs (Apr 2026)
Updates only on high-uncertainty tokens (6–26% of stream). Could reduce our TTT compute budget and allow more epochs within 10-min eval budget. Worth reading if MP-SGD TTT 4-phase approach (PR #1727) doesn't fit in time budget.

### arXiv:2412.06464 — GatedDeltaNet (ICLR 2025) — remains key reference
O(n) recurrence combining delta rule + gating. PR #1743 (WIP) is pursuing this. BPB bug pattern has killed every open GDN PR so far (#1576, #1687, #1698) — verify byte-counting before any GDN investment.

---

## HuggingFace / Community Discoveries

- **CaseOps bijective transform** is a new community pattern: three independent PRs from different authors all converged on bijective case-factoring in the last 2 days. Strong signal this is a real technique.
- **MP-SGD TTT 4 phases (PR #1727)** extends earlier 3-phase approach (PR #1700) — yahya010 found a way to fit a 4th phase in eval budget. Stack-compatible.
- **Pre-quant TTT pattern continues to be submitted** — PR #1735 is the latest. Pattern is clear: authors keep trying, dexhunter keeps flagging it. Do not implement.

---

## Recommended Actions (priority order)

1. **IMPLEMENT PR #1586 immediately** — Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 emb (15σ) + MLR=0.026. Config-level change, zero legality risk, -0.013 nats. 11 days to deadline. This is overdue.

2. **STACK PR #1667 on #1586** — Attention Output Gate (1,056 params, init to zero) + SmearGate (width=12). Expected combined delta: ~-0.019 nats total.

3. **ADD PR #1727 technique (MP-SGD TTT 4 phases)** — Appears legal (score-first per phase), +1 phase vs PR #1693 base. yahya010 reaches 1.07217 BPB. Stack with #1586+#1667.

4. **AWAIT Issue #1604 ruling on CaseOps** — If bijective CaseOps ruled legal (stronger case than casefold), dexhunter's PR #1736 at 1.06549 becomes primary target. CaseOps is reversible and BPB-accurate via sidecar; may pass where casefold fails.

5. **DO NOT IMPLEMENT**: Pre-quant TTT (#1735/#1738), casefold (#1693 component) without ruling, SLOT (#1647) without explicit risk acceptance, anything from GDN PRs until BPB bug pattern is resolved.

6. **AWAIT PR #731** — 3-seed Hedge Mixer still pending seeds 1337/2024. If merged, legal n-gram mixer available.

---

_Updated: 2026-04-19 (v14.0 — PR #1698 GDN effectively dead (BPB bug ~1.189 actual + artifact violation); CaseOps bijective tokenizer emerging as new community technique (#1729, #1736, #1738); PR #1735 pre-quant TTT flagged illegal; PR #1727 MP-SGD TTT 4-phase appears legal; merged SOTA 1.0810 Day 10 plateau; 11 days to deadline)_

---

# Parameter Golf Daily Research - 2026-04-20

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed, no change)

Rejected by @valerio-oai 2026-03-27. Train-then-score AdamW TTT 30ep on val tokens. No new comments.

## N-GRAM PR STATUS

- **PR #727** (0.9674): CLOSED — @valerio-oai ruled n-gram hash caches without proper renormalization illegal. Permanent.
- **PR #758** (1.0465): OPEN but EFFECTIVELY DEAD — MatoTeziTanka flagged Apr 12: XOR hash key incorporates target token (same violation as #727). Neural base only ~1.10–1.15 BPB. No fix submitted.
- **PR #731** (1.0400): OPEN — Reviewer says "LOOKS CLEAN" (dense count tables + Laplace smoothing, score-first per chunk). Seeds 1337 and 2024 still NOT reported as of Apr 20.

## Leaderboard

- **Merged SOTA**: 1.0810 (bigbag, PR #1493) — **DAY 11 PLATEAU**, now the longest in competition history (Apr 9 → Apr 20). 10 days to deadline (Apr 30).
- **Best open (legal, no CaseOps)**: 1.07139 (MarioPaerle, PR #1667, Attention Output Gate + SmearGate)
- **Best open (legal, incl. CaseOps if ruled legal)**: 1.06549 (dexhunter, PR #1736)
- **Our PR #771**: 1.0705 — CLOSED (illegal)

## What Changed (GitHub — Apr 20)

**New PRs filed today:**

- **PR #1751** (Pravin-dev06): Parallel-Residual + SwiGLU + 11 layers — Non-record. Best: 1.3565 BPB (not competitive).
- **PR #1750** (teslaeco): SP8192 + 3-layer recurrence + parallel residuals + legal score-first TTT — 1.08089 (3-seed mean, seeds 42/314/999). Replicates merged SOTA but does NOT beat it. No new technique.
- **PR #1749** (gracebml): GDN-Hybrid + Legal Score-First TTT + Full-Hessian GPTQ Int6 — 1.0996 (single seed, 28% of 8xH100 budget on 1xH100). Artifact 14.03 MB. **Not yet competitive; needs full 8xH100 run for valid score. Monitor.**
- **PR #1748** (elad-simbalista): Basic baseline improvement — not competitive.
- **PR #1747** (swapp1990): SP8192 + Partial RoPE (16/64) + GPTQ SDClip + SGD TTT — 1.0820 (3-seed). Worse than merged SOTA.
- **PR #1744** (MuhammedErinArchitecture): SP8192 + QK5 + Freeze10 Loss-Gated Legal TTT — 1.08886 (single seed). Not competitive.

**Key open PRs — no status change since Apr 19:**

| PR | Author | Val BPB | Technique | Status |
|----|--------|---------|-----------|--------|
| #1586 | dexhunter | 1.07493 | Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 Emb@15σ + MLR=0.026 | OPEN, no reviews |
| #1667 | MarioPaerle | 1.07139 | Attention Output Gate (1,056 params) + SmearGate (w=12) | OPEN, no reviews |
| #1727 | yahya010 | 1.07217 | MP-SGD TTT 4 phases (score-first per phase) | OPEN, no reviews |
| #1560 | dexhunter | 1.07406 | VarLen Attention (per-doc masking) + Doc-TTT (LoRA chunk=48) | OPEN, no reviews |
| #1736 | dexhunter | 1.06549 | CaseOps bijective + GatedAttn + QuantGate | OPEN, awaits Issue #1604 ruling |
| #1735 | AjAnubolu | 1.0429 | Pre-quant AdamW TTT 21ep | OPEN — **LIKELY ILLEGAL** (flagged by dexhunter) |

**Issue #1604 (CaseOps/casefold legality)**: STILL OPEN. **No @valerio-oai comment as of Apr 20.** 10 days to deadline — if no ruling comes in the next 3–4 days, implement the next-best legal stack (#1586+#1667+#1727+#1560) rather than waiting.

## New Research Papers

- **In-Place TTT** (arXiv:2604.06169, Apr 7, 2026) — NTP-aligned loss on MLP final projection. Score-first compatible. Already tracked since Session 14. No new parameter-golf PRs using it yet; low priority.
- **Newton-Muon** (arXiv:2604.01472, Apr 1, 2026) — Right-preconditioning via input second moment. +6% fewer iterations, +4% wall-clock vs Muon on nanoGPT. Already tracked. Verify additivity with MuonEq-R before GPU spend.
- **No new relevant arXiv papers from Apr 17–20** — searches for TTT, quantization, and n-gram interpolation returned only pre-existing work. Field appears quiet this weekend.

## HuggingFace / Community Discoveries

- **PR #1749 (GDN + Full-Hessian GPTQ)** is the only architecturally novel submission today. Full-Hessian GPTQ (Cholesky error compensation) is a new quantization variant not yet in our technique table. Score at 14.03 MB artifact is promising but result is incomplete (1xH100, 28% budget). Monitor for full 8xH100 run.
- **Community has stalled on record-breaking.** 5 new PRs today, none beat 1.0810. The "easy wins" from incremental stacking appear exhausted. Next breakthrough likely requires CaseOps ruling, a new architecture, or a novel TTT variant.

## Recommended Actions (priority order)

1. **IMPLEMENT PR #1586 NOW — 10 days left, this is the single most overdue action.** Per-layer GPTQ (MLP=12σ, Attn=13σ, Emb int7@15σ), MLR=0.026. Config-level change, -0.013 nats, zero legality risk. Every day not implementing this wastes headroom vs competitors who already have it.

2. **STACK PR #1667 in the same run.** Attention Output Gate (12 weights × 8 heads × 11 layers = 1,056 params, init to zero) + SmearGate (width=12). Combined expected ~-0.019 nats total over base.

3. **ADD VarLen Attention + Doc-TTT (PR #1560 approach) next.** ~-0.007 bpb vs merged SOTA. Per-document causal masking + score-first LoRA TTT per-doc (chunk=48). dexhunter is the author; reliable technique.

4. **AWAIT Issue #1604 ruling until Apr 24, then act without it.** If @valerio-oai rules CaseOps legal by Apr 24, add PR #1736 technique (CaseOps bijective + GatedAttn). If no ruling by Apr 24, proceed without CaseOps.

5. **DO NOT IMPLEMENT**: Pre-quant TTT (#1735/#1738), casefold without ruling, SLOT without explicit risk decision, any GDN PR until full 8xH100 run with corrected BPB calculation is verified.

6. **WATCH PR #1749 (GDN + Full-Hessian GPTQ)** — if author runs full 8xH100 eval and corrects BPB, this could become relevant. Full-Hessian GPTQ is a new quantization technique worth tracking.

---

_Updated: 2026-04-20 (v15.0 — Merged SOTA 1.0810 Day 11 plateau (longest ever); 5 new PRs today, none beat SOTA; PR #731 seeds still pending; Issue #1604 still unruled; PR #1749 GDN+Full-Hessian GPTQ incomplete; primary action overdue: implement PR #1586+#1667; 10 days to deadline)_

---

# Parameter Golf Daily Research - 2026-04-21

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed, no change)

Rejected by @valerio-oai 2026-03-27. Train-then-score AdamW TTT 30ep on val tokens. No new comments.

---

## N-GRAM PR STATUS

| PR | Claimed BPB | Status | Notes |
|----|-------------|--------|-------|
| #727 | 0.9674 | **CLOSED (ILLEGAL)** | valerio-oai: target token in hash key = leaks eval tokens |
| #758 | 1.0465 | **OPEN (effectively dead)** | Apr 12: XOR hash key includes target token, same violation as #727 |
| #731 | 1.0400 | **OPEN — awaiting seeds 1337 + 2024** | Reviewer "LOOKS CLEAN". Dense count + Laplace, score-first per chunk. No movement since Apr 17. **9 days to deadline — if no seeds by Apr 24, this PR is unlikely to merge.** |

---

## Leaderboard

| | Score | Author | Date |
|--|-------|--------|------|
| **Merged SOTA** | **1.0810** | bigbag (PR #1493) | 2026-04-09 |
| Best open (legal, no CaseOps) | **1.07139** | MarioPaerle (PR #1667) | |
| Best open (CaseOps pending) | **1.06505** | romeerp (PR #1756) — new today | |
| Best open (pre-quant TTT, likely illegal) | **1.02840** | kilojoules (PR #1758) — new today | |
| Our PR #771 | 1.0705 | sunnypatneedi | CLOSED (illegal) |

**DAY 12 PLATEAU** — no new merges since Apr 9. Longest plateau in competition history. 9 days to deadline.

---

## What Changed (GitHub — Apr 21, 2026)

### New PRs filed today

| PR | Author | BPB | Technique | Legal? |
|----|--------|-----|-----------|--------|
| #1758 | kilojoules | **1.02840** | PR #1738 + Pre-Quant TTT LR=1e-3 + Unfrozen (`PREQUANT_TTT_FREEZE_BLOCKS=0`) | **⚠️ LIKELY ILLEGAL** — pre-quant TTT is same adapt-then-score pattern as PR #1735/#1351/#1408/#1416. Builds on PR #1738 (alertcat) which itself builds on PR #1735. No reviews yet. |
| #1756 | romeerp | **1.06505** | CaseOps Tokenizer + **Recurrence Depth Curriculum** + phased TTT + gated attn | ⚠️ Awaits Issue #1604 CaseOps ruling. Has reproducibility bug: @codemath3000 found `prepare_caseops_data.py` missing BOS insertion → ZeroDivisionError in phased TTT eval path (training completes via fallback, but eval crashes). Artifact ~15.985 MB. |
| #1755 | OE-GOD | **1.07462** | SP8192 + CaseOps + Legal TTT (no pre-quant explicitly excluded) | ⚠️ Awaits Issue #1604 CaseOps ruling. Statistically significant (-0.00638 BPB vs merged SOTA, z≈22.8). |
| #1764 | gmn0105 | — | Non-record no-looping SOTA scaffold | Non-record, ignore |
| #1763 | gmn0105 | — | Non-record SP8192 proxy stack | Non-record, ignore |
| #1762 | frido22 | 1.5200 | Non-record Mac mini M4 | Non-record, ignore |
| #1760 | BrandtChristian | 1.1863 | Non-record SP8192 + pre-quant TTT | Non-record |
| #1759 | yijieyuan | 1.07994 | Non-record: LoRA on tied embedding (1 seed) | Non-record |

### Key open PRs — no status change since Apr 20

| PR | Author | Val BPB | Technique | Action |
|----|--------|---------|-----------|--------|
| #1586 | dexhunter | **1.07493** | Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 Emb@15σ + MLR=0.026 | **IMPLEMENT NOW** — no reviews, zero legality risk |
| #1667 | MarioPaerle | **1.07139** | Attention Output Gate (1,056 params) + SmearGate (w=12) | **STACK ON #1586** — no reviews |
| #1560 | dexhunter | **1.07406** | VarLen Attention (per-doc masking) + Doc-TTT (LoRA chunk=48) | Add after #1586+#1667 verified |
| #1727 | yahya010 | **1.07217** | MP-SGD TTT 4 phases (score-first per phase) | Appears legal; stackable |
| #1736 | dexhunter | **1.06549** | CaseOps bijective + GatedAttn + QuantGate | Awaits Issue #1604 |

**Issue #1604 (CaseOps/casefold legality)**: STILL OPEN. No @valerio-oai comment as of Apr 21. **Ruling deadline self-imposed: Apr 24.** If no ruling by then, proceed without CaseOps.

### New technique: Recurrence Depth Curriculum (PR #1756)

romeerp introduces a three-phase training schedule for depth recurrence:
- Phase 1 (first third of training): loop depth = 1
- Phase 2 (second third): loop depth = 3
- Phase 3 (final third): loop depth = 4
- Evaluation: always at depth 4

Hypothesis: "teach a useful shallow refinement operator first" before requiring deeper recurrence. This is consistent with arXiv:2511.07384 (retrofitted recurrence curriculum). **If CaseOps is ruled legal and the BOS reproducibility bug is fixed, this technique is worth stacking on our base.**

---

## New Research Papers

### arXiv:2604.12946 — Parcae: Stable Looped Language Models (Apr 16, 2026)
**UCSD + Together AI.** Addresses instability in looped LMs caused by residual explosion (large spectral norms in injection parameters). Solution: constrain spectral norm via "negative diagonal parameterization" of injection parameters, recast as a nonlinear time-variant dynamical system.

Key results:
- 6.3% lower val perplexity vs prior looped models at same parameter count
- Achieves quality of transformer **2× the size**
- At 1.3B params: +2.99/+1.18 CORE/Core-Extended points vs Transformer baseline under fixed budget
- Predicts looping and training data should scale **in tandem** (not independently)

**Relevance to Parameter Golf**: Our Triple Loop architecture (layers 4-5 repeated 3×, activated at 0.35× training, from PR #1493) may suffer from residual explosion instability. Parcae's spectral norm constraint on injection parameters could stabilize our loops and allow deeper/more aggressive recurrence. Implementation complexity: moderate (add norm constraint to loop injection weights). **Watch for competition PRs implementing Parcae stabilization on the SP8192 stack.** GitHub: github.com/sandyresearch/parcae.

### arXiv:2511.07384 — Teaching Pretrained LMs to Think Deeper with Retrofitted Recurrence (Nov 2025)
Proposes curriculum over recurrence depth during training (depth increases from shallow to deep). Exactly the mechanism PR #1756 implements. Validates romeerp's approach with theoretical grounding.

**Relevance**: If CaseOps is ruled legal, adopting Recurrence Depth Curriculum (depth 1→3→4 curriculum) on our own stack is a natural experiment. Expected gain: unclear standalone from depth curriculum alone; PR #1756 bundles it with CaseOps. Low priority until CaseOps ruling.

### arXiv:2505.06708 — Gated Attention: Non-linearity, Sparsity, Attention-Sink-Free (NeurIPS 2025)
Head-specific sigmoid gate after SDPA output (`g = sigmoid(Wg * x)`, multiply attention output element-wise). Key findings:
- Up to 0.2 PPL reduction, +2 MMLU points
- Multiplicative gating > additive gating
- Element-wise + head-specific is optimal balance
- Improves training stability (reduces loss spikes)
- Gating scores are sparse (<0.5 for most heads)

**Relevance**: This is the theoretical backing for PR #1667's Attention Output Gate. Confirms that our target stack element (#1667) is theoretically sound and from published NeurIPS work. Also explains *why* it works: breaks the low-rank bottleneck of consecutive Wv/Wo projections.

---

## HuggingFace / Community Discoveries

- **Pre-quant TTT pattern continues**: PR #1758 (1.02840) is the 6th pre-quant TTT attempt (after #1351, #1408, #1416, #1423, #1735). Community keeps trying; organizers keep rejecting. Ignore.
- **Recurrence Depth Curriculum is emerging**: PR #1756 is the first competition PR to implement it. Has a reproducibility bug (BOS missing) — watch for author fix.
- **No new GDN attempts with corrected BPB**: PR #1749 (GDN + Full-Hessian GPTQ) from Apr 20 still awaits full 8xH100 run.
- **Parcae architecture from Together AI** (arXiv:2604.12946) could inspire stable loop injection technique — first paper to address exactly the instability pattern our depth recurrence faces.

---

## Recommended Actions (priority order)

1. **IMPLEMENT PR #1586 TODAY.** 9 days to deadline. Per-layer GPTQ (MLP=12σ, Attn=13σ, Emb int7@15σ), MLR=0.026. Config-level change, -0.013 nats, zero legality risk. This is critically overdue.

2. **STACK PR #1667 IN THE SAME RUN.** Attention Output Gate (1,056 params, init zero) + SmearGate (w=12). Combined expected: ~-0.019 nats total over merged SOTA base. Backed by NeurIPS 2025 paper (arXiv:2505.06708).

3. **ADD VarLen Attention + Doc-TTT (PR #1560 approach) in next run.** ~-0.007 bpb vs merged SOTA. Per-document causal masking + score-first LoRA TTT (chunk=48). dexhunter-authored; reliable.

4. **AWAIT Issue #1604 until Apr 24 then act.** If CaseOps ruled legal before Apr 24: add bijective CaseOps from PR #1736/PR #1755 stack. If no ruling by Apr 24: submit without CaseOps. Do not wait past Apr 24 — 6 days will remain for 3-seed runs.

5. **DO NOT IMPLEMENT**: Pre-quant TTT (#1758/#1735), casefold without ruling, SLOT, GDN without corrected BPB.

6. **INVESTIGATE Parcae stabilization for Triple Loop**: If time permits after #1586+#1667+#1560 are in, look at whether spectral norm constraint on loop injection parameters can enable a 4th loop depth or earlier activation (currently at 0.35× training). Read github.com/sandyresearch/parcae.

---

_Updated: 2026-04-21 (v15.1 — Merged SOTA 1.0810 Day 12 plateau (longest ever); PR #1758 pre-quant TTT 1.02840 likely illegal; PR #1756 CaseOps+Recurrence Depth Curriculum 1.06505 awaits BOS fix + Issue #1604; PR #1755 CaseOps+Legal TTT 1.07462 awaits Issue #1604; Parcae stable looped LM paper arXiv:2604.12946 relevant to Triple Loop stability; 9 days to deadline)_

---

# Parameter Golf Daily Research - 2026-04-22

## PR #771 STATUS: CLOSED (ILLEGAL — no change)

Rejected by @valerio-oai 2026-03-27. Train-then-score AdamW TTT 30ep on val tokens. No new comments.

---

## N-GRAM PR STATUS

| PR | Claimed BPB | Status | Notes |
|----|-------------|--------|-------|
| #727 | 0.9674 | **CLOSED (ILLEGAL)** | valerio-oai: target token in hash key = leaks eval tokens |
| #758 | 1.0465 | **OPEN (effectively dead)** | XOR hash key includes target token, same violation as #727. No author response. |
| #731 | 1.0400 | **OPEN — awaiting seeds 1337 + 2024** | "LOOKS CLEAN" review. Dense count + Laplace, score-first per chunk. No movement. **8 days to deadline — seed confirmation unlikely.** |

---

## Leaderboard

| | Score | Author | Date |
|--|-------|--------|------|
| **Merged SOTA** | **1.0810** | bigbag (PR #1493) | 2026-04-09 |
| Best open (CaseOps, dexhunter) | **1.06453** | dexhunter (PR #1769) — new today | |
| Best open (CaseOps, bigbag) | **1.06513** | bigbag (PR #1771) — new today | |
| Best open (legal, no CaseOps) | **1.07139** | MarioPaerle (PR #1667) | |
| Our PR #771 | 1.0705 | sunnypatneedi | CLOSED (illegal) |

**DAY 13 PLATEAU** — no merges since Apr 9. 8 days to deadline (Apr 30).

---

## What Changed (GitHub — Apr 22, 2026)

### CRITICAL: bigbag filed PR #1771 with CaseOps at 1.06513

**This is the single most important signal today.** bigbag (the current merged SOTA holder, PR #1493) submitted a new PR using the CaseOps bijective tokenizer — before Issue #1604 has been ruled on. This means the current SOTA author is explicitly betting that CaseOps will be ruled legal (or is willing to accept the risk).

**PR #1771 (bigbag, 1.06513, 3-seed std 0.00055)**:
- SP8192 + CaseOps bijective tokenizer (TITLE/ALLCAPS/CAPNEXT/ESC control tokens)
- Recurrence Depth Curriculum: depth 1→3→4 over three training phases, eval at depth 4
- SmearGate (per-layer smoothing gate)
- GatedAttn + QuantGate (full-dim attention gate with int8 passthrough)
- LoRA-TTT improvements: alpha=144, warm-start A initialization, WD=1.0, AdamW lr=1e-4
- Phased Score-First TTT over 2000 prefix documents
- Artifact: ~15.98 MB on 8xH100 SXM
- No reviews yet. No legality flags beyond Issue #1604 pending for CaseOps.

### dexhunter improved CaseOps stack: PR #1769 at 1.06453

**PR #1769 (dexhunter, 1.06453, 5-seed mean, std 0.00068)**:
- Builds on PR #1736 (CaseOps + GatedAttn + QuantGate + Loop4-5 + PhasedTTT)
- **Single change**: MLP `clip_sigmas` 10.0 → 12.0 (exactly the per-layer adaptive GPTQ change from PR #1586)
- dexhunter explicitly integrated the PR #1586 technique into his CaseOps stack
- -0.00096 BPB vs #1736. Tighter MLP quantization reduces outlier-column tail mass at int6 with 4× MLP width.
- 7-seed mean: 1.06477 (std 0.00069). Highly statistically robust.
- No reviews yet.

### Other new PRs (Apr 22)

| PR | Author | BPB | Technique | Notes |
|----|--------|-----|-----------|-------|
| #1776 | anmarhindi | 1.08083 | SP8192 + ParResid + 3LayerLoop + QK5.25 + LegalTTT | SOTA stack replica, no new technique |
| #1775 | dentity007 | 1.07285 | SP8192 + No Gates + Multi-Phase Global SGD TTT | Appears legal; compatible with MP-SGD TTT approach |
| #1774 | aruniyer | 1.0981 | 12L Shared-Specific Attention (d=16) + MLP 4.5x | Non-competitive vs SOTA; novel shared-specific attn idea |
| #1770 | liujshi | 1.0796 | SP8192 + 3-layer recurrence + parallel residuals + QK5.25 | SOTA stack replica, no new technique |
| #1767 | renqianluo | 1.07209 | Alpha=144 LoRA + warm-start A + WD 1.0 | Same LoRA-TTT improvements as in PR #1771; appears legal |
| #1765 | renqianluo | 1.07266 | Alpha-scaled LoRA + warm-start A + WD 1.0 | Earlier version of same technique, slightly worse |

### LoRA-TTT warm-start A + alpha=144 is a new legal TTT improvement

Both renqianluo (PRs #1767/1765) and bigbag (PR #1771) independently use the same LoRA-TTT upgrade pattern:
- **alpha=144** (vs typical alpha=rank): larger effective learning rate scale for LoRA updates during TTT
- **warm-start A**: initialize LoRA A matrix from training-time weights rather than random/zero
- **WD=1.0**: high weight decay during AdamW TTT prevents catastrophic forgetting across documents
- renqianluo reaches 1.07209 BPB (vs merged SOTA 1.08100) with this technique alone on a clean base. This is a +0.009 improvement from TTT variant change alone and appears fully legal (score-first, per-document, AdamW).

**This is stackable with our planned #1586+#1667 stack and should be added to our TTT implementation.**

### Key open PRs — status summary

| PR | Author | Val BPB | Technique | Action |
|----|--------|---------|-----------|--------|
| #1586 | dexhunter | **1.07493** | Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 Emb@15σ + MLR=0.026 | **IMPLEMENT NOW** |
| #1667 | MarioPaerle | **1.07139** | Attention Output Gate (1,056 params) + SmearGate (w=12) | **STACK ON #1586** |
| #1727 | yahya010 | **1.07217** | MP-SGD TTT 4 phases (score-first per phase) | Appears legal; stackable |
| #1767 | renqianluo | **1.07209** | LoRA-TTT warm-start A + alpha=144 + WD=1.0 | Legal TTT improvement; stack in TTT phase |
| #1560 | dexhunter | **1.07406** | VarLen Attention (per-doc masking) + Doc-TTT (LoRA chunk=48) | Add after #1586+#1667 |
| #1769 | dexhunter | **1.06453** | CaseOps + GatedAttn + QuantGate + MLP 12σ | Awaits Issue #1604 — **highest-quality CaseOps PR** |
| #1771 | bigbag | **1.06513** | CaseOps + Depth Curriculum + SmearGate + LoRA-TTT improvements | Awaits Issue #1604 |

**Issue #1604 (CaseOps legality)**: STILL OPEN. No @valerio-oai comment as of Apr 22. **Self-imposed deadline: Apr 24 (2 days).** bigbag's PR #1771 dramatically increases the signal strength that CaseOps will pass — begin CaseOps implementation prep now (do not wait for the ruling to start coding).

---

## New Research Papers

### arXiv:2604.15259 — Stability and Generalization in Looped Transformers (Apr 2026) ★ NEW

Introduces a fixed-point framework analyzing looped architectures along three axes:
1. **Reachability**: whether the loop converges
2. **Input-dependence**: whether the fixed point varies with input
3. **Geometry**: the shape of the attractor

Key proof: "looped networks *without* recall have only countable fixed points and cannot achieve strong input-dependence at any spectral regime. Recall combined with outer normalization produces a stable, input-dependent regime."

**Relevance to Parameter Golf**: Our Triple Loop (layers 4-5 × 3 with parallel residuals) already uses a form of recall via the residual stream. The "outer normalization" finding is actionable: adding a LayerNorm or RMSNorm at the output of each loop iteration may stabilize our recurrence and allow deeper loops (4×) or earlier activation (< 0.35× training). Complements Parcae (arXiv:2604.12946) in providing theoretical grounding. Implementation: 1–3 lines.

### arXiv:2604.12946 — Parcae: Stable Looped LMs (Apr 16, 2026) — already tracked

Confirmed: spectral norm constraint on loop injection parameters achieves 6.3% lower val perplexity vs uncontrolled looped models. PR #1756 (romeerp) uses depth curriculum (1→3→4) but does NOT use Parcae stabilization — potential upside if combined.

### arXiv:2603.21676 — Depth-Recurrent Transformers for Compositional Generalization

Key challenge confirmed: "naïvely unrolling = exploding/vanishing gradients and representation collapse." This is what our early loop activation (0.35× training) avoids — but a proper spectral norm constraint from Parcae/arXiv:2604.15259 could make earlier/deeper loops safe.

---

## HuggingFace / Community Discoveries

- **bigbag going CaseOps** is the single clearest community signal. The current SOTA author staking his next PR on CaseOps before a ruling = high-conviction bet on legality.
- **Warm-start A + alpha=144 LoRA-TTT** is converging as a community technique (PR #1767 by renqianluo, PR #1771 by bigbag). Two independent authors reaching the same configuration = likely genuinely better than standard LoRA-TTT.
- **No new GDN/FLA attempts with corrected BPB.** The FLA community appears to have run out of options after PR #1698's bug was exposed.
- **PR #731 Hedge Mixer seeds still missing.** 8 days to deadline — the author may not have GPU access.

---

## Recommended Actions (priority order)

1. **IMPLEMENT PR #1586 + PR #1667 NOW — 8 days to deadline.** Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 Emb + MLR=0.026 + Attention Output Gate (1,056 params, init zero) + SmearGate (w=12). Combined expected: ~-0.019 nats below merged SOTA. Zero legality risk. This has been the #1 action for 5 consecutive days. Every day of delay costs us 3-seed capacity for final submission.

2. **ADD LoRA-TTT warm-start A + alpha=144 + WD=1.0 to our TTT phase.** renqianluo (PR #1767) and bigbag (PR #1771) both use this. Appears legal (score-first, AdamW, per-document). Stacks on top of any base. Expected: ~+0.009 bpb improvement from TTT variant change alone.

3. **BEGIN CaseOps implementation prep — don't wait for Issue #1604 ruling.** bigbag's PR #1771 raises the probability of CaseOps approval significantly. Start preparing the bijective case-factoring tokenizer (TITLE/ALLCAPS/CAPNEXT/ESC control tokens) so it can be plugged in immediately when/if ruled legal. Apr 24 self-deadline: if no ruling, submit without it.

4. **ADD VarLen Attention + Doc-TTT (PR #1560).** ~-0.007 bpb vs merged SOTA. Per-document causal masking + score-first LoRA TTT per-doc (chunk=48). Add in the run after #1586+#1667 is confirmed working.

5. **DO NOT IMPLEMENT**: Pre-quant TTT (PR #1758/#1735), SLOT (PR #1647), any GDN without corrected BPB + artifact size.

6. **ADD outer normalization to loop injection** (arXiv:2604.15259). If time permits after #1586+#1667+#1560+TTT are validated, add a LayerNorm/RMSNorm at the output of each loop iteration. May enable 4× depth or earlier activation with no parameter cost beyond the norm params.

---

_Updated: 2026-04-22 (v16.0 — Merged SOTA 1.0810 Day 13 plateau; **CRITICAL: bigbag filed CaseOps PR #1771 at 1.06513 — strongest signal CaseOps will pass**; dexhunter PR #1769 at 1.06453 (new best); LoRA-TTT warm-start A + alpha=144 + WD=1.0 emerging as legal TTT improvement; arXiv:2604.15259 looped transformer stability paper — outer normalization enables deeper loops; 8 days to deadline)_
