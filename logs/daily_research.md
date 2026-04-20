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
