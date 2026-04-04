# Parameter Golf Daily Research — 2026-04-04

## PR #771 STATUS: CLOSED (REJECTED)

- **Closed by**: @valerio-oai, 2026-03-27
- **Reason**: Train-then-score TTT violation — 30 epochs on all val tokens before scoring.
- **Impact**: 1.0705 result void. Base without TTT ~1.145. Must use score-first TTT ≤3 epochs.

---

## N-GRAM PR STATUS

| PR | Score | Status |
|----|-------|--------|
| #727 | 0.9674 | **CLOSED** — illegal normalization (Issue #1017) |
| #741 | 0.9850 | **CLOSED** — self-closed, same reason |
| #758 | 1.0465 | **OPEN** — no maintainer ruling; same normalization risk |
| #731 | 1.0400 | **OPEN** — no maintainer ruling; same normalization risk |

**WARNING**: PRs #758 and #731 face the same Issue #1017 ruling that killed #727/#741. Do NOT build on n-gram caches until a maintainer explicitly clears the implementation.

---

## Leaderboard

- **Merged SOTA**: 1.1147 (abaybektursun, PR #1019) — NO CHANGE
- **Best clean open PR**: 1.0807 (PR #1351, Discriminative TTT, legal)
- **Best clean arch-only open**: 1.0897 (PR #1334, SP4096 + Depth Recur + Parallel Residuals + MuonEq-R)
- **Our PR #771**: 1.0705 — CLOSED/VOID

---

## What Changed (GitHub)

### New high-value open PRs (since 2026-04-03)

| PR | BPB | Technique | Legality |
|----|-----|-----------|----------|
| **#1334** | 1.0897 | SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R | Clean — no TTT/SLOT |
| **#1344** | 1.0923 | SP4096 + Polar Express + MuonEq-R | Clean |
| **#1351** | 1.0807 | **Discriminative TTT** — per-block adaptive LR (0.3× early, 1.0× late) | Legal score-first |
| **#1333** | 1.0766 | SP4096 + Depth Recurrence + Causal SLOT-16 | SLOT unruled |
| **#1326** | 1.0896 | SP4096 + Depth Recur + Parallel Residuals + MuonEq-R + Legal TTT | Score-first, check |
| **#1318** | 1.0096 | TTT-AdamW + SLOT L-BFGS25 + GPTQ | SLOT unruled |
| **#1350** | 1.0046 | L-BFGS Causal SLOT | SLOT unruled; likely eval budget overrun |
| #1319 | 0.6951 | SLOT-64 | Eval takes 825s — over time budget |

### SLOT legality — still unruled

- Issue #140 thread is active but @valerio-oai has NOT issued a ruling.
- @abaybektursun (merged SOTA holder) **removed SLOT from his own stack** after finding causality issues.
- PR #1333 author self-flagged and submitted a clean backup (PR #1334, 1.0897) without SLOT.
- PR #1303 (SLOT-16, 0.9462) has community causality flags (Issue #1240), no maintainer ruling.
- **Verdict: Do NOT spend GPU on SLOT until @valerio-oai rules explicitly in Issue #140.**

### New techniques to track (2026-04-04)

1. **Depth Recurrence + Parallel Residuals** — shared layer weights iterated N times + parallel residual connections. PRs #1334, #1326, #1333 at 1.08–1.09 bpb range.
2. **MuonEq-R** — modified Muon optimizer variant. Appears alongside depth recurrence in top submissions.
3. **Discriminative TTT** (PR #1351, 1.0807) — per-block adaptive LR during score-first TTT. Early transformer blocks get 0.3× LR, later blocks 1.0× LR. -0.010 bpb vs flat LR TTT. **High EV, legal, adopt.**
4. **Pre-quant AdamW TTT** (PR #1306, 1.0846) — TTT applied to full-precision weights BEFORE GPTQ quantization. 6 epochs, -0.022 bpb. Novel, no legality flags yet.
5. **Causal SLOT** (PR #1306) — δ optimization restricted to already-scored tokens only. -0.009 bpb. Lower risk than standard SLOT, but still awaiting @valerio-oai ruling.

---

## New Research Papers

| Paper | arXiv | Action |
|-------|-------|--------|
| **ExoFormer** (Exogenous Anchor Attention) | 2601.08131 (Jan 2026) | **NEW — read.** Extends Value Residual to Q/K/V with learned mixing coefficients. Est. -0.005 to -0.010 bpb. Medium complexity, no legality risk. Verify 16MB fit. |
| **Muon Optimizer State Quantization** | 2509.23106 (Sep 2025) | **NEW — low risk.** Int8 blockwise quantization of Muon optimizer states. 62% memory reduction during training → allows slightly larger model in same GPU memory. Low complexity. |
| **SLOT** | 2505.12392 (May 2025) | -0.021 bpb. BLOCKED — legality DISPUTED (Issue #1240). Await @valerio-oai. |
| **LaCT** (Large Chunk TTT) | 2505.23884 (May 2025) | High complexity (architecture overhaul + meta-learning). Skip. |
| **E2E-TTT for Long Context** | 2512.23675 (Dec 2025) | High complexity, meta-learning required. Skip. |
| **XSA** (Exclusive Self-Attention) | 2603.09078 (Mar 2026) | Already in stack. Read for implementation refinements only. |
| **Muon is Scalable** | 2502.16982 (Feb 2025) | Already in stack (WD=0.085 validated). No new action. |
| **pQuant** (Decoupled Linear QAT) | 2602.22592 (Feb 2026) | Sub-4-bit QAT. May enable int5/int4 with lower quality loss. High complexity. |
| **Induction-head N-gram** | 2411.00066 (Oct 2024) | N-gram interpolation still faces Issue #1017 ruling. Skip until organizer confirms. |

---

## Recommended Action

**Priority order:**

1. **Verify SLOT legality** — check Issue #140 and Issue #1240 for @valerio-oai ruling before any GPU spend.

2. **Implement Discriminative TTT** (PR #1351 technique) — per-block adaptive LR for score-first TTT. Legal, -0.010 bpb, adopt on existing stack.

3. **Adopt SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R** — PR #1334 achieves 1.0897 without any TTT or SLOT. New clean architecture path.

4. **Evaluate Pre-quant TTT** (PR #1306) — TTT before GPTQ. Novel, no legality flags. Watch for reviewer feedback before GPU spend.

5. **Do NOT build on n-gram cache** — PRs #758/#731 still open but will likely be closed under Issue #1017.

6. **Gap to close**: Merged SOTA 1.1147 → our immediate target ~1.080 (Discriminative TTT + depth recurrence + SP4096). Must beat by ≥0.005 nats (p<0.01, 3 seeds).

---

_Updated: 2026-04-04_

---

# Parameter Golf Daily Research — 2026-04-03

## PR #771 STATUS: CLOSED (REJECTED — CONFIRMED)

Rejected by @valerio-oai on 2026-03-27. Reason: train-then-score violation — AdamW TTT 30ep adapted on all val tokens, then scored the same tokens. Illegal. Pre-TTT base was ~1.145 bpb. **No appeal possible.**

---

## N-GRAM PR STATUS

| PR | Score | Technique | Status |
|----|-------|-----------|--------|
| #727 | 0.9674 | Hashed n-gram eval cache | **CLOSED** — illegal normalization (2026-03-27) |
| #741 | — | Hashed n-gram cache (variant) | **CLOSED** — same ruling |
| #758 | 1.0465 | 7-gram backward-looking eval cache | **OPEN** — no reviews yet |
| #731 | 1.0400 | Score-first n-gram + hedging | **OPEN** — no reviews, author claims legality |

---

## Leaderboard

| Track | Score | Author | PR | Date |
|-------|-------|--------|-----|------|
| **Merged SOTA** | **1.1147** | abaybektursun | #1019 | 2026-03-25 |
| Best open (SLOT unreviewed) | **0.9462** | anthony-maio | #1303 | 2026-04-03 |
| Best open (lower risk) | **1.0819** | MatoTeziTanka | #1289 | 2026-04-03 |
| Our PR #771 | 1.0705 | — | #771 | CLOSED/REJECTED |

---

## What Changed (GitHub) — 2026-04-03

### HEADLINE: PR #1303 claims 0.9462 bpb (Causal SLOT-16 + QK-Gain 4.0 + XSA-11)
- Author: anthony-maio
- Techniques: Causal SLOT-16, QK-Gain 4.0, XSA all 11 layers, LeakyReLU² + lzma base
- SLOT implementation: per-sample δ [bsz, 1, 512] + logit bias, 16 AdamW steps, cosine LR, scored-position masking
- Eval time: ~384s (within 600s budget); Artifact: 15.74–15.83 MB
- **Status: OPEN, no reviews — legality UNCONFIRMED**

### PR #1306: Causal SLOT + Pre-quant TTT (1.0846 bpb)
- **Causal SLOT**: δ optimization restricted to context-only positions. -0.009 bpb. Addresses Issue #1240.
- **Pre-quant AdamW TTT**: TTT on full-precision weights BEFORE GPTQ. 6 epochs, -0.022 bpb. Novel.
- Eval time: ~551s. Artifact: ~15.95 MB.
- **Status: OPEN, no reviews. Lower legality risk than standard SLOT.**

### Other notable open PRs (2026-04-03):
| PR | Score | Technique |
|----|-------|-----------|
| #1289 | 1.0819 | PROTEUS v1.6 (Scylla + Parallel) |
| #1296 | 1.0897 | SP4096 + Depth Recurrence + Parallel |
| #1291 | 1.0925 | Vocab4096 + MLP4.0x + SLOT |
| #1285 | 1.0912 | MuonEq-R + Depth Recurrence + GPTQ |
| #1218 | 1.0979 | Vocab4096 + MLP4x + GPTQ (no TTT) |

### SLOT legality (Issue #1240):
- Standard SLOT (PR #1176): δ on all positions including future tokens → causality violation
- Causal SLOT (PR #1306): δ only on already-scored tokens → -0.009 bpb, awaiting ruling
- PR #1303 scored-position masking: similar to Causal SLOT, claims 0.9462 bpb
- **No @valerio-oai ruling on Causal SLOT as of 2026-04-03**

---

## Session Notes (2026-04-03)

- Merged leaderboard moved last on 2026-03-25 (PR #1019). 10 days without merge.
- Pre-quant TTT (PR #1306) is the most novel technique. Watch for reviews.
- Competition deadline: April 30, 2026. 27 days remaining.

---

*Updated: 2026-04-03 (daily research agent)*

---

# Parameter Golf Daily Research — 2026-04-01

## CRITICAL ALERTS

- **PR #771 CLOSED (REJECTED).** Our AdamW TTT 30ep submission was ruled illegal by @valerio-oai on 2026-03-27. Reason: train-then-score violation — we adapted on all val tokens for 30 epochs, then evaluated on those same tokens. Not score-first. Our pre-TTT base was ~1.145 bpb.
- **N-GRAM CACHE RULING REVERSED.** All hashed n-gram cache approaches ruled ILLEGAL on 2026-03-27 (Issue #1017). Reason: produce unnormalized probability distributions. PRs #727, #741 closed. The "CONFIRMED LEGAL" note from 2026-03-25 is now void.
- **Score-first TTT at ≤3 epochs IS legal.** PR #1176 uses 3-epoch Muon-TTT with score-first ordering and is under review at 1.0914 bpb.
- **Merged SOTA moved to 1.1147** (PR #1019, from previous 1.1194).

---

## PR #771 STATUS: CLOSED (REJECTED — ILLEGAL TTT)

- **Closed by**: @valerio-oai, 2026-03-27
- **Reason**: "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on" — violates score-first TTT constraint.
- **Impact**: Our 1.0705 result is void. Base without TTT was ~1.145. Must redesign TTT to score-first.

---

## N-GRAM PR STATUS

| PR | Score | Technique | Status |
|----|-------|-----------|--------|
| #727 | 0.9674 | Multi-order backoff (2-7) + entropy-adaptive alpha | **CLOSED** — normalization violation (Issue #1017) |
| #741 | 0.9850 | Cosine TTT + multi-order n-gram cache | **CLOSED** (self-closed) — same normalization issue |
| #731 | 1.0400 | Score-first TTT + n-gram (self-assessed legal) | **OPEN** — no reviewer comments yet |
| #758 | 1.0465 | 11L XSA-all + 7-gram cache | **OPEN** — no reviewer comments yet |
| #1094 | 0.3958 | BackoffNgramMixer orders 2-10, 4M buckets | **OPEN** — legality disputed by @kooshi, author updated |

**WARNING**: PRs #731 and #758 still open but face same normalization risk as closed PRs.

---

_Updated: 2026-04-01_
