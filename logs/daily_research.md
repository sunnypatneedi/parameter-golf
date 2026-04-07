# Parameter Golf Daily Research - 2026-04-07

## PR #771 STATUS: CLOSED (REJECTED)

Rejected by @valerio-oai on 2026-03-27: "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." No appeal path. Our AdamW TTT 30ep approach is fully void.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache, unnormalized — ruled out Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Same ruling as #727 — author self-closed Mar 30 |
| #758 | 1.0465 | **OPEN** | 7-gram backward-looking cache, legality unresolved |
| #731 | 1.0400 | **OPEN** | 5-expert Hedge Mixer, legality unresolved ("same risk" noted) |

---

## Leaderboard

- **Merged SOTA**: 1.1147 val_bpb (abaybektursun, PR #1019, 2026-03-25) — **UNCHANGED**
- **Best open legal PR**: 1.08014 (PR #1420, abaybektursun, SP8192 + Triple Loop + N-gram Tilt + Fused Kernels)
- **PR #1430 (today)**: 0.39642 — OPEN, **LIKELY ILLEGAL** (see analysis below)
- **Our PR #771**: 1.0705 — CLOSED/REJECTED

---

## What Changed (GitHub)

### CRITICAL: PR #1430 — 0.39642 val_bpb (renqianluo, Apr 7, 2026) ⚠️ LIKELY ILLEGAL

The most dramatic claim in competition history. **Status: OPEN, no organizer reviews yet.**

**Techniques:**
1. **Per-Sample SLOT**: Each sequence gets `[bsz,1,512]` hidden delta + `[bsz,1,1024]` logit bias = 1536 params/sequence. AdamW 24 steps, cosine LR 0.432→0.001, β₁=0.6, β₂=0.5
2. **Causal Backoff N-gram Mixer (order 2–22)**: 4M hash buckets, entropy-adaptive alpha: `0.20 + 0.55 * sigmoid(2*(H-2.5))`. Claims "causal" (only backward-looking tokens).
3. **TTT**: Second pass over first 10% of chunks, AdamW 1 epoch, lr=0.0001, freezing blocks 0–9
4. **GPTQ damp=0.005**

**Legality analysis:**
- **N-gram hash cache (4M buckets)**: Almost certainly illegal — same pattern as #727/#741 which were closed for "hashed n-gram caches that don't renormalize correctly." Unless renormalized over full vocab (unlikely with 4M hash buckets), this will be ruled out.
- **Per-Sample SLOT (δ-vector)**: Still UNRULED under Issue #140. @valerio-oai has not issued a ruling. SLOT standard variant was showing -0.021 bpb when @abaybektursun removed it pre-emptively. Per-sample SLOT is even more aggressive.
- **TTT structure**: "Second pass over first 10% of chunks" — needs score-first validation. If it trains on already-scored tokens, it may be legal, but the "second pass" framing is ambiguous.

**Verdict**: Do NOT implement until @valerio-oai rules on this PR. High probability of rejection on n-gram grounds alone. Watch for organizer response.

**Artifact sizes**: 15.86–15.90MB (within 16MB). Timing: train~600s, eval~587–595s (within budget). 3-seed mean: 0.39642.

### Other New PRs (Apr 6–7, 2026)

PRs #1421–1444 filed Apr 6–7. Most are likely stacking variations on PR #1334/#1420 architecture. Highlighted:
- **#1421** (X-Abhishek-X, 1.0925): EMA decay 0.9965 tuning on PR #1334 base — LEGAL
- **#1422–#1444**: Not individually reviewed; check for new sub-1.08 entries

### Issue #140 Status

No new rulings from @valerio-oai on:
- SLOT delta-vector (standard or per-sample)
- Causal SLOT
- ETLB (Eval-Time Logit Bias)

All three remain UNRULED. **Do not spend GPU on any of these until ruled.**

---

## New Research Papers

No breakthrough new papers found beyond those already tracked in CLAUDE.md. Confirmed papers:

| Priority | Paper | arXiv ID | Δ bpb est. | Risk |
|----------|-------|----------|-----------|------|
| **NOW** | MuonEq: Balancing Before Orthogonalization | 2603.28254 | ~-0.005 | Low |
| **NOW** | Compute-Optimal QAT (cooldown+QAT fusion) | 2509.22935 | ~-0.002 | Low |
| Medium | Sparse Growing Transformer (SGT) | 2603.23998 | saves FLOP budget | Medium |
| Medium | LaCT: Large Chunk TTT | 2505.23884 | better GPU util | Medium |
| Medium | Two-Scale Latent Dynamics / early-exit recurrence | 2509.23314 | frees eval budget | Medium |
| Watch | Newton-Muon | 2604.01472 | ~+4-6% steps | High (new) |

New paper confirmed existing:
- **"Thinking Deeper, Not Longer" (arXiv:2603.21676)**: Depth-recurrent transformer with silent thinking objective + LayerScale + identity-biased recurrence. Not directly applicable (focuses on compositional generalization, not compression), but confirms depth recurrence direction.

---

## HuggingFace / Community Discoveries

- None found today beyond GitHub PR activity.
- NGPU-LM (arXiv:2505.22857) — GPU-accelerated n-gram LM for context biasing in ASR. Interesting for fast n-gram hash table implementation, but legal path requires proper normalization (not hashed).

---

## Recommended Action

**Primary target: Implement PR #1420 stack (unchanged from yesterday)**
- SP8192 + Triple Loop (17 virtual layers) + N-gram Tilt (normalized, legal) + Fused Kernels
- Expected: ~1.080 bpb
- All legal. Start with SP8192 + Triple Loop + N-gram Tilt, confirm bpb, then add fused kernels.

**Layer 2: Legal Score-First TTT (PR #1413 method)**
- All blocks, 3ep, lr=0.005, score-first (inference_mode scoring before update)
- Expected: ~-0.003 bpb → ~1.077 combined

**DO NOT implement (legality concerns):**
- PR #1430 techniques (per-sample SLOT, N-gram order-22 hash, TTT second pass) — await ruling
- SLOT in any form (Issue #140 unruled)
- ETLB (unruled)
- Pre-quant TTT (all variants illegal)
- N-gram hash cache without proper normalization

**Watch:**
- PR #1430 organizer response — if ruled LEGAL, per-sample SLOT alone could be worth ~0.7 bpb. Extremely unlikely given past n-gram rulings. Monitor Issue #140 and PR #1430 comments.
- PRs #1422–1444 for new legal techniques.

**Best reachable legal target: ~1.075–1.077 bpb** (PR #1420 stack + Legal TTT)
Delta vs merged SOTA: ~0.037–0.040 nats (well above 0.005 threshold)

---

_Updated: 2026-04-07 (v11.0 — PR #1430 flagged: 0.39642 bpb claim, likely illegal; merged SOTA unchanged 1.1147; legal path remains PR #1420 stack)_
