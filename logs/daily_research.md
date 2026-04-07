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

| PR | Author | Score | Notes | Legal? |
|----|--------|-------|-------|--------|
| **#1437** | dexhunter | **1.08091** | SP8192 + Parallel Residuals + 3-Layer Recurrence + N-gram Tilt | **YES — but reveals causality bug (see below)** |
| #1423 | aryanbhosale | 1.0791 | SP8192 + Pre-Quant TTT + QK-Gain 5.0 | **ILLEGAL — flagged by abaybektursun: "fine-tuning on val data for 6 epochs before quantization"** |
| #1424 | OnlyJundong | 1.0858 | Extended compute (50K steps) | Non-record (>10 min) |
| #1421 | X-Abhishek-X | 1.0925 | Depth Recurrence + EMA tuning | Legal |
| #1435 | AbhayAnandUCSD | 1.0980 | Depth Recurrence + BigramHash + EMA 0.9965 | No flags |
| #1440 | Mertyandimata | 1.1026 | EngramLite + Mousse + Progressive Depth Recurrence + TTT | No flags |

### ⚠️ N-gram Tilt Causality Bug (PR #1437)

**PR #1437 independently found and disclosed a causality bug in the N-gram Tilt kernel that also affects PR #1420.** dexhunter's results:
- Pre-fix (bugged): 1.07807 bpb
- Post-fix (correct causal): 1.08091 bpb

This means **PR #1420's reported 1.08014 may include a non-causal n-gram tilt implementation.** Impact: ~0.003 bpb worse when bug is fixed. When implementing N-gram Tilt, use the corrected kernel from PR #1437 (not #1420). PR #1420 may face a correction request from reviewers.

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
| Medium | LaCT: Large Chunk TTT | 2505.23884 | ~-0.003 to -0.008 (GPU util 0→70%) | Medium |
| Medium | Sparse Growing Transformer (SGT) | 2603.23998 | ~-0.001 to -0.003 indirect (saves FLOP) | Medium |
| Medium | Two-Scale Latent Dynamics / early-exit recurrence | 2509.23314 | ~-0.001 indirect | Medium |
| Watch | Newton-Muon | 2604.01472 | ~+4-6% steps | Medium (new, Apr 2026) |
| Watch | **MUD (MomentUm Decorrelation)** | **2603.17970** | **+20-50% throughput** | **Medium — quality vs speed tradeoff** |
| Watch | **Mousse** | **2603.09697** | **~-0.002 to -0.003** | **Medium-Hard — Kronecker overhead** |
| Skip | ByteFlow (byte-level LM, no tokenizer) | 2603.03583 | N/A | Not applicable — incompatible with eval metric |

### NEW Papers Not Previously Tracked

**MUD: MomentUm Decorrelation (arXiv:2603.17970, Mar 2026)**
Replaces Muon's Newton-Schulz polar decomposition with triangular (Cholesky-like) whitening via Gauss-Seidel solves. FLOPs per step ~12× smaller than Muon's Newton-Schulz. Results: 1.3–2.6× peak tokens/sec vs. Muon on most settings; up to 3× on GPT-2 large on A100; 10–50% wall-clock improvement in time-to-perplexity.
- **Competition impact**: If the throughput gain holds on H100s, could mean +20–50% more training steps. However, per-step convergence quality vs. MuonEq-R is unknown. Run a 1xH100 ablation: MuonEq-R vs. MUD vs. Mousse before committing.
- **Implementation**: Medium — replace Newton-Schulz iteration with triangular solve.

**Mousse: Rectifying the Geometry of Muon (arXiv:2603.09697, Mar 2026)**
Adds Kronecker-factored (Shampoo-style) preconditioning to Muon's polar update. Operates in a whitened coordinate system; polar decomposition applied after whitening.
- ~12% reduction in training steps vs. Muon on 160M–800M models.
- **Competition impact**: ~-0.002 to -0.003 bpb. Overhead of Kronecker factors at 8xH100 scale unknown. Lower priority than MuonEq-R; try only if MuonEq-R plateaus.

**Confirmed existing papers (no change):**
- "Thinking Deeper, Not Longer" (arXiv:2603.21676): Not applicable — focuses on compositional generalization with 20+ recurrence steps + silent objective. Triple Loop in PR #1420 is the competition-tuned variant.

---

## HuggingFace / Community Discoveries

- None found today beyond GitHub PR activity.
- NGPU-LM (arXiv:2505.22857) — GPU-accelerated n-gram LM for context biasing in ASR. Interesting for fast n-gram hash table implementation, but legal path requires proper normalization (not hashed).

---

## Recommended Action

**Primary target: Implement PR #1420 stack with N-gram Tilt bug fix**
- SP8192 + Triple Loop (17 virtual layers) + N-gram Tilt (**use PR #1437 corrected kernel**) + Fused Kernels
- Expected: ~1.081 bpb (slightly worse than PR #1420 due to causality fix, consistent with PR #1437's 1.08091)
- All legal. Start with SP8192 + Triple Loop + N-gram Tilt (fixed), confirm bpb, then add fused kernels.
- **Critical**: Do NOT copy N-gram Tilt code directly from PR #1420 — use PR #1437's corrected causal implementation.

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

_Updated: 2026-04-07 (v11.2 — added MUD (2603.17970) and Mousse (2603.09697) to Watch list; N-gram Tilt bug use PR #1437 kernel; merged SOTA unchanged 1.1147)_
