# Parameter Golf Daily Research - 2026-03-26

## PR #771 STATUS: OPEN — NO REVIEWS YET

- Claimed val_bpb: **1.0705** (3-seed mean, σ=0.0009)
- Technique: AdamW TTT 30ep cosine decay, per-layer LR (MLP-out 3×, MLP-in 0.5×) on PR #549 base
- No reviewer assigned. Metadata fix commit included (seeds/track fields corrected).

---

## N-GRAM PR STATUS

| PR | Technique | Score | Status |
|----|-----------|-------|--------|
| #727 | Multi-order backoff (2–7) + entropy-adaptive alpha | 0.9674 | OPEN |
| #758 | XSA-all + 7-gram backward-looking eval cache | 1.0465 | OPEN |
| #731 | 5-expert Hedge Mixer (unigram/bigram/trigram/entropy + neural) | 1.0400 | OPEN |
| #868 | Budgeted two-pass N-gram backoff (order-12) | 0.1181 | OPEN |
| #869 | Two-pass score-first 9-gram cache | 0.1290 | OPEN |
| #870 | BROADSIDE: full-rescore all 62M tokens (order 2-12) | **0.0935** | OPEN |

---

## Leaderboard

- **Merged SOTA**: 1.1194 — abaybektursun (PR #549: LeakyReLU² + Legal Score-First TTT + Parallel Muon)
- **Best open PR (unmerged)**: 0.0935 — simon-marcus (PR #870, legality disputed)
- **Credible legal open PRs**: 0.1181 (PR #868), 0.1290 (PR #869), 0.1315 (PR #853)
- **Our PR #771**: 1.0705 — OPEN, no reviews

---

## What Changed (GitHub) — CRITICAL SHIFT

### The N-gram Two-Pass Revolution (2026-03-25 to 2026-03-26)

The entire competition landscape has been upended. N-gram backward-looking cache (confirmed legal 2026-03-25 by @valerio-oai) has been extended to **two-pass full-rescore**, dropping scores from ~1.12 BPB to **sub-0.10 BPB** — a 10× improvement over merged SOTA.

**How it works (two-pass):**
1. Pass 1: Run standard sliding-window eval, storing per-token model probabilities. Build complete n-gram cache (orders 2–12) from all 62M scored tokens.
2. Pass 2: Rescore all 62M tokens against the full cache using NumPy.

**Key open questions — legality unclear:**
- PR #870 author self-flags: each token's own n-gram contributes to its own score ("self-inclusion"). This may violate the spirit of backward-looking-only rules.
- The Mar 24–25 enforcement sweep closed 25+ PRs for similar eval-time data violations.
- No official ruling on two-pass full-rescore has been issued yet. **Do NOT submit this variant until @valerio-oai explicitly rules it legal.**

**Score-first two-pass (PR #868, #869) appears safer**: Pass 1 scores each chunk with partial cache, then Pass 2 rescores with complete cache. This is closer to what was confirmed legal.

### Other Notable New PRs

| PR | Technique | Score |
|----|-----------|-------|
| #857 | 15L Depth Recurrence + LeakyReLU² + Cosine TTT | 1.1093 |
| #852 | Hymba-11L (SSM hybrid) | 1.1189 |
| #862 | DenseFormer + VRL + XSA last 4 layers | — |
| #860 | Learned Routing + Two-Pass N-gram Rescoring | — |
| #853 | Two-Pass Order-12 N-gram Backoff + 256K Chunks | 0.1315 |

**Depth Recurrence (PR #857, 1.1093)**: First credible depth-recurrence result — better than 1.2092 we noted in Session 3. The 15L variant with cosine TTT appears viable. Monitor for reviewer confirmation.

---

## Research Paper Scan

### Test-Time Training

| Paper | arXiv ID | Date | Relevance |
|-------|----------|------|-----------|
| End-to-End TTT for Long Context | 2512.23675 | Dec 2025 | Compresses context into weights via next-token prediction — meta-learned init analogous to our AdamW TTT |
| Specialization after Generalization (TTT theory) | 2509.24510 | Sep 2025 | Explains WHY TTT works: domain specialization for underparameterized models; calibrates TTT budget allocation |
| LaCT: Test-Time Training Done Right | 2505.23884 | May 2025 | **Already tracked.** Large-chunk (2K–1M) TTT for better GPU util — batch-size insight for H100 |

### N-gram / Neural LM Interpolation

| Paper | arXiv ID | Date | Relevance |
|-------|----------|------|-----------|
| NGPU-LM: GPU-Accelerated N-Gram LM | 2505.22857 | May 2025 | GPU-parallel n-gram data structures + neural interpolation — directly relevant to N-gram eval cache |
| Lossless Compression via Next-Token Prediction | 2505.06297 | May 2025 | N-gram redundancy in text; BPB theory background |

### Quantization-Aware Training

| Paper | arXiv ID | Date | Relevance |
|-------|----------|------|-----------|
| **pQuant: Decoupled Linear QAT** | **2602.22592** | **Feb 2026** | **NEW** — addresses "parameter democratization" in extreme low-bit QAT; split-layer approach could improve GradQuant for sub-int5 |
| Compute-Optimal QAT | 2509.22935 | Sep 2025 | Fuse QAT with LR cooldown (not separate) — reclaim training budget within 10-min constraint |
| SiLQ: Simple LLM QAT | 2507.16933 | Jul 2025 | <0.1% compute overhead QAT outperforming SOTA methods — simplicity crucial at fixed 10-min budget |
| LieQ: Layer-wise PTQ for Small LMs | 2508.03332 | Aug 2025 | Per-layer functional saliency for mixed-precision quant — directly applicable to GradQuant adaptive int5/6/7 |

### Most Actionable New Finds

1. **pQuant (2602.22592, Feb 2026)**: Freshest QAT paper — split-layer 1-bit dominant + compact auxiliary. Could unblock sub-int5 quantization that currently hurts quality.
2. **Compute-Optimal QAT (2509.22935)**: Merging QAT into warmdown phase instead of post-hoc could free 30-60s of training budget.
3. **NGPU-LM (2505.22857)**: GPU-parallel n-gram — if N-gram eval cache becomes dominant strategy, GPU acceleration matters for 600s eval budget.

---

## Recommended Actions (Priority Order)

### IMMEDIATE (today)

1. **Do NOT submit two-pass full-rescore yet.** Wait for @valerio-oai ruling on PR #870's self-inclusion issue. If ruled illegal, full-rescore is closed. If legal, it's worth implementing immediately (158s eval, 0.0935 BPB).

2. **Study PR #868 code carefully.** Score-first two-pass (0.1181 BPB) appears closer to legal — it only rescores using cache built from prior tokens at each point, then a full second pass. This is the safer implementation to attempt first.

3. **Reframe our target.** Our PR #771 (1.0705) is now 10× worse than the frontier. Even if two-pass is ruled partially illegal, single-pass n-gram cache (PR #727: 0.9674) already beats us by 0.10. We need n-gram interpolation integrated into our next submission regardless.

### SHORT TERM (next GPU run)

4. **Implement score-first single-pass n-gram on our v5.1 stack** (PR #727 approach: multi-order 2–7 backoff + entropy-adaptive alpha). Expected delta: ~0.1 BpB improvement on top of our 1.0705 → target ~0.97.

5. **If two-pass ruled legal**, implement the full two-pass rescore on our stack. PR #868 architecture + our AdamW TTT could combine for sub-0.11 BPB.

6. **Monitor PR #857** (15L depth recurrence, 1.1093). If verified, depth recurrence may deserve another look in combination with n-gram cache.

### WATCH LIST

- @valerio-oai response on PR #870 legality — **check daily**
- PR #868/#869/#870 merge status — first to merge sets the new SOTA baseline
- Any enforcement sweep targeting two-pass submissions

---

## Key Numbers to Remember

| What | Value |
|------|-------|
| Merged SOTA | 1.1194 (PR #549) |
| Our PR #771 | 1.0705 (open, no reviews) |
| Best single-pass n-gram (open) | 0.9674 (PR #727) |
| Best two-pass n-gram (legal unclear) | 0.0935 (PR #870) |
| Best two-pass score-first (likely legal) | 0.1181 (PR #868) |
| Gap: us vs score-first two-pass frontier | ~0.95 BPB |

---

*Updated: 2026-03-26 (daily research agent)*

---

# Parameter Golf Daily Research - 2026-03-31

## PR #771 STATUS: CLOSED — RULE VIOLATION

valerio-oai ruling: *"you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission."*

Fix: score each chunk under frozen weights FIRST, then apply TTT update. Never re-score adapted tokens.

---

## N-GRAM PR STATUS

| PR | val_bpb | Technique | Status |
|----|---------|-----------|--------|
| #727 | 0.9674 | Multi-order n-gram backoff (2-7) + entropy-adaptive alpha | **CLOSED — illegal** ("does not renormalize correctly", "leaks eval tokens") |
| #758 | 1.0465 | 7-gram backward-looking eval cache (alpha=0.40), no TTT | OPEN, no review |
| #731 | 1.0400 | Hedge Mixer (neural+unigram+bigram+trigram+entropy) + AdamW TTT | OPEN, no review |
| #798 | 0.5466 | Order-adaptive entropy gating, per-order thresholds | OPEN, pending review |

Issue #677: no conclusive organizer ruling on what constitutes a legal n-gram implementation.

---

## Leaderboard

**Merged SOTA**: 1.1147 (PR #1019, abaybektursun, 2026-03-25) — Self-Gen GPTQ + XSA-all + BigramHash 3072×112

| PR# | val_bpb | Technique | Legal? |
|-----|---------|-----------|--------|
| #798 | 0.5466 | Order-adaptive entropy n-gram gating | Pending |
| #1184 | **0.9485** | Scylla tokenizer (998 tokens) + Full GPTQ + XSA-all + FA3 | Pending |
| #1185 | 0.9641 | LeakyReLU² + Score-First TTT + N-gram Backoff (2-9) | Pending, disputed |
| #731 | 1.0400 | Hedge Mixer + AdamW TTT | Pending |
| #758 | 1.0465 | 7-gram eval cache (no TTT) | Pending |
| #1180 | 1.0577 | SR-CM-P2Loss + residual mixing | Pending, metric review |
| #771 | 1.0705 | AdamW TTT 30ep (ours) | **CLOSED** |
| #1176 | 1.0914 | QK-Gain 4.0 + Muon-TTT + SLOT | Pending |
| #1019 | 1.1147 | Self-Gen GPTQ + XSA-all + BigramHash | **MERGED SOTA** |

---

## New Technique Discoveries

### 1. Scylla Tokenizer (PR #1184 — 0.9485 BPB)
- 998-token custom vocabulary (vs standard sp1024 = 1024 tokens)
- Pure architecture — no n-gram eval cache dependency
- Achieves 0.9485 with Full GPTQ + XSA-all + FA3 on top
- Zero legal risk (no eval-time technique concerns)
- Requires new tokenizer artifact + data preprocessing changes

### 2. Score-First TTT (Legal Protocol — our fix)
- Record loss for each chunk under FROZEN weights
- Then update weights via TTT on that chunk
- Never go back and re-score adapted tokens
- ~30-line refactor of our eval loop

### 3. SLOT Eval Technique (PR #1176 — ~-0.023 bpb)
- At eval time: optimize additive delta `δ ∈ R^{512}` at last hidden layer
- 8 AdamW steps, lr=0.005 per chunk; model weights frozen
- Appears fully legal (no weight updates to base model)
- ~30 lines implementation

### 4. QK-Gain 4.0 (PR #1176)
- Query-key scaling hyperparameter = 4.0 (validated via 45-experiment sweep)
- 1-line change, part of PR #1176's 0.023 bpb total gain

### 5. Full GPTQ (PR #1019, merged SOTA)
- Hessian + Cholesky error compensation (vs GPTQ-lite proxy Hessian)
- Better quantization quality, ~0.010-0.020 bpb over GPTQ-lite
- Must run within training window (eval-time GPTQ remains illegal)

---

## Research Papers

| Paper | arXiv ID | Relevance |
|-------|----------|-----------|
| End-to-End TTT for Long Context | 2512.23675 | Legal TTT design: compress context into MLP weights before scoring |
| LaCT (TTT Done Right) | 2505.23884 | Large-chunk TTT architecture reference |
| Compute-Optimal QAT | 2509.22935 | Cooldown+QAT fusion — optimal quantization timing |
| N-gram Is Back (Residual) | 2210.14431 | N-gram as residual learner — potentially legal framing |

---

## Recommended Action

1. **Fix score-first TTT** (score → frozen → update protocol). Smoke-test on MLX. Resubmit.
2. **Add SLOT** to eval loop (~-0.023 bpb, ~30 lines, appears legal).
3. **Study PR #1184 diff** for Scylla tokenizer accessibility before GPU spend.
4. **Hold n-gram implementation** — PR #727 closed as illegal. Wait for issue #677 ruling.

*Updated: 2026-03-31*
