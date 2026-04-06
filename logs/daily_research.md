# Parameter Golf Daily Research - 2026-04-06

## PR #771 STATUS: CLOSED (REJECTED)

Rejected by @valerio-oai on 2026-03-27: "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." No appeal path. Our AdamW TTT 30ep approach is fully void.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache, unnormalized — ruled out Mar 27 |
| #741 | — | **CLOSED** (illegal) | Same ruling as #727 |
| #758 | 1.0465 | **OPEN** | 7-gram backward-looking cache, legality unresolved |
| #731 | 1.0400 | **OPEN** | 5-expert Hedge Mixer, legality unresolved |

---

## Leaderboard

- **Merged SOTA**: 1.1147 val_bpb (abaybektursun, PR #1019, 2026-03-25) — **UNCHANGED**
- **Best open legal PR**: 1.08014 (PR #1420, abaybektursun, SP8192 + Triple Loop + N-gram Tilt)
- **Best open any PR**: 1.07948 (PR #1416, SP8192 + Pre-Quant TTT — likely illegal)
- **Our PR #771**: 1.0705 — CLOSED/REJECTED

---

## What Changed (GitHub)

### New High-Value PRs (since 2026-04-05)

| PR | Author | Score | Techniques | Legal? |
|----|--------|-------|-----------|--------|
| **#1420** | abaybektursun | **1.08014** | SP8192 + Triple Loop + N-gram Tilt + Fused Kernels | **YES** |
| #1413 | dexhunter | 1.08279 | SP8192 + QK-Gain 5 + Legal Score-First TTT (3ep) | **YES** |
| #1421 | X-Abhishek-X | 1.0925 | EMA decay 0.9965 tuning on PR #1334 base | **YES** |
| #1415 | bigbag | 1.0913 | SP4096 + 3-Layer Recurrence + ETLB (no TTT/SLOT claimed) | Pending (ETLB) |
| #1399 | AnubhavBharadwaaj | 1.0898 | Pre-Quant TTT + ETLB | ILLEGAL (pre-quant TTT) |
| #1416 | erichroepke | 1.07948 | SP8192 + Pre-Quant TTT | ILLEGAL |
| #1408 | aamodbhatt | 1.0800 | dTTT (10ep pre-quant discriminative) | LIKELY ILLEGAL |
| #1406 | aamodbhatt | 1.0887 | Depth Recurrence + Discriminative Pre-Quant TTT | LIKELY ILLEGAL |

### Critical Technique Details

**PR #1420 — N-gram Tilt (NEW, LEGAL)**
- C++ open-addressing hash table, built from scored tokens only (backward-looking, causal)
- `p_tilt(t) = p_model(t) · exp(β · 1[t==hint]) / Z` — properly normalized via partition function Z
- Delivers **-0.0029 bpb** improvement, zero artifact cost
- This is the legal replacement for the illegal hash cache approach

**PR #1420 — Triple Loop (17 virtual layers)**
- Layers 4-5 repeated 3× (not 2× as in PR #1334), giving virtual 17-layer effective depth
- Loop activation at 0.35× training (earlier than PR #1334's step 3000)

**PR #1420 — Fused Kernels**
- Forward: Triton TMA fuses `leaky_relu(fc(x), 0.5).square()` — eliminates 302MB HBM intermediates
- Backward: CUTLASS 3.x Epilogue Visitor Tree fuses `(grad_out @ proj.weight) * act_grad`
- Result: **+10% throughput** = **+127 extra training steps** within 600s budget

**PR #1420 — SP8192 vocab**
- Larger than SP4096; further frees embedding budget vs SP4096
- Combined effect vs PR #1334 (SP4096): ~0.009 bpb improvement (1.08014 vs 1.0897)

**PR #1413 — Legal Score-First TTT (all blocks)**
- `torch.inference_mode()` scoring before any update per chunk
- All 11 blocks trainable (not frozen), lr=0.005, 3 epochs
- Adds **-0.003 bpb** over base without TTT

**ETLB (Eval-Time Logit Bias, PR #1399/#1415)**
- Learns bias vector `b ∈ ℝ^vocab` added to output logits during sliding window eval
- 5 SGD steps on context tokens (previously scored), then scores stride tokens
- Warm-start bias carried forward window-to-window; no weight modification
- Standalone impact: **-0.0019 bpb**
- Legality: reviewer questioned SLOT resemblance. Author argues it's post-LM-head (no hidden state leakage). **No ruling yet from @valerio-oai.**

**MuonEq-R**
- Cited in PR #1334 as arXiv:2603.28254 (2026 paper)

---

## New Research Papers

Key paper flagged from PR analysis:
- **MuonEq-R** (arXiv:2603.28254, 2026) — Muon variant with equalized updates/rotation. Used in PR #1334 (1.0897), #1344. Already implemented in competitor code; read PR #1334 for implementation.

*(Full arxiv search results pending from background agent)*

---

## HuggingFace / Community Discoveries

- None found today beyond GitHub PR activity.

---

## Recommended Action

**Primary target: Implement PR #1420 stack**
Techniques: SP8192 + Triple Loop (3× recurrence, 17 virtual layers) + N-gram Tilt + Fused Kernels
- Expected: ~1.080 bpb (confirmed 1.08014 by abaybektursun, 5-seed)
- All techniques legal
- Start with SP8192 + Triple Loop + N-gram Tilt first. Confirm bpb. Then add fused kernels (complex Triton/CUTLASS — add last).

**Layer 2: Legal Score-First TTT (PR #1413 method)**
- All blocks, 3ep, lr=0.005, score-first
- Expected: ~-0.003 bpb → ~1.077 combined

**Layer 3: ETLB (BLOCKED — await ruling)**
- File question to @valerio-oai in Issue #140 before spending GPU time
- If legal: ~-0.002 bpb → ~1.075

**Do NOT implement**:
- Pre-quant TTT in any form (illegal)
- dTTT 10ep pre-quant (PR #1408 — likely illegal)
- PR #731/#758 n-gram (unresolved, high risk)

**Best reachable legal target: ~1.075–1.077 bpb**
Delta vs merged SOTA: ~0.037–0.040 nats (well above 0.005 threshold)
