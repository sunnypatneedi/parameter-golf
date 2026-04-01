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

**WARNING**: PRs #731 and #758 still open but face same normalization risk as closed PRs. PR #1094 at 0.3958 is almost certainly going to face legal scrutiny — if it survives it would be the new all-time best.

---

## Leaderboard

### Merged Leaderboard
| Score | Author | Technique | Date |
|-------|--------|-----------|------|
| **1.1147** | abaybektursun | AR Self-Gen GPTQ + XSA-all (PR #1019) | 2026-03-25 |
| 1.1194 | abaybektursun | LeakyReLU² + Legal TTT + Parallel Muon | 2026-03-23 |
| 1.1228 | signalrush | 11L EMA + GPTQ-lite + warmdown3500 | 2026-03-22 |

**Merged SOTA**: 1.1147 (down from 1.1194, improved by 0.0047)

### Best Open PRs (as of 2026-04-01)
| PR | Score | Technique | Legal Status |
|----|-------|-----------|-------------|
| #1094 | **0.3958** | BackoffNgramMixer orders 2-10, sliding window | Under dispute |
| #731 | 1.0400 | Score-first legal n-gram | Open, no ruling |
| #758 | 1.0465 | 11L XSA-all + 7-gram | Open, no ruling |
| **#1176** | **1.0914** | QK-Gain 4.0 + Muon-TTT 3ep + SLOT | Open, SLOT causality flagged |
| #1218 | **1.09785** | 4096-vocab + 4×MLP + XSA-all + GPTQ | Open, no comments |
| #1217 | 1.1027 | Context-Only SLOT + QK_GAIN=5.0 | Open |
| #1219 | 1.1084 | Window attn + mixed seq_len | Open |
| #1209 | 1.1064 | Full GPTQ + Score-First TTT + SLOT | Open |

**Our PR #771**: 1.0705 — **CLOSED/REJECTED**

---

## What Changed Since 2026-03-25

### GitHub
1. **Legality wave on 2026-03-27**: All n-gram cache PRs using hashed distributions closed. Normalization requirement added to competition rules (Issue #1017). Overturns our previous "CONFIRMED LEGAL" assessment.
2. **New PRs #1209-#1224**: Multiple sub-1.1 submissions from new participants. Key standouts:
   - **PR #1218** (clarkkev, 1.09785): Achieves sub-1.1 with NO TTT — 4096 vocab + 4×MLP + GPTQ + XSA-all. Demonstrates the architecture path to sub-1.1.
   - **PR #1176** (1.0914): QK-Gain 4.0 validated across 45 experiments (-0.006 bpb). Score-first 3-epoch Muon-TTT (not AdamW). SLOT adds -0.021 bpb but causality concerns raised.
   - **PR #1217** (bigbag, 1.1027): "Context-Only SLOT" — a constrained SLOT variant that may address causality.
3. **Depth recurrence non-record submitted**: PR #363 merged as non-record (1.2092), confirming Lesson #12.
4. **Ternary quantization non-record** (PR #640): 1.1570 BPB from 73.7M ternary model. Not competitive but novel.

---

## New Research Papers

### Directly Applicable

**SLOT: Sample-specific LM Optimization at Test-time** (arXiv:2505.12392, May 2025)
- **Technique**: Adds lightweight δ-vector to final hidden layer only. Runs few AdamW steps (8 steps, lr=0.005) minimizing cross-entropy on input prompt. Weights frozen — only δ adapts.
- **Application**: PR #1176 reports -0.021 bpb. The key question for parameter-golf is whether "input prompt" = already-scored tokens satisfies score-first. @kooshi raised causality concerns but authors argue frozen weights + detached hiddens = legal.
- **Implementation cost**: ~30 lines — add δ vector, run mini-optimize during sliding eval
- **Expected impact**: -0.015 to -0.025 bpb if legal ruling confirmed

**TTT Done Right / LaCT** (arXiv:2505.23884) — Already in our reference. PR #771 implemented this. The issue wasn't the paper's approach — it was our eval ordering bug (train-then-score instead of score-first).

**Layer-wise QAT for SLMs — LieQ** (arXiv:2508.03332, Aug 2025)
- **Technique**: Mixed-precision across layers based on information-effectiveness metric. Keeps uniform bit-width within each layer (hardware-friendly). int7 on high-saliency layers, int5 on redundant layers.
- **Application**: Could recover 0.002-0.004 bpb vs uniform int6, or free ~150KB for extra capacity.
- **Implementation cost**: ~60 lines to compute layer saliency + assign bits

### General Context (Lower Priority)
- **N-gram Residual Learning** (arXiv:2210.14431): Train neural LM to fit residual over n-gram. Architecturally interesting but requires training-time changes and n-gram normalization (same legal issue).
- **EfficientQAT** (arXiv:2407.11062): Block-wise QAT. More expensive than our current approach but useful if we need to push quantization quality.

---

## HuggingFace / Community

- No parameter-golf-specific HuggingFace posts found.
- GitHub Issue #140 remains the primary community hub. As of 2026-04-01 latest notes: official SOTA 1.1147, best pending 0.3958 (PR #1094, legality TBD), best standard-stack score 1.0914 (PR #1176).
- **EBLS from PR #796** (seen in previous report): Status unknown — PR not checked today, n-gram wave likely closed it.

---

## Recommended Action

### Immediate (before next GPU session)

1. **Verify SLOT legality**: Check Issue #1017 or comment thread on PR #1176 for @valerio-oai ruling on SLOT. If legal, SLOT is -0.021 bpb for ~30 lines of code. This is the highest-ROI unlock on the table.

2. **Study PR #1218 diff**: clarkkev achieves 1.09785 with NO TTT — just 4096-vocab + 4×MLP + XSA-all + GPTQ + WD=0.085. This is a clean architecture path that avoids ALL the TTT legality risk. Read their exact config.

3. **Fix our TTT to score-first**: PR #771 was illegal because we ran all epochs THEN scored. The fix is: score token → record loss → update on scored token → move to next token. At 3 epochs this is legal per PR #1176. Do NOT run 30 epochs (budget blowout + likely still illegal).

### Next GPU Experiment Priority

**Option A (lowest risk)**: Port PR #1218 approach — 4096 vocab + 4×MLP + XSA-all + GPTQ + WD=0.085. No TTT. Expected: ~1.098 bpb. Cost: $8 (1-seed smoke test).

**Option B (higher upside)**: Score-first 3-epoch TTT + QK-Gain 4.0 + SLOT (if legal) on our existing 11L base. Expected: ~1.09 bpb. Risk: SLOT legality still uncertain.

**Do NOT**: Attempt n-gram cache until normalization issue solved (i.e., proper renormalization on every backoff step). The approach is powerful but every implementation tried so far has been illegal.

### Technique Stack for Next Submission

| Technique | Source | Expected Δbpb | Legal Status |
|-----------|--------|--------------|-------------|
| 4096-vocab | PR #1218 | ~-0.02 | Legal |
| 4×MLP | PR #1218 | ~-0.01 | Legal |
| XSA-all 11L | Multiple | -0.002 to -0.005 | Legal |
| GPTQ + WD=0.085 | PR #1218 | ~-0.01 | Legal |
| QK-Gain 4.0 | PR #1176 | -0.006 | Legal |
| Score-first TTT 3ep | PR #1176 | -0.003 | Legal |
| SLOT (δ-vector) | PR #1176 | -0.021 | VERIFY FIRST |

**Conservative target** (no SLOT): ~1.08 bpb
**Optimistic target** (with SLOT): ~1.06 bpb

---

## Updated Strategy Summary

The competition has bifurcated:
- **N-gram path** (0.39-1.04 bpb): Powerful but most implementations illegal due to normalization. PRs #731, #758 still open — wait for ruling.
- **Architecture/clean path** (1.09-1.11 bpb): No legality risk. PR #1218 shows 1.09785 without TTT. PR #1176 shows 1.0914 with light TTT + SLOT.

Our plan: Build on the architecture path first (fast, cheap, low-risk), then add score-first TTT + SLOT once SLOT legality is confirmed.
