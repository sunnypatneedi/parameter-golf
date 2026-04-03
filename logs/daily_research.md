# Parameter Golf Daily Research — 2026-04-03

---

## PR #771 STATUS: CLOSED (REJECTED — CONFIRMED)

Rejected by @valerio-oai on 2026-03-27. Reason: train-then-score violation — AdamW TTT 30ep adapted on all val tokens, then scored the same tokens. Illegal. Pre-TTT base was ~1.145 bpb. **No appeal possible.** This is recorded in our CLAUDE.md and strategy has pivoted.

---

## N-GRAM PR STATUS

| PR | Score | Technique | Status |
|----|-------|-----------|--------|
| #727 | 0.9674 | Hashed n-gram eval cache | **CLOSED** — illegal normalization (2026-03-27) |
| #741 | — | Hashed n-gram cache (variant) | **CLOSED** — same ruling |
| #758 | 1.0465 | 7-gram backward-looking eval cache | **OPEN** — no reviews yet |
| #731 | 1.0400 | Score-first n-gram + hedging | **OPEN** — no reviews, author claims legality |

PRs #758 and #731 remain open with no official rulings from @valerio-oai as of 2026-04-03.

---

## Leaderboard

| Track | Score | Author | PR | Date |
|-------|-------|--------|-----|------|
| **Merged SOTA** | **1.1147** | abaybektursun | #1019 | 2026-03-25 |
| Best open (legal?) | **0.9462** | anthony-maio | #1303 | 2026-04-03 |
| Best open (lower risk) | **1.0819** | MatoTeziTanka | #1289 | 2026-04-03 |
| Our PR #771 | 1.0705 | — | #771 | CLOSED/REJECTED |

**Merged SOTA UNCHANGED** from last session (still PR #1019 at 1.1147). The git log confirms no new merges beyond PR #1019.

---

## What Changed (GitHub) — 2026-04-03 Activity

A flood of new PRs submitted today. Key findings:

### HEADLINE: PR #1303 claims 0.9462 bpb (SLOT-16 + QK-Gain 4.0 + XSA-11)
- Author: anthony-maio
- Techniques: Causal SLOT-16, QK-Gain 4.0, XSA all 11 layers, LeakyReLU² + lzma base
- SLOT implementation: per-sample δ [bsz, 1, 512] + logit bias [bsz, 1, 1024], 16 AdamW steps, cosine LR 0.008→0.0008
- Key causality claim: "scored-position masking (last stride=64 tokens per non-first window)" — only optimizes on tokens already evaluated in prior windows
- Eval time: ~384s total (within 600s budget)
- Artifact: 15.74–15.83 MB (within limit)
- **Status: OPEN, no reviews yet — legality UNCONFIRMED**
- RISK: msisovic contested SLOT causality in Issue #1240, calling standard SLOT a "100% violation rate." PR #1303's scored-position masking may resolve this, but needs @valerio-oai ruling.

### PR #1306: Causal SLOT + Pre-quant TTT (1.0846 bpb) — resouer
- **Causal SLOT** (vs standard SLOT): restricts δ optimization to "context-only positions — tokens already scored in previous windows." Yields **-0.009 bpb**. Explicitly addresses the Issue #1240 violation.
- **Pre-quant AdamW TTT**: Apply TTT to full-precision weights BEFORE GPTQ quantization, not after. 6 epochs, -0.022 bpb. Author reports post-quant SGD failed in 25 documented cases.
- Eval time: ~551s. Artifact: ~15.95 MB.
- **Status: OPEN, no reviews. Lower legality risk than standard SLOT.**

### Other notable open PRs today:
| PR | Score | Technique | Notes |
|----|-------|-----------|-------|
| #1289 | 1.0819 | PROTEUS v1.6 (Scylla + Parallel) | Open, no reviews |
| #1296 | 1.0897 | SP4096 + Depth Recurrence + Parallel | Open |
| #1291 | 1.0925 | Vocab4096 + MLP4.0x + SLOT | Matches our arch target |
| #1286 | 1.0963 | Lucky IV | Open |
| #1285 | 1.0912 | MuonEq-R + Depth Recurrence + GPTQ | Open |
| #1176 | 1.0914 | QK-Gain 4.0 + Muon-TTT 3ep + SLOT | Disputed SLOT legality |
| #1218 | 1.0979 | Vocab4096 + MLP4x + GPTQ (no TTT) | Our arch target base |
| #1302 | 1.1079 | Split-LR + N-gram Agreement + GPTQ | n-gram agreement (different from cache) |

### SLOT legality status (Issue #1240 summary):
- Standard SLOT (PR #1176): δ optimized on all positions including unseen future tokens → causality violation, 100% violation rate documented
- Causal SLOT (PR #1306): δ only on already-scored tokens → causality preserved, -0.009 bpb
- PR #1303 scored-position masking: similar to Causal SLOT, claims 0.9462 bpb — much larger gain, mechanism needs verification
- **No @valerio-oai ruling yet on Causal SLOT or PR #1303's approach**

---

## New Research Papers

### SLOT (arXiv:2505.12392) — Yang Hu et al., Westlake University, May 2025
- Per-sample δ-vector added to final hidden layer before output head; optimized at test-time via CE loss on input prompt
- **Relevance**: Directly implemented in PRs #1176, #1303, #1306. The -0.021 bpb gain (standard SLOT) reduces to -0.009 (causal variant). Must verify legal causal implementation.
- **Implementation complexity**: ~50 lines. Low.

### LaCT: Test-Time Training Done Right (arXiv:2505.23884) — Zhang et al., May 2025
- Large chunk TTT (2K–1M tokens), GPU utilization up to 70% vs <5% for small-chunk TTT
- Nonlinear state size up to 40% of model params
- **Relevance**: Could enable more efficient score-first TTT (faster epochs = more epochs in budget). Currently only PRs #481/#503 use small-chunk TTT. LaCT could enable 3+ legal epochs within 10-min budget.
- **Implementation complexity**: "few dozen lines of pure PyTorch." Medium (needs architecture integration).

### pQuant: Decoupled Linear QAT (arXiv:2602.22592) — Feb 2026
- Low-bit QAT via decoupled linear quantization; targets sub-4-bit LLMs
- **Relevance**: Our GPTQ currently at int6. pQuant might enable int5/int4 with lower quality loss, freeing artifact bytes for more parameters.
- **Implementation complexity**: High (replaces quantization core).

### EfficientQAT (arXiv:2407.11062) — July 2024
- Block-wise QAT + end-to-end quantization parameter training with 4096 calibration samples
- **Relevance**: Basis for our current GPTQ approach. Already incorporated indirectly via PR #1019.

---

## HuggingFace / Community Discoveries

- No new relevant HuggingFace blog posts found specific to parameter-constrained LM competition.
- "Pre-quant TTT" (PR #1306) is a novel community discovery: TTT before quantization avoids quantization noise in gradients. 6 epochs at full precision → then GPTQ. Worth tracking.
- Multiple teams converging on Vocab4096 + 4×MLP + XSA-all architecture (PRs #1218, #1291, #1287) — confirms this is the right base.

---

## Recommended Action

**Priority order for next GPU session:**

1. **FIRST: Verify SLOT legality.** Check Issue #1240 for @valerio-oai ruling on Causal SLOT before any GPU spend. If no ruling, post a direct question tagging @valerio-oai with the PR #1306 implementation (context-only positions). Do NOT spend GPU time on SLOT until confirmed legal.

2. **Build PR #1218 arch base (low risk, high EV):** Vocab4096 + 4×MLP + XSA-all + GPTQ + WD=0.085 = 1.0979 bpb. This is confirmed by PR #1291 at 1.0925. Start here — it beats merged SOTA by 0.017 nats with no legality risk.

3. **Add QK-Gain 4.0 on top of #2:** PR #1176 documents -0.006 bpb from 45-experiment sweep. Low risk, small but validated gain.

4. **Add score-first Muon-TTT 3ep on top of #3:** Legal per PR #1176's approach. -0.003 bpb. Keep within time budget.

5. **If Causal SLOT approved:** Add Causal SLOT (PR #1306 implementation, scored-position masking only). Expected -0.009 bpb (causal) vs -0.021 (standard). 

6. **If Pre-quant TTT passes legality review:** PR #1306 reports -0.022 bpb from AdamW TTT before GPTQ. Novel, no flags yet. Could stack with QK-Gain + arch improvements to reach sub-1.05.

7. **Target ceiling:** PR #1303 at 0.9462 bpb if scored-position SLOT approved. Even with causal-only SLOT and full arch path, sub-1.07 bpb seems achievable within legal bounds.

---

## Session Notes

- Merged leaderboard moved last on 2026-03-25 (PR #1019). No merges in 9 days.
- Multiple sub-1.1 open PRs suggest next merge wave is coming. Submit before the merge flood closes the gap.
- Pre-quant TTT (PR #1306) is the most novel technique seen today. Watch for reviews.
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
