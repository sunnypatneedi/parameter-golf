# Parameter Golf — Competition Context for Claude

## Competition Rules
- **Goal**: Train the best 16MB language model in ≤10 minutes on 8×H100.
- **Metric**: bits-per-byte (val_bpb) on FineWeb validation set — lower is better.
- **Tokenizer**: SentencePiece sp1024 (1024-token vocabulary).
- **Artifact limit**: ≤16MB (compressed weights, tokenizer, any eval artifacts).
- **Compute limit**: ≤600 seconds wall-clock training on 8×H100 SXM.
- **Eval budget**: Eval time does NOT count against the 600s training limit — but TTT (test-time training) adapting on eval tokens DOES count, and must be backward-looking (score-first, then adapt on already-scored chunks).
- **GPTQ calibration rule (new, 2026-03-24)**: Hessian/calibration for GPTQ must be done within the 600s training window. Using training data during eval phase is disallowed.
- **Hardware**: Must be 8×H100 SXM (not A100, not A800). Non-H100 runs are non-record.

## Current SOTA (as of 2026-03-27) — PARADIGM SHIFT: N-GRAM REVOLUTION

- **Merged SOTA: 1.1194** — abaybektursun, PR #549, merged 2026-03-23
  - Stack: LeakyReLU(0.5)² + Legal Score-First TTT + Parallel Muon + 11L XSA4 + EMA + PartialRoPE + GPTQ-lite int6
- **Our PR #771: 1.0705** — open, no reviews (AdamW TTT 30ep cosine + per-layer LR on PR #549 base)
- **Best open score-first two-pass N-gram: 0.1181** — PR #868 (order-12 backoff, budgeted two-pass)
- **Best open full-rescore N-gram: 0.0935** — PR #870 (BROADSIDE, legality DISPUTED — await @valerio-oai ruling)
- **Extreme n-gram (verify legality): 0.0274** — PR #945 (Order-16 oracle + trained gate); **0.0881** — PR #961 (Order-12 + phrase cache)

**⚠️ CRITICAL (2026-03-25)**: Backward-looking N-gram eval cache confirmed legal by @valerio-oai. Two-pass N-gram rescoring now achieves sub-0.12 BPB — a **10× improvement** over merged SOTA. All competitive submissions must use N-gram interpolation.

**⚠️ CRITICAL (2026-03-27)**: Architecture quality collapses in n-gram regime. PR #961 confirms: "54× larger model gap → <0.001 BPB after cache application." Focus is n-gram cache design, not architecture.

### N-gram Technique Summary
- **Single-pass backward N-gram** (PR #727, 0.9674): multi-order backoff orders 2–7, entropy-adaptive alpha `0.05 + 0.55*sigmoid(2*(H-4.0))` — **CONFIRMED LEGAL**
- **Score-first two-pass** (PR #868, 0.1181): Pass 1 scores each chunk with partial cache, Pass 2 rescores with complete 62M-token cache — **LIKELY LEGAL**
- **Full-rescore** (PR #870, 0.0935): rescores all 62M tokens including self — **LEGALITY DISPUTED, DO NOT IMPLEMENT until @valerio-oai rules**
- **Order-16 oracle + trained gate** (PR #945, 0.0274): `nn.Linear(512,17)` gate, complementary training — **VERIFY LEGALITY**

## Our Baseline
- **1.1249** (PR #486 reproduced)

## Confirmed Technique Deltas (ablated on the PR #414 stack)
| Technique | Delta BPB | Source |
|-----------|----------|--------|
| LeakyReLU(0.5)² in MLP | **-0.003** | PR #493, #549 ablation |
| XSA on all 11 layers (vs last 4) | -0.0016 | PR #609 |
| Legal TTT (freeze=0, 3 epochs) | -0.0024 | PR #549 |
| BigramHash 2048→3072 | -0.0009 | PR #549 ablation |
| TTT freeze=2→0 | -0.0004 | PR #549 ablation |
| Parallel Muon / Parameter Banking | ±0.0000 (speed only) | PR #399, #549 |
| Soft-Round QAT (tanh) | ~-0.001 (estimated) | PR #606, 1.1162 |

## Competition Strategy

**Merged leaderboard SOTA**: 1.1194 val_bpb (abaybektursun, 2026-03-23) — LeakyReLU² + Legal Score-First TTT + Parallel Muon on PR #549 base
**Previous SOTA**: 1.1228 (signalrush, 2026-03-22) — superseded
**Our PR #771**: 1.0705 val_bpb — OPEN, no comments yet

### Current Best Path (updated 2026-03-27)
1. **Check PR #870 legality ruling daily** — if full-rescore approved, that's the entire game (0.0935 BPB).
2. **Implement score-first two-pass N-gram** (PR #868 approach) on our PR #771 stack. Target: ~0.11 BPB.
3. **If two-pass ruled illegal**, implement single-pass N-gram (PR #727 approach). Target: ~0.97 BPB.
4. Architecture improvements (XSA, TTT tuning) are now secondary to eval strategy.

### Previous Best Path (superseded by N-gram revolution)
1. Start from PR #549 merged SOTA stack.
2. **Layer in**: XSA-all (XSA_LAST_N=11, confirmed -0.0016), Soft-Round QAT (-0.001 est).
3. **TTT**: Legal score-first TTT, freeze=0, 3 epochs, cosine LR decay.
4. **Target**: Beat 1.1144 (SOTA - 0.005).

### Our approach (v5.1 — current PR #771 base)
1. **AdamW TTT** (30 epochs, cosine, per-layer LR) — -0.04 to -0.06 bpb (from PR #481)
2. **XSA on all 11 layers** — exclusive self-attention, -0.002 to -0.005 bpb (from PR #503)
3. **Value Residual (ResFormer)** — blend V vectors from layer 0, 22 params (from PR #486)
4. **GradQuant** — gradient-guided adaptive Int5/6/7 quantization (from PR #486)
5. **TrigramHash(4096)** — 3-gram context embedding (from PR #486)
6. **Partial RoPE (16/64)**, **LN Scale**, **EMA (0.997) + SWA (every 50)**, **11 layers**

**Next step (v6.0 — n-gram upgrade)**:
Add multi-order backoff cache (orders 2–7) + entropy-adaptive alpha on PR #771 base:
- `α = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))` where H = model entropy
- Score-first eval only (confirmed legal by @valerio-oai)
- Target: 1.0705 → ~0.97 bpb (estimated -0.10 from PR #727 delta vs similar base)
- Longer term: score-first two-pass (PR #868) → target ~0.11 bpb

**Key reference PRs**: #727 (0.9674, multi-order backoff 2–7), #741 (0.9850, simpler variant), #868 (0.1181, score-first two-pass), #870 (0.0935, full-rescore, legality disputed), #945 (0.0274, Order-16 + trained gate), #481 (1.0970, best TTT ref)

**Abandoned approaches**: LoRA TTT (hurts), product quantization (SWA-incompatible), larger vocab (embedding cost), custom Triton kernels (poor EV), int4 without QAT (quality-destructive at this scale), eval stride=32 (exceeds time budget with 30-epoch TTT), depth recurrence (PR #363, 1.2092 bpb).

### Key Architectural Decisions (Settled)
- 11 layers, 512d hidden, 8H/4KV GQA
- BigramHash vocab 1536 (upgraded from 2048)
- LeakyReLU(0.5)² activation (confirmed best activation)
- XSA on all 11 layers (not just last 4)
- EMA(0.997) + SWA(every 50 steps)
- PartialRoPE (16/64 dims)
- LN Scale 1/√(layer+1)
- VE(dim=128) on layers 9-10
- GPTQ-lite int6 + lzma (calibrate within 600s)
- WARMDOWN_ITERS=3500

### Legal TTT Protocol
- Score-first: use `torch.inference_mode()` during scoring (no gradient tracking)
- Then adapt on already-scored chunk (SGD lr=0.002, momentum=0.9, 3 epochs)
- Chunk size: 32,768 tokens
- All blocks unfrozen (freeze=0)
- Last chunk scored but never trained on

### Negative Results (Don't Retry)
- Value Residual Learning: neutral/negative on this stack
- Gated Attention variants (various): neutral
- Hadamard rotation: neutral
- Larger BigramHash (>3072): diminishing returns
- SWA interval < 50: no gain
- GPTQ calibration outside 600s: ILLEGAL

---

## Technique Reference

| Technique | Approx Δ bpb | Status |
|-----------|-------------|--------|
| **Score-first two-pass N-gram (order 12)** | **~-0.96 from 1.07** | **Next target (PR #868, 0.1181)** |
| **Multi-order N-gram cache (2–7) + entropy-adaptive α** | **~-0.10 from 1.07** | **Fallback (PR #727, 0.9674)** |
| **Order-16 oracle + trained gate + complementary loss** | **~-1.04 from 1.07** | **Stretch goal (PR #945, 0.0274)** |
| **AdamW TTT (30 ep, cosine, per-layer LR)** | **-0.04 to -0.06** | **In SOTA + our PR #771** |
| Sliding window eval (stride=64) | -0.032 | In SOTA |
| TrigramHash + ValueResidual + GradQuant | -0.023 | In SOTA (PR #486) |
| 3× MLP expansion | -0.015 | In SOTA |
| Int6 QAT + GradQuant adaptive Int5/6/7 | -0.010 | In SOTA |
| **XSA (all 11 layers)** | **-0.002 to -0.005** | **In our PR #771** |
| SmearGate + BigramHash(4096) | -0.006 | In SOTA |
| Value Residual (ResFormer) | -0.005 to -0.017 | In SOTA |
| 11 layers | -0.003 | In SOTA |
| EMA (0.997) + SWA (every 50) | -0.002 | In SOTA |
| Partial RoPE (16/64) + LN Scale | -0.002 | In SOTA |
| Orthogonal init + Muon WD=0.04 | -0.003 | In SOTA |
| LoRA TTT | **+0.004 (HURTS)** | **Abandoned** |
| Depth recurrence | **+0.08 (HURTS)** | **Abandoned (PR #363)** |

---

## Experiment Tracking

One row per run in `logs/experiments.md`:
```
Date | Exp ID | Change | val_bpb (slide) | Artifact bytes | Steps | Hypothesis → Verdict
```

Rules:
- Change ONE thing per run
- Record negative results explicitly
- 3 seeds only for submission-quality results
- Current byte headroom: ~660 KB (SOTA artifact is 15.34MB / 16.00MB with GradQuant)

---

## Key Constraints Cheat Sheet

| Constraint | Value |
|-----------|-------|
| Artifact size | < 16,000,000 bytes (code + compressed model) |
| Training time | 10 minutes on 8xH100 SXM |
| Eval time | 10 minutes on 8xH100 SXM (separate budget) |
| Network during eval | Prohibited |
| Val data during training | Prohibited |
| TTT rule | Only on tokens already evaluated |
| SOTA improvement threshold | >=0.005 nats, p<0.01, 3 seeds |
| Competition deadline | April 30, 2026 |

---

## Lessons Learned

### Session 1 (2026-03-22)
1. **Ship experiments first, debate strategy second.** Time-box planning to 30 min. Run a GPU experiment in the first hour, not the fifth.
2. **Always use `nohup` for RunPod commands.** SSH drops on 15-min runs. Pattern: `nohup bash -c 'CMD > /workspace/run.log 2>&1' &`
3. **Never launch parallel torchrun on the same pod.** Two jobs on 8xH100 corrupt each other. Run sequentially.
4. **1xH100 cannot run SOTA-class models.** Only use for baseline-scale experiments or code debugging. Always use 8xH100 for SOTA work.
5. **The leaderboard moves daily.** Check BOTH merged leaderboard AND open PRs before every session.
6. **TTT gains diminish on stronger bases.** -0.075 on 1.16 base → -0.022 on 1.11 base. Always verify TTT improvement on YOUR architecture first.
7. **Stride=32 is not significant.** Tested 3 seeds: only -0.0005 nats over stride=64. Don't revisit.

### Session 2 (2026-03-23)
8. **NEVER ship unverified quantization code.** GPTQ caused 0.18 bpb quant penalty (expected 0.003). Always compare pre-quant vs post-quant bpb before adding new quant methods. Quantization bugs are silent killers.
9. **First GPU run = UNMODIFIED baseline.** Establish baseline numbers before adding ANY changes. Then add ONE change at a time. Shipping 578 new lines in one run made debugging impossible.
10. **Compute TTT time budget before setting epochs.** `epochs × batches × time/batch`. 20 epochs × 71 batches × ~1s = 1420s. Basic math catches budget blowouts.
11. **Check disk quota before downloading data.** RunPod disk quotas are per-pod, not per-filesystem. 80 shards = ~16GB. Verify space first.
12. **Depth recurrence is falsified.** PR #540 got 1.2092 bpb (worse than 1.2244 baseline). Do not attempt.

### Session 3 (2026-03-24)
13. **In-Place TTT is HARMFUL.** Loss INCREASES (2.63+, going up not down). MLP output projections are NOT good TTT targets at this scale. Do not attempt.
14. **GradQuant int5/int6 mix exceeds 16MB.** Even without int7, the artifact was 34KB over. Use uniform int6 or match PR #414's exact quantization scheme.
15. **PR #486 baseline reproduced at 1.1249** (vs reported 1.1233). Within seed variance. This is our verified baseline.
16. **The v7.0 incremental plan works.** Run 0→1→2→3 from PR #414 base. Each run adds ONE thing. Stop doing moonshots with 500+ new lines.

### Daily Research 2026-03-26
- The PR #414 stack (signalrush) is the stable foundation. Don't drift from it.
- Warmdown duration is a huge lever (8k steps ≈ -0.101 BPB in unlimited compute setting, but under 10-min budget WARMDOWN_ITERS=3500 is optimal).
- TTT adds ~-0.0024 BPB but costs ~410s of eval time — fits within budget.
- Enforcement is active: check rule compliance before running expensive experiments.
- **N-gram two-pass is a 10× win** (0.0935–0.1181 BPB) — eval strategy now dominates architecture choices.
- **Check legality before implementing** any new eval-time technique — enforcement sweep closed 25+ PRs on 2026-03-24/25.

### Daily Research 2026-03-27
17. **N-gram eval cache has taken over the competition.** Scores of 0.0274 and 0.0881 bpb achievable with Order-12/16 n-gram oracles. PR #961: architecture quality collapses to <0.001 bpb difference after cache. Architecture optimization is secondary to n-gram cache design.
18. **Entropy-adaptive alpha is the key to legal n-gram blending.** `α = 0.05 + 0.55 * sigmoid(2*(H-4.0))` — trust n-grams when model is uncertain. Hindsight selection was disqualified; entropy-based blending is confirmed legal.
19. **Extended TTT (>3 epochs) risks memorization.** Community analysis shows data memorization starts above ~3 epochs. Our 30-epoch config is in this regime — verify TTT is genuinely adaptive, not memorizing.
20. **Merged SOTA updated to 1.1194** (abaybektursun, 2026-03-23). Was 1.1228. Our PR #771 at 1.0705 is the best unmerged architecture-track submission.

## Golden Rules

Every change must answer: "Does this lower val_bpb within the 16MB/10-min constraints?" If the answer is unclear, run a quick experiment on 1xH100 before investing more time. Compression and eval tricks are as valuable as architecture changes. The cheapest experiment that gives signal is the best experiment. Speed > perfection — submit early, iterate after.

_Updated: 2026-03-27 (v8.0 — Two-pass n-gram + extreme n-gram dominance; PR #770 legality watch)_
