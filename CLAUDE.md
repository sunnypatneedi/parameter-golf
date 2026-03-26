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

## Current SOTA (as of 2026-03-26) — PARADIGM SHIFT: N-GRAM REVOLUTION

- **Merged SOTA: 1.1194** — abaybektursun, PR #549, merged 2026-03-23
  - Stack: LeakyReLU(0.5)² + Legal Score-First TTT + Parallel Muon + 11L XSA4 + EMA + PartialRoPE + GPTQ-lite int6
- **Our PR #771: 1.0705** — open, no reviews (AdamW TTT 30ep cosine + per-layer LR on PR #549 base)
- **Best open score-first two-pass N-gram: 0.1181** — PR #868 (order-12 backoff, budgeted two-pass)
- **Best open full-rescore N-gram: 0.0935** — PR #870 (BROADSIDE, legality DISPUTED — await @valerio-oai ruling)

**⚠️ CRITICAL (2026-03-25)**: Backward-looking N-gram eval cache confirmed legal by @valerio-oai. Two-pass N-gram rescoring now achieves sub-0.12 BPB — a **10× improvement** over merged SOTA. All competitive submissions must use N-gram interpolation.

### N-gram Technique Summary
- **Single-pass backward N-gram** (PR #727, 0.9674): multi-order backoff orders 2–7, entropy-adaptive alpha `0.05 + 0.55*sigmoid(2*(H-4.0))` — **CONFIRMED LEGAL**
- **Score-first two-pass** (PR #868, 0.1181): Pass 1 scores each chunk with partial cache, Pass 2 rescores with complete 62M-token cache — **LIKELY LEGAL**
- **Full-rescore** (PR #870, 0.0935): rescores all 62M tokens including self — **LEGALITY DISPUTED, DO NOT IMPLEMENT until @valerio-oai rules**

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

### Current Best Path (updated 2026-03-26)
1. **Check PR #870 legality ruling daily** — if full-rescore approved, that's the entire game (0.0935 BPB).
2. **Implement score-first two-pass N-gram** (PR #868 approach) on our PR #771 stack. Target: ~0.11 BPB.
3. **If two-pass ruled illegal**, implement single-pass N-gram (PR #727 approach). Target: ~0.97 BPB.
4. Architecture improvements (XSA, TTT tuning) are now secondary to eval strategy.

### Previous Best Path (superseded by N-gram revolution)
1. Start from PR #549 merged SOTA stack.
2. **Layer in**: XSA-all (XSA_LAST_N=11, confirmed -0.0016), Soft-Round QAT (-0.001 est).
3. **TTT**: Legal score-first TTT, freeze=0, 3 epochs, cosine LR decay.
4. **Target**: Beat 1.1144 (SOTA - 0.005).

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

## Lessons Learned
- The PR #414 stack (signalrush) is the stable foundation. Don't drift from it.
- Warmdown duration is a huge lever (8k steps ≈ -0.101 BPB in unlimited compute setting, but under 10-min budget WARMDOWN_ITERS=3500 is optimal).
- TTT adds ~-0.0024 BPB but costs ~410s of eval time — fits within budget.
- Enforcement is active: check rule compliance before running expensive experiments.
- XSA-all vs XSA4 is an easy win that may not yet be in our stack — verify and add.
- **N-gram two-pass is a 10× win** (0.0935–0.1181 BPB) — eval strategy now dominates architecture choices.
- **Check legality before implementing** any new eval-time technique — enforcement sweep closed 25+ PRs on 2026-03-24/25.
- See `logs/daily_research.md` for full 2026-03-26 research report.
