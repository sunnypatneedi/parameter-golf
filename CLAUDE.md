# Parameter Golf — Competition Context

## Competition Rules
- Train the best 16MB language model in **10 minutes on 8×H100**
- Metric: **val_bpb** (bits per byte on validation set, lower is better)
- Model artifact must be ≤16MB after quantization + compression
- Eval must be score-first and backward-looking (no oracle, no future token access)
- TTT allowed if legal (score chunk → train on already-scored chunk only)
- No training-data access at eval time

## Competition State (as of 2026-03-25)

### Merged SOTA
**1.1194** — LeakyReLU² + Legal Score-First TTT + Parallel Muon (PR #549, abaybektursun, 2026-03-23)
- Built on PR #414 stack (EMA + GPTQ-lite + warmdown3500)
- LeakyReLU(0.5)² activation: -0.003 BPB vs relu²
- Legal TTT: SGD(lr=0.002, momentum=0.9), 3 epochs/chunk, 32K-token chunks, ~409s TTT time
- Pre-TTT bpb: 1.1218 → Post-TTT: 1.1194

### Our Baseline
**1.1249** — PR #486 reproduced

### Leaderboard Progression (top records only)
| Score | Technique | PR |
|------:|-----------|-----|
| 1.1194 | LeakyReLU² + Legal TTT + Parallel Muon | #549 |
| 1.1228 | 11L EMA + GPTQ-lite + warmdown3500 | #414 |
| 1.1248 | 11L Partial RoPE + LN Scale + EMA + XSA4 | #315 |
| 1.1271 | 11L XSA4 + EMA + Int6 MLP3x | #287 |
| 1.2244 | Naive Baseline | — |

## Current Stack (PR #549 base)
- 11 layers, 512d, MLP 3× expansion
- LeakyReLU(0.5)² activation
- XSA on last 4 layers (XSA4)
- Int6 QAT, zstd-22 compression
- BigramHash(2048)
- EMA replacing SWA
- GPTQ-lite calibration
- Parallel Muon optimizer (Parameter Banking)
- Warmdown 3500 steps
- Legal Score-First TTT (~409s)
- Sliding window eval (stride=64)
- Partial RoPE (16/64)

## Open PRs to Watch (2026-03-25)
| PR | Score | Technique | Status |
|----|------:|-----------|--------|
| #728 | 1.1142 | Val-Calibrated GPTQ + BigramHash 3072×112 + XSA-all | Open, compliance review |
| #740 | 1.0909 | XSA-all + 5-gram eval cache (α=0.20) | Open, legality TBD |
| #741 | 0.9850 | Cosine TTT + Multi-order N-gram Cache | Open, legality TBD |
| #727 | 0.9674 | Multi-order N-gram Backoff (2–7 gram) + entropy-adaptive α | Open, "First Legal Sub-1.0" |

## Competition Strategy

### Immediate Target (no n-gram cache)
Beat **1.1142** via PR #728 techniques:
1. Expand BigramHash: 2048 → 3072×112
2. Val-Calibrated GPTQ (64 fwd passes on val, no weight updates — legal as read-only compression)
3. XSA-all (extend from XSA-last-4 to all layers)

### If N-gram Eval Cache Gets Merged
The 5-gram eval cache (fixed α=0.20) delivers ~-0.079 BPB. This would be the single largest technique to add.
- Target: ~1.040–1.09 range
- Implement score-first: score token → update cache → repeat

### Technique Delta Reference
| Technique | Delta BPB |
|-----------|-----------|
| LeakyReLU(0.5)² vs relu² | -0.003 |
| Legal TTT (409s, SGD) | -0.0025 |
| Val-Calibrated GPTQ | ~-0.005 |
| BigramHash 3072×112 vs 2048 | ~-0.002 |
| 5-gram eval cache (α=0.20) | ~-0.079 |
| Multi-order N-gram (2–7) + entropy-α | ~-0.15 |

## Lessons Learned
- **GPTQ calibration must use val data or training data within the 600s window** — post-training calibration on training data was ruled illegal
- **TTT is legal** if score-first (inference_mode for scoring, train only on already-scored chunks)
- **N-gram cache is disputed** — PR #659 was closed; PRs #740/#727 claim legal compliance via score-first updates. Wait for maintainer ruling before building on top.
- **XSA-all > XSA-last-4**: PR #728 uses XSA on all layers, beating XSA-last-4
- **Parallel Muon** is worth ~6,950 steps vs ~7,100 baseline (faster but more steps per budget)
- Artifact size is a hard constraint: must fit BigramHash + model weights + compression overhead in ≤16MB
