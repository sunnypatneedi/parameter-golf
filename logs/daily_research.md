# Parameter Golf Daily Research - 2026-03-29

## PR #771 STATUS: CLOSED (RULE VIOLATION)

**Critical**: PR #771 was closed. Reason: multi-epoch TTT adapted the model to eval tokens, then reported validation metrics on those **same** tokens. This violates the rule that "adapting" and "scoring" must use distinct data.
- The n-gram eval cache that @valerio-oai confirmed "legal" on 2026-03-25 was subsequently revisited and found to violate normalization rules (see N-gram section below).
- **Action required**: Full strategy pivot away from n-gram and score-first multi-epoch TTT.

---

## N-GRAM PR STATUS — ALL INVALIDATED

On 2026-03-27, @valerio-oai closed **33+ n-gram PRs**. Reason: hashed n-gram cache implementations scored only the correct token without full-vocabulary normalization, making scores appear artificially low. When implemented correctly with full renormalization, n-gram achieves **~1.51 BPB** — worse than the neural baseline.

| PR | Reported BPB | Status |
|----|-------------|--------|
| PR #727 | 0.9674 | **CLOSED** — valerio-oai: "hashed n-gram caches leak eval tokens" |
| PR #758 | 1.0465 | **OPEN** — contains 7-gram component, at risk of closure |
| PR #731 | 1.0400 | Unknown (not checked today) |
| PR #741 | Unknown | Unknown (not checked today) |
| PR #1076 | 0.0109 | **CLOSED** — "Packed Causal N-gram + Dirichlet Backoff", closed after legality challenge |
| PR #1083 | 0.4961 | **OPEN** — ClownCar + X-WING N-gram Oracle (unreviewed) |

**Key lesson**: N-gram cache is a dead end. Do not invest GPU time here.

---

## Leaderboard

**Merged SOTA**: 1.1194 — abaybektursun (PR #549: LeakyReLU² + Legal Score-First TTT + Parallel Muon, 2026-03-23)
*(CLAUDE.md was outdated — had 1.1228. PR #549 merged and is now the record.)*

**Best open legitimate PR**: 1.1122 — dexhunter (PR #1060: Coprime-Stride Loader + Full Hessian GPTQ + XSA-all 11 layers)

**Our PR #771**: 1.0705 — **CLOSED** (rule violation)

---

## What Changed (GitHub)

### Merged since last check
- PR #549 (1.1194) — LeakyReLU² + Legal Score-First TTT + Parallel Muon — **NEW SOTA**
- PR #640 (1.1570) — Ternary quantization (73.7M params, U-Net, 8192 BPE)

### Key open PRs (legitimate)
- **PR #1060** (1.1122) — Coprime-Stride Loader + Full Hessian GPTQ (Cholesky error compensation) + XSA-all 11L — currently best open record candidate
- **PR #1072** (1.117, draft) — Fused LeakyReLU² + Online GPTQ + Parallel Muon
- **PR #1060 technique breakdown**:
  - Coprime-stride block sampling across shards (diverse batches, no pattern repetition)
  - Full GPTQ with Cholesky error compensation (vs GPTQ-lite clip search)
  - XSA extended to all 11 layers (not just last 4)

### N-gram collapse
- 33+ PRs closed by @valerio-oai on 2026-03-27
- All sub-1.0 scores in last week's scan were from buggy n-gram normalization
- Current real frontier: ~1.11 BPB

---

## New Research Papers

**pQuant** (arXiv:2602.22592, Feb 2026) — Decoupled QAT for sub-2-bit models. Splits linear layers into 1-bit dominant branch + high-precision saliency branch. Could apply to MLP layers in our 16MB budget, but implementation complexity is high (~200 lines). Expected Δ bpb: unclear, likely -0.003 to -0.008 if adapted to int5/6 range.

**GPTQ Geometry** (arXiv:2507.18553, ICLR 2026) — Shows GPTQ = Babai's Nearest Plane on Hessian lattice. Yields tighter error bounds + better GPU kernels. Directly relevant to PR #1060's Full Hessian GPTQ. Implementing full Hessian vs GPTQ-lite (our current approach) could give -0.005 to -0.010 bpb.

**RAMP** (arXiv:2603.17891, 2026) — RL-based per-layer mixed-precision using activation statistics. Too complex for competition timeline (~300 lines + RL training overhead).

**LaCT** (arXiv:2505.23884) — TTT Done Right: chunk-based context compression into fast weights. Architecture-level change, high implementation risk for competition.

---

## HuggingFace / Community Discoveries

- None found relevant beyond what's in GitHub PRs.
- The n-gram invalidation event has reset the competitive landscape to pure neural techniques.

---

## Recommended Action

**Immediate (today)**:
1. **Our PR #771 is dead.** Do not try to fix the TTT violation — the approach is fundamentally banned. The n-gram strategy is also dead.
2. **New target**: beat PR #1060 (1.1122 BPB) with a legal pure-neural stack.

**Strategy pivot — v8.0 plan**:
Build on PR #549 base (merged SOTA, 1.1194) and add:
1. **Full Hessian GPTQ** (Cholesky compensation, like PR #1060) — expect -0.005 to -0.010 bpb
2. **XSA all 11 layers** (from PR #503, confirmed -0.002 to -0.005) — ensure full coverage
3. **Coprime-stride data loader** (from PR #1060) — diverse batches, ~20 lines, low risk
4. **AdamW TTT (legal score-first, 30ep)** — keep from #549 base
5. **Value Residual + TrigramHash(4096)** — from PR #486

Priority: run PR #549 base first to get verified baseline, then add Full GPTQ, then XSA-all. Each as separate 1xH100 test before 8xH100 submission run.

**Do NOT**: attempt n-gram interpolation, multi-epoch TTT with re-scoring, or eval-time GPTQ calibration.

---

# Parameter Golf Daily Research — 2026-03-25

## Alerts

- **COMPETITION HAS EXPLODED.** Open PRs now reach 0.5466 bpb (PR #798). Our 1.0705 is no longer competitive among open submissions.
- **N-gram eval cache is CONFIRMED LEGAL** — but hindsight selection (comparing n-gram vs model on ground truth) is BANNED. Fixed-weight or entropy-adaptive alpha only.
- **Eval-time GPTQ BANNED.** Quantization calibration must happen within the training window, not eval. PRs #606, #615, #626, #639, #656 were closed for this.
- **Multi-epoch TTT with min-loss selection BANNED.** Token adaptation before evaluation = training on val set.
- **PR #771 still OPEN, no reviews yet.** No action required, but it will likely not merge given the score gap to new submissions.

