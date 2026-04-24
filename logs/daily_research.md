# Parameter Golf Daily Research - 2026-04-24

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed)

@valerio-oai ruling (2026-03-27): "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." Train-then-score ordering permanently disqualified. No appeal path. Score of 1.0705 is void.

---

## N-GRAM PR STATUS

- **PR #727**: CLOSED — ruled illegal by @valerio-oai ("disallowed due to hashed n-gram caches which do not renormalize correctly"). Permanent.
- **PR #758**: OPEN but effectively dead — reviewer MatoTeziTanka found XOR hash key includes target token (same pattern as #727). Author has not fixed.
- **PR #731** (Hedge Mixer — dense count tables + Laplace smoothing): OPEN — reviewer said "LOOKS CLEAN", score-first per chunk confirmed. Seeds 1337 and 2024 still PENDING as of Apr 20. Still not merged.

---

## Leaderboard

- **Official Merged SOTA (README)**: 1.0810 — bigbag (PR #1493, Apr 9). Day **15 plateau** — longest in competition history.
- **Disputed Scylla record**: 0.9485 — icryo — committed to `track_10min_16mb/` on Apr 23 but README **not updated**. Byte accounting dispute (PR #1271 corrects to ~1.1289 bpb). Organizers merged the folder but have not added to README leaderboard. Treat as **UNVERIFIED**.
- **Retroactive records added Apr 23-24** (not in README, old PRs from March): dexhunter 1.1122 (PR #1060) and aamodbhatt 1.1179 (PR #1148).
- **Our PR #771**: CLOSED/ILLEGAL.
- **Target**: ≤1.0760 bpb (beat merged SOTA by ≥0.005 nats). 6 days to deadline (Apr 30).

---

## What Changed (GitHub)

### Issue #1604 (CaseOps/Casefold ruling)
**NO @valerio-oai ruling as of Apr 24.** Self-imposed deadline was Apr 24 — it has now passed with no response. Field is paralyzed waiting; 9+ open PRs depend on this ruling. **Decision: proceed with clean legal stack NOW. Do not wait.**

### New Open PRs (Apr 21-24, ranked by interest)

| PR | Author | Score | Technique | Legality |
|----|--------|-------|-----------|---------|
| **#1795** | OE-GOD | **1.01252** | SP4096 + byte-level PPM order-4 adaptive-λ mixture | Fixed (gate frozen before observing byte); appears legal |
| **#1797** | dexhunter | **1.06157** | PR #1787 base + SmearGate + LQER Asym | No flags |
| **#1801** | leon2k2k2k | **1.06287** | PR #1787 base + Sparse Gate + Updated Frozen Carry | No flags |
| **#1787** | nprime06 | **1.06335** | PR #1736 + Polar Express NS + MIN_LR=0.10 + Sparse Attn Gate + Fused CE + TTT alpha=144/warm-A/WD=1.0 | No flags — **new best base PR** |
| **#1802** | aamodbhatt | **1.0771** | SP8192 + Polar Express NS + Multi-Phase Global TTT | No flags |
| **#1796** | simon-marcus | **1.08056** | Scylla tokenizer (~998 tokens) + Legal Score-First TTT | Open; no legality flags |
| #1807 | davie2009kh | 1.07037 | SP8192 + Pre-Quant TTT 3-Epoch | ⚠️ Pre-quant TTT — likely illegal |
| #1801 | — | — | (included above) | — |

### Key PR detail: PR #1787 (nprime06, 1.06335) — NEW BASE TO TRACK
This is the new community-consensus best base PR, superseding PR #1736:
- **Polar Express Newton-Schulz** — adaptive NS polynomial (ICLR 2026 paper). Replaces fixed 5-step NS in Muon.
- **MIN_LR=0.10** — warmdown floor at 0.10×LR instead of 0. Prevents over-decay.
- **Sparse Attention Gate** — head-output gate, ~96 params/layer (very lightweight vs PR #1667's 1,056 params).
- **Triton fused cross-entropy kernel** — training-time efficiency.
- **LoRA-TTT upgrades from PR #1767** — alpha=144, warm-start A, WD=1.0 (already in our stack plan).
- Artifact ≤15.94 MB, train ≤599.57s, eval ≤525.7s. All clean.

### Key PR detail: PR #1795 (OE-GOD, 1.01252) — WATCH CLOSELY
- Classical PPM (Prediction by Partial Matching) order-4 as byte-level predictor, mixed with neural LM via adaptive-λ gate.
- PPM updates counts only AFTER scoring each byte — score-first compliant.
- Initial gate was target-conditioned (flagged by reviewer nprime06) — **fixed** by freezing gate before observing byte.
- Score 1.01252 vs merged SOTA 1.0810 = **−0.069 bpb**. If legal and verified: new SOTA by massive margin.
- **Risk**: PPM "adapts" to validation bytes sequentially (like legal TTT). But it's a pure count model (no parameters), just accumulating statistics. Legal precedent unclear — similar to legal TTT or to illegal pre-quant TTT?
- **Do NOT implement** until organizer reviews or PR merges.

### Scylla Tokenizer (PR #1184 / PR #1796)
- ~998-token TokenMonster-based vocabulary (~byte-level).
- PR #1184 (icryo, 0.9485) committed to `track_10min_16mb/` Apr 23 — byte accounting dispute from PR #1271 says corrected score ~1.1289. README not updated.
- PR #1796 (simon-marcus, 1.08056) — separate Scylla implementation + legal TTT — open, appears legal.
- **Do NOT invest in Scylla until byte accounting dispute is resolved.**

---

## New Research Papers

### Polar Express (arXiv:2505.16932, ICLR 2026)
- **Authors**: Noah Amsel, David Persson, Christopher Musco, Robert M. Gower
- **Technique**: Optimal matrix sign method — dynamically adapts polynomial update rule each NS iteration. Outperforms fixed 5-step Newton-Schulz. Super-exponential convergence, ~2× faster than NS when σ_min ≈ ℓ.
- **Relevance**: Drop-in replacement for Newton-Schulz in Muon. PR #1787 and PR #1802 both use it.
- **Competition impact**: ~+5-10% effective step quality improvement. Low-risk config change.
- **Implementation**: Replace NS coefficient tuple in Muon with Polar Express adaptive updates. Reference: PR #1787 code.

### Gram Newton-Schulz (Dao-AILab, 2026)
- **Authors**: Jack Zhang, Noah Amsel, Berlin Chen, Tri Dao
- **Technique**: Iterates on small symmetric Gram matrix XX^T instead of full rectangular M. Lower FLOPs, enables symmetric GEMM kernels. pip installable (`pip install gram-newton-schulz`).
- **Relevance**: Alternative drop-in for NS in Muon. Complementary to Polar Express.
- **Competition impact**: Unknown standalone impact in competition setting. Likely similar to Polar Express.
- **Implementation**: pip install + replace NS call. Very low effort.

### LQER: Low-Rank Quantization Error Reconstruction (arXiv:2402.02446)
- **Technique**: Combines GPTQ quantization with low-rank approximation to recover capability. Activation-induced scale matrix drives SV distribution of quantization error. W4A8 without grid search or gradients.
- **Relevance**: PR #1797 uses "LQER Asym" — asymmetric variant. dexhunter achieves 1.06157 stacking this on PR #1787. Could improve our post-GPTQ quality.
- **Competition impact**: Unknown standalone vs our per-layer adaptive GPTQ. Likely complementary.

---

## HuggingFace / Community Discoveries

- dexhunter has now stacked to 1.06157 via PR #1787 (new base) → #1797. His submissions remain the most reliable in the competition (3-5 seeds, all artifacts clean).
- The community has converged on PR #1787 as the new best clean base (replaces PR #1736). Our stack plan should update to build on PR #1787 techniques.
- aamodbhatt filed a retroactive record (1.1179, PR #1148, Muon-TTT + Entropy-Adaptive Epochs) — shows Muon optimizer in TTT loop is an established approach (not novel).

---

## Recommended Actions (Priority Order, 6 days to deadline)

1. **IMPLEMENT NOW: Polar Express NS + MIN_LR=0.10** — 2 hyperparameter/config changes from PR #1787. Zero legality risk. Likely +5-10% effective steps and better warmdown floor. Stack on top of #1586+#1667+#1560+#1727+LoRA-TTT.

2. **IMPLEMENT NOW: Full clean legal stack** — Issue #1604 deadline passed with no ruling. Stop waiting. Build on PR #1493 SOTA with: Per-Layer Adaptive GPTQ (#1586) + Attention Output Gate + SmearGate (#1667) + VarLen Attention + Doc-TTT (#1560) + MP-SGD TTT 4-phase (#1727) + LoRA-TTT warm-start A + alpha=144 + WD=1.0 (#1767). Target: ~1.068-1.072.

3. **WATCH: PR #1795 (PPM mixture, 1.01252)** — Monitor for organizer review. If @valerio-oai confirms legal, this alone beats everything by 0.069 bpb. Do NOT implement before ruling (similar risk profile to SLOT).

4. **WATCH: PR #731 (Hedge Mixer)** — Seeds 1337+2024 pending. If both confirm ~1.04 and merged, provides legal n-gram mixer blueprint. Low priority given 6-day deadline.

5. **DO NOT implement**: CaseOps (no ruling), Scylla (byte accounting dispute), PPM mixture (no ruling), pre-quant TTT (illegal), PR #1735/#1738/#1758 (illegal chain).

---

*Research session: 2026-04-24 | Next check: 2026-04-25 | Days to deadline: 6*
