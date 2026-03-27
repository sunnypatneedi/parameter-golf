# Daily Parameter Golf Research — 2026-03-25

## Alerts

- **COMPETITION HAS EXPLODED.** Open PRs now reach 0.5466 bpb (PR #798). Our 1.0705 is no longer competitive among open submissions.
- **N-gram eval cache is CONFIRMED LEGAL** — but hindsight selection (comparing n-gram vs model on ground truth) is BANNED. Fixed-weight or entropy-adaptive alpha only.
- **Eval-time GPTQ BANNED.** Quantization calibration must happen within the training window, not eval. PRs #606, #615, #626, #639, #656 were closed for this.
- **Multi-epoch TTT with min-loss selection BANNED.** Token adaptation before evaluation = training on val set.
- **PR #771 still OPEN, no reviews yet.** No action required, but it will likely not merge given the score gap to new submissions.

## Leaderboard

**Merged SOTA**: 1.1194 bpb (PR #549, abaybektursun, 2026-03-23)

**Our PR #771**: 1.0705 bpb — Open, awaiting review. Beats merged SOTA by 0.049 but far behind open PR frontier.

### Top Open PRs (the real competition)

| PR# | val_bpb | Technique | Seeds | Status |
|-----|---------|-----------|-------|--------|
| #798 | **0.5466** | Order-adaptive entropy gating + BackoffNgramMixer (per-order ent_centers) | 3 | Open |
| #796 | **0.6567** | Prefill cache + 7-gram entropy-adaptive + EBLS | ? | Open |
| #770 | **0.6672** | 11L + eval-time multi-order n-gram cache (2-7), entropy-adaptive alpha | 1 | Open |
| #795 | **0.8881** | 11L + order-adaptive 11-gram | ? | Open |
| #797 | **0.8960** | 7-gram n-gram cache | ? | Open |
| #792 | **1.0340** | 11L LeakyReLU² + XSA-all + Full GPTQ + 5-gram | ? | Open |
| #727 | **0.9674** | Multi-order n-gram backoff (2-7) + entropy-adaptive alpha | 3 | Open |
| #741 | **0.9850** | Cosine TTT + multi-order n-gram cache | ? | Open |
| #758 | **1.0465** | N-gram no TTT | ? | Open |
| #771 | **1.0705** | AdamW TTT 30ep cosine + per-layer LR (ours) | 3 | Open |

**Key pattern**: Every sub-1.0 submission uses n-gram eval cache. The top submissions use ORDER-ADAPTIVE entropy gating with per-order thresholds. Pure TTT without n-gram is no longer competitive.

## New Techniques Found

### 1. Order-Adaptive Entropy Gating (PR #798 — 0.5466 bpb)
- **Source**: Open PR, 3-seed validated
- **Delta estimate**: -0.52 bpb vs our base (!)
- **How it works**: Per-order entropy centers: `{7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}`. Higher-order n-grams activate at lower entropy (high confidence), lower-order at higher entropy. Builds on BackoffNgramMixer from PR #779.
- **Evidence quality**: STRONG (3-seed, 15.99MB artifact, compliant)
- **Legality**: Legal — score-first caching, no hindsight selection
- **Implementation cost**: Medium — need backoff n-gram mixer + per-order entropy gating

### 2. BackoffNgramMixer (PR #779 foundation)
- **Source**: Referenced by PR #798
- **How it works**: Multi-order n-gram cache with highest-order-first cascading fallback on miss. Orders 2-7 with backoff.
- **Delta estimate**: -0.10 to -0.16 bpb (base technique before entropy gating)

### 3. Entropy-Adaptive Alpha Mixing (PR #727 — 0.9674 bpb)
- **Source**: Open PR, 3-seed validated
- **Formula**: `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
- **Evidence**: Ablation shows +0.0151 bpb gain over fixed alpha=0.40
- **This is the simpler version** — PR #798's per-order centers are the upgrade

### 4. Prefill Cache + EBLS (PR #796 — 0.6567 bpb)
- **Source**: Open PR
- **Unclear what EBLS is** — needs investigation. Could be a major technique.

## Technique Legality Updates (Issue #140, 2026-03-25)

1. **N-gram eval cache**: LEGAL (score-first, backward-looking only)
2. **Hindsight selection** (comparing n-gram vs model on ground truth): **BANNED**
3. **Eval-time GPTQ calibration**: **BANNED** (must fit in training window)
4. **Multi-epoch TTT with min-loss selection**: **BANNED** (= training on val set)
5. **Fixed-weight blending or entropy-adaptive alpha (model uncertainty, not labels)**: LEGAL

## Recommended Action Plan

### Priority 1: Implement Order-Adaptive Entropy Gating N-gram Cache on our base

**Theory of victory**: PR #798 achieves 0.5466 on a standard 11L base. Our base (1.0705 with AdamW TTT) is stronger than average. Adding order-adaptive n-gram should yield 0.50-0.60 bpb range. Even a conservative implementation (just multi-order backoff + basic entropy-adaptive alpha like PR #727) should get us to 0.85-0.95 bpb.

**Implementation plan**:
1. Start with PR #727's approach (simpler): multi-order backoff (2-7) + `alpha = 0.05 + 0.55 * sigmoid(2 * (H - 4.0))`
2. Then upgrade to PR #798's per-order entropy centers
3. Ensure score-first compliance (cache only from already-evaluated tokens)

**We already have v9a/v9b code locally** that targets this. Key files:
- `records/track_10min_16mb/2026-03-25_sunnypatneedi_v2/train_gpt_v9a_11gram_no_ttt.py`
- `records/track_10min_16mb/2026-03-25_sunnypatneedi_v2/train_gpt_v9b_11gram_mini_ttt.py`

**RunPod commands** (after pod creation on 8xH100 SXM):
```bash
# Setup
pip install zstandard --break-system-packages
python3 -c "import zstandard; print('zstd OK')"

# Clone and checkout
cd /workspace
git clone https://github.com/sunnypatneedi/parameter-golf.git
cd parameter-golf

# Copy the n-gram version to test
cp records/track_10min_16mb/2026-03-25_sunnypatneedi_v2/train_gpt_v9a_11gram_no_ttt.py train_gpt.py

# 1-seed smoke test
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Check artifact size
python3 -c "import os; s=os.path.getsize('artifact.tar.gz'); print(f'{s:,} bytes ({s/1e6:.2f} MB) — {\"PASS\" if s < 16_000_000 else \"FAIL\"}')"

# If seed 42 bpb < 1.0 AND artifact < 16MB, run 3-seed validation
# (use the run_3seeds.sh pattern from PR #771 submission)
```

**Expected result**: 0.85-1.00 bpb (conservative), 0.55-0.75 (if per-order gating works well)
**Abort criteria**: If seed 42 bpb > 1.05, the n-gram implementation has a bug — debug before spending more.
**Estimated cost**: $8 (1-seed) to $33 (3-seed) on 8xH100

### Priority 2: Study PR #798 implementation in detail

Before GPU spend, WebFetch the PR #798 diff to understand the exact BackoffNgramMixer + entropy gating code. Port the key logic into our v9a/v9b scripts. The per-order entropy centers are the key innovation:
```python
ent_centers = {7: 3.0, 6: 3.2, 5: 3.5, 4: 3.8, 3: 4.2, 2: 4.5}
```

### Priority 3: Investigate "EBLS" technique from PR #796

PR #796 achieves 0.6567 with "Prefill Cache + EBLS". EBLS is unknown — could be a significant technique worth understanding.

## Code Changes Made

No new code written in this report cycle. Existing local versions v9a (11gram no TTT) and v9b (11gram mini TTT) were already prepared in previous session.

## Papers & Community

### Relevant Papers

**Directly applicable to our competition work:**

- **SLOT: Sample-specific LM Optimization at Test-time (arXiv:2505.12392)**: Adds a lightweight parameter vector δ to the final hidden layer, optimized on the input prompt via cross-entropy. Few optimization steps, caches last-layer features. **Potential unlock**: This is a legal TTT variant — adapts per-sample at test time by minimizing loss on the prompt itself (backward-looking). Could complement n-gram cache. The "light-weight δ on final hidden layer" is architecturally different from our current full-model TTT. Worth investigating whether SLOT-style adaptation + n-gram outperforms AdamW TTT + n-gram.

- **N-gram Residual Learning (arXiv:2210.14431)**: Trains a neural LM to fit the *residual* between an n-gram LM and the true distribution, rather than the full distribution. The neural model only needs to learn what the n-gram can't predict. **Potential unlock**: If we trained our base model with n-gram residual awareness, the neural+n-gram combination at eval time would be tighter. This is a training-time change, not just eval-time — could be worth 0.01-0.03 bpb over naive interpolation. Medium implementation cost.

- **LaCT / TTT Done Right (arXiv:2505.23884)**: ICLR 2026 Oral. Already in our technique reference — cosine + per-layer LR recipe. Our PR #771 implements this.

- **E2E TTT (arXiv:2512.23675)**: Meta-learns TTT initialization at train time. Interesting but the meta-learning phase likely exceeds our 10-min training budget. Not directly applicable.

**Quantization-specific (for squeezing more model into 16MB):**

- **LieQ (arXiv:2508.03332)**: Layer-wise mixed-precision PTQ for small LMs. Keeps uniform bit-width within each layer but mixes precision across layers based on an information-effectiveness metric. **Potential unlock**: Instead of uniform int6 everywhere, use int7 on critical layers and int5 on redundant ones. Could recover 0.002-0.005 bpb at same artifact size, or free ~200KB for a larger n-gram cache. Low implementation cost.

- **pQuant (arXiv:2602.22592)**: Decoupled linear QAT for sub-2-bit. Aggressive but our int6 regime is higher — less applicable. Note for future if we need to go lower.

- **SLMQuant (arXiv:2511.13023)**: Systematic benchmark showing SLMs are uniquely sensitive to quantization. Confirms our Lesson #8 — small models need careful quant. Validates our approach of QAT over PTQ.

**Expert mixing / ensemble theory:**

- **Lossless Compression via Next-Token Prediction (arXiv:2505.06297)**: Uses LLM predictions + arithmetic coding for lossless compression. The ensemble approach (multiple predictors) is conceptually what n-gram + neural model does. Confirms the competition meta is sound.

- **PEER: Parameter Efficient Expert Retrieval**: Uses product keys to route through millions of single-neuron experts. Interesting architecture but likely too expensive for our 10-min budget. File away for future.

### Community

- **HuggingFace**: No parameter-golf-specific community posts. TTT and test-time scaling blog posts reference the E2E TTT paper but no novel techniques.
- **DeepWiki (openai/parameter-golf)**: Has a wiki-style breakdown of the competition but blocked for direct fetch. Could contain technique cataloging.
- **Reddit**: No significant parameter golf threads found. GitHub Issue #140 remains the primary community discussion hub.
- **Competition coverage**: algo-mania.com and aitoolsclub.com have general awareness articles, no new techniques.

## Strategic Assessment

**Our position**: Our 1.0705 beats the merged SOTA (1.1194) but is ranked ~10th among open PRs. The competition has moved to n-gram territory. Without n-gram cache, we cannot compete.

**The meta**: Order-adaptive entropy gating + multi-order n-gram backoff (2-7 or 2-11) is the dominant technique. TTT is a secondary boost. The winning formula appears to be: strong 11L base + n-gram cache + entropy-adaptive mixing.

**Next session priority**: Study PR #798 diff → port order-adaptive entropy gating to our base → test on RunPod → submit.

**Budget note**: At $33/attempt, we have ~10 attempts left before deadline. Each attempt should now include n-gram cache. Pure architecture or TTT experiments without n-gram are no longer worth GPU time.
