# Parameter Golf Daily Research - 2026-04-11

## PR #771 STATUS: CLOSED (REJECTED) — FINAL

@valerio-oai: "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." No new comments. No appeal path. Fully dead.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache, unnormalized |
| #741 | 0.9850 | **CLOSED** (illegal) | Self-closed, same ruling |
| #758 | 1.0465 | **OPEN** (flagged) | MatoTeziTanka flagged: TTT contradiction + unnormalized n-gram; effectively dead, no organizer ruling |
| #731 | 1.0400 | **OPEN** | 5-expert Hedge Mixer, no new ruling |

---

## Leaderboard

- **Merged SOTA**: **1.0810** val_bpb (bigbag, PR #1493, 2026-04-09) — **NO CHANGE** since yesterday
- **Best open legal PRs (new today)**:
  - PR #1541 (bigbag, **1.07785**): SP8192 + Improved Parallel Residuals + Muon 0.97 — under clarification on hash embed flag
  - PR #1540 (aryanbhosale, **1.0777**): SP8192 + VarLen Attention + Doc-Independent LoRA TTT — batch ordering question resolved by author
  - PR #1533 (aryanbhosale, **1.0790**): SP8192 + Banking + Triple Recurrence
  - PR #1523 (EthanYangTW, 1.0778): still open with ⚠️ hash embedding flag
- **Best open (illegal)**: PR #1539 (translatingthename, 1.0587): Pre-Quant AdamW TTT — same ruling pattern as #771; flagged by reviewer
- **Bogus claim**: PR #1545 (Abhishek8108, 1.028): BUG — double-counting inflates byte count ~14%; real score ~1.18 BPB
- **Non-record**: PR #1535 (newjordan, 1.07424983) is a 4-HOUR run; 10-min legal version = 1.135 BPB

**Target unchanged**: ≤1.0760 bpb (beat merged SOTA 1.0810 by ≥0.005 nats)

---

## What Changed (GitHub)

### New Open PRs (Apr 10–11, 2026)

| PR | Author | Score | Key Technique | Legality |
|----|--------|-------|---------------|----------|
| **#1541** | bigbag | **1.07785** | Improved Parallel Residuals (cross-lane learned scalars) + Muon 0.97 + MATRIX_LR=0.03 | ⚠️ hash embed flag (logs show `ttt_hash_embed: True`) — author clarification pending |
| **#1540** | aryanbhosale | **1.0777** | VarLen Attention (within-doc only) + Doc-Independent LoRA TTT rank-96 (resets per batch) + Triton TMA MLP (+5%) | **Appears legal** — batch ordering concern resolved; LoRA resets to zero each batch |
| #1539 | translatingthename | 1.0587 | Pre-Quant AdamW TTT | **ILLEGAL** — flagged as structurally identical to #1376; train-before-score |
| #1533 | aryanbhosale | 1.0790 | SP8192 + Banking + Triple Recurrence | Legal — but no new techniques vs. plan |
| #1532 | nogakeren | 1.0803 | SP8192 + 3-Layer Recurrence + QK-Gain | Legal |
| #1535 | newjordan | 1.07424 | 7F+3C depth-recurrent hybrid | **Non-record** (4 hours); 10-min = 1.135 |
| #1545 | Abhishek8108 | ~~1.028~~ | GDN + SWA hybrid | **BUG**: byte double-counting inflates BPB by 14%; real ~1.18 |

### Technique Details: PR #1541 (bigbag) — Improved Parallel Residuals
- Cross-lane routing: attention and MLP outputs go to **both** lanes via learned scalars, not same-lane only
- Starts at layer 7 (same as merged SOTA)
- Muon momentum 0.97 + MATRIX_LR = 0.03 (co-tuned pair)
- Builds on PR #1493 merged SOTA stack
- **Note**: bigbag is the merged SOTA author — his next PR is the one to watch for merge

### Technique Details: PR #1540 (aryanbhosale) — Doc-Independent LoRA TTT
- VarLen attention masks attention to within-document tokens only (no cross-doc contamination)
- Rank-96 LoRA adapter initialized to zero each batch, trained score-first during eval
- Adapter is discarded after each document — zero state leakage across docs
- This is fundamentally different from our abandoned LoRA TTT (which was static, not score-first)
- Fused Triton TMA MLP: +5% throughput
- **Legality verdict**: Appears score-first compliant; batch ordering concern resolved

---

## New Research Papers

No new breakthrough papers today beyond those already tracked.

| Paper | arXiv ID | Notes |
|-------|----------|-------|
| pQuant: Decoupled Linear QAT | 2602.22592 | Feb 2026; 1-bit params packed UINT8; not applicable to our int6 GPTQ |
| MuonEq-R | 2603.28254 | Already tracked — confirmed row/column normalization before Newton-Schulz; O(m+n) overhead |
| Newton-Muon | 2604.01472 | Already tracked — WATCH; ~+4-6% steps |

---

## HuggingFace / Community Discoveries

None today.

---

## Recommended Action

**No change to strategy from 2026-04-10 report. Refined priorities:**

1. **WATCH PR #1541** (bigbag, 1.07785): Cross-lane Improved Parallel Residuals is a new technique not in our plan. If merged, it sets a new record and raises our target. Hash embed flag needs resolution — check again tomorrow.

2. **ADOPT PR #1540's LoRA TTT approach**: Doc-independent rank-96 LoRA resetting per batch is categorically different from the abandoned training-time LoRA TTT. Score-first, zero artifact size cost. Worth adding to our stack after ANS compression and banking.

3. **Avoid PR #1539 pattern**: Pre-quant AdamW TTT confirmed flagged — same ruling as #771.

4. **Priority stack remains** (from Apr 10 report):
   - ANS weight compression (PR #1510) — 1.6MB freed, HIGH PRIORITY
   - Parameter Banking + Parallel Muon (PR #1523) — +5.2% throughput
   - Per-Pass Loop Embeddings (PR #1518) — reduces quant gap
   - Muon 0.97 + QK-Gain 5.25 — free wins (2-line change)

---

_Updated: 2026-04-11 (v11.5 — PR #1541 bigbag 1.07785 and PR #1540 aryanbhosale 1.0777 new open PRs; PR #1539 pre-quant illegal; PR #1545 BPB bug; LoRA TTT doc-independent approach appears legal; no merged SOTA change)_

---

# Parameter Golf Daily Research - 2026-04-10

## PR #771 STATUS: CLOSED (REJECTED) — CONFIRMED

Same ruling as 2026-04-07. AdamW TTT 30ep was train-then-score. No new comments. No appeal path.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache, unnormalized — closed Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Author self-closed Mar 30, unnormalized hash |
| #758 | 1.0465 | **OPEN** | 7-gram backward-looking cache, no organizer ruling |
| #731 | 1.0400 | **OPEN** | 5-expert Hedge Mixer, no ruling ("same risk" noted) |

---

## Leaderboard — ⚠️ MAJOR UPDATE

- **Merged SOTA**: **1.0810** val_bpb (bigbag, PR #1493, 2026-04-09) — **DROP FROM 1.1147 (WAS STALE)**
- **Best open legal PR**: ~1.0778 (PR #1523, EthanYangTW, Triple Recurrence + Banking + Fused MLP)
- **Best open with SLOT**: 1.0766 (PR #1333, aryanbhosale, Causal SLOT-16, no ruling)
- **Best open (pre-quant TTT, likely illegal)**: 1.0632 (PR #1517, RulinShao)
- **Our PR #771**: 1.0705 — CLOSED/REJECTED
- **New target**: beat 1.0810 by ≥0.005 → need **≤1.0760 bpb**

### Recently Merged Records (Apr 6–9, 2026)

| PR | Author | Score | Date | Key Technique |
|----|--------|-------|------|---------------|
| **#1493** | bigbag | **1.0810** | Apr 9 | SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT |
| #1477 | aryanbhosale | 1.0822 | Apr 8 | SP8192 + Parallel Residuals + Score-First TTT |
| #1413 | dexhunter | 1.0828 | Apr 6 | SP8192 + QK-Gain 5.0 + Legal Score-First TTT |
| #1412 | Robby Sneiderman | 1.0835 | Apr 6 | SP8192 + Parallel Residuals + Hessian-aware SDClip |
| #1394 | Kevin Clark | 1.0856 | Apr 5 | SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip |

Four records merged since last report. The competition moved from 1.1147 → 1.0810 in 3 weeks. Pace is accelerating.

---

## What Changed (GitHub)

### Best Open PRs (as of Apr 10)

| PR | Author | Score | Techniques | Legality |
|----|--------|-------|------------|----------|
| **#1523** | EthanYangTW | **1.0778** | Triple Recurrence (blocks 3-4-5 repeated) + Parameter Banking + Fused MLP (Triton TMA) + Muon 0.97 + Eval-Time Hash Embedding | ⚠️ Eval-Time Hash Embedding potentially illegal |
| **#1518** | abaybektursun | **1.078825** | Wider Loop (blocks 3-5 × 3 passes) + Per-Pass Embeddings + Tap-In V6 + Legal TTT | ⚠️ Tap-In V6 legality unclear |
| **#1514** | dexhunter | **1.07983** | SP8192 + Muon 0.97 + N-gram Tilt (causal) + Legal TTT | **Legal** |
| **#1333** | aryanbhosale | 1.0766 | Causal SLOT-16 (16-step AdamW delta on scored positions) | ⚠️ Unruled |
| **#1517** | RulinShao | 1.0632 | Depth Recurrence + Banked Muon + Pre-Quant TTT 18ep | **ILLEGAL** — pre-quant TTT |

### New Technique Deep-Dives

**1. ANS Weight Compression (PR #1510, OE-GOD, OPEN)**
- Replaces LZMA with per-layer rANS (range Asymmetric Numeral Systems) encoding
- Uses per-layer histogram frequencies to encode int6 symbols at near-entropy-limit
- **Claimed: 1.6MB lossless savings** → within 11KB of theoretical entropy limit
- **Impact: ~2.2M extra parameters within the same 16MB budget**
- No model quality change — purely compression efficiency
- No organizer flags, no legality issue
- **RECOMMENDATION: HIGH PRIORITY — implement before next GPU run**

**2. Per-Pass Loop Embeddings (PR #1518, abaybektursun)**
- 3 learned 512-dim vectors added to residual stream before each loop pass
- Allows model to differentiate loop iterations without weight sharing workaround
- **Reduces quantization gap from 0.0131 → 0.0114** (tighter post-quant performance)
- Δbpb: small standalone, compounds with GPTQ calibration improvement
- Low implementation cost (~10 lines)
- **RECOMMENDATION: ADD to next run after ANS compression**

**3. Wider Depth Recurrence (PR #1518)**
- Loops blocks 3-5 (3 blocks) × 3 passes = 9 total executions, vs blocks 4-5 (2 blocks) × current
- Same compute but more parameter diversity per loop pass
- Better post-quantization performance (fewer params per block = less GPTQ error)
- **RECOMMENDATION: Evaluate vs. current Triple Loop in ablation (1xH100)**

**4. Parameter Banking + Parallel Muon (PR #1523, EthanYangTW)**
- Consolidates 66 weight matrices into 4 contiguous memory banks
- Enables batched Newton-Schulz iterations: **15× faster optimizer step**
- Throughput: +3.8% from banking alone, +5.2% combined with Fused MLP
- Zero impact on model quality; purely systems optimization
- Pairs naturally with existing Fused Kernels (Triton TMA)
- **RECOMMENDATION: HIGH PRIORITY — add +5.2% throughput for free steps**

**5. Muon Momentum 0.97 (PRs #1514, #1523)**
- Reduce momentum from default 0.99 to 0.97
- Δbpb: **-0.0004** standalone
- Zero implementation cost (1 hyperparameter change)
- **RECOMMENDATION: Change immediately**

**6. QK-Gain 5.25 (PR #1493, merged SOTA)**
- Current plan uses 5.0 from PR #1334/#1420
- Monotonic improvement: 5.25 was tested in the merged SOTA (1.0810)
- **RECOMMENDATION: Change 5.0 → 5.25**

**7. Tap-In V6 (PR #1518, abaybektursun) — LEGALITY UNCLEAR**
- Document-local matching at eval time: scans backward in same document for matching phrase continuations, nudges probability distribution upward
- Effectively a document-scope n-gram-style hint (not pre-loaded cache)
- ⚠️ Needs @valerio-oai ruling before implementing. Similar spirit to n-gram tilt but operating on already-seen document context. May be legal if causal.
- **RECOMMENDATION: Do NOT implement until ruled on. Watch PR #1518 for organizer feedback.**

**8. Eval-Time Hash Embedding (PR #1523) — SUSPECT**
- 16384×512 bigram hash table with zero initialization, trained during inference
- This is adapting parameters during evaluation — similar pattern to illegal pre-quant TTT
- ⚠️ May violate "no adaptation before scoring" rule
- **RECOMMENDATION: Do NOT implement. High rejection risk.**

**9. BPB-Weighted Training Loss (PR #1519, definenoob) — WITHDRAWN**
- Weight each token's CE loss by UTF-8 byte count to align training with BPB metric
- Author retracted ("had an out of date repo")
- Only effective for small vocabs (not SP8192)
- **SKIP**

---

## New Research Papers

| Priority | Paper | arXiv ID | Δ bpb est. | Notes |
|----------|-------|----------|-----------|-------|
| **HIGH** | rANS for Neural Network Feature Compression | 2511.11664 | — | Theory behind PR #1510 ANS compression; confirms approach is sound |
| **HIGH** | Newton-Muon Optimizer | 2604.01472 | ~+6% fewer steps | Interprets Muon as Newton; 4% wall-clock reduction; already in CLAUDE.md "Watch" |
| Medium | MuonBP (Block-Periodic Muon) | openreview | ~+8% throughput | Block-periodic Newton-Schulz; bridges throughput gap; less data-efficient than Muon |
| Medium | LaCT: Large Chunk TTT | 2505.23884 | GPU util 0→70% | Large-chunk post-quant TTT; already tracked |
| Low | End-to-End TTT for Long Context | 2512.23675 | N/A | Long-context meta-learning; not applicable at our scale |

**No new breakthrough papers found beyond those already tracked in CLAUDE.md.**

---

## HuggingFace / Community Discoveries

- None today beyond GitHub PR activity.

---

## Recommended Action

**New target: ≤1.0760 bpb (beat merged SOTA 1.0810 by ≥0.005 nats)**

**Priority 0 — Zero-cost wins (do before next GPU run):**
1. Muon momentum 0.97 (was 0.99) — free -0.0004 bpb
2. QK-Gain 5.25 (was 5.0) — free improvement per merged SOTA

**Priority 1 — Systems gains (no quality regression):**
3. ANS weight compression (PR #1510) — 1.6MB freed → +2.2M params capacity; implement lossless
4. Parameter Banking + Parallel Muon (PR #1523) — +5.2% throughput → ~+30 extra training steps

**Priority 2 — Architecture improvements (1xH100 ablation first):**
5. Per-Pass Loop Embeddings (PR #1518) — reduces quant gap; low implementation cost
6. Wider Recurrence (blocks 3-5 × 3 passes vs blocks 4-5) — ablate vs current Triple Loop

**Do NOT implement:**
- Pre-Quant TTT any form (PR #1517) — illegal
- Eval-Time Hash Embedding (PR #1523) — suspect legality
- Tap-In V6 (PR #1518) — await ruling
- N-gram hash cache (PRs #727, #741 pattern) — illegal

**Watch:**
- PR #1518 for Tap-In V6 organizer ruling from @valerio-oai
- PR #1333 for Causal SLOT-16 ruling (if ruled legal, implement for additional -0.003 bpb)
- PR #1523 being evaluated by organizers (Eval-Time Hash Embedding may be flagged)

---

_Updated: 2026-04-10 (v11.4 — Merged SOTA updated 1.1147→1.0810; 4 new records; new target ≤1.0760; ANS compression HIGH priority; Parameter Banking HIGH priority)_

---

# Parameter Golf Daily Research - 2026-04-07

## PR #771 STATUS: CLOSED (REJECTED)

Rejected by @valerio-oai on 2026-03-27: "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." No appeal path. Our AdamW TTT 30ep approach is fully void.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache, unnormalized — ruled out Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Same ruling as #727 — author self-closed Mar 30 |
| #758 | 1.0465 | **OPEN** | 7-gram backward-looking cache, legality unresolved |
| #731 | 1.0400 | **OPEN** | 5-expert Hedge Mixer, legality unresolved ("same risk" noted) |

---

## Leaderboard

- **Merged SOTA**: 1.1147 val_bpb (abaybektursun, PR #1019, 2026-03-25) — **UNCHANGED**
- **Best open legal PR**: 1.08014 (PR #1420, abaybektursun, SP8192 + Triple Loop + N-gram Tilt + Fused Kernels)
- **PR #1430 (today)**: 0.39642 — OPEN, **LIKELY ILLEGAL** (see analysis below)
- **Our PR #771**: 1.0705 — CLOSED/REJECTED

---

## What Changed (GitHub)

### CRITICAL: PR #1430 — 0.39642 val_bpb (renqianluo, Apr 7, 2026) ⚠️ LIKELY ILLEGAL

The most dramatic claim in competition history. **Status: OPEN, no organizer reviews yet.**

**Techniques:**
1. **Per-Sample SLOT**: Each sequence gets `[bsz,1,512]` hidden delta + `[bsz,1,1024]` logit bias = 1536 params/sequence. AdamW 24 steps, cosine LR 0.432→0.001, β₁=0.6, β₂=0.5
2. **Causal Backoff N-gram Mixer (order 2–22)**: 4M hash buckets, entropy-adaptive alpha: `0.20 + 0.55 * sigmoid(2*(H-2.5))`. Claims "causal" (only backward-looking tokens).
3. **TTT**: Second pass over first 10% of chunks, AdamW 1 epoch, lr=0.0001, freezing blocks 0–9
4. **GPTQ damp=0.005**

**Legality analysis:**
- **N-gram hash cache (4M buckets)**: Almost certainly illegal — same pattern as #727/#741 which were closed for "hashed n-gram caches that don't renormalize correctly." Unless renormalized over full vocab (unlikely with 4M hash buckets), this will be ruled out.
- **Per-Sample SLOT (δ-vector)**: Still UNRULED under Issue #140. @valerio-oai has not issued a ruling. SLOT standard variant was showing -0.021 bpb when @abaybektursun removed it pre-emptively. Per-sample SLOT is even more aggressive.
- **TTT structure**: "Second pass over first 10% of chunks" — needs score-first validation. If it trains on already-scored tokens, it may be legal, but the "second pass" framing is ambiguous.

**Verdict**: Do NOT implement until @valerio-oai rules on this PR. High probability of rejection on n-gram grounds alone. Watch for organizer response.

**Artifact sizes**: 15.86–15.90MB (within 16MB). Timing: train~600s, eval~587–595s (within budget). 3-seed mean: 0.39642.

### Other New PRs (Apr 6–7, 2026)

| PR | Author | Score | Notes | Legal? |
|----|--------|-------|-------|--------|
| **#1437** | dexhunter | **1.08091** | SP8192 + Parallel Residuals + 3-Layer Recurrence + N-gram Tilt | **YES — but reveals causality bug (see below)** |
| #1423 | aryanbhosale | 1.0791 | SP8192 + Pre-Quant TTT + QK-Gain 5.0 | **ILLEGAL — flagged by abaybektursun: "fine-tuning on val data for 6 epochs before quantization"** |
| #1424 | OnlyJundong | 1.0858 | Extended compute (50K steps) | Non-record (>10 min) |
| #1421 | X-Abhishek-X | 1.0925 | Depth Recurrence + EMA tuning | Legal |
| #1435 | AbhayAnandUCSD | 1.0980 | Depth Recurrence + BigramHash + EMA 0.9965 | No flags |
| #1440 | Mertyandimata | 1.1026 | EngramLite + Mousse + Progressive Depth Recurrence + TTT | No flags |

### ⚠️ N-gram Tilt Causality Bug (PR #1437)

**PR #1437 independently found and disclosed a causality bug in the N-gram Tilt kernel that also affects PR #1420.** dexhunter's results:
- Pre-fix (bugged): 1.07807 bpb
- Post-fix (correct causal): 1.08091 bpb

This means **PR #1420's reported 1.08014 may include a non-causal n-gram tilt implementation.** Impact: ~0.003 bpb worse when bug is fixed. When implementing N-gram Tilt, use the corrected kernel from PR #1437 (not #1420). PR #1420 may face a correction request from reviewers.

### Issue #140 — CLOSED (Apr 6)

**Major update**: @notapplica closed the issue on Apr 6. @valerio-oai **never commented in Issue #140**. All official rulings came from PR comments and Issue #677.

**SLOT status — dramatically different from what we believed:**
- **9 record PRs use SLOT variants** with no organizer rejection.
- **PR #1333** (aryanbhosale, 1.0766 BPB): Causal SLOT-16, filed as record, OPEN — best legal-ish open PR
- **PR #1229** (scored-position SLOT, 0.9300 BPB): Extraordinary; no organizer rejection
- @abaybektursun's self-removal was a personal choice (causality concern), not an official ruling
- @valerio-oai has never banned SLOT in any venue

**Implication**: SLOT is de facto in use. The risk is @valerio-oai could rule on any SLOT PR at any time and retroactively reject it. Two tracks:
- **Safe track**: PR #1437 stack + legal TTT → ~1.081 bpb, zero rejection risk
- **SLOT track**: PR #1333 approach (Causal SLOT-16) → ~1.077 bpb or better, non-trivial rejection risk

**ETLB**: Still unruled, not mentioned in Issue #140 at all.

---

## New Research Papers

No breakthrough new papers found beyond those already tracked in CLAUDE.md. Confirmed papers:

| Priority | Paper | arXiv ID | Δ bpb est. | Risk |
|----------|-------|----------|-----------|------|
| **NOW** | MuonEq: Balancing Before Orthogonalization | 2603.28254 | ~-0.005 | Low |
| **NOW** | Compute-Optimal QAT (cooldown+QAT fusion) | 2509.22935 | ~-0.002 | Low |
| Medium | LaCT: Large Chunk TTT | 2505.23884 | ~-0.003 to -0.008 (GPU util 0→70%) | Medium |
| Medium | Sparse Growing Transformer (SGT) | 2603.23998 | ~-0.001 to -0.003 indirect (saves FLOP) | Medium |
| Medium | Two-Scale Latent Dynamics / early-exit recurrence | 2509.23314 | ~-0.001 indirect | Medium |
| Watch | Newton-Muon | 2604.01472 | ~+4-6% steps | Medium (new, Apr 2026) |
| Watch | **MUD (MomentUm Decorrelation)** | **2603.17970** | **+20-50% throughput** | **Medium — quality vs speed tradeoff** |
| Watch | **Mousse** | **2603.09697** | **~-0.002 to -0.003** | **Medium-Hard — Kronecker overhead** |
| Skip | ByteFlow (byte-level LM, no tokenizer) | 2603.03583 | N/A | Not applicable — incompatible with eval metric |

### NEW Papers Not Previously Tracked

**MUD: MomentUm Decorrelation (arXiv:2603.17970, Mar 2026)**
Replaces Muon's Newton-Schulz polar decomposition with triangular (Cholesky-like) whitening via Gauss-Seidel solves. FLOPs per step ~12× smaller than Muon's Newton-Schulz. Results: 1.3–2.6× peak tokens/sec vs. Muon on most settings; up to 3× on GPT-2 large on A100; 10–50% wall-clock improvement in time-to-perplexity.
- **Competition impact**: If the throughput gain holds on H100s, could mean +20–50% more training steps. However, per-step convergence quality vs. MuonEq-R is unknown. Run a 1xH100 ablation: MuonEq-R vs. MUD vs. Mousse before committing.
- **Implementation**: Medium — replace Newton-Schulz iteration with triangular solve.

**Mousse: Rectifying the Geometry of Muon (arXiv:2603.09697, Mar 2026)**
Adds Kronecker-factored (Shampoo-style) preconditioning to Muon's polar update. Operates in a whitened coordinate system; polar decomposition applied after whitening.
- ~12% reduction in training steps vs. Muon on 160M–800M models.
- **Competition impact**: ~-0.002 to -0.003 bpb. Overhead of Kronecker factors at 8xH100 scale unknown. Lower priority than MuonEq-R; try only if MuonEq-R plateaus.

**Confirmed existing papers (no change):**
- "Thinking Deeper, Not Longer" (arXiv:2603.21676): Not applicable — focuses on compositional generalization with 20+ recurrence steps + silent objective. Triple Loop in PR #1420 is the competition-tuned variant.

---

## HuggingFace / Community Discoveries

- None found today beyond GitHub PR activity.
- NGPU-LM (arXiv:2505.22857) — GPU-accelerated n-gram LM for context biasing in ASR. Interesting for fast n-gram hash table implementation, but legal path requires proper normalization (not hashed).

---

## Recommended Action

**Two tracks — explicit decision required before GPU spend:**

**Track A (safe, zero rejection risk):**
- SP8192 + Triple Loop + N-gram Tilt (**use PR #1437 corrected causal kernel**) + Legal Score-First TTT (3ep, PR #1413) + Fused Kernels
- Expected: ~1.078 bpb. Delta vs merged SOTA: ~0.036 nats.
- Critical: Do NOT copy N-gram Tilt from PR #1420 — use PR #1437's corrected implementation.

**Track B (higher EV, explicit rejection risk):**
- Causal SLOT-16 + same base (equivalent to PR #1333, 1.0766 BPB)
- Issue #140 CLOSED Apr 6; @valerio-oai never ruled on SLOT; 9 open record PRs use it.
- Risk: @valerio-oai could close any SLOT PR at any time without prior warning.
- **Recommendation**: File Track A first as the safe submission. Then file Track B as a second PR if Track A is clean. Two shots at the leaderboard.

**Do NOT implement:**
- PR #1430 techniques (per-sample SLOT + N-gram order-22 hash) — await organizer ruling
- ETLB (unruled)
- Pre-quant TTT (all variants illegal)

**Watch:**
- PR #1430 for organizer response — if N-gram hash is ruled legal this would change everything (unlikely)
- PR #1333 and PR #1229 for any SLOT rulings from @valerio-oai on the actual PRs

**Best reachable target:**
- Track A: ~1.078 bpb (guaranteed legal)
- Track B: ~1.073–1.077 bpb (SLOT risk)
- Both beat merged SOTA 1.1147 by >0.035 nats — well above the 0.005 threshold

---

_Updated: 2026-04-07 (v11.3 — Issue #140 CLOSED; SLOT de facto in use (9 PRs); PR #1333 (1.0766 causal SLOT) new best-open; two-track strategy; merged SOTA unchanged 1.1147)_
