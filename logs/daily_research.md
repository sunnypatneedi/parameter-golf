# Parameter Golf Daily Research - 2026-04-27

## PR #771 STATUS: CLOSED (ILLEGAL — no change)

@valerio-oai ruling (2026-03-27): train-then-score AdamW TTT 30ep = instant disqualification. Permanent.

---

## N-GRAM PR STATUS

- **PR #727**: CLOSED — permanent (illegal hash cache, no renormalization).
- **PR #758**: OPEN but dead — XOR hash key includes target token (flagged by MatoTeziTanka Apr 12). No fix from author.
- **PR #731** (Hedge Mixer — dense count tables + Laplace smoothing): OPEN — "LOOKS CLEAN" per reviewer. Seeds 1337 and 2024 still PENDING. No merge. No new activity.

---

## Leaderboard

- **Official Merged SOTA (README)**: **1.0810** — bigbag (PR #1493, Apr 9). **Day 18 plateau** — longest in competition history. No merges since Apr 9.
- **Scylla**: Officially removed by OpenAI commit `7427de2` (Apr 26). Definitively dead. Merged SOTA is 1.0810.
- **Our PR #771**: CLOSED/ILLEGAL.
- **Target**: ≤1.0760 bpb. **3 days to deadline (Apr 30).**

---

## What Changed Since Apr 26 (GitHub)

### PPM-D Wave — Now Dominant Competition Technique

Multiple PRs filed Apr 26–27 all use byte-level PPM mixture. dexhunter independently validated the technique at 1.0322 (PR #1857) before self-closing in favor of the earlier-filed PR #1850. This cross-validation by the most reliable competition contributor is strong signal the mechanism works. No organizer ruling yet.

| PR | Author | Score | Technique | Status | Notes |
|----|--------|-------|-----------|--------|-------|
| **#1848** | newjordan | **0.87980** | "12L SP4096 + brotli + mixed-int + score-first TTT" | OPEN | ⚠️ **BPB RISK** — extraordinary score; sibling PR #1846 (0.87206) CLOSED same day with no explanation; no community BPB verification; do NOT implement |
| **#1846** | newjordan | **0.87206** | "Raphe" (similar to #1848) | **CLOSED Apr 27** | Self-closed, no explanation; artifact 13.49MB — **suspicious closure suggests BPB bug** |
| **#1854** | ndokutovich | **0.90236** | PPM-D byte mixture on PR #1797 base | OPEN | Explicitly documents Issue #1017 compliance (causality + normalized + score-before-update + single pass). 15.95MB artifact. No reviewer comments. |
| **#1858** | G3sparky | **0.9946** | Score-First TTT + PPM-D byte mixture | OPEN | ⚠️ **PARTIAL DATA** — @dexhunter flagged: score computed on first 8M tokens only (~20% of full 40.5M val set). NOT comparable to full leaderboard. Do not track. |
| **#1852** | G3sparky | **1.0282** | Pre-Quant TTT + Void Compass | OPEN | Pre-quant TTT = **ILLEGAL**. |
| **#1850** | someone114514 | **1.00495** | Strict Full-Val Byte PPM Mixture | OPEN | 15.997MB (2,567 bytes under cap). Score-before-update documented. Priority watch — preceded #1857. |
| **#1857** | dexhunter | **1.0322** | PR #1787 + SmearGate + LQER + PPM-D (OpenMP parallelized) | **CLOSED** | Self-closed: yielded to PR #1850 (earlier filing). dexhunter independently validates PPM-D mechanism. High credibility signal. |
| **#1861** | Hetul803 | ~0.8997 est | SkipQuant Adapter TTT + PPM-D | **CLOSED** | Closed before any reviews. Possible BPB issue. |
| **#1863** | seinare | unknown | Arterial Mixer (N-Lane Transformer) | **CLOSED** | Architecture-only attempt, no competitive score. |

### Clean Legal PRs (Apr 26–27)

| PR | Author | Score | Technique | Notes |
|----|--------|-------|-----------|-------|
| **#1855** | codemath3000 | **1.06108** | SP8192 + LQER Asym int4 + Sparse Attn Gate + **SmearGate BOS Fix** + lrzip compression | PR #1797 base. SmearGate BOS fix addresses prev-token leak at document boundaries. Clean. |
| **#1851** | aquariouseworkman | **1.06128** | **SmearGate BOS Fix** + PR #1787 Base + Phased TTT | BOS fix confirmed independently by two authors. |
| **#1826** | EthanYangTW | **1.0770** | SP8192 + Polar Express + SmearGate + AttnOutGate + 4ep TTT | 4ep TTT flag. Appears clean otherwise. |
| **#1809** | PranavViswanath | **1.0800** | SP8192 + Gram-NS + Polar Express + 3-Layer Recurrence + Parallel Residuals | Gram-NS + Polar Express both in same submission. |

### SmearGate BOS Bug — IMPORTANT FIX

Both PR #1855 and PR #1851 independently fix the same bug: SmearGate's prev-token lookback leaks **across document boundaries** at BOS positions (BOS token has no prev-token to look back at). Fix: mask the prev-token term wherever `current_token == BOS`. **Any implementation of SmearGate must include this fix.**

### PR #1835 (anmarhindi, 1.00136) — 24h Watch Period Elapsed

- Still OPEN. No reviewer comments. No @valerio-oai ruling on PPM-D legality.
- No community BPB bug flagged after 24+ hours. Unlike GDN/Scylla BPB bugs (caught within hours by community), PR #1835 has survived initial scrutiny.
- **Assessment**: Technique credibility HIGH. Legality ruling is the only remaining gate.

### Issue #1604 (CaseOps ruling) — Day 14 Silence

No @valerio-oai response. 14 days since filing, 3 days since self-deadline passed. Do NOT wait. Proceed with clean legal stack only.

---

## New Research Papers

### arXiv:2604.07822 — "Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers" (Apr 2026)
- Models looped transformers as implicit solvers for fixed-point equations. Each "loop step" acts as a refinement pass toward a converged representation. Provides geometric analysis of why additional loop iterations improve performance on reasoning tasks.
- **Relevance**: Directly supports our Triple Loop (3× recurrence). Suggests that longer loops can encode "depth-first" reasoning, making depth curriculum (arXiv:2511.07384) even more motivated. Also supports Parcae-style outer normalization to stabilize fixed-point convergence.
- **Action**: No immediate code change. Confirms the architecture direction.

### Gram Newton-Schulz (Dao-AILab)
- PR #1809 (PranavViswanath, 1.0800) uses both Gram-NS and Polar Express NS simultaneously — confirms they are complementary, not redundant. Score of 1.0800 vs PR #1493 base of 1.0810 suggests minimal gain on top of existing stack, but it's a free optimizer change.
- **Caveat**: Requires CUDA 12.9+ and PyTorch 2.7.1+. Check hardware before using.

---

## HuggingFace / Community Discoveries

- **dexhunter validating PPM-D** (PR #1857, self-closed) is the strongest community signal yet that PPM-D byte mixture is a real technique. When the most reliable author in the competition independently reproduces a technique at 1.0322 before deferring to an earlier PR, the mechanism is confirmed.
- **PR #1846 closure pattern matches BPB bug**: newjordan's PR #1846 (0.87206) was closed the same day it was filed (Apr 27), with no explanation and an unusually small 13.49MB artifact. This pattern (extraordinary score + fast self-close) strongly resembles prior BPB bug cases. PR #1848 (0.87980, still open) should be treated with same skepticism.
- **Competition final sprint**: 15+ new PRs opened Apr 26–27 alone. Merge decisions will likely happen in final 2 days. File before Apr 29 to maximize review window.

---

## Recommended Actions (Priority Order, 3 days to deadline)

1. **GPU RUN TODAY — final submission window**:
   - PR #1787 base (Polar Express NS + MIN_LR=0.10 + Fused CE)
   - Per-Layer Adaptive GPTQ MLP=12σ/Attn=13σ + int7 Emb@15σ (PR #1586)
   - Attention Output Gate **+ SmearGate with BOS fix** (PR #1667 + #1855/#1851 BOS fix)
   - LoRA-TTT warm-start A + alpha=144 + WD=1.0 (PR #1767)
   - Target: ~1.066–1.072 bpb. File as PR before Apr 29.

2. **WATCH PR #1854 (0.90236) and PR #1850 (1.00495)** for organizer review. If PPM-D is ruled legal (same as score-first TTT: score-before-update), add as pure eval-time layer on top of submitted artifact. No retraining needed.

3. **DO NOT implement**:
   - PR #1848 (0.87980) — BPB risk (sibling PR #1846 closed same-day)
   - PR #1846 (0.87206) — CLOSED, likely BPB bug
   - PR #1858 (0.9946) — partial data (8M/40.5M tokens only)
   - Pre-quant TTT: PR #1852 and all others — illegal
   - CaseOps: Issue #1604 unruled
   - PR #1813 (Scylla 0.94166) — parent PR reverted by OpenAI

4. **Low-priority**:
   - PR #731 (Hedge Mixer, 1.0400): if seeds confirm, provides n-gram mixer blueprint. 3 days left — unlikely to merge in time.
   - PR #1812 (4ep TTT, 1.0729): if organizer rules 4ep legal, upgrade from 3ep for free gain. Watch for ruling.

---

*Research session: 2026-04-27 | Days to deadline: 3*

---

# Parameter Golf Daily Research - 2026-04-26

## PR #771 STATUS: CLOSED (ILLEGAL — no change)

@valerio-oai ruling (2026-03-27): train-then-score AdamW TTT 30ep = instant disqualification. Permanent. Score of 1.0705 is void.

---

## N-GRAM PR STATUS

- **PR #727**: CLOSED — permanent (illegal hash cache, no renormalization).
- **PR #758**: OPEN but dead — XOR hash key includes target token (flagged by MatoTeziTanka). No fix.
- **PR #731** (Hedge Mixer — dense count tables + Laplace smoothing): OPEN — "LOOKS CLEAN" per reviewer. Seeds 1337 and 2024 still PENDING. Not merged. No new activity.

---

## Leaderboard

- **Official Merged SOTA (README)**: **1.0810** — bigbag (PR #1493, Apr 9). **Day 17 plateau** — longest in competition history.
- **CRITICAL (TODAY)**: Upstream commit `7427de2` (Apr 26, Alex Zhao / OpenAI): **Scylla 0.9485 (PR #1184, icryo) REMOVED as invalid** ("Remove invalid Scylla record"). Also removed a non-record Muon TTT submission. The disputed record is officially dead. Merged SOTA is **definitively 1.0810**.
- **Our PR #771**: CLOSED/ILLEGAL.
- **Target**: ≤1.0760 bpb. **4 days to deadline (Apr 30).**

---

## What Changed Since Apr 25 (GitHub)

### Organizer Action — Scylla Reverted (CRITICAL)
Commit `7427de2` by Alex Zhao (OpenAI, Apr 26 00:49 EDT):
- "Remove invalid Scylla record" — deletes `track_10min_16mb/2026-03-31_Scylla_FullGPTQ_XSA_FA3/` folder (PR #1184, 0.9485 BPB)
- "Remove non-record Muon TTT submission"
- Updates README leaderboard accordingly (confirmed 1.0810 SOTA)

**Impact**: Any Scylla-based claim (PR #1813, 0.94166) now faces **increased organizer scrutiny** given the parent PR was reverted. Do NOT invest in Scylla. PR #1813 BPB risk is now effectively confirmed by proxy.

### New PRs opened Apr 26 (TODAY)

| PR | Author | Score | Technique | Legality |
|----|--------|-------|-----------|---------|
| **#1835** | anmarhindi | **1.00136** | SP8192 + PPM-D order-5 byte mixture (binary-λ gate) | ⚠️ WATCH — most credible extraordinary claim yet; score-first documented; 24h for community BPB check |
| **#1834** | ghrua | **1.08034** | NgramRes (3-gram MLP, α=0.3, +0.6M params) + Sliding-Window Attn (layers 0-3, window=512) + Legal TTT | Appears legal — references PR #1493 precedent |
| **#1837** | X-Abhishek-X | 1.07063 | Non-record: E2E TTT (full-model SGD per chunk) | Non-record, 10min+ |
| **#1836** | hardik-bhalekar | Unknown | "hardik_top5_run submission package" | Score unknown |
| **#1826** (DRAFT) | EthanYangTW | 1.0770 | SP8192 + Polar Express + SmearGate + AttnOutGate + 4ep TTT | Draft — 4ep legality pending |

**PR #1835 detail** (highest priority to monitor):
- PPM-D (Prediction by Partial Matching, order-5) at byte level. Binary-λ gate: λ=0.05 (trust PPM) when PPM top-symbol ≥0.9, else λ=0.9 (trust NN). Mix in probability space before BPB.
- Score-first: PPM state updated **after** each byte is scored — explicitly compliant per author.
- Artifact: **15,993,020 bytes** (6,980 bytes under 16 MB cap — very tight).
- 3-seed mean 1.00136, std 0.00111.
- No BPB bug flags in visible comments (as of Apr 26 morning). Substantially different technique from the GDN/Scylla BPB bugs.
- **Wait 24-48h for community review before implementing.**

**PR #1834 detail** (stackable, modest):
- NgramRes: small 3-gram MLP component (+0.6M params) mixed with main model output via learned α=0.3.
- Sliding-Window Attention: window=512 on layers 0-3 only, full causal attention on remaining layers.
- Achieves 1.08034 — slight beat of merged SOTA (1.0810) but below our target of ≤1.0760.
- **Legality**: appears clean. Could stack NgramRes onto our stack.

### Status of previously-tracked PRs (Apr 26 update)

| PR | Author | Score | Status | Notes |
|----|--------|-------|--------|-------|
| **#1813** | djeidy | 0.94166 | OPEN | Scylla-based — parent (PR #1184) reverted today as INVALID. Treat as dead. |
| **#1812** | EthanNing | 1.0729 | OPEN | 4ep TTT: reviewer questioned attribution (may not be from 4ep). No organizer ruling. |
| **#1797** | dexhunter | 1.06157 | OPEN | Best clean dexhunter PR. SmearGate + LQER Asym on PR #1787. No new comments. |
| **#1787** | nprime06 | 1.06335 | OPEN | Best community base. GPTQ timing concern raised — submitter says no gradients in GPTQ step. No organizer ruling. |
| **#1667** | MarioPaerle | 1.07139 | OPEN | Attention Output Gate + SmearGate. Clean. Stack on #1586. |
| **#1727** | yahya010 | 1.07217 | OPEN | MP-SGD TTT 4-phase. Appears legal. Stackable. |
| **#1795** | OE-GOD | 1.01252 | OPEN | PPM order-4 mixture. No organizer ruling. DO NOT implement. |
| **#1771** | bigbag | 1.06513 | OPEN | CaseOps + Depth Curriculum. Awaits Issue #1604. |
| Issue #1604 | — | — | **NO RULING** | 12+ days silence. Self-deadline Apr 24 passed. Proceed without CaseOps. |

---

## New Research Papers

### Gram Newton-Schulz — Chebyshev variant (arXiv:2506.10935)
- "Accelerating Newton-Schulz Iteration for Orthogonalization via Chebyshev-type Polynomials"
- Theoretically derives optimal 3rd-order NS coefficients via Chebyshev's alternance theorem. Applies Remez algorithm for higher-degree polynomials.
- Complementary to Polar Express (arXiv:2505.16932) — different derivation path, both improve convergence.
- **Relevance**: If achievable on H100 SXM pods, could be a slight upgrade over standard Polar Express NS.
- **Action**: Read implementation notes before trying. Verify CUDA/PyTorch requirements same as Gram-NS.

### End-to-End TTT for Long Context (arXiv:2512.23675, Dec 2025)
- Next-token prediction as TTT objective, compresses context into weights as model reads validation.
- For 3B models on 164B tokens: scales with context length equivalent to full attention Transformer.
- **Relevance**: Provides theoretical grounding for our score-first TTT approach. The "compress-as-you-read" framing is exactly our legal TTT paradigm.
- **Action**: No implementation needed — we're already aligned with this paradigm.

---

## HuggingFace / Community Discoveries

- Competition is at **final sprint**: 7 new PRs opened Apr 26 alone (PRs #1831–1837). Submission rate increasing dramatically in final 4 days.
- PPM-D technique (PR #1835) is a new variant of PPM beyond the order-4 approach in PR #1795. "D" suffix likely = "depth" or "deterministic" weighting. If community confirms clean BPB: sub-1.001 is legitimate and would represent a ~0.079 bpb improvement over merged SOTA.
- PR #1834 NgramRes approach (+0.6M params for 3-gram MLP) could complement our architecture — adds n-gram signal without the legality questions of hash caches.

---

## Recommended Actions (Priority Order, 4 days to deadline)

1. **GPU RUN TODAY — clean legal stack** (4 days remain, no more delays):
   - Base: PR #1493 + Polar Express NS + MIN_LR=0.10 (PR #1787 changes)
   - Per-Layer Adaptive GPTQ MLP=12σ/Attn=13σ + int7 Emb@15σ (PR #1586)
   - Attention Output Gate + SmearGate 1,056 params init-0 (PR #1667)
   - LoRA-TTT warm-start A + alpha=144 + WD=1.0 (PR #1767)
   - Target: ~1.068–1.072 bpb. Beats merged SOTA by ≥0.008 nats with margin.

2. **Monitor PR #1835** (PPM-D, 1.00136) for 24h: If community confirms no BPB bug by Apr 27, this is the single most important technique to add. Implementation: pure eval-time, no artifact weight change needed. Could be added on top of any clean stack.

3. **DO NOT implement**:
   - PR #1813 (Scylla 0.94166) — parent PR reverted as invalid today
   - CaseOps (no ruling, Issue #1604 silent 12+ days)
   - PR #1795 (PPM order-4, no organizer ruling)
   - Any pre-quant TTT variant (illegal)
   - PR #1758, #1735, #1738 (illegal chain)

4. **Low-priority watch**:
   - PR #731 (Hedge Mixer, 1.0400) — 2 seeds pending. If merged before Apr 30, NgramRes from PR #1834 confirms n-gram mixtures are a viable path.
   - PR #1812 (4ep TTT, 1.0729) — if 4ep gets an organizer green light, upgrade our TTT from 3→4ep for free improvement.

---

*Research session: 2026-04-26 | Next check: 2026-04-27 | Days to deadline: 4*

---

# Parameter Golf Daily Research - 2026-04-25

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed, no change)

Same as Apr 24. @valerio-oai ruling (2026-03-27): train-then-score AdamW TTT 30ep = instant disqualification. Permanent.

---

## N-GRAM PR STATUS

- **PR #727**: CLOSED — permanent (valerio-oai: hash caches don't renormalize correctly).
- **PR #758**: OPEN but dead — XOR hash key includes target token, flagged by MatoTeziTanka. No fix.
- **PR #731** (Hedge Mixer — dense count tables + Laplace smoothing): OPEN — still awaiting seeds 1337 and 2024. "LOOKS CLEAN" per reviewer. No merge yet.

---

## Leaderboard

- **Official Merged SOTA (README)**: **1.0810** — bigbag (PR #1493, Apr 9). **Day 16 plateau** — longest in competition history.
- **Disputed Scylla in repo**: 0.9485 (PR #1184, icryo) — folder in `track_10min_16mb/` but README not updated, byte accounting disputed (~1.1289 corrected per PR #1271). Treat as **UNVERIFIED**.
- **Our PR #771**: CLOSED/ILLEGAL.
- **Target**: ≤1.0760 bpb. **5 days to deadline (Apr 30).**

---

## What Changed Since Apr 24 (GitHub)

### New PRs opened Apr 25 (TODAY)

| PR | Author | Score | Technique | Legality |
|----|--------|-------|-----------|---------|
| **#1813** | djeidy | **0.94166** | Scylla QK5.25 + depth recurrence + full GPTQ int6 + LZMA | ⚠️ EXTRAORDINARY — no reviews yet, watch for BPB bug |
| **#1812** | EthanNing | **1.0729** | SP8192 + LegalTTT **4ep** + split MLP WD (mlp=0.115/attn=0.095) | ⚠️ 4ep beyond ≤3ep safe threshold — score-first claimed but needs ruling |

**PR #1813 detail**: Claims QK5.25, depth recurrence on layers 3-5, reduced bigram dim, full GPTQ int6 + LZMA. All artifacts 15.85–15.87 MB (under 16 MB). No legality flags visible. Score is extraordinary — 0.94166 would be competition-leading. Pattern matches prior Scylla claims with BPB bugs. Wait for community review before tracking.

**PR #1812 detail**: 4-epoch score-first TTT. Author explicitly states "Score-first TTT only. No SLOT, no pre-quant TTT." 4ep is above the ≤3ep threshold established by PR #1413 precedent. May be legal (PR #1557 cites PR #1514 as 5ep precedent) but risky. If confirmed legal, 1.0729 is a clean beat of merged SOTA.

### Status of previously-tracked PRs (Apr 25 update)

- **PR #1795** (PPM mixture, 1.01252): OPEN, **still no organizer ruling**. OE-GOD has flagged the legality question explicitly in submission.json. Do NOT implement.
- **PR #1797** (dexhunter, 1.06157): OPEN, clean. dexhunter pushed a CaseOps byte-counting fix on Apr 24 — reported metric unchanged. Still best clean dexhunter PR.
- **PR #1787** (nprime06, 1.06335): OPEN. One reviewer raised GPTQ timing concern; no organizer ruling. Still the community-consensus best base.
- **PR #1807** (davie2009kh, 1.07037): "Parallel pre-quant TTT" explicitly in title and technique — **ILLEGAL** despite author claiming no pre-quant TTT leakage. Same violation pattern as #1351/#1408. Do NOT track.
- **PR #1667** (Attention Output Gate + SmearGate, 1.07139): OPEN, no new reviews as of Apr 25.
- **PR #1727** (MP-SGD TTT 4-phase, 1.07217): OPEN, no new reviews as of Apr 25.
- **Issue #1604** (CaseOps ruling): **STILL NO @valerio-oai response**. 12 days silence. Proceed without CaseOps.

---

## New Research Papers

### arXiv:2604.21215 — "The Recurrent Transformer: Greater Effective Depth and Efficient Decoding" (Apr 23, 2026)
- Each layer attends to KV pairs computed off its own activations → layerwise recurrent memory, standard autoregressive decoding cost.
- On 150M and 300M param C4 pretraining: improves cross-entropy vs parameter-matched Transformer baseline with *fewer layers*.
- **Relevance**: Directly describes our Triple Loop mechanism. Suggests our 3× depth recurrence approach has strong theoretical grounding. Also supports adding per-loop-iteration KV retention. May help motivate depth 4× with outer normalization.
- **Action**: Read architecture section before next architecture experiment. No GPU run needed.

### arXiv:2604.11791 — "A Mechanistic Analysis of Looped Reasoning Language Models" (Apr 2026)
- Proves each layer in a looped model converges to a **distinct fixed point**; loop follows a consistent cyclic trajectory.
- Recurrent blocks learn distinct inference stages that repeat in depth.
- **Relevance**: Supports outer normalization (arXiv:2604.15259) to stabilize each fixed point. Our Triple Loop should benefit from RMSNorm at loop output (suggested in CLAUDE.md). ~1–3 lines of code.
- **Action**: Low-risk 1-liner after base stack is validated.

### Gram Newton-Schulz (Dao-AILab, github.com/Dao-AILab/gram-newton-schulz)
- Iterates on symmetric Gram matrix XX^T instead of rectangular M → lower FLOPs, symmetric GEMM kernels.
- "Up to 2× faster" Newton-Schulz. pip installable. Drop-in for NS in Muon.
- **CAVEAT**: Requires **Hopper or Blackwell GPU + CUDA 12.9+ + PyTorch 2.7.1+**. H100 SXM nodes typically run CUDA 12.x — may work, but driver version is uncertain. Must verify CUDA version on RunPod pod before attempting.
- **Action**: Check CUDA version on next RunPod pod (`nvcc --version`). If CUDA ≥12.9 and PyTorch ≥2.7.1: test as Polar Express NS alternative. If not: skip.

---

## HuggingFace / Community Discoveries

- dexhunter fixed a CaseOps byte-counting script (Apr 24) — shows ongoing maintenance of his stack. PR #1797 remains the most reliable dexhunter submission.
- PR #1813 (djeidy, 0.94166) is the second extraordinary Scylla-style claim this week. Pattern: large score, no reviews, potential BPB bug. Do not act until BPB verified by community (typically 1–3 days).
- Competition has entered final sprint phase: 4 new PRs opened Apr 25 (PRs #1812–1815), confirming acceleration.

---

## Recommended Actions (Priority Order, 5 days to deadline)

1. **GPU RUN TODAY**: Full clean legal stack on PR #1493 base. Implement in order:
   - Polar Express NS + MIN_LR=0.10 (from PR #1787 — 2 config changes)
   - Per-Layer Adaptive GPTQ MLP=12σ/Attn=13σ + int7 Emb@15σ (PR #1586)
   - Attention Output Gate + SmearGate (PR #1667)
   - LoRA-TTT warm-start A + alpha=144 + WD=1.0 (PR #1767)
   - Target: ~1.068–1.072 bpb. This alone beats merged SOTA by ≥0.005 nats.

2. **Monitor PR #1813** (Scylla 0.94166): if community reviewer confirms no BPB bug and artifact is clean, this changes the competition. Check again in 24–48 hours.

3. **Monitor PR #1812** (4ep TTT, 1.0729): if score confirmed and 4ep ruling is clarified, this gives a cleaner path to 1.07xx without additional stack work. Check for organizer comment.

4. **DO NOT implement**: CaseOps (no ruling), PPM (no ruling), pre-quant TTT (illegal), Scylla (BPB unverified), Gram-NS (hardware constraint uncertain).

5. **Deadline**: Apr 30. Today is Apr 25. 5 days left. No more research delays.

---

*Research session: 2026-04-25 | Next check: 2026-04-26 | Days to deadline: 5*

---

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
