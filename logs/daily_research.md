# Parameter Golf Daily Research - 2026-04-16

## PR #771 STATUS: CLOSED (REJECTED) — no change

@valerio-oai ruling (confirmed): "adapting model to eval tokens with TTT for multiple epochs, then reporting val numbers on those same tokens." No appeal path.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache — ruled out Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Author self-closed, same illegality |
| #758 | 1.0465 | **OPEN** (dead) | XOR hash key includes target token — same violation as #727. No new activity. |
| #731 | 1.0400 | **OPEN** | Dense-count + Laplace smoothing. MatoTeziTanka "LOOKS CLEAN." Seed 42 only; seeds 1337+2024 pending. 6104 steps, 15,999,919 bytes. |

---

## Leaderboard

**Merged SOTA: 1.0810 (bigbag, PR #1493) — DAY 7 UNCHANGED.**

Last upstream commit: `75700cb` April 9, 2026. Longest plateau since the Apr 5–9 acceleration wave. No new records in 7 days. Expect a merge wave before deadline (April 30 = 14 days).

### Best Open PRs (updated Apr 16)

| PR | Score | Author | Technique | Legal? |
|----|-------|--------|-----------|--------|
| #1670 | **1.05970** | dexhunter | Casefold V4 + Multi-Phase Global SGD TTT | **AWAIT CASEFOLD RULING** |
| #1647 | **1.0616** | powerpratik | SLOT-4 + TTT + 3-Layer Recurrence + Parallel Residuals | ⚠️ SLOT unruled |
| #1585 | **1.0639** | codemath3000 | Casefold Tokenizer + Parallel Residuals + Systems Opt | **AWAIT RULING** |
| #1578 | **1.0668** | mikeapedia | Custom Casefold BPE retrain | **AWAIT RULING** |
| #1560 | **1.07406** | dexhunter | VarLen Attention + Doc-TTT | **YES** |
| #1586 | **1.07493** | dexhunter | Per-Layer Adaptive GPTQ + int7 Emb + MLR=0.026 | **YES** |
| #1667 | **1.07139** | MarioPaerle | SmearGate + Attention Output Gate (1,056 params) + Legal TTT | **YES — no reviews yet, appears clean** |
| #1610 | **1.0728** | romeerp | VarLenAttn + PhasingTTT | YES (low EV) |
| #1584 | **1.0752** | codemath3000 | Systems Opt (fused Muon + batched EMA + loader prealloc) | **YES** |
| #1555 | **1.07636** | andrewbaggio1 | TMA Megakernel + Tap-In (min_match=1) | Tap-In unconfirmed |
| #1541 | **1.07785** | bigbag | Improved Parallel Residuals + Muon 0.97 | ⚠️ hash embed flag |
| #1540 | **1.0777** | aryanbhosale | VarLen + Doc-Independent LoRA TTT rank-96 | **YES** |

**Target**: ≤1.0760 bpb. 14 days remaining (April 30 deadline).

---

## What Changed (GitHub — Apr 15–16, 2026)

### No new merges. Day 7 plateau continues.

### New Open PRs (filed Apr 14–16)

**PR #1670** (dexhunter, **1.05970**, new best open) — ⚠️ AWAIT CASEFOLD RULING
- Casefold V4: lowercase normalization before SP8192 tokenization ("reduces vocabulary entropy")
- Multi-Phase Global SGD TTT: 3 phases across 2000 prefix documents (builds on PR #1626)
- std dev 0.00031 (3-seed), artifact ~15.20 MB
- TTT phase ordering unclear (score-first vs. train-then-score not explicit in docs)
- **Depends on casefold ruling at Issue #1604** (open, no @valerio-oai comment yet)
- **Do NOT implement until casefold ruled legal**

**PR #1667** (MarioPaerle, **1.07139**) — ✅ CLEAN, APPEARS LEGAL
- Attention Output Gate: lightweight per-head multiplicative gate on attention output; 1,056 new params (12 weights × 8 heads × 11 layers); initialized to zero → scale=1.0 at start
- SmearGate: reintroduced with input dependence (Modded Nano GPT style), width=12
- Legal score-first TTT, 3ep, LR=0.005, SGD
- 3-seed mean 1.07139 (std 0.00082), artifact 15.927 MB (max 15.94 MB)
- No organizer feedback; self-certified compliance
- **Stack this on PR #1586 for potential additive improvement**

**PR #1647** (powerpratik, **1.0616**) — ⚠️ RISKY (SLOT)
- SLOT-4: per-window delta-vector logit bias, 4 AdamW steps
- Standard SLOT (not Causal SLOT-16)
- No reviews yet from any reviewer
- Do NOT implement until SLOT receives organizer ruling

**PR #1671** (souro26, 1.3827): Token-wise gating — well above baseline, skip
**PR #1666** (mrbese, 1.1531): BESE 288-vocab tokenizer — not competitive

### Issue #1604 (casefold tokenizer legality): Still OPEN
- Filed Apr 13 by mikeapedia; no @valerio-oai comment as of Apr 16
- Core question: does NFKC + lowercase on validation corpus constitute invalid benchmark manipulation?
- Three community members debating; no ruling

---

## New Research Papers

| Priority | Paper | arXiv ID | Date | Key Technique | Applicability |
|----------|-------|----------|------|---------------|--------------|
| Watch | Self-Calibrating LMs via TTT Discriminative Distillation (SECL) | 2604.09624 | Apr 2026 | TTT pipeline that reduces ECE via discriminative distillation; score-first compatible | Targets calibration (ECE), not BPB. Low direct impact on our metric. |
| Already tracked | End-to-End TTT for Long Context | 2512.23675 | Dec 2025 | Compresses context to weights at test time via next-token prediction; scales with context length | Relevant to Doc-TTT quality; LaCT (2505.23884) is the higher-EV variant already in plan |
| Already tracked | Newton-Muon | 2604.01472 | Apr 2026 | +6% fewer steps, +4% wall-clock vs standard Muon | Verify additive with MuonEq-R before GPU spend |
| Skip | LieQ (layer-wise quant for small LMs) | 2508.03332 | Aug 2025 | Canonical division of labour across layers for PTQ; 2-bit target | Not applicable — we use int6/int7 GPTQ, not sub-4-bit regime |

No new breakthrough papers today. arXiv:2604.09624 (SECL) is the sole new find; low direct impact.

---

## HuggingFace / Community

No new relevant blog posts. dexhunter filed PR #1670 (1.05970) — their third top-10 PR (#1560, #1586, #1670). MarioPaerle is a new submitter worth watching (PR #1667 technique is clean and implementable).

---

## Recommended Action

**No change to core strategy. Two additions: PR #1667 Attention Output Gate is now a candidate to stack; casefold watch continues.**

Priority order for next GPU run:
1. **Implement PR #1586** (per-layer GPTQ: MLP=12σ, Attn=13σ, Emb int7@15σ; MLR=0.026). Config-level change, -0.01266 nats confirmed, zero legality risk.
2. **Add VarLen Attention + Doc-TTT** (PR #1560 approach): -0.007 bpb. Combined target with #1: ~1.062–1.068 bpb.
3. **Evaluate PR #1667 Attention Output Gate + SmearGate** on same run or follow-up: 1,056 extra params, no legality concerns. If additive with #1586 + #1560, expected combined ~1.065–1.070.
4. **Watch PR #1731** — if third seed confirms 1.0400 BPB and merges, Hedge Mixer (legal n-gram interpolation) is adoptable.
5. **Watch Issue #1604** — if casefold ruled legal, PR #1670 (dexhunter, 1.05970) jumps to highest-EV action; reset target to ≤1.0499.

**Do NOT implement**: Casefold (#1670, #1585, #1578 — await ruling), SLOT (#1647 — unruled), PR #758 (dead), AdamW multi-epoch TTT, pre-quant TTT.

---

_Updated: 2026-04-16 (merged SOTA 1.0810 Day 7 no change; PR #1667 MarioPaerle new clean PR (1.07139, Attention Output Gate + SmearGate); PR #1670 dexhunter new best open (1.05970) but pending casefold ruling; PR #1647 SLOT-4 (1.0616) risky; casefold Issue #1604 open; 14 days remaining)_

---

# Parameter Golf Daily Research - 2026-04-15

## PR #771 STATUS: CLOSED (REJECTED) — no change

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache — no change |
| #741 | 0.9850 | **CLOSED** (illegal) | No change |
| #758 | 1.0465 | **OPEN** (dead) | No new activity. XOR hash includes target token; effectively dead. |
| #731 | 1.0400 | **OPEN** | Dense-count tables + Laplace smoothing. Awaiting seeds 1337+2024. No update. |

---

## Leaderboard

**Merged SOTA: 1.0810 (bigbag, PR #1493) — DAY 6 UNCHANGED.**

Last upstream commit: `75700cb 2026-04-09 15:22 PDT` (PR #1511, leaderboard README). Zero new records since Apr 9.

This is the longest plateau since the Apr 5–9 acceleration wave (4 records in 4 days). Either the field is stuck, or a wave of PRs is being prepared for end-of-month push. **15 days to deadline.**

Best open PRs (no changes from Apr 14):

| PR | Score | Author | Technique | Legal? |
|----|-------|--------|-----------|--------|
| #1585 | **1.0639** | codemath3000 | Casefold Tokenizer + Parallel Residuals + Systems Opt | **AWAIT RULING** |
| #1578 | **1.0668** | mikeapedia | Casefold BPE retrain | **AWAIT RULING** |
| #1560 | **1.07406** | dexhunter | VarLen Attention + Doc-TTT | **YES** |
| #1586 | **1.07493** | dexhunter | Per-Layer Adaptive GPTQ + int7 Emb + MLR=0.026 | **YES** |
| #1584 | **1.0752** | codemath3000 | Systems Opt (fused Muon + batched EMA + loader prealloc) | **YES** |
| #1555 | **1.07636** | andrewbaggio1 | TMA Megakernel + Tap-In (min_match=1) | Tap-In unconfirmed |
| #1541 | **1.07785** | bigbag | Improved Parallel Residuals + Muon 0.97 | ⚠️ hash embed flag |
| #1540 | **1.0777** | aryanbhosale | VarLen + Doc-Independent LoRA TTT rank-96 | **YES** |
| #1610 | **1.0728** | romeerp | VarLenAttn + PhasingTTT | **YES** (low EV) |

**Target**: ≤1.0760 bpb. 15 days remaining.

---

## What Changed (GitHub — Apr 14–15, 2026)

**No new merges. No new high-priority PRs detected via web search.** Day 6 plateau continues.

Checked via: `git log upstream/main -5` (Apr 9 is most recent) + web search for new submissions.

### PRs to watch for movement:
- PR #1586 (per-layer GPTQ) — highest probability of merging next given 3-seed confirmation + zero flags
- PR #1541 (bigbag improved residuals) — hash embed flag must clear first; bigbag is the merged-SOTA author so organizers watch his PRs closely
- Casefold PRs (#1585, #1578) — ruling pending from @valerio-oai; if ruled legal, would reset our target to ≤1.0589

---

## New Research Papers

| Priority | Paper | arXiv ID | Date | Key Technique | Competition Relevance |
|----------|-------|----------|------|---------------|----------------------|
| **Add to plan** | **Newton-Muon Optimizer** | **2604.01472** | Apr 2, 2026 | Right-preconditioning by input second moment; surrogate quadratic model. Reaches target val loss in **6% fewer steps**, 4% less wall-clock vs standard Muon | **NOT YET IN PLAN.** Drop-in Muon replacement. At our budget (~4800 steps), 6% ≈ +288 extra effective steps. Small but free. Compatible with MuonEq-R base; verify they don't conflict before adding. |
| Already tracked | In-Place TTT | 2604.06169 | Apr 7, 2026 | MLP final-projection fast weights + NTP-aligned loss + chunk-wise updates | Score-first compatible. Key distinction from Session 3: uses NTP loss not reconstruction loss. Lesson #13 ("HARMFUL") used reconstruction loss on a different model. Could retry with NTP-aligned loss before dismissing permanently. Low priority until base stack is confirmed. |
| Already tracked | PRISM | 2602.10796 | Feb 2026 | Parallelizable iterative residual correction; 174× vs serial | Architectural inspiration for Triple Loop improvement — read before next recurrence change |
| Already tracked | Ouroboros | 2604.02051 | Apr 2, 2026 | Hypernetwork-generated per-step LoRA modulation for recursive blocks | 9.2M extra params overhead; likely too expensive for 16MB budget. Watch for competition PR. |
| Already tracked | Mousse | 2603.09697 | Mar 2026 | Kronecker-factored preconditioning for Muon; ~12% fewer steps | Higher EV than Newton-Muon but more overhead |

---

## HuggingFace / Community

No new relevant blog posts or model releases. Web search for "parameter-golf 1.06 OR 1.05" returned only PR list page — no new scores below 1.06 surfacing publicly.

---

## Recommended Action

**No strategy change from Apr 14. One addition: add Newton-Muon to technique tracking.**

Priority order:
1. **Next GPU run: Implement PR #1586** (per-layer GPTQ + int7 emb + MLR=0.026). Expected: ~1.068–1.070 bpb. Config changes only: `clip_sigmas={'mlp': 12.0, 'attn': 13.0, 'emb': 15.0}, MATRIX_LR=0.026, emb_bits=7`.
2. **Same run: Add VarLen Attention + Doc-TTT (PR #1560 approach).** Combined expected: ~1.062–1.068 bpb.
3. **Watch PR #1541** — if hash embed flag clears and it merges, new target becomes ≤1.0728.
4. **Newton-Muon (arXiv:2604.01472)**: Evaluate as a Muon swap in a follow-up run. +288 effective steps at our scale. Check if MuonEq-R and Newton-Muon are additive or redundant before GPU spend.
5. **Do NOT implement**: Casefold (#1585, await ruling), PR #758 (dead), any AdamW TTT.

---

_Updated: 2026-04-15 (merged SOTA 1.0810 Day 6 no change; no new PRs; Newton-Muon arXiv:2604.01472 added as new tracked technique (+6% effective steps); 15 days remaining)_

---

# Parameter Golf Daily Research - 2026-04-14

## PR #771 STATUS: CLOSED (REJECTED) — no change

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache |
| #741 | 0.9850 | **CLOSED** (illegal) | Same ruling |
| #758 | 1.0465 | **OPEN** (dead) | Flagged Apr 12 by MatoTeziTanka (XOR hash includes target token). No change. |
| #731 | 1.0400 | **OPEN** | Dense-count tables + Laplace smoothing. Community "LOOKS CLEAN." Single seed. Awaiting seeds 1337+2024. |

---

## Leaderboard

**No new records. Merged SOTA: 1.0810 (bigbag, PR #1493) — Day 5 unchanged.**

Best open PRs (from today's GitHub scan):

| PR | Score | Author | Technique | Legal? |
|----|-------|--------|-----------|--------|
| #1585 | **1.0639** | codemath3000 | Casefold Tokenizer + Parallel Residuals | **AWAIT RULING** |
| #1610 | **1.0728** | romeerp | VarLenAttn + PhasingTTT | **YES (new today)** |
| #1586 | **1.07493** | dexhunter | Per-Layer Adaptive GPTQ + int7 Emb | **YES** |
| #1560 | **1.07406** | dexhunter | VarLen Attention + Doc-TTT | **YES** |
| #1541 | **1.07785** | bigbag | Improved Parallel Residuals + Muon 0.97 | Hash embed flag pending |

**Target**: ≤1.0760 bpb. **16 days remaining (April 30 deadline).**

---

## What Changed (GitHub — Apr 13–14, 2026)

### No new merged records (Day 5 plateau)

Upstream `git log upstream/main -3` shows only PR #1511 (April leaderboard README update). Nothing new merged since Apr 9.

### New Open PR

**PR #1610** (romeerp, 1.0728, VarLenAttn + PhasingTTT) — open, no organizer comments
- Combines Variable-Length Attention with a two-phase global SGD approach
- Phase 1: LoRA-based TTT on 2000 already-scored documents
- Pause: global SGD on those scored documents (not new tokens)
- Phase 2: resume evaluation on remaining documents
- Score-first: tokens are scored before any adaptation runs on them
- Delta vs PR #1530 base: **-0.00055760 bpb** — very low EV
- All 3 seeds within 600s eval budget and under 16MB
- **Assessment**: Legal but minimal gain. Do not prioritize over PR #1586.

### Other new PRs (low interest)
- **PR #1619** (SP8192 + AdamW TTT): likely illegal — AdamW TTT same pattern as rejected PR #771
- **PR #1616** (QK-Gain 5.5 + deeper recurrence): open, no score listed; testing QK-Gain above 5.25
- **PR #1620** (1.66 BPB, squeeze architecture): non-competitive baseline submission

---

## New Research Papers

| Priority | Paper | arXiv ID | Date | Key technique | Competition relevance |
|----------|-------|----------|------|---------------|----------------------|
| **Read now** | PRISM: Parallel Residual Iterative Sequence Model | 2602.10796 | Feb 2026 | Iterative non-linear correction within parallelizable linear recurrence; 174× throughput vs serial | Our depth recurrence + parallel residuals (PR #1493) is the same motivation. Read to see if PRISM's correction phase improves recurrence quality without extra params. |
| Watch | Ouroboros: Dynamic Weight Gen for Recursive Transformers | 2604.02051 | Apr 2026 | Input-conditioned LoRA modulation (hypernetwork) per recurrence step | Could make 3× recurrence loops more expressive; adds hypernetwork params (16MB budget risk). Watch for any competition PR adopting this. |
| Already tracked | In-Place TTT | 2604.06169 | Apr 7, 2026 | NTP-aligned TTT objective | Read before next TTT implementation |
| Already tracked | LaCT | 2505.23884 | May 2025 | Large-chunk TTT | In plan (PR #1560 approach) |

---

## HuggingFace / Community

No new blog posts or model releases relevant to competition today.

---

## Recommended Action

**No strategy change from Apr 13. Priorities unchanged:**

1. **Implement PR #1586 (per-layer GPTQ + int7 emb) in next GPU run.** -0.01266 bpb, 3-seed confirmed, zero legality risk. Change: `clip_sigmas MLP=12.0, Attn=13.0, Emb(int7)=15.0; MATRIX_LR=0.026`.
2. **Add VarLen Attention + Doc-TTT (PR #1560 approach)** as the architecture change in the same run. Combined expected: ~1.062–1.068 bpb.
3. **Read PRISM (arXiv:2602.10796)** before next recurrence architecture decision.
4. **Watch PR #1541 (bigbag, 1.07785)** — if hash embed flag clears and it merges, target tightens to ≤1.0728.
5. **Watch PR #1610 PhasingTTT** — legal but low EV (-0.0006 bpb); only adopt if everything else is in.

**Do NOT implement** yet: Casefold (#1585, await ruling), PR #1619 (likely illegal AdamW TTT), PR #758 (dead).

---

_Updated: 2026-04-14 (merged SOTA 1.0810 Day 5 no change; PR #1610 PhasingTTT new legal open PR (low EV); PRISM arXiv:2602.10796 relevant paper; Ouroboros arXiv:2604.02051 watch; 16 days remaining)_

---

# Parameter Golf Daily Research - 2026-04-13

## PR #771 STATUS: CLOSED (REJECTED) — CONFIRMED

@valerio-oai ruling stands: "adapting model to eval tokens with TTT for multiple epochs, then reporting val numbers on those same tokens is not an allowable submission." No appeal path.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache — ruled out Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Author self-closed, same illegality |
| #758 | 1.0465 | **OPEN** ⚠️ effectively dead | MatoTeziTanka (Apr 12): XOR hash key includes target token → same normalization violation as #727. "Neural base model ~1.10–1.15 without cache." Do NOT track further. |
| #731 | 1.0400 | **OPEN** | Dense-count tables + Laplace smoothing, score-first per chunk; reviewer "LOOKS CLEAN"; awaiting seeds 1337+2024 |

---

## Leaderboard

**No change. Merged SOTA: 1.0810 (bigbag, PR #1493, 2026-04-09)**

Upstream `git log --oneline upstream/main -10` shows most recent merge is PR #1511 (leaderboard update, Apr leaderboard README). No new records merged since Apr 9.

**Best open PRs (updated today):**

| PR | Score | Author | Technique | Legal? |
|----|-------|--------|-----------|--------|
| #1576 | ~~1.01671~~ → ~1.16–1.18 | joshkmartinez | GDN-Hybrid + SWA | **BPB BUG** — reviewer confirmed double-count of space bytes from parent PR #1545; actual score ~1.16–1.18. Do NOT implement. |
| #1585 | **1.0639** | codemath3000 | Casefold Tokenizer + Parallel Residuals + Systems Opt | **LEGALITY DEBATED** — modifying val corpus (NFKC + lowercase); await organizer ruling |
| #1578 | **1.0668** | mikeapedia | Custom Casefold Tokenizer (BPE retrained on casefolded text) | **LEGALITY DEBATED** — same issue as #1585 |
| #1586 | **1.07493** | dexhunter | Per-Layer Adaptive GPTQ + int7 Emb + MLR 0.026 | **YES — no flags** |
| #1560 | **1.07406** | dexhunter | VarLen Attention + Doc-TTT | **YES** |
| #1584 | **1.0752** | codemath3000 | Improved Parallel Residuals + Systems Opt (fused Muon, batched EMA, loader prealloc) | **YES** |
| #1540 | **1.0777** | aryanbhosale | VarLen Attn + Doc-Independent LoRA TTT rank-96 | **YES** |
| #1541 | **1.07785** | bigbag | Improved Parallel Residuals + Muon 0.97 | ⚠️ hash embed flag pending |
| #1555 | **1.07636** | andrewbaggio1 | TMA Megakernel + Improved Parallel Residuals + Tap-In | Tap-In legality unconfirmed |
| #1437 | **1.08091** | dexhunter | N-gram Tilt (causality-fixed) | **YES** |

**Target**: ≤1.0760 bpb (beats 1.0810 by ≥0.005 nats). **17 days remaining (April 30 deadline).**

---

## What Changed (GitHub — Apr 12–13, 2026)

### No New Merged PRs
Last merge was PR #1511 (Apr leaderboard README, no new record). SOTA unchanged.

### New Open PRs (filed Apr 12–13)

**PR #1586** (dexhunter, 1.07493) — **HIGH PRIORITY: implement this**
- Per-layer GPTQ clip sigmas: MLP=12.0σ, Attention=13.0σ (vs uniform previously)
- int7 Embeddings at 15.0σ: saves ~530 KB vs int8
- MLR (matrix learning rate) = 0.026 (vs default 0.022), tuned via sweeps
- Artifact ~15.93 MB; -0.01266 nats vs merged SOTA (>2× the 0.005 threshold)
- No legality concerns raised

**PR #1584** (codemath3000, 1.0752) — Systems-only, **~20 extra steps free**
- Fused Muon kernel: Muon optimizer steps computed with kernel fusion
- Batched EMA: exponential moving average operations batched
- Loader prealloc: data loader memory pre-allocated
- No ML changes; claim: "0.005 nats waived for systems-only optimization"
- Builds on PR #1529 (dual-lane parallel residuals)

**PR #1585** (codemath3000, 1.0639) — Casefold Tokenizer — ⚠️ AWAIT RULING
- NFKC normalization + lowercasing applied to training corpus AND validation corpus
- Custom SentencePiece BPE retrained on normalized text
- Achieves ~10% better byte compression → lower BPB mechanically
- **Key concern**: modifying what bytes are counted in the validation set denominator. 3 participants debated; no organizer ruling yet.
- CUTLASS EVT build required (extra dependency)

**PR #1578** (mikeapedia, 1.0668) — Custom Casefold Tokenizer — ⚠️ AWAIT RULING
- Eliminates ~21.1% of SP8192 vocab (case-duplicate tokens)
- Refills 374 freed slots with optimized subwords
- Same legality debate as #1585

**PR #1576** (joshkmartinez, 1.01671) — GDN-Hybrid — ⚠️ BPB BUG
- Inherits BPB calculation bug from parent PR #1545
- Space token double-count inflates denominator byte count ~14%
- Reviewer estimate: actual ~1.16–1.18 BPB, not 1.01671
- No organizer response yet; do NOT build on this

**PR #1564** — CLOSED (voluntarily by author, superseded by PR #1575)

---

## New Research Papers

| Priority | Paper | arXiv ID | Date | Technique | Applicability |
|----------|-------|----------|------|-----------|--------------|
| **Watch** | In-Place Test-Time Training | 2604.06169 | 2026-04-07 | Replaces generic TTT reconstruction loss with NTP-aligned objective + chunk-wise updates. Score-first compatible (NTP IS the evaluation criterion). Scales to 128k context on 4B model. | Could improve legal post-quant score-first TTT quality; implementation adds NTP loss alignment to TTT LoRA updates. Estimate -0.001 to -0.002 bpb vs current TTT. |
| Already tracked | LaCT (Test-Time Training Done Right) | 2505.23884 | 2025-05 | Large-chunk TTT — PR #1560 Doc-TTT appears to be LaCT-style | In plan |
| Low | N-gram Is Back (residual learning) | 2210.14431 | 2022 | Neural model fits residual of n-gram distribution | Interesting but complex; n-gram Tilt (PR #1437) is simpler path already in plan |
| Low | Lightweight Adaptive Mixture of Neural+N-gram LMs | 1804.07705 | 2018 | Small network predicts mixture weight per timestep | Superseded by N-gram Tilt approach |

No breakthrough papers found beyond what was tracked Apr 12. arXiv:2604.06169 is the one new relevant paper.

---

## HuggingFace / Community Discoveries

- **codemath3000** filed 3 PRs in one day (#1583, #1584, #1585) — a systems-optimization sweep. The CUTLASS EVT dependency in #1585 is a potential build concern for reproducibility.
- **dexhunter** remains the most active legal submitter: #1560 (1.07406), #1586 (1.07493). Both clean.
- Casefold tokenizer debate is active (3+ community members discussing legality). Organizer ruling expected soon.

---

## Recommended Action

**Priority 1 — Implement immediately (before next GPU run):**
- **Per-Layer Adaptive GPTQ from PR #1586**: Change GPTQ clip_sigmas to MLP=12.0, Attn=13.0, Emb=int7@15.0. Saves 530KB (more parameter budget). Zero legality risk. Expected: -0.0050 to -0.013 nats vs current stack. This is a config-level change, not an architecture change.
- **MLR = 0.026**: Change MATRIX_LR from 0.022 → 0.026 (co-tuned with per-layer GPTQ in PR #1586).

**Priority 2 — Architecture (next GPU run):**
- VarLen Attention + Doc-TTT (PR #1560 approach): -0.007 bpb vs merged SOTA
- With per-layer GPTQ: combined target ~1.068–1.072 bpb

**Priority 3 — Monitor:**
- PR #731 (n-gram Hedge Mixer, 1.0400): If third seed confirms and it merges, a clean legal n-gram mixer is available
- PR #1585 casefold tokenizer: If organizer rules it legal, -0.017 bpb for essentially free
- arXiv:2604.06169 (In-Place TTT): Read paper; may improve TTT quality with same legal budget
- PR #1541 (bigbag, 1.07785): Hash embed flag — if cleared, this is the next likely merge that tightens our target to ≤1.0728

**Do NOT implement:**
- Casefold Tokenizer (#1578, #1585): Await organizer ruling on val corpus modification
- GDN-Hybrid (#1576): BPB bug not resolved; actual performance likely ~1.17
- PR #758 n-gram: Flagged dead by MatoTeziTanka Apr 12; same illegality as #727

---

_Updated: 2026-04-13 (merged SOTA unchanged 1.0810; PR #758 effectively dead via Apr 12 flag; PR #1576 GDN-Hybrid has BPB bug; PR #1586 per-layer GPTQ is highest-EV safe action; 17 days remaining)_

---

# Parameter Golf Daily Research - 2026-04-12

## PR #771 STATUS: CLOSED (REJECTED)

Same as last session. Rejected for train-then-score TTT. No action needed.

---

## N-GRAM PR STATUS

| PR | Score | Status | Notes |
|----|-------|--------|-------|
| #727 | 0.9674 | **CLOSED** (illegal) | Hashed n-gram cache — ruled out Mar 27 |
| #741 | 0.9850 | **CLOSED** (illegal) | Same as #727 — author self-closed |
| #758 | 1.0465 | **OPEN** | Major legality concerns flagged (hash key includes target token, TTT contradiction) |
| #731 | 1.0400 | **OPEN** | Dense-count tables + Laplace smoothing — reviewer says "LOOKS CLEAN"; awaiting seeds 1337+2024 |

---

## Leaderboard

**MERGED SOTA HAS CHANGED SIGNIFICANTLY** (was 1.1147 on 2026-04-07, now **1.0810** as of 2026-04-09):

| Rank | Score | Author | PR | Technique | Date |
|------|-------|--------|-----|-----------|------|
| 1 | **1.0810** | bigbag | #1493 | SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT | 2026-04-09 |
| 2 | 1.0822 | aryanbhosale | #1477 | SP8192 + Parallel Residuals + Score-First TTT | 2026-04-08 |
| 3 | 1.0828 | dexhunter | #1413 | SP8192 + QK-Gain 5.0 + Legal Score-First TTT | 2026-04-06 |
| 4 | 1.0835 | Robby955 | #1412 | SP8192 + Parallel Residuals + Hessian-Aware SDClip + Progressive Recurrence | 2026-04-06 |
| 5 | 1.0856 | clarkkev | #1394 | SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip | 2026-04-05 |
| 6 | 1.0897 | aryanbhosale | #1334 | SP4096 + Depth Recurrence + Parallel Residuals + MuonEq-R | 2026-04-04 |

**Best open PRs (unmerged):**

| PR | Score | Author | Technique | Legal? |
|----|-------|--------|-----------|--------|
| #1564 | **1.01710** | joshkmartinez | GDN-Hybrid (Gated DeltaNet + SWA), NO TTT/SLOT | **YES** (safe, Track-A) |
| #1560 | **1.07406** | dexhunter | VarLen Attention + Triton Fused MLP + Doc-TTT + Warmdown 0.75 | **YES** |
| #1555 | **1.07636** | andrewbaggio1 | TMA Megakernel + Improved Parallel Residuals + Tap-In min_match=1 | Yes |
| #1557 | **1.07730** | ndokutovich | SP8192 + Improved Parallel Residuals + Muon 0.97 + TTT 5ep | Yes (refs PR #1514) |
| #1561 | **1.07830** | EthanYangTW | SP8192 + Triple Recurrence + Banking + Fused MLP + Score-First TTT | Yes |
| #1333 | **1.07660** | aryanbhosale | Causal SLOT-16 on PR #1334 base | RISK (unruled) |
| #1437 | **1.08091** | dexhunter | SP8192 + Parallel Residuals + 3-Layer Recurrence + Causal N-gram Tilt | **YES** |

**Our PR #771**: CLOSED/REJECTED

**To beat merged SOTA by ≥0.005 nats**: need val_bpb ≤ 1.0760

---

## What Changed (GitHub — since 2026-04-07)

### 6 New Merged PRs
All merged 2026-04-05 to 2026-04-09. The stack is now:
- SP8192 vocab (required)
- 3-Layer Depth Recurrence (layers 4-5, 3×, activated at 0.35× training)
- Parallel Residuals (layers 7-10, GPT-J style)
- QK-Gain 5.25 (up from 5.0)
- GPTQ Embeddings + SDClip (int8 emb save ~4MB)
- Legal Score-First TTT (WD=0.095, EMA=0.9965, MLR=0.022, warmdown=0.72)

### New Techniques Observed in Open PRs

**1. VarLen Attention / Per-Document Causal Masking** (PR #1560, dexhunter, 1.07406 BPB)
- Disables cross-document attention: attention does not bleed across document boundaries
- Enables true per-document TTT (Doc-TTT): score-first adaptation resets per document
- LoRA chunk size=48, Muon momentum=0.97, warmdown=0.75
- Combined with Triton Fused MLP kernel
- **Estimated impact**: -0.009 bpb vs current merged SOTA (1.0810→1.074)

**2. TMA Megakernel + Tap-In Unigram Matching** (PR #1555, andrewbaggio1, 1.07636 BPB)
- TMA Megakernel: Triton Hopper TMA fused MLP, **+10.5% throughput = ~200 extra training steps** in 600s
- Tap-In min_match=1: activates at 21% of positions vs 1.7% at min_match=3, large activation increase
- "Improved Parallel Residuals" from related submission
- **Tap-In appears to be a unigram-level n-gram hint mechanism** — needs legality verification; may be similar to N-gram Tilt

**3. GDN-Hybrid Architecture** (PR #1564, joshkmartinez, 1.01710 BPB) ⚠️ WATCH CAREFULLY
- 5 Gated DeltaNet (linear attention) layers + shared SWA components
- SP1024 tokenizer, NO TTT, NO SLOT, NO eval-time adaptation — pure architecture
- MuonEq-R + AdamW training, GPTQ int6, zstd-22
- **1.01710 BPB is extraordinary** — would be best safe submission by far
- Status: OPEN, not yet reviewed by organizers
- Based on PR #1545 (GDN-Hybrid foundation); PR #1370 (GDN-only, 1.003 non-record) is supporting evidence
- **Risk**: If build on SP1024, may be beatable with SP8192 GDN-Hybrid. Verify training time ≤10 min.

**4. Parameter Banking** (PR #1561): Groups model parameters for efficient computation alongside recurrence. Combined with Fused MLP for 1.0783.

### Technique Stack Reaching Diminishing Returns
PR #1493 (merged SOTA, 1.0810) uses all safe legal techniques. The delta to the best open safe PR (#1560, 1.07406) is 0.007 bpb — still beatable, but the next big jump likely requires either:
a) GDN-Hybrid architecture rewrite (PR #1564 approach)
b) SLOT (PR #1333, risky)

---

## Apr 11 Update (PRs #1541, #1540, #1545 BPB Bug)

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

| Priority | Paper | ID | Technique | Δ bpb est. | Notes |
|----------|-------|----|-----------|-----------|-------|
| **Watch** | LaCT: Test-Time Training Done Right | arXiv:2505.23884 | Large Chunk TTT, GPU util 0→70%, O(n) scaling | ~-0.003 to -0.008 | PR #1560 "Doc-TTT" may be LaCT-style; dexhunter already implementing |
| Watch | E2E TTT for Long Context | arXiv:2512.23675 | Meta-learned next-token prediction TTT; compresses context to weights | uncertain | Complex to adapt to competition setup |
| Skip | Gated Delta Networks (arXiv:2412.06464) | Architecture paper for GDN — backs PR #1564 | | Already in PR #1564 |

No new papers found beyond what was tracked on 2026-04-07. Competition activity is moving faster than new arXiv papers.

No new breakthrough papers today beyond those already tracked.

| Paper | arXiv ID | Notes |
|-------|----------|-------|
| pQuant: Decoupled Linear QAT | 2602.22592 | Feb 2026; 1-bit params packed UINT8; not applicable to our int6 GPTQ |
| MuonEq-R | 2603.28254 | Already tracked — confirmed row/column normalization before Newton-Schulz; O(m+n) overhead |
| Newton-Muon | 2604.01472 | Already tracked — WATCH; ~+4-6% steps |

---

## HuggingFace / Community Discoveries

- None found. GitHub PR activity is the primary signal.

---

## Recommended Action

**Target**: val_bpb ≤ 1.0760 (beats new merged SOTA 1.0810 by ≥0.005 nats). Deadline: April 30 = **18 days remaining**.

**Priority 1 — Architecture decision (decide NOW, don't delay)**

Option A: **Incremental stack** (SP8192 + Triple Recurrence + Parallel Residuals + QK-Gain 5.25 + GPTQ Emb + SDClip + VarLen Attn + Doc-TTT)
  - Build on merged PR #1493 stack + add VarLen Attention + Doc-TTT from PR #1560
  - Expected: ~1.074–1.075 bpb (PR #1560 shows 1.07406 without N-gram Tilt)
  - Add N-gram Tilt (PR #1437 kernel) for additional ~-0.003 bpb → ~1.071 bpb
  - **Safe, zero rejection risk, beatable path**

Option B: **GDN-Hybrid architecture** (Gated DeltaNet + SWA, PR #1564 approach)
  - Wait for PR #1564 to receive organizer review OR replicate from PR #1564/#1545 code
  - 1.01710 BPB would be a massive leap; upgrade to SP8192 likely adds another -0.009 bpb
  - **High EV but requires architecture rewrite; verify 10-min budget first**

Option C: **SLOT track** (Causal SLOT-16, PR #1333 approach, 1.0766 BPB)
  - Issue #140 closed, @valerio-oai never ruled; 9 record PRs use SLOT
  - Delta vs Option A: ~-0.003 extra bpb at cost of rejection risk

**Recommendation**: Start with Option A (incremental, safe). Run 1xH100 validation of VarLen Attention + Doc-TTT addition to current PR #1493 stack. Monitor PR #1564 for organizer review — if approved, pivot to GDN-Hybrid.

**Do NOT implement:**
- Tap-In (PR #1555): Verify legality — mechanism touches token-level unigram cache; may be same pattern as N-gram Tilt or may be illegal
- PR #1430 techniques (per-sample SLOT + order-22 n-gram hash) — pending ruling

**Newly prioritized technique for next GPU run:**
1. VarLen Attention (per-document masking) — easy add to existing stack, -0.007 bpb
2. Doc-TTT with LoRA chunk size=48 — extends legal TTT, -0.003 bpb
3. TMA Megakernel (Triton Hopper) — +200 steps = ~-0.002 bpb additional training

---

_Updated: 2026-04-12 (merged SOTA NOW 1.0810 since 2026-04-09; 6 new merged PRs; GDN-Hybrid 1.01710 open; VarLen+Doc-TTT 1.07406 open; 18 days to deadline)_

---

### Apr 11 Recommended Actions (context)

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
