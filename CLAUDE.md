# CLAUDE.md — Parameter Golf AI Agent Instructions

---

## TL;DR

**Parameter Golf**: Train the best language model that fits in a 16MB artifact and trains in under 10 minutes on 8xH100 SXM GPUs, scored by compression quality (bits-per-byte) on FineWeb validation.

**Challenge**: https://github.com/openai/parameter-golf | https://openai.com/index/parameter-golf/

**Core Delivery**: Lowest val_bpb score | 16MB artifact constraint | 10-min train budget | 10-min eval budget | Tokenizer-agnostic BPB metric

**NOT**: A general LLM training framework | A production inference system | A data engineering project. This is a constrained optimization competition — every decision optimizes for val_bpb within the 16MB/10-min constraints.

**Stack**: Python 3 + PyTorch (CUDA/H100) + MLX (Apple Silicon local dev) + SentencePiece + zstd/zlib compression

**Repo**: Single-repo with baseline scripts at root and competition submissions in `records/`

---

## Critical Rules

1. **16MB artifact limit**: code (`train_gpt.py`) + compressed model weights must be < 16,000,000 bytes (decimal, not MiB). Check artifact size on EVERY experiment.
2. **No network during eval**: The artifact must be fully self-contained. No downloads, no API calls during evaluation.
3. **Validation data is sacred**: NEVER access validation data during training. Test-time training is ONLY allowed on validation tokens already evaluated (already graded).
4. **train_gpt.py is the submission**: All counted code lives in this single file. Submissions are self-contained folders in `records/`.
5. **Don't edit baseline scripts for competition work**: `train_gpt.py` (root) and `train_gpt_mlx.py` are onboarding scripts. Competition work goes in `records/` folders.
6. **Statistical significance required**: New SOTA must beat existing by >=0.005 nats with p<0.01 across 3 seeds.
7. **MLX is for learning, not tuning**: MLX and CUDA have different numerical paths (float32 vs bf16 Muon). Never trust absolute bpb numbers from MLX runs.
8. **Always run the quantization roundtrip**: Post-quant val_bpb is the submission score, not pre-quant.
9. **Shut down RunPod pods when idle** — $3+/hr adds up fast.
10. Plan before building — non-trivial changes get a written hypothesis first.

---

## Security & Safety

### Destructive Operations

**PROHIBITED without explicit user confirmation**:
- **RunPod**: Deleting pods with unsaved work, terminating running experiments
- **Git**: push --force, reset --hard, deleting branches with experiment results
- **Data**: Deleting downloaded dataset shards (16GB+ redownload)

### Agent Boundaries

**NEVER autonomously**: spend RunPod credits (always confirm before launching pods) | modify the root `train_gpt.py` or `train_gpt_mlx.py` for competition purposes | submit PRs to the upstream repo | delete experiment logs

**ALWAYS**: track experiment results with hypothesis and verdict | verify artifact size < 16,000,000 bytes | run 3 seeds before claiming a result | check competition rules before novel eval approaches

---

## Context Documents

| File | When to Read |
| ---- | ------------ |
| `README.md` | Challenge rules, leaderboard, submission process, FAQ |
| `documents/raise-the-floor.md` | When output quality drops or agent oscillates between good and bad |
| `documents/testing-guide.md` | Before designing experiments or validating results |
| `data/README.md` | Dataset download, tokenizer variants, shard format |
| `records/track_10min_16mb/2026-03-22_FullStack_v51/README.md` | Our v5.1 submission (full stack) |
| PR #486 (branch `pr-486`) | Current SOTA: TrigramHash + ValueResidual + GradQuant + TTT |
| PR #503 (branch `pr-503`) | XSA on all layers + legal score-first TTT + Partial RoPE |
| PR #481 (branch `pr-481`) | Best TTT reference: cosine + per-layer LR |
| PR #490 (branch `pr-490`) | Value Residual + Gated Attention + TTT |
| `records/track_10min_16mb/2026-03-20_10L_Int5MLP_.../README.md` | Previous SOTA (1.1428) techniques and ablation |
| `records/track_10min_16mb/2026-03-17_LoRA_TTT/README.md` | LoRA TTT reference (abandoned — hurts) |

---

## Project Structure

```
parameter-golf/
├── train_gpt.py              # Baseline CUDA training script (1126 lines) — DO NOT edit for competition
├── train_gpt_mlx.py          # Baseline MLX script for local dev (1104 lines) — DO NOT edit for competition
├── requirements.txt           # Python dependencies reference
├── data/
│   ├── cached_challenge_fineweb.py   # Dataset downloader (supports sp1024/sp2048/sp4096)
│   ├── datasets/                      # Downloaded training shards + validation
│   └── tokenizers/                    # SentencePiece models
├── records/
│   ├── track_10min_16mb/              # Competition submissions (17 entries)
│   │   ├── 2026-03-17_NaiveBaseline/  # Baseline: 1.2244 val_bpb
│   │   ├── 2026-03-20_10L_Int5MLP_*/  # SOTA: 1.1428 val_bpb
│   │   └── ...
│   └── track_non_record_16mb/         # Unlimited compute submissions
└── logs/                              # Training run logs
```

**Commands**:
```bash
# Local (MLX, Apple Silicon)
RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 python3 train_gpt_mlx.py

# RunPod (CUDA, 1xH100)
torchrun --standalone --nproc_per_node=1 train_gpt.py

# RunPod (CUDA, 8xH100 — final validation only)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

---

## Session Protocol

**Start**: Check current SOTA on leaderboard (README.md) → Review experiment log → Read active plan

**End**: Log experiment results (hypothesis, numbers, verdict) → Stop RunPod pod if running → Update plan if approach changed

---

## Competition Strategy

**Merged leaderboard SOTA**: **1.0810 val_bpb** (bigbag, PR #1493, 2026-04-09) — NO CHANGE (confirmed Apr 24, **Day 15 plateau** — longest in competition history). NOTE: Scylla 0.9485 record folder was committed to `track_10min_16mb/` on Apr 23 (PR #1184, icryo) but README not updated and score is DISPUTED (PR #1271 corrects to ~1.1289 bpb via byte accounting fix). Treat merged SOTA as 1.0810 until README is updated.
**Best open legal PRs (Apr 24 update)**:
  - PR #1795 (OE-GOD, **1.01252**): SP4096 + byte-level PPM order-4 adaptive-λ mixture — score-first (PPM updates after scoring each byte); initial gate legality concern **FIXED** (gate frozen before observing byte). Extraordinary score — **WATCH CLOSELY for organizer ruling before implementing**.
  - PR #1797 (dexhunter, **1.06157**): PR #1787 base + SmearGate + LQER Asym — **dexhunter's new best, no flags**. Stack on this.
  - PR #1801 (leon2k2k2k, **1.06287**): PR #1787 base + Sparse Gate + Updated Frozen Carry — no flags.
  - PR #1787 (nprime06, **1.06335**): PR #1736 + **Polar Express NS** + MIN_LR=0.10 + Sparse Attn Gate + Fused CE + TTT alpha=144/warm-start A/WD=1.0 — **NEW BEST CLEAN BASE PR**. No legality flags. Polar Express NS = ICLR 2026 paper (arXiv:2505.16932).
  - PR #1771 (bigbag, **1.06513**): CaseOps + Recurrence Depth Curriculum (1→3→4) + SmearGate + GatedAttn + LoRA-TTT (alpha=144, warm-start A, WD=1.0) — ⚠️ Awaits Issue #1604.
  - PR #1769 (dexhunter, **1.06453**): CaseOps + GatedAttn + QuantGate + Loop4-5 + MLP clip_sigmas 10→12 — ⚠️ Awaits Issue #1604.
  - PR #1802 (aamodbhatt, **1.0771**): SP8192 + Polar Express NS + Multi-Phase Global TTT — no flags, stackable.
  - PR #1758 (kilojoules, **1.02840**): PR #1738 + Pre-Quant TTT LR=1e-3 + Unfrozen — **⚠️ ILLEGAL** (pre-quant TTT). Do NOT track.
  - PR #1698 (arsenis-cmd, **~~1.00995~~**): GatedDeltaNet (FLA) — **DEAD**: BPB bug + artifact violation. Do NOT track.
  - PR #1738 (alertcat, **1.03540**): CaseOps V15 + PR #1735 Pre-Quant TTT — **⚠️ BUILDS ON ILLEGAL PR #1735**. Do NOT track.
  - PR #1756 (romeerp, **1.06505**): CaseOps + Recurrence Depth Curriculum — ⚠️ Awaits Issue #1604 + BOS bug.
  - PR #1755 (OE-GOD, **1.07462**): SP8192 + CaseOps + Legal TTT — ⚠️ Awaits Issue #1604.
  - PR #1736 (dexhunter, **1.06549**): CaseOps + GatedAttn + QuantGate + SP8192 — superseded by PR #1769. Await Issue #1604.
  - PR #1693 (dexhunter, **1.05733**): Casefold V4 + AttnOutGate + SmearGate + Multi-Phase SGD TTT — ⚠️ Await casefold ruling.
  - PR #1767 (renqianluo, **1.07209**): LoRA-TTT warm-start A + alpha=144 + WD=1.0 — **LEGAL**. Stack with #1586+#1667.
  - PR #1775 (dentity007, **1.07285**): No Gates + Multi-Phase Global SGD TTT — appears legal; stackable.
  - PR #1667 (MarioPaerle, **1.07139**): SmearGate + Attention Output Gate (1,056 params, 12×8×11 heads) + Legal TTT — **CLEAN**. Backed by arXiv:2505.06708.
  - PR #1727 (yahya010, **1.07217**): MP-SGD TTT 4 phases + QK-Gain 5.25 — **LEGAL** (score-first per phase); stackable.
  - PR #1586 (dexhunter, **1.07493**): Per-Layer Adaptive GPTQ (MLP=12σ, Attn=13σ) + int7 Emb (15σ) + MLR=0.026 — **CLEAN, implement immediately**.
  - PR #1560 (dexhunter, **1.07406**): VarLen Attention + Doc-TTT — appears legal (no reviews yet).
  - PR #1584 (codemath3000, **1.0752**): Systems-only (fused Muon + batched EMA + loader prealloc), ~20 extra steps.
  - PR #1555 (andrewbaggio1, **1.07636**): TMA Megakernel + Improved Parallel Residuals + Tap-In min_match=1.
  - PR #1540 (aryanbhosale, **1.0777**): VarLen Attention + Doc-Independent LoRA TTT rank-96 + Triton TMA — appears legal.
  - PR #1735 (AjAnubolu, **1.0429**): Pre-Quant AdamW TTT 21ep — **⚠️ ILLEGAL**. Do NOT track.
  - PR #1576 (joshkmartinez, **~~1.01671~~**): GDN-Hybrid — **BPB BUG**. Do NOT track.
  - PR #1687 (resouer, **~~1.04090~~**): FLA — **CLOSED: BPB BUG**. Do NOT track.
  - PR #1585 (codemath3000, **1.0639**): Casefold Tokenizer — **LEGALITY DEBATED** (Issue #1604); await ruling.
  - PR #1647 (powerpratik, **1.0616**): SLOT-4 + TTT + 3-Layer Recurrence + Parallel Residuals — ⚠️ standard SLOT risk.
**Best open with SLOT**: ~1.0616 val_bpb (PR #1647, powerpratik, SLOT-4) — no reviews yet
**Best open (illegal)**: 1.0429 (PR #1735, pre-quant TTT)
**Issue #1604 (CaseOps ruling)**: **SELF-DEADLINE PASSED (Apr 24) — NO @valerio-oai response in 11 days**. Proceed with clean legal stack NOW. Do not wait.
**Target**: Beat 1.0810 merged SOTA by >=0.005 nats → need **≤1.0760 bpb**. Best reachable (legal, no CaseOps): ~1.065–1.068 (clean stack #1787 base + #1586 + #1667 + #1560 + #1727 + LoRA-TTT warm-start A + Polar Express NS + MIN_LR). **6 days to deadline (Apr 30). Stop waiting — implement NOW.**

**CRITICAL LEGALITY UPDATES**:
- **PR #771 REJECTED (2026-03-27)** — Our AdamW TTT 30ep was train-then-score. All 30-epoch TTT results void.
- **N-gram hash cache ILLEGAL** — PRs #727, #741 closed. PR #758: MatoTeziTanka (Apr 12) flagged XOR hash key includes target token = same illegality as #727. Effectively dead. PR #731 open (dense count tables + Laplace smoothing, reviewer says "LOOKS CLEAN", awaiting 3rd seed).
- **N-gram Tilt IS LEGAL (PR #1420)** — Normalized via softmax Z. **⚠️ PR #1420 has causality bug — use PR #1437's corrected implementation.**
- **Score-first TTT IS LEGAL** — ≤3ep confirmed (PR #1413). PR #1557 cites PR #1514 as precedent for 5ep — status uncertain; use ≤3ep to be safe.
- **Pre-quant TTT ILLEGAL (all variants)** — PR #1351, #1416, #1408, #1423. Do NOT use.
- **SLOT δ-vector: Issue #140 CLOSED (Apr 6), NO organizer ban** — @valerio-oai NEVER commented in Issue #140. 9 record PRs use SLOT. Risk remains. Implement only if willing to accept rejection risk.
- **ETLB UNRULED** — PR #1399/#1415; no ruling; -0.0019 bpb standalone. Await before implementing.
- **GDN-Hybrid (PR #1576)**: OPEN but **BPB calculation bug confirmed (Apr 13)** — space token double-count from parent PR #1545 inflates byte count ~14%; actual ~1.16–1.18 BPB. PR #1564 CLOSED (superseded by PR #1575 by same author). Monitor PR #1575/#1576 for bug fix/organizer response before investing.
- **VarLen Attention + Doc-TTT (PR #1560)**: No legality flags — per-document masking is architectural, score-first TTT per-doc. Still awaiting review.
- **Tap-In unigram matching (PR #1555)**: Legality UNCONFIRMED — verify before implementing (may be similar to n-gram approaches).
- **Casefold Tokenizer (PR #1578, #1585)**: LEGALITY DEBATED (Apr 13) — modifying validation corpus bytes via case normalization may constitute invalid benchmark manipulation. Await @valerio-oai ruling before implementing.
- **Per-Layer Adaptive GPTQ (PR #1586)**: NO LEGALITY FLAGS — safe config change, implement immediately.

**Current best-stack approach (PR #1787 as new base, Apr 24 update)**:
1. **SP8192 vocab** — beats SP4096 by ~0.009 bpb [merged SOTA]
2. **Triple Loop (17 virtual layers)** — layers 4-5 repeated 3×, activated at 0.35× training [merged SOTA]
3. **Parallel Residuals (layers 7-10)** — GPT-J style [merged SOTA]
4. **MuonEq-R + Polar Express Newton-Schulz** — MuonEq-R (arXiv:2603.28254) + Polar Express adaptive NS coefficients (arXiv:2505.16932, ICLR 2026) replacing fixed 5-step NS — PR #1787
5. **4× MLP expansion** [merged SOTA]
6. **GPTQ Embeddings (int7@15σ) + SDClip** — PR #1586; saves ~530KB vs int8 Emb
7. **Per-Layer Adaptive GPTQ clip** — MLP=12σ, Attn=13σ (PR #1586) — **IMPLEMENT NOW**
8. **QK-Gain 5.25** — up from 5.0 [merged SOTA]
9. **WD=0.095, EMA=0.9965, warmdown=0.72, MLR=0.026, MIN_LR=0.10** — MLR from PR #1586; **MIN_LR=0.10 warmdown floor from PR #1787** (prevents over-decay)
10. **N-gram Tilt** — use PR #1437 corrected kernel only
11. **Legal Score-First TTT (post-quant, ≤3ep)** — lr=0.005, all blocks; upgrade to alpha=144 + warm-start A + WD=1.0 (PR #1767)
12. **VarLen Attention (per-document causal masking)** — PR #1560, ~-0.007 bpb — **add next**
13. **Doc-TTT (per-document score-first TTT)** — PR #1560, chunk size=48, Muon 0.97 — **add next**
14. **Attention Output Gate + SmearGate (PR #1667)** — 1,056 extra params (12×8×11 heads); multiplicative per-head gate init to zero; backed by arXiv:2505.06708
15. **MP-SGD TTT 4 phases (PR #1727)** — score-first each phase; stackable
16. **TMA Megakernel (Triton TMA fused MLP)** — PR #1555, +10.5% throughput = ~200 extra steps — add after base validated
17. **CaseOps bijective tokenizer** — TITLE/ALLCAPS/CAPNEXT/ESC control tokens; BPB on original UTF-8 via sidecar — **Issue #1604 self-deadline PASSED with NO ruling. Do NOT block GPU runs on this. Implement clean stack now.**

**Key reference PRs**: #1493 (merged SOTA 1.0810), #1769 (1.06453, dexhunter, best CaseOps — await ruling), #1771 (1.06513, bigbag, CaseOps+Depth Curriculum+LoRA-TTT improvements — await ruling), #1667 (1.07139, Attention Output Gate+SmearGate — clean, stack on #1586), #1767 (1.07209, renqianluo, LoRA-TTT warm-start A+alpha=144 — appears legal), #1586 (1.07493, per-layer GPTQ — implement now), #1560 (1.07406, best open safe — VarLen+Doc-TTT), #1584 (1.0752, systems opt — fused Muon/EMA/prealloc), #1555 (1.07636, TMA Megakernel+Tap-In), #1437 (1.08091, causal-fixed N-gram Tilt kernel — use this), #1413 (1.08279, SP8192+Legal TTT), #1334 (1.0897, arch reference)

**Abandoned approaches**: Training-time static LoRA TTT (hurts), product quantization (SWA-incompatible), custom Triton kernels (poor EV — REVERTED: PR #1420 shows +10% via Triton TMA, revisit after base works), int4 without QAT (quality-destructive), eval stride=32 (time budget), AdamW TTT 30ep (illegal), n-gram hash cache (illegal), pre-quant TTT any form (illegal), Eval-Time Hash Embedding trained at inference (suspect illegal — same adapt-then-score pattern), Tap-In V6 document-local matching (await ruling), GDN-Hybrid #1576 (BPB bug — actual ~1.17 not 1.01671).
**NOTE**: Doc-Independent LoRA TTT (PR #1540, rank-96, resets per batch, score-first) is categorically DIFFERENT from abandoned LoRA TTT and appears legal — consider adopting.

---

## Technique Reference

| Technique | Approx Δ bpb | Status |
|-----------|-------------|--------|
| **Pre-quant TTT (any form, before GPTQ)** | — | **ILLEGAL — PR #1351, #1408, #1416, #1423 all illegal** |
| **Standard SLOT δ-vector (arXiv:2505.12392)** | **-0.021** | **DE FACTO IN USE — Issue #140 CLOSED (Apr 6); 9 record PRs use SLOT variants; no organizer rejection** |
| **Causal SLOT-16 (scored-position delta only)** | **-0.009** | **DE FACTO IN USE — PR #1333 (aryanbhosale, 1.0766 BPB, open record); PR #1229 (0.9300 BPB). No organizer rejection.** |
| **Scored-Position SLOT (PR #1229)** | **~-0.18 vs base** | **Extraordinary — 0.9300 BPB; no organizer rejection; causality concern still present** |
| **ETLB (Eval-Time Logit Bias)** | **-0.0019** | **UNRULED — PR #1399/#1415; await before implementing** |
| **N-gram Tilt (PR #1437 kernel)** | **-0.0029** | **LEGAL — properly normalized via Z; causal; zero artifact cost. PR #1420 has causality bug — use PR #1437** |
| **VarLen Attention + Doc-TTT** | **~-0.007** | **LEGAL — PR #1560 (dexhunter, 1.07406 BPB); per-document causal masking + score-first TTT per-doc; LoRA chunk=48** |
| **TMA Megakernel (Triton Hopper fused MLP)** | **+200 steps (~-0.002)** | **LEGAL — PR #1555; +10.5% throughput; add after base validated** |
| **Tap-In Unigram Matching (min_match=1)** | **~-0.009** | **LEGALITY UNCONFIRMED — PR #1555; 21% activation rate; verify before implementing** |
| **Attention Output Gate + SmearGate (PR #1667)** | **~-0.006 bpb (vs merged SOTA)** | **APPEARS LEGAL — PR #1667 (MarioPaerle, 1.07139 BPB); per-head multiplicative gate (1,056 params, init to zero); SmearGate width=12; no reviews; stack on PR #1586** |
| **Per-Layer Adaptive GPTQ (MLP=12σ, Attn=13σ) + int7 Emb** | **-0.013 nats (-0.0046 bpb)** | **LEGAL — PR #1586 (dexhunter, 1.07493 BPB); MLP tighter clip, Attn looser, int7 emb saves 530KB; MLR=0.026; IMPLEMENT IMMEDIATELY** |
| **Systems Opt (fused Muon + batched EMA + loader prealloc)** | **~+20 steps (~-0.001 bpb)** | **LEGAL — PR #1584 (codemath3000, 1.0752); pure kernel/memory efficiency; no ML changes** |
| **CaseOps Bijective Tokenizer** | **~-0.014 bpb (est.)** | **LEGALITY DEBATED — PR #1769 (dexhunter, 1.06453), #1771 (bigbag, 1.06513); reversible case-factoring (TITLE/ALLCAPS/CAPNEXT/ESC control tokens); BPB on original UTF-8 via byte sidecar; bigbag filing PR #1771 = strongest signal of legal approval; await Issue #1604 ruling (Apr 24 self-deadline); begin implementation prep now** |
| **LoRA-TTT warm-start A + alpha=144 + WD=1.0** | **~+0.009 bpb improvement vs standard TTT** | **APPEARS LEGAL — PR #1767 (renqianluo, 1.07209) and PR #1771 (bigbag) both use this; score-first per-document AdamW; warm-start A from training weights; add to our TTT phase** |
| **MP-SGD TTT 4 phases** | **~-0.009 bpb (est.)** | **APPEARS LEGAL — PR #1727 (yahya010, 1.07217); score-first each phase, all under torch.no_grad() before update; extends 3-phase approach; stackable** |
| **Polar Express Newton-Schulz (arXiv:2505.16932, ICLR 2026)** | **~+5-10% step quality** | **IMPLEMENT — Drop-in replacement for fixed 5-step NS in Muon; adapts polynomial update rule each NS iteration; ~2× faster convergence vs NS when σ_min ≈ ℓ. PR #1787 uses it (1.06335 BPB). Zero legality risk.** |
| **MIN_LR warmdown floor** | **~+steps quality** | **IMPLEMENT — Set MIN_LR=0.10 (warmdown to 10% of peak LR, not 0). PR #1787. Easy 1-line change.** |
| **PPM Byte-Level Adaptive Mixture (PR #1795)** | **~-0.069 bpb (vs 1.0810)** | **WATCH — 1.01252 BPB (OE-GOD). Classical PPM order-4 + neural LM byte-level mixture with adaptive-λ gate. Score-first: PPM updates counts only after scoring each byte. Gate legality concern was fixed. Await @valerio-oai ruling before implementing.** |
| **Gram Newton-Schulz (Dao-AILab 2026)** | **similar to Polar Express** | **WATCH — Iterates on small symmetric Gram matrix XX^T, lower FLOPs vs standard NS. pip installable. Alternative to Polar Express.** |
| **Looped Transformer Outer Normalization (arXiv:2604.15259)** | **unknown standalone** | **WATCH — Apr 2026; proof that "recall + outer normalization = stable looped regime"; adding LayerNorm/RMSNorm at loop output may enable deeper loops (4×) or earlier activation; ~1–3 lines** |
| **Casefold Tokenizer (NFKC + lowercase BPE retrain)** | **~-0.017 bpb** | **LEGALITY DEBATED — PR #1578 (1.0668), #1585 (1.0639); modifying val corpus byte count raises comparability concern; await @valerio-oai ruling** |
| **GDN-Hybrid Architecture (Gated DeltaNet + SWA)** | **~~-0.064 vs merged SOTA~~ → BPB BUG** | **BPB CALCULATION BUG (Apr 13 confirmed) — PR #1576 space-token double-count; actual ~1.16–1.18, not 1.01671. PR #1698 (arsenis-cmd) also has same bug + artifact size violation. All GDN PRs effectively dead.** |
| **Triple Loop (3× depth recurrence)** | **~-0.009 vs 2×** | **IN MERGED SOTA — PR #1493 (1.0810); 17 virtual layers; activate at 0.35× training** |
| **SP8192 vocab** | **~-0.009 vs SP4096** | **IN MERGED SOTA — PR #1493** |
| **GPTQ Embeddings (int8) + SDClip** | **~-0.003 + artifact** | **IN MERGED SOTA — PR #1394; saves ~4MB artifact budget** |
| **QK-Gain 5.25** | **~-0.001 vs 5.0** | **IN MERGED SOTA — PR #1493** |
| **Legal Score-First TTT (all blocks, 3ep)** | **-0.003** | **IN MERGED SOTA — PR #1413; lr=0.005; ≤3ep safe; 5ep cited in PR #1557 (refs #1514) — use ≤3ep to be safe** |
| **Parallel Residuals (layers 7-10)** | **~-0.008** | **IN MERGED SOTA — PR #1493** |
| **MuonEq-R optimizer** | **~-0.005** | **IN MERGED SOTA — arXiv:2603.28254** |
| **4× MLP expansion** | **~-0.01** | **IN MERGED SOTA** |
| SP4096 vocab | ~-0.02 vs SP1024 | Superseded by SP8192 |
| Sliding window eval (stride=64) | -0.032 | In SOTA |
| AR Self-Gen GPTQ calibration | ~-0.005 | In older merged SOTA (PR #1019) |
| XSA (all 11 layers) | -0.002 to -0.005 | In older merged SOTA |
| EMA decay 0.9965 (vs 0.997) | ~-0.002 | In merged SOTA (PR #1493 uses 0.9965) |
| 3× MLP expansion | -0.015 | In older SOTA |
| Int6 QAT | -0.010 | In SOTA |
| SmearGate + BigramHash(4096) | -0.006 | In older SOTA |
| Value Residual (ResFormer) | -0.005 to -0.017 | In older SOTA |
| Gated DeltaNet (PR #1370) | ~-0.11 vs baseline | **Non-record (>10 min) — but PR #1564 GDN-Hybrid claims 10-min compliance at 1.01710** |
| **Cooldown+QAT fusion (arXiv:2509.22935)** | **~-0.002** | **WATCH — LR decay jointly with QAT; no artifact size change** |
| **LaCT large-chunk TTT (arXiv:2505.23884)** | GPU util 0→70% | WATCH — PR #1560 Doc-TTT may be LaCT-style; dexhunter already implementing |
| **SGT sparse depth recurrence (arXiv:2603.23998)** | saves FLOP budget | Watch — reduces Triple Loop FLOP overhead |
| **Newton-Muon (arXiv:2604.01472)** | **~+6% steps (~+288 steps at our scale, ~-0.001 bpb)** | **WATCH — Apr 2, 2026; right-preconditioning via input second moment; 6% fewer iterations + 4% wall-clock vs Muon on nanoGPT speedrun; drop-in Muon swap. Verify additive with MuonEq-R before GPU spend.** |
| MUD/MomentUm Decorrelation (arXiv:2603.17970) | +20-50% throughput | WATCH — triangular Cholesky whitening; 1.3–2.6× tokens/sec vs Muon |
| Mousse (arXiv:2603.09697) | ~-0.002 to -0.003 | WATCH — Kronecker-factored preconditioning for Muon; ~12% fewer steps |
| Infini-gram interpolation (arXiv:2401.17377) | large but legal unclear | WATCH — suffix array ∞-gram, normalized |
| **Parcae stable loop injection (arXiv:2604.12946)** | **~6.3% lower perplexity vs prior looped models** | **WATCH — Apr 16, 2026 (UCSD + Together AI); constrains spectral norm of loop injection via negative diagonal parameterization; prevents residual explosion in our Triple Loop; may enable depth 4 or earlier activation. GitHub: github.com/sandyresearch/parcae** |
| **Recurrence Depth Curriculum (arXiv:2511.07384)** | **unknown standalone** | **WATCH — PR #1756 implements depth 1→3→4 over training thirds; theoretical backing confirmed; await CaseOps ruling + BOS bug fix before adopting** |
| AdamW TTT (30 ep, train-then-score) | — | **ILLEGAL (PR #771 rejected)** |
| N-gram hash cache | — | **ILLEGAL (normalization, Issue #1017)** |
| LoRA TTT | **+0.004 (HURTS)** | **Abandoned** |

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
12. **Simple depth recurrence (PR #540) is falsified.** Got 1.2092 bpb (worse than baseline). However, **Depth Recurrence + Parallel Residuals** in PR #1334 achieves 1.0897 — different implementation. The key is parallel residual connections alongside recurrence. See PR #1334 before dismissing.

### Session 3 (2026-03-24)
13. **In-Place TTT is HARMFUL.** Loss INCREASES (2.63+, going up not down). MLP output projections are NOT good TTT targets at this scale. Do not attempt.
14. **GradQuant int5/int6 mix exceeds 16MB.** Even without int7, the artifact was 34KB over. Use uniform int6 or match PR #414's exact quantization scheme.
15. **PR #486 baseline reproduced at 1.1249** (vs reported 1.1233). Within seed variance. This is our verified baseline.
16. **The v7.0 incremental plan works.** Run 0→1→2→3 from PR #414 base. Each run adds ONE thing. Stop doing moonshots with 500+ new lines.

### Session 4 (2026-04-01)
17. **Score-first is non-negotiable for TTT.** PR #771 was rejected because we ran 30 epochs of TTT on all val tokens, THEN evaluated on those same tokens. The rule: score the token first, update on scored token, move on. Maximum ~3 epochs to stay within time budget. "Train-then-score" ordering = instant rejection.
18. **N-gram hash cache is illegal without proper normalization.** Issue #1017 (2026-03-27) ruled all hashed n-gram caches out of the record track — they don't produce normalized probability distributions. PRs #727, #741 closed. The "CONFIRMED LEGAL" status from earlier sessions is void. Any n-gram implementation must renormalize properly on every backoff step.
19. **Verify legality of novel eval techniques before GPU spend.** SLOT (δ-vector on final hidden layer) shows -0.021 bpb but causality concerns raised. As of 2026-04-04, still unruled — @abaybektursun removed it from his own SOTA stack. Check Issue #140 for @valerio-oai ruling before spending GPU.
20. **4096 vocab + 4×MLP achieves sub-1.1 WITHOUT TTT.** PR #1218 (clarkkev) gets 1.09785 via architecture alone — 4096 vocab, 4×MLP, XSA-all, GPTQ, WD=0.085. No TTT needed. This is a low-risk, high-EV path.

### Session 5 (2026-04-04)
21. **Discriminative TTT is legal and high-EV.** PR #1351 (1.0807 bpb) uses per-block adaptive LR during score-first TTT: early transformer blocks get 0.3× LR, later blocks 1.0× LR. Adds -0.010 bpb vs flat-LR TTT. Adopt in next GPU run.
22. **Depth Recurrence + Parallel Residuals is the new architecture frontier.** PR #1334 (1.0897, clean, no TTT/SLOT) uses shared-weight layers iterated N times with parallel residuals and MuonEq-R optimizer. Better than adding more distinct layers. Investigate before GPU spend.
23. **MuonEq-R replaces standard Muon in top submissions.** Appears in PR #1334, #1344, #1326. Likely a Muon variant with equalized updates or rotation. Read PR #1334 code before next run.
24. **Best reachable target without SLOT: ~1.080.** PR #1351 (Discriminative TTT) + PR #1334 (Depth Recur + Parallel Res + MuonEq-R) combined should reach 1.080–1.085 range. Both techniques are legal. This beats merged SOTA by ~0.035 nats — well above the 0.005 threshold.

### Session 6 (2026-04-05)
25. **Pre-quant AdamW TTT is ILLEGAL.** PR #1351 author (resouer) self-closed on 2026-04-05 citing pre-quant TTT as "pre-eval adaptation on validation data." Every prediction depends on the model having memorized targets across 6 full epochs on the exact validation set. Do NOT use pre-quant TTT. Remove -0.022 bpb entry from consideration.
26. **Discriminative TTT technique is separable from pre-quant TTT.** The per-block adaptive LR (0.3× early, 1.0× late) technique in PR #1351 is distinct from the illegal pre-quant component. Score-first post-quant discriminative TTT ≤3ep remains legal.
27. **PR #1334 is now the cleanest path to sub-1.09 bpb.** Zero legality flags, 1.0897 bpb, fully documented. Merged SOTA delta: 0.0250 bpb. Primary implementation target.
28. **Gated DeltaNet (PR #1370, 1.003 bpb) is a new architecture to watch.** Linear attention O(n) complexity, no legality flags. Not in our current plan but could be strong base alternative.

### Session 7 (2026-04-06)
29. **N-gram Tilt is the legal n-gram path.** PR #1420 uses properly normalized next-token hints: `p_tilt(t) = p_model(t) · exp(β · 1[t==hint]) / Z`. Causal (backward-looking only). -0.0029 bpb, zero artifact cost. Replaces the illegal hash cache approach.
30. **Triple Loop (3×) beats Double Loop (2×).** PR #1420 repeats layers 4-5 three times (17 virtual layers) vs PR #1334's twice (13 virtual layers), activated earlier at 0.35× training. Combined with SP8192: 1.08014 bpb vs 1.0897.
31. **SP8192 beats SP4096 by ~0.009 bpb.** Upgrade vocab target from SP4096 to SP8192. PR #1420 and #1413 both confirm.
32. **Fused kernels are now worth implementing.** PR #1420 shows Triton TMA (forward) + CUTLASS 3.x (backward) yields +10% throughput = +127 steps in 600s budget. Earlier "poor EV" assessment was wrong at this scale. Add after base stack is confirmed working.
33. **abaybektursun is still the leader to track.** He holds merged SOTA (PR #1019, 1.1147) AND best open legal PR (PR #1420, 1.08014). His techniques reliably work. Read his PRs first.
34. **All pre-quant TTT variants are illegal.** PRs #1408 (dTTT 10ep), #1406 (discriminative pre-quant), #1416 (SP8192 + pre-quant) all use pre-quant TTT. Same illegality as PR #1351. Do not implement any of these.
35. **ETLB is promising but awaits ruling.** -0.0019 bpb, post-LM-head bias vector, uses context tokens (already scored). Ask @valerio-oai in Issue #140 before GPU spend. If legal, easy -0.002 bpb with no weight modification.

## Golden Rules

Every change must answer: "Does this lower val_bpb within the 16MB/10-min constraints?" If the answer is unclear, run a quick experiment on 1xH100 before investing more time. Compression and eval tricks are as valuable as architecture changes. The cheapest experiment that gives signal is the best experiment. Speed > perfection — submit early, iterate after.

**NEW (2026-04-01)**: Before any eval-time technique, answer: "Does the model see the ground truth label before scoring the token?" If yes, it's illegal. If no, verify with @valerio-oai before spending GPU time.

### Session 8 (2026-04-07)
36. **N-gram Tilt in PR #1420 has a causality bug.** PR #1437 (dexhunter) independently found and fixed it. Pre-fix: 1.07807 bpb; post-fix: 1.08091 bpb. Always use PR #1437's corrected kernel. Do NOT copy the n-gram tilt implementation from PR #1420 directly.
37. **Extraordinary score claims (0.396 bpb) are almost always illegal.** PR #1430 by renqianluo claims 0.39642 via per-sample SLOT + N-gram order-22 hash + TTT. N-gram hash without proper normalization = same pattern as #727/#741 (closed). Per-sample SLOT (196K params per sequence trained on val data) strains the spirit of eval rules. Expect rejection.
38. **The competition is accelerating.** PRs #1421–1444 filed in one day (Apr 6–7). Check PRs list every session — the landscape shifts fast.
39. **Issue #140 closed (Apr 6). @valerio-oai never ruled there.** All official rulings came from direct PR comments and Issue #677. SLOT has never been officially banned — 9 record PRs use it. Causal SLOT-16 (PR #1333, 1.0766 BPB) is a credible target. The SLOT causality concern is real but unresolved. Accept the risk explicitly if implementing.
40. **Scored-position SLOT (PR #1229) reached 0.9300 BPB.** This is the highest-quality legal-ish technique in the competition. If @valerio-oai rules it legal, it would leapfrog everything. High risk, high reward — do NOT implement without explicit decision to accept rejection risk.

### Session 9 (2026-04-10)
41. **Merged SOTA jumped from 1.1147 → 1.0810 in 3 days.** Four records merged Apr 5–9: PR #1394 (1.0856), #1412 (1.0835), #1477 (1.0822), #1493 (1.0810). Competition pace is accelerating — check the leaderboard at the start of EVERY session; stale SOTA numbers are dangerous.
42. **ANS weight compression (PR #1510) is the single highest-EV non-model change available.** rANS per-layer histogram encoding replaces LZMA; 1.6MB freed = 2.2M extra params at int6. Zero model quality change, no legality risk. Implement before next GPU run.
43. **Parameter Banking + Parallel Muon (+5.2% throughput) is the highest-EV optimizer change.** Consolidate 66 matrices into 4 contiguous banks; batch Newton-Schulz iterations; no data-quality loss. Combine with existing Triton TMA kernels for up to 15% total throughput gain.
44. **Muon 0.97 and QK-Gain 5.25 are free wins.** Both confirmed in open/merged PRs. Change both before any GPU run; cost is two hyperparameter edits.
45. **"Eval-time adaptation" is a growing legality concern.** PR #1517 (Pre-Quant TTT 18ep) and PR #1523 (Eval-Time Hash Embedding trained at inference) both likely illegal — same adapt-before-score pattern that killed PR #1351, #1408, #1416, #1423. When ANY technique modifies parameters or a trainable table during evaluation before scoring the current token, treat it as illegal until ruled otherwise.
46. **Tap-In V6 is interesting but unruled.** abaybektursun's document-local phrase matching (PR #1518) could be legal (backward-looking, causal). Explicitly NOT the same as n-gram hash cache (no hash table, no unnormalized distribution). But await @valerio-oai ruling before implementing.
47. **New target is ≤1.0760 bpb**, not 1.1142. CLAUDE.md has been updated. Any experiment plan using the old SOTA as target is obsolete.

### Session 10 (2026-04-11)
48. **PR #771 is fully closed — no appeal.** @valerio-oai's ruling is explicit: train-then-score ordering on val tokens across multiple epochs = instant rejection. The 30ep AdamW TTT with cosine LR and per-layer LR adjustments is permanently void.
49. **PR #758 (7-gram cache, 1.0465 bpb) is effectively dead.** Reviewer MatoTeziTanka flagged BOTH a TTT contradiction (code runs train-then-score by default) AND n-gram unnormalized distribution. Even if the author fixes one, both issues apply. Don't track this PR for inspiration.
50. **PR #1545 BPB bug: extraordinary claims require BPB verification.** GDN-Hybrid claimed 1.028 bpb, but the byte-counting function has a +1 double-count for space tokens inflating total bytes ~14%. Actual score ~1.18 bpb. ALWAYS check the BPB calculation code when a score seems too good to be true.
51. **Doc-Independent LoRA TTT (PR #1540) is a legal path.** Rank-96 LoRA adapter initialized to zero before each batch, trained score-first, discarded after each document. No state leaks across documents. This is fundamentally different from the abandoned static LoRA TTT (which was applied at training time, not eval). Evaluate after ANS compression + banking are in.
52. **Improved Parallel Residuals (PR #1541) is bigbag's next move.** The merged SOTA author's new open PR introduces cross-lane routing with learned scalars. Monitor for merge — if it merges, the merged SOTA drops to ~1.0778 and our target tightens to ≤1.0728.
53. **MATRIX_LR = 0.03 pairs with Muon momentum 0.97.** Both PRs #1541 and #1523 co-tune these. When reducing momentum from 0.99 → 0.97, also reduce MATRIX_LR. Check whether our base config uses 0.03 or 0.05.

_Updated: 2026-04-11 (v11.5 — PR #1541 bigbag 1.07785 + PR #1540 aryanbhosale 1.0777 new open PRs; doc-independent LoRA TTT appears legal; PR #1545 BPB bug; MATRIX_LR 0.03 pairs with momentum 0.97; no merged SOTA change)_
### Session 11 (2026-04-12)
54. **Merged SOTA jumped from 1.1147 to 1.0810 in 5 days.** Six PRs merged between Apr 4–9 (PRs #1334, #1285, #1394, #1412, #1413, #1477, #1493). The competition accelerated dramatically. Check leaderboard every session before planning — yesterday's target may already be beaten.
55. **The merged SOTA stack is now fully defined: SP8192 + Triple Recurrence + Parallel Residuals + QK-Gain 5.25 + GPTQ Emb (int8) + SDClip + WD=0.095 + EMA 0.9965 + Legal TTT.** PR #1493 (bigbag) at 1.0810. Any new submission must beat this cleanly. Target: ≤1.0760.
56. **VarLen Attention (per-document masking) is the next clear win.** PR #1560 (dexhunter) achieves 1.07406 BPB by adding per-document causal masking + Doc-TTT (per-document score-first LoRA TTT, chunk=48) on top of the PR #1413 stack. -0.009 bpb vs merged SOTA. Implement this next.
57. **GDN-Hybrid (PR #1564) at 1.01710 BPB is extraordinary — watch closely.** Gated DeltaNet + SWA architecture, no TTT/SLOT, SP1024. If organizers approve, this represents a ~0.064 bpb architectural leap with no eval-time techniques. Do not implement until organizer review; replicate if approved.
58. **TMA Megakernel (Triton Hopper) gives +200 training steps.** PR #1555 shows +10.5% throughput on H100 via TMA-fused MLP kernel. Worth implementing after VarLen+Doc-TTT is verified. Combined with Tap-In (min_match=1, 21% activation), PR #1555 reaches 1.07636.
59. **Do NOT implement Tap-In before verifying legality.** "Tap-In Unigram Matching" from PR #1555 activates at 21% of positions vs 1.7% at min_match=3. Mechanism involves token-level unigram cache — may be similar to n-gram approaches. Verify it's properly normalized before GPU spend.
60. **PR #731 n-gram is now looking clean.** Dense count tables + Laplace smoothing (not hash caches). Reviewer said "LOOKS CLEAN" — waiting on seeds 1337 and 2024 to confirm 1.0400 BPB. If merged, this gives a legal n-gram mixer alternative.
61. **18 days remain. Prioritize safe incremental improvements over risky architecture rewrites.** VarLen+Doc-TTT (PR #1560 approach) is the lowest-risk path to beating the target. File that first, then consider GDN-Hybrid rewrite if approved.

_Updated: 2026-04-12 (v12.1 — merged SOTA 1.0810 (PR #1493, Apr 9); 6 new merges; GDN-Hybrid 1.01710 open; VarLen+Doc-TTT 1.07406 open; target ≤1.0760; 18 days remaining)_
### Session 12 (2026-04-13)
62. **PR #758 n-gram is effectively dead.** MatoTeziTanka (Apr 12) flagged the 7-gram cache XOR hash key includes target token — same normalization/leakage violation as PRs #727/#741. The reviewer explicitly states the neural base is ~1.10–1.15 without the cache. Stop tracking #758.
63. **GDN-Hybrid BPB bug confirmed (PR #1576).** Space token double-count inherited from PR #1545 inflates byte denominator ~14%, making 1.01671 actually ~1.16–1.18 BPB. No organizer response yet. PR #1564 was voluntarily closed (superseded by PR #1575). Extraordinary GDN-Hybrid claims are FALSE until the author provides corrected byte-counting code.
64. **Per-Layer Adaptive GPTQ (PR #1586) is the highest-EV immediate action.** dexhunter's PR achieves 1.07493 (3-seed mean, std 0.00078) by differentiating GPTQ clip_sigmas: MLP=12.0, Attn=13.0, Emb int7@15.0σ. Saves 530KB vs int8 Emb, MLR=0.026. -0.01266 nats vs merged SOTA (>2× the 0.005 threshold). No legality concerns. This is a config-level change that should be in our submission.
65. **Casefold Tokenizer legality is actively contested.** PR #1578 (1.0668) and #1585 (1.0639) apply NFKC+lowercase to the validation corpus, reducing what bytes need to be predicted. Three community members debated it; no organizer ruling as of Apr 13. The improvement is real (~-0.017 bpb) but the legality is uncertain — do NOT implement until @valerio-oai rules.
66. **Systems optimizations (PR #1584) give ~20 extra steps for free.** Fused Muon kernel + batched EMA + loader prealloc = same training budget with ~20 extra gradient steps. Pure engineering, no model changes. Worth including before next submission.
67. **arXiv:2604.06169 In-Place TTT (Apr 7) is worth reading.** Replaces TTT's generic reconstruction loss with a next-token-prediction-aligned objective, enabling chunk-wise updates compatible with score-first paradigm. Could improve legal TTT quality. Read before next TTT implementation.
68. **Merged SOTA held at 1.0810 for 4 days (Apr 9–13).** This is the longest gap since competition acceleration began. Either the field is catching up, or a wave of PRs is being prepared. Expect merges in next 2–3 days given the 8 open PRs in range.

_Updated: 2026-04-13 (v12.2 — merged SOTA 1.0810 confirmed; PR #758 dead; GDN-Hybrid BPB bug confirmed; PR #1586 per-layer GPTQ highest-EV immediate action; Casefold Tokenizer legality debated; 17 days remaining)_

### Session 13 (2026-04-14)
69. **Merged SOTA still 1.0810 — now 5 days without a new record.** Longest plateau since the Apr 5–9 acceleration wave. Eight open PRs are in striking range (1.062–1.077); the next merge will likely set the new SOTA. Track PR #1586 (per-layer GPTQ, 1.0749), #1585 (Casefold, 1.0639 pending ruling), and PR #1541 (bigbag's improved parallel residuals, 1.07785).
70. **PR #1610 (PhasingTTT, 1.0728) is a new legal TTT approach.** romeerp's PR combines VarLen Attention with a two-phase global SGD: LoRA TTT on 2000 scored docs, pause, global SGD update on those already-scored tokens, resume. Score-first compliant. But EV is low (-0.0006 bpb vs base); do not prioritize over PR #1586.
71. **PRISM (arXiv:2602.10796, Feb 2026) is architecturally relevant.** Parallel Residual Iterative Sequence Model: performs iterative non-linear correction within a parallelizable linear recurrence, achieving 174× throughput vs serial solvers. Our depth recurrence + parallel residuals approach in PR #1493 is the same spirit. Read before any recurrence architecture change — may reveal how to improve recurrence quality without extra parameters.
72. **Ouroboros (arXiv:2604.02051, Apr 2026) is worth monitoring.** Generates per-step LoRA modulation dynamically via a hypernetwork (input-conditioned, not static per-layer). Could make our 3× recurrence loops more expressive. Implementation adds hypernetwork parameters — must verify 16MB fits before attempting. Watch for any competition PR implementing this.
73. **16 days remain. Implement PR #1586 (per-layer GPTQ + int7 emb) before any other change.** -0.01266 bpb, verified 3-seed, zero legality risk. This is the single fastest path to beating the merged SOTA. Do not wait for Casefold or Hedge Mixer rulings before running the GPU experiment.

_Updated: 2026-04-14 (v12.3 — merged SOTA 1.0810 Day 5 no change; PR #1610 PhasingTTT legal but low EV; PRISM arXiv:2602.10796 relevant to recurrence design; Ouroboros arXiv:2604.02051 watch; 16 days remaining)_

### Session 14 (2026-04-15)
74. **Merged SOTA 1.0810 enters Day 6 plateau — longest since competition acceleration.** `git log upstream/main` confirms last commit was Apr 9 15:22 PDT. No new records or merges detected via git or web search. Eight open PRs remain in range. Expect imminent merge wave.
75. **Newton-Muon (arXiv:2604.01472, Apr 2) is a drop-in Muon swap worth testing.** Right-preconditioning by input second moment gives 6% fewer iterations + 4% wall-clock vs standard Muon on nanoGPT speedrun benchmark. At our ~4800-step budget, 6% ≈ +288 effective steps ≈ small free bpb gain. NOT currently in technique table — added today. Verify it is additive with MuonEq-R (our current optimizer) before spending GPU; they may be redundant.
76. **In-Place TTT (arXiv:2604.06169) is NOT the same as Session 3's failed attempt.** Session 3 used reconstruction loss on MLP output projections and saw loss blow up (2.63+). The Apr 7 paper uses an NTP-aligned loss, which is theoretically grounded for autoregressive LM. The "HARMFUL" lesson (#13) should not prevent trying In-Place TTT with NTP-aligned loss on a modern base. Low priority now; revisit after PR #1586 + VarLen+Doc-TTT are confirmed.
77. **No new open PRs filed Apr 14–15 with competitive scores.** Web search and git log show nothing new. PR #1619 (likely illegal AdamW TTT) and PR #1616 (QK-Gain 5.5) are low-interest. The competitive field is in a holding pattern — same 8 PRs as yesterday.

_Updated: 2026-04-15 (v12.4 — merged SOTA 1.0810 Day 6 no change; Newton-Muon arXiv:2604.01472 added (+6% effective steps, verify vs MuonEq-R); In-Place TTT (2604.06169) NTP-aligned loss distinguishes it from Session 3 failure; 15 days remaining)_

### Session 15 (2026-04-16)
78. **Merged SOTA 1.0810 — Day 7 plateau, longest in competition history.** Seven days since last merge (Apr 9). With 14 days to deadline, the field appears to be preparing a late push. Do not take the plateau as stability — a wave of merges is likely imminent given 8+ open PRs in the 1.062–1.078 range.
79. **PR #1667 (MarioPaerle, 1.07139) is a new clean stackable technique.** Attention Output Gate: 1,056 parameter multiplicative gate on attention output heads (12 weights × 8 heads × 11 layers), initialized to zero so scale starts at 1.0. SmearGate reintroduced (width=12, input-dependent). Legal score-first TTT (3ep, SGD, LR=0.005). Artifact 15.927 MB. No legality flags. Stack this on top of PR #1586 before next GPU run.
80. **PR #1670 (dexhunter, 1.05970) is the new best open PR — but depends on casefold ruling.** Casefold V4 + Multi-Phase Global SGD TTT achieves 1.05970 (std 0.00031, 3-seed). The Casefold legality question (Issue #1604) has no @valerio-oai ruling as of Apr 16. Do NOT implement until ruled. If casefold is approved, this becomes the primary target and resets our goal to ≤1.0499.
81. **PR #1647 (powerpratik, 1.0616) uses standard SLOT-4 — high risk.** Delta-vector logit bias optimized 4 AdamW steps per window. No organizer reviews yet. Standard SLOT (not causal SLOT-16). Risk: @valerio-oai could rule at any time. Only implement if willing to accept rejection.
82. **PR #731 (Hedge Mixer, 1.0400) is close to merge — 2 seeds pending.** Dense-count tables + Laplace smoothing + 5-expert ensemble. Reviewer confirmed score-first per chunk and said "LOOKS CLEAN." Seeds 1337 and 2024 are the only remaining gate. If both seeds confirm ~1.04, this merges and gives us a legal n-gram mixer blueprint.
83. **dexhunter now holds 3 of the top-5 open legal PRs (#1560, #1586, #1670).** Highly reliable submitter with zero legality flags across all PRs. Copy techniques from his PRs with confidence.

_Updated: 2026-04-16 (v12.5 — merged SOTA 1.0810 Day 7; PR #1667 Attention Output Gate new clean stackable tech; PR #1670 dexhunter 1.05970 best open but casefold pending; PR #1647 SLOT-4 risky; PR #731 seeds pending; 14 days remaining)_

### Session 16 (2026-04-17)
84. **PR #1698 (arsenis-cmd, GatedDeltaNet FLA, 1.00995 BPB) is the most significant new open PR — but has an artifact size concern.** Artifacts are 16.6MB decimal (16,600,916 bytes) vs the 16,000,000-byte competition limit. Author claims "under 16 MiB" (16,777,216 bytes). If organizers enforce 16 MB decimal, PR is disqualified. If organizers accept 16 MiB (binary), PR is clean. Do NOT invest GPU time in GDN rewrite until artifact limit is resolved.
85. **The BPB bug pattern (SP leading-space double-count) keeps recurring.** PR #1687 (resouer, claimed 1.04090) is the 4th instance — @bigbag caught it again. Actual score ~1.22 BPB. Any extraordinary score claim from an FLA/GDN PR must be verified against SP byte-counting at line ~192–205 in training script before tracking.
86. **dexhunter's PR #1693 supersedes PR #1670 for casefold.** Stacks AttnOutGate + SmearGate on Casefold V4 + Multi-Phase TTT for 1.05733 (down from 1.05970). If casefold is ruled legal, our target resets to ≤1.0523. Still awaiting @valerio-oai ruling on Issue #1604.
87. **Merged SOTA at Day 8 plateau — expect imminent wave.** Eight open PRs within 1.062–1.078 BPB range (no GDN). If PR #1698 artifact is ruled compliant and merges, new merged SOTA becomes ~1.010 and our entire incremental strategy is obsolete. Check leaderboard at session start every day.
88. **PR #731 Hedge Mixer still needs 2 seeds.** "LOOKS CLEAN" from reviewer. Seeds 1337 and 2024 pending. If confirmed ~1.04 and merged, legal n-gram mixer blueprint available with dense-count tables + Laplace smoothing.

_Updated: 2026-04-17 (v13.0 — PR #1698 GatedDeltaNet FLA 1.00995 flagged artifact size concern; PR #1687 CLOSED BPB bug; PR #1693 dexhunter 1.05733 casefold leader; merged SOTA 1.0810 Day 8; 13 days remaining)_

### Session 17 (2026-04-19)
89. **PR #1698 (GDN FLA) is now effectively dead.** dexhunter confirmed BPB bug in `build_sentencepiece_luts` (leading-space double-count, same pattern as PR #1545/1576/1687) — corrected actual score is ~1.189 BPB, not 1.00995. Artifact size also confirmed over-limit (16.47–16.60MB vs 16MB decimal). Author has no GPU access. Do NOT track.
90. **CaseOps bijective tokenizer is a distinct and potentially legal approach.** Three PRs (#1729 romeerp, #1736 dexhunter, #1738 alertcat) use a reversible case-factoring transform: capitalization encoded as control tokens (TITLE/ALLCAPS/CAPNEXT/ESC), fully reconstructible, BPB scored on original UTF-8 bytes via byte sidecar. Unlike casefold (which permanently discards case), CaseOps is lossless and has a stronger legality argument. Awaiting Issue #1604 @valerio-oai ruling. dexhunter (most reliable submitter) achieves 1.06549 with CaseOps + GatedAttn + QuantGate (PR #1736).
91. **PR #1735 (AjAnubolu, 1.0429) is pre-quant TTT — flagged by dexhunter as illegal.** "pre_quant_adamw_ttt runs 21 AdamW epochs over the full val stream before the final BPB is scored on the same tokens." Same adapt-then-score pattern as PR #1351/#1408/#1416. PR #1738 (alertcat, 1.03540) builds on PR #1735 — both scores likely void. Do NOT implement.
92. **MP-SGD TTT 4 phases (PR #1727, yahya010, 1.07217) appears legal.** Each phase scores under `torch.no_grad()` before any SGD update, all four phases fit within the 600s eval budget. Extends earlier 3-phase approach. Stackable with #1586+#1667.
93. **Merged SOTA plateau now 10 days (Apr 9 → Apr 19) — deadline is 11 days away.** With 8+ open PRs in 1.062–1.078 range, a merge wave is overdue. Implementing #1586 immediately is critical — every day without it is wasted headroom.

_Updated: 2026-04-19 (v14.0 — PR #1698 GDN effectively dead (BPB bug ~1.189 + artifact violation); CaseOps bijective tokenizer new community technique (#1729/#1736/#1738); PR #1735 pre-quant TTT flagged illegal; PR #1727 MP-SGD TTT 4-phase appears legal at 1.07217; merged SOTA 1.0810 Day 10; 11 days remaining)_

### Session 18 (2026-04-21)
94. **Pre-quant TTT pattern (PR #1758, 1.02840) is the 6th attempt — still illegal.** kilojoules tuned PR #1738's pre-quant TTT with LR=1e-3 and unfrozen blocks to reach 1.02840. No reviews filed; no organizer response. Same adapt-then-score violation as PR #1351/#1408/#1416/#1423/#1735. The community keeps submitting; organizers keep rejecting. Do not track.
95. **Recurrence Depth Curriculum (PR #1756, romeerp, 1.06505) is a new technique with theoretical backing.** Three-phase training schedule: depth 1 (first third) → depth 3 (second third) → depth 4 (final third), eval fixed at depth 4. Grounded in arXiv:2511.07384 (retrofitted recurrence curriculum). Two issues: (a) awaits Issue #1604 CaseOps ruling; (b) `prepare_caseops_data.py` missing BOS insertion causes ZeroDivisionError in phased TTT eval path (@codemath3000 flagged Apr 21). Watch for fix. If CaseOps ruled legal and bug fixed, this is a strong stack candidate.
96. **Parcae (arXiv:2604.12946, UCSD + Together AI, Apr 16) directly addresses our Triple Loop instability.** Looped models suffer residual explosion from large spectral norms in injection parameters. Parcae constrains spectral norm via negative diagonal parameterization, achieving quality of a transformer **2× the size** at equal parameter count. Our PR #1493 Triple Loop (layers 4-5 × 3, activated at 0.35×) may be leaving performance on the table due to this same instability. If time permits after #1586+#1667+#1560 are validated, investigate adding Parcae-style spectral norm constraint on loop injection weights. GitHub: github.com/sandyresearch/parcae.
97. **Attention Output Gate (PR #1667) is backed by NeurIPS 2025 research (arXiv:2505.06708).** Head-specific sigmoid SDPA output gate breaks the low-rank bottleneck of consecutive Wv/Wo projections, yielding up to 0.2 PPL reduction. Multiplicative form (as in PR #1667) is optimal. Confirms our implementation target is theoretically sound. Implement with #1586.
98. **Issue #1604 (CaseOps/casefold ruling) has been open 8 days with no @valerio-oai response.** Self-impose a deadline: if no ruling by Apr 24 (6 days before competition close), proceed with the clean legal stack (#1586+#1667+#1560+#1727) rather than waiting. CaseOps has a stronger legal argument than casefold (bijective, lossless, BPB on original bytes), but the clock is running.
99. **Merged SOTA Day 12 plateau is now confirmed longest in competition history.** No merges since Apr 9. The 8+ open PRs between 1.062–1.078 remain unreviewed. The organizers may be reviewing multiple PRs in batch. Expect a wave when rulings come (especially Issue #1604). Check leaderboard at the start of every session.

_Updated: 2026-04-21 (v15.1 — merged SOTA 1.0810 Day 12; PR #1758 pre-quant TTT 1.02840 likely illegal; PR #1756 CaseOps+Recurrence Depth Curriculum 1.06505 has BOS bug + Issue #1604 pending; PR #1755 CaseOps+Legal TTT 1.07462 pending Issue #1604; Parcae arXiv:2604.12946 relevant to Triple Loop stability; Attention Output Gate backed by arXiv:2505.06708; Issue #1604 self-deadline Apr 24; 9 days remaining)_

### Session 19 (2026-04-22)
100. **bigbag filed PR #1771 with CaseOps (1.06513) — the most important signal of the competition.** The current merged SOTA holder explicitly bet on CaseOps legality before Issue #1604 is ruled on. His stack: CaseOps + Recurrence Depth Curriculum (1→3→4) + SmearGate + GatedAttn + LoRA-TTT (alpha=144, warm-start A, WD=1.0). 3-seed std 0.00055. This is the single strongest prior that CaseOps will be approved. Begin implementation prep immediately — do not wait for the ruling to start coding.
101. **LoRA-TTT warm-start A + alpha=144 + WD=1.0 is a new legal TTT improvement, independently discovered by two authors.** Both renqianluo (PR #1767, 1.07209) and bigbag (PR #1771) use: warm-start A matrix from training weights (not random/zero), alpha=144 (high effective LR scale), WD=1.0 AdamW during eval. renqianluo reaches 1.07209 on a clean legal base. This is ~+0.009 improvement from the TTT config change alone. Stack with #1586+#1667 immediately.
102. **dexhunter improved to 1.06453 (PR #1769) with a one-line change: MLP clip_sigmas 10.0→12.0.** This is exactly the per-layer adaptive GPTQ change from PR #1586 applied to the CaseOps stack. Confirms that #1586's GPTQ technique stacks onto CaseOps without interference. 5-seed mean, highly statistically robust.
103. **arXiv:2604.15259 (Apr 2026) proves outer normalization stabilizes looped transformers.** "Recall combined with outer normalization produces a stable, input-dependent looped regime." Practical action: add LayerNorm or RMSNorm at the output of each loop iteration in our Triple Loop. May allow depth 4×  or earlier activation (<0.35×). ~1–3 lines of code. Add after #1586+#1667+#1560+TTT are validated.
104. **Day 13 plateau = the field is waiting for Issue #1604.** Almost all new PRs use CaseOps. The competition is blocked on a single organizer ruling. If @valerio-oai approves CaseOps, there will be a massive merge wave. If rejected, the clean non-CaseOps stack (#1586+#1667+#1560+#1727+LoRA-TTT) is still ~1.068–1.072, which beats merged SOTA by >0.005 nats.

_Updated: 2026-04-22 (v16.0 — merged SOTA 1.0810 Day 13; **bigbag CaseOps PR #1771 (1.06513) — strongest signal CaseOps will pass**; dexhunter PR #1769 (1.06453) new best; LoRA-TTT warm-start A+alpha=144+WD=1.0 appears legal, confirmed by 2 independent authors; arXiv:2604.15259 outer normalization for stable loops; 8 days remaining)_

### Session 20 (2026-04-24)
105. **Scylla 0.9485 committed to track_10min_16mb/ on Apr 23, but README not updated and score is DISPUTED.** PR #1184 (icryo, "Scylla TokenMonster ~998 tokens + Full GPTQ + XSA-all + FA3") merged Apr 23 as a record folder commit. PR #1271 had earlier identified a byte accounting error; corrected actual score ~1.1289 bpb. Despite the dispute, organizers merged the folder without updating the README leaderboard. Treat merged SOTA as 1.0810 until README changes. Do NOT implement Scylla until dispute is fully resolved.
106. **PR #1787 (nprime06, 1.06335) is the new community-consensus best clean base, replacing PR #1736.** Key new ingredient: Polar Express Newton-Schulz (arXiv:2505.16932, ICLR 2026) — adaptive polynomial replacing fixed 5-step NS in Muon. Also adds MIN_LR=0.10 warmdown floor and Triton fused cross-entropy. PR #1797 (dexhunter, 1.06157) stacks SmearGate + LQER Asym on #1787. Implement Polar Express NS + MIN_LR immediately — they are pure config changes.
107. **PR #1795 (OE-GOD, 1.01252) — byte-level PPM order-4 mixture — is the most interesting new technique but needs an organizer ruling.** PPM (Prediction by Partial Matching) is a classical compressor that accumulates byte statistics score-first. Mixed with neural LM via adaptive-λ gate. Initial legality concern (gate conditioned on observed byte) was FIXED by freezing gate before observation. If legal, this alone beats merged SOTA by 0.069 bpb. Risk: organizers may view sequential PPM adaptation (even score-first) as equivalent to pre-quant TTT. WATCH but do NOT implement before ruling.
108. **Issue #1604 self-deadline (Apr 24) passed with zero @valerio-oai response — 11 days of silence.** The CaseOps wait is over. Proceed with the clean legal stack immediately. Every day waiting for this ruling is a wasted GPU run with 6 days left.
109. **Retroactive record additions (Apr 23-24) confirm the competition was richer than the README suggested.** Three old-PR records were pushed to main: Scylla 0.9485 (PR #1184, Mar 31), dexhunter 1.1122 (PR #1060), aamodbhatt 1.1179 (PR #1148). These don't change the official leaderboard but show organizers are processing backlog. Expect more README updates soon.
110. **Polar Express Newton-Schulz (arXiv:2505.16932) is a drop-in Muon improvement now proven in competition.** ICLR 2026 paper by Amsel et al. Adapts NS polynomial update rule each iteration, super-exponentially converging. Used in PR #1787 (1.06335) and PR #1802 (1.0771 with Multi-Phase TTT). Implement by replacing the fixed NS coefficient tuple in Muon with Polar Express adaptive updates. Zero legality risk.

_Updated: 2026-04-24 (v17.0 — merged SOTA 1.0810 Day 15; Scylla 0.9485 in repo but disputed; PR #1787 (1.06335) new clean base with Polar Express NS; PR #1795 (1.01252) PPM mixture needs ruling; Issue #1604 deadline passed — implement clean stack NOW; 6 days remaining)_
