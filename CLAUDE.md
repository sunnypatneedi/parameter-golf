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

**Merged leaderboard SOTA**: 1.1194 val_bpb (abaybektursun, PR #549, 2026-03-23)
**Best open PR (unmerged)**: 1.1122 val_bpb (PR #1060 dexhunter — Coprime-Stride Loader + Full Hessian GPTQ + XSA-all 11L)
**Our PR #771**: CLOSED (rule violation — multi-epoch TTT re-scored same eval tokens; see Lessons Learned #17)
**Target**: Beat PR #1060 (1.1122) with a legal pure-neural stack. SOTA improvement threshold >=0.005.

**CRITICAL RULE (2026-03-27 enforcement)**: N-gram eval caches are **BANNED** — hashed n-gram normalization bug invalidated 33+ PRs. Correctly normalized n-gram achieves only ~1.51 BPB (worse than baseline). Do not attempt. Multi-epoch TTT where you score then adapt is **BANNED** — score-first means score a token ONCE, then optionally adapt afterward, never re-score.

**The AdamW TTT revolution**: LoRA TTT hurts (+0.004 bpb). AdamW TTT with aggressive config gives **-0.04 to -0.06 bpb** — the single biggest unlock. Legal config: score-first (score each token exactly once, never re-score), then adapt. Config: 30 epochs, cosine LR decay, lr=0.0005, per-layer LR (MLP output 3x, input 0.5x). PR #549 is the verified legal reference.

**Our approach (v8.0 — pure neural, post-n-gram-collapse)**:
Build on PR #549 base (merged SOTA). Add in order:
1. **Full Hessian GPTQ** (Cholesky error compensation, like PR #1060) — expected -0.005 to -0.010 bpb
2. **XSA all 11 layers** (from PR #503) — -0.002 to -0.005 bpb
3. **Coprime-stride data loader** (diverse batch sampling, ~20 lines, from PR #1060) — low risk
4. **AdamW TTT (30ep, cosine, legal score-first)** — keep from PR #549
5. **Value Residual + TrigramHash(4096) + GradQuant** — from PR #486

**Key reference PRs**: #549 (SOTA 1.1194, legal TTT ref), #1060 (1.1122, Full GPTQ + XSA-all ref), #503 (XSA ref), #486 (Value Residual + TrigramHash + GradQuant ref)

**Abandoned approaches**: LoRA TTT (hurts), n-gram eval cache (BANNED — normalization bug), multi-epoch TTT with re-scoring (BANNED), eval-time GPTQ calibration (BANNED), product quantization (SWA-incompatible), larger vocab (embedding cost), custom Triton kernels (poor EV), int4 without QAT (quality-destructive), eval stride=32 (time budget exceeded).

---

## Technique Reference

| Technique | Approx Δ bpb | Status |
|-----------|-------------|--------|
| **AdamW TTT (30 ep, cosine, legal score-first)** | **-0.04 to -0.06** | **In SOTA (PR #549)** |
| Sliding window eval (stride=64) | -0.032 | In SOTA |
| TrigramHash + ValueResidual + GradQuant | -0.023 | In SOTA (PR #486) |
| 3× MLP expansion | -0.015 | In SOTA |
| **Full Hessian GPTQ** (Cholesky comp.) | **-0.005 to -0.010** | **Target (PR #1060)** |
| Int6 QAT + GradQuant adaptive Int5/6/7 | -0.010 | In SOTA |
| **XSA (all 11 layers)** | **-0.002 to -0.005** | **Target (PR #1060)** |
| SmearGate + BigramHash(4096) | -0.006 | In SOTA |
| Value Residual (ResFormer) | -0.005 to -0.017 | In SOTA |
| LeakyReLU² activation | ~-0.003 | In SOTA (PR #549) |
| Coprime-stride data loader | ~-0.001 | Target (PR #1060) |
| 11 layers | -0.003 | In SOTA |
| EMA (0.997) + SWA (every 50) | -0.002 | In SOTA |
| Partial RoPE (16/64) + LN Scale | -0.002 | In SOTA |
| Orthogonal init + Muon WD=0.04 | -0.003 | In SOTA |
| LoRA TTT | **+0.004 (HURTS)** | **Abandoned** |
| N-gram eval cache | **BANNED** | **Invalidated 2026-03-27** |
| Multi-epoch TTT with re-scoring | **BANNED** | **PR #771 closed for this** |

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

### Session 4 (2026-03-29)
17. **Our PR #771 was closed for TTT rule violation.** Multi-epoch TTT that adapts on eval tokens then re-scores them is BANNED. Legal score-first means: score each token ONCE, then optionally adapt — never re-score. PR #549 is the verified legal reference.
18. **N-gram eval cache is DEAD.** @valerio-oai closed 33+ PRs on 2026-03-27. The hashed n-gram implementations had a normalization bug (only scored correct token, not full vocab). Correct n-gram = ~1.51 BPB (worse than baseline). Do not attempt any n-gram interpolation.
19. **Real frontier is ~1.11 BPB, not sub-1.0.** All sub-1.0 scores in the past week were from the n-gram normalization bug. After invalidation, best legitimate open PR is PR #1060 at 1.1122. Our target is now 1.11 → sub-1.10.
20. **Full Hessian GPTQ beats GPTQ-lite.** PR #1060 uses Cholesky error compensation for full Hessian GPTQ vs our GPTQ-lite. This is the main technique gap to close.

## Golden Rules

Every change must answer: "Does this lower val_bpb within the 16MB/10-min constraints?" If the answer is unclear, run a quick experiment on 1xH100 before investing more time. Compression and eval tricks are as valuable as architecture changes. The cheapest experiment that gives signal is the best experiment. Speed > perfection — submit early, iterate after.

_Updated: 2026-03-23 (v6.0 — LoRA TTT + In-Place TTT moonshot, GPTQ disabled after Run 1 failure)_
