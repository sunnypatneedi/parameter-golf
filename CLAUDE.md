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

**Merged leaderboard SOTA**: 1.1147 val_bpb (abaybektursun, PR #1019, 2026-03-25)
**Best open legal PR**: 1.0914 val_bpb (PR #1176, QK-Gain 4.0 + Muon-TTT 3ep + SLOT)
**Best open any PR**: 0.3958 (PR #1094, BackoffNgramMixer — legality disputed)
**Target**: Beat 1.1147 merged SOTA by >=0.005 nats.

**CRITICAL LEGALITY UPDATES (2026-03-27)**:
- **PR #771 REJECTED** — Our AdamW TTT 30ep was train-then-score, not score-first. All 30-epoch TTT results are void.
- **N-gram cache ILLEGAL** — Hashed n-gram caches ruled out (Issue #1017): produce unnormalized distributions. PRs #727, #741 closed. PRs #731, #758 still open pending ruling.
- **Score-first TTT ≤3 epochs IS LEGAL** — PR #1176 uses 3-epoch Muon-TTT with score-first ordering.

**Current approach (architecture path — no TTT legality risk)**:
1. **4096 vocab** — larger tokenizer, +embedding params freed from compression (from PR #1218)
2. **4× MLP expansion** — vs 3× in older SOTA (from PR #1218)
3. **XSA on all 11 layers** — exclusive self-attention (from PRs #503, #1019)
4. **GPTQ + WD=0.085** — Hessian-aware quantization (from PR #1218)
5. **QK-Gain 4.0** — hyperparameter, -0.006 bpb, 45-experiment validated (from PR #1176)
6. **Score-first Muon-TTT 3ep** — legal light TTT (from PR #1176)
7. **SLOT δ-vector** — optimize additive δ at final hidden layer, -0.021 bpb (arXiv:2505.12392) — **VERIFY LEGALITY BEFORE GPU SPEND**

**Key reference PRs**: #1019 (merged SOTA 1.1147), #1176 (1.0914, score-first TTT + SLOT), #1218 (1.09785, no TTT clean arch)

**Abandoned approaches**: LoRA TTT (hurts), product quantization (SWA-incompatible), custom Triton kernels (poor EV), int4 without QAT (quality-destructive), eval stride=32 (time budget), AdamW TTT 30ep (illegal train-then-score), n-gram hash cache (illegal normalization).

---

## Technique Reference

| Technique | Approx Δ bpb | Status |
|-----------|-------------|--------|
| **SLOT δ-vector (arXiv:2505.12392)** | **-0.021** | **Target — verify legality** |
| **QK-Gain 4.0** | **-0.006** | **Target (PR #1176)** |
| **Score-first Muon-TTT 3ep** | **-0.003** | **Legal (PR #1176)** |
| 4096 vocab | ~-0.02 | Target (PR #1218) |
| 4× MLP expansion | ~-0.01 | Target (PR #1218, vs 3×) |
| Sliding window eval (stride=64) | -0.032 | In SOTA |
| AR Self-Gen GPTQ calibration | ~-0.005 | In merged SOTA (PR #1019) |
| XSA (all 11 layers) | -0.002 to -0.005 | In merged SOTA |
| 3× MLP expansion | -0.015 | In older SOTA |
| Int6 QAT | -0.010 | In SOTA |
| SmearGate + BigramHash(4096) | -0.006 | In older SOTA |
| Value Residual (ResFormer) | -0.005 to -0.017 | In older SOTA |
| 11 layers | -0.003 | In SOTA |
| EMA (0.997) + SWA (every 50) | -0.002 | In SOTA |
| Partial RoPE (16/64) + LN Scale | -0.002 | In SOTA |
| Orthogonal init + Muon WD=0.04 | -0.003 | In SOTA |
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
12. **Depth recurrence is falsified.** PR #540 got 1.2092 bpb (worse than 1.2244 baseline). Do not attempt.

### Session 3 (2026-03-24)
13. **In-Place TTT is HARMFUL.** Loss INCREASES (2.63+, going up not down). MLP output projections are NOT good TTT targets at this scale. Do not attempt.
14. **GradQuant int5/int6 mix exceeds 16MB.** Even without int7, the artifact was 34KB over. Use uniform int6 or match PR #414's exact quantization scheme.
15. **PR #486 baseline reproduced at 1.1249** (vs reported 1.1233). Within seed variance. This is our verified baseline.
16. **The v7.0 incremental plan works.** Run 0→1→2→3 from PR #414 base. Each run adds ONE thing. Stop doing moonshots with 500+ new lines.

### Session 4 (2026-04-01)
17. **Score-first is non-negotiable for TTT.** PR #771 was rejected because we ran 30 epochs of TTT on all val tokens, THEN evaluated on those same tokens. The rule: score the token first, update on scored token, move on. Maximum ~3 epochs to stay within time budget. "Train-then-score" ordering = instant rejection.
18. **N-gram hash cache is illegal without proper normalization.** Issue #1017 (2026-03-27) ruled all hashed n-gram caches out of the record track — they don't produce normalized probability distributions. PRs #727, #741 closed. The "CONFIRMED LEGAL" status from earlier sessions is void. Any n-gram implementation must renormalize properly on every backoff step.
19. **Verify legality of novel eval techniques before GPU spend.** SLOT (δ-vector on final hidden layer) shows -0.021 bpb but causality concerns raised. Check Issue #140 or PR discussion for @valerio-oai ruling before investing $33+ on a GPU run.
20. **4096 vocab + 4×MLP achieves sub-1.1 WITHOUT TTT.** PR #1218 (clarkkev) gets 1.09785 via architecture alone — 4096 vocab, 4×MLP, XSA-all, GPTQ, WD=0.085. No TTT needed. This is a low-risk, high-EV path.

## Golden Rules

Every change must answer: "Does this lower val_bpb within the 16MB/10-min constraints?" If the answer is unclear, run a quick experiment on 1xH100 before investing more time. Compression and eval tricks are as valuable as architecture changes. The cheapest experiment that gives signal is the best experiment. Speed > perfection — submit early, iterate after.

**NEW (2026-04-01)**: Before any eval-time technique, answer: "Does the model see the ground truth label before scoring the token?" If yes, it's illegal. If no, verify with @valerio-oai before spending GPU time.

_Updated: 2026-04-01 (v7.0 — PR #771 rejected, n-gram illegal, pivoting to arch path + legal TTT)_
