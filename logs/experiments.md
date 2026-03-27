# Experiment Log — Parameter Golf

| Date | Exp ID | Change | val_bpb (slide) | Artifact bytes | Steps | ms/step | Hypothesis → Verdict |
|------|--------|--------|----------------|----------------|-------|---------|---------------------|
| 2026-03-22 | mlx_smoke | Baseline MLX 200 iters | 2.4081 | 13,169,730 | 200 | ~155 | Crash test → Works |
| 2026-03-22 | ex1_baseline | Baseline MLX 500 iters | 2.1850 | — | 500 | ~155 | Baseline reference → Recorded |
| 2026-03-22 | ex1_fewer_layers | 5 layers MLX 500 iters | 2.1845 | — | 500 | ~100 | Fewer layers worse? → No difference at 500 iters |
| 2026-03-22 | ex1_wider_mlp | 3x MLP MLX 500 iters | 2.1868 | — | 500 | ~155 | Wider MLP better? → Slightly worse (undertrained) |
| 2026-03-22 | baseline_1gpu | Baseline 1xH100 10min | 1.3412 | 13,169,730 | ~1,235 | ~485 | 1xH100 reference → 1.34 (fewer steps than 8xH100) |
| 2026-03-22 | sota_1gpu | SOTA on 1xH100 10min | 1.5223 | 15,668,352 | ~670 | ~900 | SOTA on 1xH100 → WORSE than baseline (too few steps, eval >10min). 1xH100 cannot run SOTA. |
| 2026-03-22 | sota_8gpu_verify | SOTA on 8xH100 (Phase 0) | 1.1463 | — | ~6,700 | ~89 | Reproduce SOTA → 1.1463 (within 0.004 of reference 1.1428). Phase 0 PASSED. |
| 2026-03-22 | stride32_8gpu | SOTA + EVAL_STRIDE=32, 8xH100 | **1.1430** | — | ~6,587 | ~91 | Stride=32 better than 64? → **YES, -0.0033 bpb, -0.0054 nats.** Eval 341s (within budget). |
| 2026-03-23 | pr486_baseline | PR #486 UNMODIFIED, 8xH100, seed 42 | **1.1249** | 13,327,625 | 5,148 | ~116 | Establish unmodified baseline → **1.1249 bpb. Pre-TTT: 1.1911. TTT gain: -0.066. Artifact 13.3MB (2.67MB headroom). Eval 524s total (within 600s).** |
| 2026-03-23 | v6_fixed | v6.0 full stack (GPTQ off, TTT 5ep, In-Place fixed), 8xH100 | **1.3174** | 16,034,682 | 5,075 | ~118 | v6.0 better than baseline? → **NO. +0.193 bpb WORSE. In-Place TTT DESTROYED model (avg_loss increasing). Artifact 34KB over 16MB. Eval >1000s (over budget). ABANDON In-Place TTT.** |
| 2026-03-25 | v8_seed42 | v8.0 AdamW TTT on PR #549, seed 42, 8xH100 | **1.0706** | 17,121,847 | ~6000 | ~86 | AdamW TTT on SOTA? → **YES, -0.075 bpb from TTT. But artifact used zlib (17.1MB, OVER). Seed 42 only.** |
| 2026-03-25 | v8_seed1337_v2 | v8.0 retry after cache clear, seed 1337, 8xH100 | **1.0699** | 15,757,968 | ~6000 | ~86 | Retry with zstd + cleared cache → **PASS. 15.76MB artifact, no crash, 1.0699 bpb.** |
| 2026-03-25 | v8_seed2024_v2 | v8.0 retry, seed 2024, 8xH100 | **1.0702** | 16,105,594 | ~6000 | ~86 | Seed 2024 → **1.0702 bpb but artifact 16.11MB (+106KB over). Needs fix.** |
