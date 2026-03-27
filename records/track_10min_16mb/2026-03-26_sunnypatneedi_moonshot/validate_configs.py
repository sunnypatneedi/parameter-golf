#!/usr/bin/env python3
"""
validate_configs.py — CPU-only config validator. No GPU, no training data needed.

Estimates parameter counts and artifact sizes for each config in MOONSHOT_SPACE
using analytical formulas derived from the adaptive quantization scheme in
train_gpt_v10_moonshot.py.

Quantization scheme (adaptive_quantize):
  - blocks.*.mlp.*  (large 2D, >65536 params): ternary int8  → ~0.25 bytes/param post-zstd-22
  - blocks.*.attn.* (large 2D, >65536 params): int4 int8     → ~0.50 bytes/param post-zstd-22
  - tok_emb / bigram.embed / ve_shared.embed / other large:
                                                int6 int8     → ~0.75 bytes/param post-zstd-22
  - small tensors (<=65536 params, non-control): float16     → ~1.80 bytes/param
  - control tensors (attn_scale, mlp_scale, resid_mix, q_gain,
                     skip_weights, smear, ve_layer_scales, ve_shared.scale):
                                                float32       → ~3.20 bytes/param

Compression ratios calibrated from:
  - PR #486 baseline: ~30M params, int6 uniform, 15.34MB artifact → ~0.51 bytes/param
  - Ternary (3 distinct values) compresses ~5-7x better than random int8 under zstd-22
  - Int4 (15 values) compresses ~2x better than int6

Architecture fixed params (matching train_gpt_v10_moonshot.py defaults):
  vocab_size=1024, num_layers=11, num_heads=8, num_kv_heads=4
  bigram_vocab_size=6144, bigram_dim=128, ve_dim=128, ve_layers="7,8,9,10" (4 layers)
  gated_attention=True, INT8_KEEP_FLOAT_MAX_NUMEL=65536

Usage:
    python3 validate_configs.py
    python3 validate_configs.py --n_configs 20 --seed_offset 42
"""
import argparse
import json
import random
from pathlib import Path

# ── Architecture constants (must match train_gpt_v10_moonshot.py defaults) ────
NUM_LAYERS      = 11
VOCAB_SIZE      = 1024
NUM_HEADS       = 8
NUM_KV_HEADS    = 4      # kv_dim = num_kv_heads * head_dim = D/2
BIGRAM_VOCAB    = 6144
BIGRAM_DIM      = 128
VE_DIM          = 128
VE_LAYERS       = 4      # "7,8,9,10"
GATED_ATTN      = True
SMALL_THRESHOLD = 65_536

# Code file sizes in bytes (wc -c)
CODE_BYTES_MOONSHOT = 82_935
CODE_BYTES_SAFE     = 82_956

# Bytes-per-param after zstd level 22 (empirically calibrated)
BPP_TERNARY  = 0.25   # MLP weights:  3-value int8, excellent zstd
BPP_INT4     = 0.50   # Attn weights: 15-value int8, good zstd
BPP_INT6     = 0.75   # Embed/other:  63-value int8, moderate zstd
BPP_FLOAT16  = 1.80   # Small float16 passthrough (weakly compressible)
BPP_FLOAT32  = 3.20   # Control tensors float32 (small, weakly compressible)

# Fixed overhead: torch.save pickle headers + dict metadata + zstd frame
OVERHEAD_BYTES = 35_000

LIMIT = 16_000_000

# ── Search spaces (must match auto_experiment.py) ─────────────────────────────
SAFE_SPACE = {
    "hedge_eta":          [0.05, 0.1, 0.2, 0.3],
    "ngram_delta":        [0.001, 0.01, 0.05, 0.1],
    "ngram_alpha_center": [0.5, 0.6, 0.7, 0.8],
    "ngram_max_order":    [8, 10, 11],
    "swa_every":          [30, 50, 80],
    "ema_decay":          [0.995, 0.997, 0.999],
}

MOONSHOT_SPACE = {
    **SAFE_SPACE,
    "model_dim": [576, 608, 640],
    "mlp_mult":  [2.5, 3.0, 3.5],
}


def sample_config(space: dict, seed: int) -> dict:
    rng = random.Random(seed)
    return {k: rng.choice(v) for k, v in space.items()}


def estimate_artifact_size(model_dim: int, mlp_mult: float, script: str = "moonshot") -> dict:
    """
    Analytically estimate artifact size for a given (model_dim, mlp_mult) config.
    All parameter counts are exact; compression ratios are empirical estimates.
    """
    D = model_dim
    M = mlp_mult
    head_dim = D // NUM_HEADS          # e.g. 640/8 = 80
    kv_dim   = NUM_KV_HEADS * head_dim  # e.g. 4*80 = 320 = D/2

    # ── Large MLP matrices → ternary ─────────────────────────────────────────
    # Per block: fc (D × M*D) + proj (M*D × D)  →  2*M*D² params
    mlp_params_per_block = 2 * int(M * D * D)
    mlp_params_total     = mlp_params_per_block * NUM_LAYERS

    # Per-row float16 scales: fc has D rows, proj has M*D rows → D*(1+M) per block
    mlp_scale_params = int(D * (1 + M)) * NUM_LAYERS

    # ── Large Attention matrices → int4 ──────────────────────────────────────
    # c_q: D×D,  c_k: D×kv_dim,  c_v: D×kv_dim,  proj: D×D
    attn_shapes = [D*D, D*kv_dim, D*kv_dim, D*D]
    attn_int4_params   = sum(n for n in attn_shapes if n > SMALL_THRESHOLD) * NUM_LAYERS
    attn_float16_small = sum(n for n in attn_shapes if n <= SMALL_THRESHOLD) * NUM_LAYERS

    # Per-row scales: c_q/c_k/c_v/proj each have D rows → 4*D per block
    attn_scale_params = 4 * D * NUM_LAYERS

    # attn_gate (gated_attention): D×8 weight + 8 bias → always ≤65536 → float16
    if GATED_ATTN:
        attn_float16_small += (D * 8 + 8) * NUM_LAYERS

    # q_gain: 8 per block → control tensor (float32), not quantized
    q_gain_params = 8 * NUM_LAYERS

    # ── Embeddings / other large → int6 ──────────────────────────────────────
    tok_emb_n    = VOCAB_SIZE * D           # always large (e.g. 1024*640=655360)
    bigram_emb_n = BIGRAM_VOCAB * BIGRAM_DIM  # 6144*128=786432, always large
    ve_emb_n     = VOCAB_SIZE * VE_DIM      # 1024*128=131072, always large

    # bigram.proj: BIGRAM_DIM × D
    bigram_proj_n = BIGRAM_DIM * D
    if bigram_proj_n > SMALL_THRESHOLD:
        bigram_proj_int6 = bigram_proj_n
        bigram_proj_f16  = 0
    else:
        bigram_proj_int6 = 0
        bigram_proj_f16  = bigram_proj_n

    # ve_shared.proj: VE_DIM × kv_dim (e.g. 128*320=40960 for D=640 → always ≤65536)
    ve_proj_n = VE_DIM * kv_dim
    if ve_proj_n > SMALL_THRESHOLD:
        ve_proj_int6 = ve_proj_n
        ve_proj_f16  = 0
    else:
        ve_proj_int6 = 0
        ve_proj_f16  = ve_proj_n

    int6_params_total = tok_emb_n + bigram_emb_n + ve_emb_n + bigram_proj_int6 + ve_proj_int6

    # Per-row scales for int6 tensors
    int6_scale_params = (VOCAB_SIZE            # tok_emb rows
                         + BIGRAM_VOCAB        # bigram.embed rows
                         + VOCAB_SIZE          # ve_shared.embed rows
                         + (BIGRAM_DIM if bigram_proj_int6 else 0)
                         + (VE_DIM    if ve_proj_int6  else 0))

    # ── Small float16 tensors ─────────────────────────────────────────────────
    small_f16_params = (attn_float16_small
                        + bigram_proj_f16
                        + ve_proj_f16
                        + 1)   # bigram.scale (1 param, not a control tensor)

    # ── Control tensors (float32 passthrough) ─────────────────────────────────
    # Per block: attn_scale(D) + mlp_scale(D) + resid_mix(2D) = 4D
    ctrl_block  = 4 * D * NUM_LAYERS
    # Global: skip_weights(5×D) + smear.gate(D) + ve_shared.scale(1) + ve_layer_scales(4) + q_gain(88)
    num_encoder = NUM_LAYERS // 2
    num_skip    = min(num_encoder, NUM_LAYERS - num_encoder)
    ctrl_global = num_skip * D + D + 1 + VE_LAYERS + q_gain_params
    ctrl_params_total = ctrl_block + ctrl_global

    # ── Total parameter count ─────────────────────────────────────────────────
    total_params = (mlp_params_total + mlp_scale_params
                    + attn_int4_params + attn_scale_params
                    + int6_params_total + int6_scale_params
                    + small_f16_params
                    + ctrl_params_total)

    # ── Compressed size estimate ──────────────────────────────────────────────
    sz_ternary   = mlp_params_total   * BPP_TERNARY
    sz_t_scale   = mlp_scale_params   * BPP_FLOAT16
    sz_int4      = attn_int4_params   * BPP_INT4
    sz_a_scale   = attn_scale_params  * BPP_FLOAT16
    sz_int6      = int6_params_total  * BPP_INT6
    sz_i_scale   = int6_scale_params  * BPP_FLOAT16
    sz_f16_small = small_f16_params   * BPP_FLOAT16
    sz_ctrl      = ctrl_params_total  * BPP_FLOAT32

    model_compressed = int(sz_ternary + sz_t_scale + sz_int4 + sz_a_scale
                           + sz_int6 + sz_i_scale + sz_f16_small + sz_ctrl
                           + OVERHEAD_BYTES)

    code_bytes  = CODE_BYTES_MOONSHOT if script == "moonshot" else CODE_BYTES_SAFE
    total_bytes = model_compressed + code_bytes
    headroom    = LIMIT - total_bytes

    return {
        "model_dim":            D,
        "mlp_mult":             M,
        "total_params":         total_params,
        "mlp_params":           mlp_params_total,
        "attn_int4_params":     attn_int4_params,
        "int6_params":          int6_params_total,
        "small_f16_params":     small_f16_params,
        "ctrl_fp32_params":     ctrl_params_total,
        "model_compressed_bytes": model_compressed,
        "code_bytes":           code_bytes,
        "total_bytes":          total_bytes,
        "headroom_bytes":       headroom,
        "fits_16mb":            total_bytes < LIMIT,
        "pct_used":             round(100.0 * total_bytes / LIMIT, 1),
        "breakdown_mb": {
            "ternary_mlp":   round(sz_ternary  / 1e6, 2),
            "int4_attn":     round(sz_int4     / 1e6, 2),
            "int6_embed":    round(sz_int6     / 1e6, 2),
            "scales_f16":    round((sz_t_scale + sz_a_scale + sz_i_scale + sz_f16_small) / 1e6, 2),
            "ctrl_fp32":     round(sz_ctrl     / 1e6, 2),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Validate moonshot configs — no GPU needed")
    parser.add_argument("--n_configs",   type=int, default=20)
    parser.add_argument("--seed_offset", type=int, default=42)
    args = parser.parse_args()

    HERE     = Path(__file__).parent
    log_path = HERE / "experiments.jsonl"

    print("=" * 74)
    print("VALIDATE CONFIGS — artifact size estimates (CPU-only, no training data)")
    print("=" * 74)
    print(f"Compression assumptions: ternary={BPP_TERNARY} B/p  int4={BPP_INT4} B/p  "
          f"int6={BPP_INT6} B/p  (post zstd-22)")
    print(f"Limit: {LIMIT:,} bytes  |  Code overhead: {CODE_BYTES_MOONSHOT:,} bytes")
    print()

    # ── 1. All (D, M) combos in MOONSHOT_SPACE ────────────────────────────────
    print("── Model capacity by (D, mlp_mult) ────────────────────────────────────────")
    print(f"{'D':>5}  {'M':>5}  {'Params':>12}  {'Model+zstd':>12}  {'Total':>12}  "
          f"{'Headroom':>10}  {'%Used':>6}  Fits?")
    print("-" * 74)

    arch_results: dict[tuple, dict] = {}
    for D in [576, 608, 640]:
        for M in [2.5, 3.0, 3.5]:
            r = estimate_artifact_size(D, M, "moonshot")
            arch_results[(D, M)] = r
            hk  = r["headroom_bytes"] / 1024
            tag = "YES" if r["fits_16mb"] else "NO ❌"
            print(f"  {D:>4}  {M:>5.1f}  {r['total_params']:>12,}  "
                  f"{r['model_compressed_bytes']:>12,}  "
                  f"{r['total_bytes']:>12,}  "
                  f"{hk:>+9.0f}K  "
                  f"{r['pct_used']:>5.1f}%  {tag}")
    print()

    # ── 2. Breakdown for D=640 ────────────────────────────────────────────────
    print("── Size breakdown for D=640 configs ───────────────────────────────────────")
    for M in [2.5, 3.0, 3.5]:
        r = arch_results[(640, M)]
        b = r["breakdown_mb"]
        fits = "FITS" if r["fits_16mb"] else "TOO BIG"
        print(f"  D=640 M={M}: ternary={b['ternary_mlp']}MB  int4={b['int4_attn']}MB  "
              f"int6={b['int6_embed']}MB  scales={b['scales_f16']}MB  "
              f"ctrl={b['ctrl_fp32']}MB  → {fits}")
    print()

    # ── 3. All seeded configs ─────────────────────────────────────────────────
    print(f"── All {args.n_configs} seeded moonshot configs (seed_offset={args.seed_offset}) "
          f"──────────────────────")
    print(f"{'Seed':>5}  {'D':>4}  {'M':>5}  {'Total B':>12}  {'%':>5}  Fits?  "
          f"n-gram / hedge / train")
    print("-" * 74)

    all_records = []
    for i in range(args.n_configs):
        seed = args.seed_offset + i
        cfg  = sample_config(MOONSHOT_SPACE, seed)
        D, M = cfg["model_dim"], cfg["mlp_mult"]
        r    = arch_results[(D, M)]
        tag  = "YES" if r["fits_16mb"] else "NO ❌"
        info = (f"n{cfg['ngram_max_order']}_δ{cfg['ngram_delta']}_"
                f"α{cfg['ngram_alpha_center']}  "
                f"η{cfg['hedge_eta']}  "
                f"ema{cfg['ema_decay']}_swa{cfg['swa_every']}")
        print(f"  {seed:>4}  {D:>4}  {M:>5.1f}  {r['total_bytes']:>12,}  "
              f"{r['pct_used']:>4.1f}%  {tag:>5}  {info}")
        all_records.append({
            "seed":   seed,
            "config": cfg,
            "estimate": {
                "total_params":     r["total_params"],
                "total_bytes":      r["total_bytes"],
                "headroom_bytes":   r["headroom_bytes"],
                "fits_16mb":        r["fits_16mb"],
                "pct_used":         r["pct_used"],
                "breakdown_mb":     r["breakdown_mb"],
            },
            "status": "estimated_ok" if r["fits_16mb"] else "estimated_too_large",
        })
    print()

    # ── 4. v10_safe baseline (D=512, int6, no ternary) ───────────────────────
    print("── v10_safe reference (D=512, M=3.0, uniform int6) ───────────────────────")
    r_safe = estimate_artifact_size(512, 3.0, "safe")
    print(f"  total_params={r_safe['total_params']:,}  "
          f"total_bytes={r_safe['total_bytes']:,}  "
          f"headroom={r_safe['headroom_bytes']/1024:+.0f}K  "
          f"{'FITS' if r_safe['fits_16mb'] else 'TOO BIG'}")
    print(f"  (Note: v10_safe uses int6 for all large matrices, BPP={BPP_INT6})")
    print()

    # ── 5. Summary ────────────────────────────────────────────────────────────
    fitting  = [rec for rec in all_records if rec["estimate"]["fits_16mb"]]
    too_big  = [rec for rec in all_records if not rec["estimate"]["fits_16mb"]]

    print("── Summary ────────────────────────────────────────────────────────────────")
    print(f"  Configs that fit (<16MB):  {len(fitting)}/{args.n_configs}")
    print(f"  Configs too large (>16MB): {len(too_big)}/{args.n_configs}")
    if too_big:
        print(f"  Too large seeds: {[r['seed'] for r in too_big]}")

    print()
    print("  Top 5 by param capacity (largest models within budget):")
    for rec in sorted(fitting, key=lambda x: x["estimate"]["total_params"], reverse=True)[:5]:
        cfg = rec["config"]
        e   = rec["estimate"]
        print(f"    seed={rec['seed']}  D={cfg['model_dim']}  M={cfg['mlp_mult']}  "
              f"params={e['total_params']:,}  bytes={e['total_bytes']:,}  ({e['pct_used']}%)")

    print()
    print("  Max D that fits per mlp_mult:")
    for M in [2.5, 3.0, 3.5]:
        for D in [640, 608, 576]:
            r = arch_results.get((D, M))
            if r and r["fits_16mb"]:
                print(f"    M={M}: max D={D}  ({r['pct_used']}% used, "
                      f"{r['headroom_bytes']/1024:.0f}K headroom)")
                break
        else:
            print(f"    M={M}: no D in {{576,608,640}} fits")

    # ── 6. Write experiments.jsonl ────────────────────────────────────────────
    with open(log_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")
    print()
    print(f"Wrote {len(all_records)} records → {log_path}")


if __name__ == "__main__":
    main()
