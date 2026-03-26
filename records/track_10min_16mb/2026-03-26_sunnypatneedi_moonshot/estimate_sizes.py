#!/usr/bin/env python3
"""Estimate artifact sizes for different model configs.

Calibration: SOTA is 512d/11L/int6, actual artifact ~15.34MB.
  Raw params (with tied lm_head excluded): ~51.7M
  Raw bytes at int6: 51.7M * 6/8 = 38.8MB
  Actual artifact: 15.34MB → compression ratio ≈ 0.39
  Using compression_ratio=0.39 and NO lm_head double-count (weight-tied).
"""

VOCAB = 50304
CODE_BYTES = 50_000         # ~50KB for train_gpt.py code portion
COMP_RATIO = 0.39           # calibrated: zstd on int-quantized weights
LIMIT_MB = 15.5             # conservative limit (16MB hard limit - margin)

def estimate_params(model_dim, n_layers, n_head, n_kv_head, mlp_mult):
    head_dim = model_dim // n_head
    # Embeddings (wte only — lm_head is weight-tied)
    embed = VOCAB * model_dim
    per_layer = 0
    per_layer += model_dim * (n_head * head_dim)        # q_proj
    per_layer += model_dim * (n_kv_head * head_dim)     # k_proj
    per_layer += model_dim * (n_kv_head * head_dim)     # v_proj
    per_layer += (n_head * head_dim) * model_dim        # o_proj
    per_layer += model_dim * int(model_dim * mlp_mult)  # mlp.fc
    per_layer += int(model_dim * mlp_mult) * model_dim  # mlp.proj
    per_layer += model_dim * 2                          # rmsnorm
    return embed + per_layer * n_layers

def estimate_artifact_bytes(total_params, avg_bits):
    weight_bytes = (total_params * avg_bits) / 8
    return weight_bytes * COMP_RATIO + CODE_BYTES

# Verify calibration
sota_params = estimate_params(512, 11, 8, 4, 3.0)
sota_art = estimate_artifact_bytes(sota_params, 6)
print(f"[CALIBRATION] SOTA params={sota_params/1e6:.1f}M  estimated artifact={sota_art/1e6:.2f}MB  (actual ~15.34MB)")
print()

configs = [
    # (name, model_dim, n_layers, n_head, n_kv_head, mlp_mult, avg_bits)
    # Baseline / current SOTA
    ("SOTA (512d, 11L, int6)",               512,  11,  8,  4, 3.0, 6.0),
    # v10-moon: ternary MLP + int4 attn + int6 embed (~3.2 avg)
    ("v10-moon (640d, 11L, ~3.2b)",          640,  11, 10,  5, 3.0, 3.2),
    ("v10-moon (640d, 13L, ~3.2b)",          640,  13, 10,  5, 3.0, 3.2),
    ("v10-moon (768d, 11L, ~3.2b)",          768,  11, 12,  6, 3.0, 3.2),
    ("v10-moon (768d, 11L, ~2.5b)",          768,  11, 12,  6, 3.0, 2.5),
    # Ternary-heavy (~2.0 avg bits)
    ("ternary (640d, 11L, ~2.0b)",           640,  11, 10,  5, 3.0, 2.0),
    ("ternary (768d, 11L, ~2.0b)",           768,  11, 12,  6, 3.0, 2.0),
    ("ternary (768d, 13L, ~2.0b)",           768,  13, 12,  6, 3.0, 2.0),
    ("ternary (896d, 11L, ~2.0b)",           896,  11, 14,  7, 3.0, 2.0),
    ("ternary (1024d, 11L, ~2.0b)",         1024,  11, 16,  8, 3.0, 2.0),
    ("ternary (1024d, 13L, ~2.0b)",         1024,  13, 16,  8, 3.0, 2.0),
    ("ternary (1280d, 11L, ~2.0b)",         1280,  11, 20, 10, 3.0, 2.0),
    # BitNet extreme (1.58 bits)
    ("bitnet (768d,  11L, 1.58b)",           768,  11, 12,  6, 3.0, 1.58),
    ("bitnet (1024d, 11L, 1.58b)",          1024,  11, 16,  8, 3.0, 1.58),
    ("bitnet (1024d, 13L, 1.58b)",          1024,  13, 16,  8, 3.0, 1.58),
    ("bitnet (1280d, 11L, 1.58b)",          1280,  11, 20, 10, 3.0, 1.58),
    ("bitnet (1536d, 11L, 1.58b)",          1536,  11, 24, 12, 3.0, 1.58),
    ("bitnet (1536d, 13L, 1.58b)",          1536,  13, 24, 12, 3.0, 1.58),
    ("bitnet (1792d, 11L, 1.58b)",          1792,  11, 28, 14, 3.0, 1.58),
    ("bitnet (2048d, 11L, 1.58b)",          2048,  11, 32, 16, 3.0, 1.58),
    # Int4 uniform
    ("int4 (640d, 11L, 4b)",                 640,  11, 10,  5, 3.0, 4.0),
    ("int4 (768d, 11L, 4b)",                 768,  11, 12,  6, 3.0, 4.0),
    ("int4 (640d, 13L, 4b)",                 640,  13, 10,  5, 3.0, 4.0),
    # Int5
    ("int5 (576d, 11L, 5b)",                 576,  11,  9,  4, 3.0, 5.0),
    ("int5 (640d, 11L, 5b)",                 640,  11, 10,  5, 3.0, 5.0),
]

print(f"{'Config':<42} {'Params_M':>8} {'Raw_MB':>8} {'Est_MB':>8} {'Fits':>5} {'Headroom':>10}")
print("-" * 88)

results = []
for name, dim, layers, nh, nkv, mlp_m, bits in configs:
    params = estimate_params(dim, layers, nh, nkv, mlp_m)
    artifact = estimate_artifact_bytes(params, bits)
    raw_mb = (params * bits / 8) / 1e6
    est_mb = artifact / 1e6
    fits = est_mb < LIMIT_MB
    headroom = LIMIT_MB - est_mb
    results.append((name, params, est_mb, fits, headroom))
    marker = "<<<" if fits else ""
    print(f"{name:<42} {params/1e6:>7.1f}M {raw_mb:>7.2f} {est_mb:>7.2f} {'YES' if fits else 'NO':>5} {headroom:>9.2f}MB  {marker}")

print("\n\n=== CONFIGS THAT FIT < 15.5MB (sorted by param count desc) ===\n")
fitting = [(n, p, e, h) for n, p, e, f, h in results if f]
fitting.sort(key=lambda x: -x[1])
if fitting:
    for name, params, est_mb, headroom in fitting:
        print(f"  {name:<42} {params/1e6:>7.1f}M params  {est_mb:>6.2f}MB  headroom: {headroom:>5.2f}MB")
else:
    print("  (none — all configs exceed 15.5MB)")

print("\n\n=== RECOMMENDED FOR MORNING H100 RUN ===")
print("1st: v10-safe (512d, int6) — baseline, verify n-gram + hedge mixer")
print("2nd: Best fitting ternary/mixed config from above")
print("3rd: Largest BitNet config that fits (if any)")
