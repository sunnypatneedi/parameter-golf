"""
validate_configs.py — Artifact size estimator for Parameter Golf 16MB track.

Reference: 27.7M params @ int6 = 15.9MB → compression ratio = 0.80
(accounts for metadata, codebook overhead, real-world packing inefficiency)
"""

import math

VOCAB_SIZE = 50304
MAX_BYTES = 16 * 1024 * 1024  # 16MB

# Bits per weight for each quant scheme
QUANT_BITS = {
    "int6":    6.0,
    "int4":    4.0,
    "mixed":   3.1,   # ternary MLP + int4 attn + int6 embed, avg ~3.1b
    "ternary": 2.0,   # ~log2(3) = 1.585, practical overhead → ~2.0b
    "bitnet":  1.58,  # BitNet 1.58b
}

# Empirical compression ratio calibrated from SOTA (27.7M params = 15.9MB int6)
# raw_bytes = params * bits / 8  →  actual_bytes = raw_bytes * ratio
SOTA_PARAMS   = 27_700_000
SOTA_MB       = 15.9
SOTA_BITS     = QUANT_BITS["int6"]
COMPRESSION   = SOTA_MB * 1024 * 1024 / (SOTA_PARAMS * SOTA_BITS / 8)
# COMPRESSION ≈ 0.80


def count_params(model_dim: int, n_layers: int, mlp_mult: float) -> dict:
    """Break down parameter count by component."""
    d = model_dim
    mlp_hidden = int(d * mlp_mult)

    # With tied embeddings embed == out_proj, so we count once
    embed_tied = VOCAB_SIZE * d          # single shared matrix

    # Attention: Q, K, V, O projections (no bias, standard)
    attn = n_layers * 4 * d * d

    # MLP: two linear layers
    mlp = n_layers * 2 * d * mlp_hidden

    # LayerNorm / RMSNorm — negligible but included
    norms = n_layers * 2 * d + d        # pre-attn, pre-mlp per layer + final

    total_tied   = embed_tied + attn + mlp + norms
    total_untied = VOCAB_SIZE * d + VOCAB_SIZE * d + attn + mlp + norms

    return {
        "embed":        embed_tied,
        "attn":         attn,
        "mlp":          mlp,
        "norms":        norms,
        "total_tied":   total_tied,
        "total_untied": total_untied,
    }


def estimate_size_mb(params: int, quant: str) -> float:
    bits = QUANT_BITS[quant]
    raw_bytes = params * bits / 8
    actual_bytes = raw_bytes * COMPRESSION
    return actual_bytes / (1024 * 1024)


def fits(mb: float) -> str:
    return "FITS" if mb <= 16.0 else f"NO ({mb:.2f}MB)"


def main():
    configs = [
        # label,                       dim,  layers, mlp_mult, quant,     tied
        ("SOTA baseline",              512,  11,     4.0,      "int6",    True),
        ("Safe + hedge mixer",         512,  11,     4.0,      "int6",    True),
        ("Mixed (640d 11L)",           640,  11,     4.0,      "mixed",   True),
        ("Ternary (640d 11L)",         640,  11,     4.0,      "ternary", True),
        ("BitNet (768d 11L)",          768,  11,     4.0,      "bitnet",  True),
        # Sweep: dim
        ("int6  512d  9L",             512,   9,     4.0,      "int6",    True),
        ("int6  512d 11L",             512,  11,     4.0,      "int6",    True),
        ("int6  512d 13L",             512,  13,     4.0,      "int6",    True),
        ("int6  640d 11L",             640,  11,     4.0,      "int6",    True),
        ("int6  640d 13L",             640,  13,     4.0,      "int6",    True),
        ("int6  768d 11L",             768,  11,     4.0,      "int6",    True),
        ("int4  640d 11L",             640,  11,     4.0,      "int4",    True),
        ("int4  768d 11L",             768,  11,     4.0,      "int4",    True),
        ("int4  768d 13L",             768,  13,     4.0,      "int4",    True),
        ("int4 1024d 11L",            1024,  11,     4.0,      "int4",    True),
        ("mixed 768d 11L",             768,  11,     4.0,      "mixed",   True),
        ("mixed 768d 13L",             768,  13,     4.0,      "mixed",   True),
        ("mixed 1024d 11L",           1024,  11,     4.0,      "mixed",   True),
        ("ternary 768d 11L",           768,  11,     4.0,      "ternary", True),
        ("ternary 768d 13L",           768,  13,     4.0,      "ternary", True),
        ("ternary 1024d 11L",         1024,  11,     4.0,      "ternary", True),
        ("ternary 1024d 13L",         1024,  13,     4.0,      "ternary", True),
        ("bitnet  768d 11L",           768,  11,     4.0,      "bitnet",  True),
        ("bitnet  768d 13L",           768,  13,     4.0,      "bitnet",  True),
        ("bitnet 1024d 11L",          1024,  11,     4.0,      "bitnet",  True),
        ("bitnet 1024d 13L",          1024,  13,     4.0,      "bitnet",  True),
        ("bitnet 1024d 15L",          1024,  15,     4.0,      "bitnet",  True),
        # MLP multiplier sweep
        ("int6  512d 11L mlp3x",       512,  11,     3.0,      "int6",    True),
        ("ternary 768d 11L mlp3x",     768,  11,     3.0,      "ternary", True),
        ("bitnet 1024d 11L mlp3x",    1024,  11,     3.0,      "bitnet",  True),
        # Untied embedding variants (for reference)
        ("int6  512d 11L untied",      512,  11,     4.0,      "int6",    False),
        ("bitnet 1024d 11L untied",   1024,  11,     4.0,      "bitnet",  False),
    ]

    results = []
    for label, dim, layers, mlp_mult, quant, tied in configs:
        p = count_params(dim, layers, mlp_mult)
        total = p["total_tied"] if tied else p["total_untied"]
        mb    = estimate_size_mb(total, quant)
        results.append((mb, total, label, dim, layers, mlp_mult, quant, tied, p))

    results.sort(key=lambda x: x[1])  # sort by param count

    header = (f"{'Label':<35} {'dim':>5} {'L':>3} {'mlp':>4} {'quant':>8} "
              f"{'tied':>5} {'params':>10} {'size_MB':>8}  status")
    sep = "=" * (len(header) + 4)
    print(sep)
    print("  Parameter Golf 16MB Track — Config Validator")
    print(f"  Compression ratio: {COMPRESSION:.4f}  "
          f"(calibrated from SOTA 27.7M@int6=15.9MB)")
    print(sep)
    print(header)
    print("-" * (len(header) + 4))

    recommended = []
    for mb, total, label, dim, layers, mlp_mult, quant, tied, p in results:
        status = fits(mb)
        tie_str = "yes" if tied else "no"
        flag = ""
        if mb <= 16.0 and total > 20_000_000:
            flag = "  ★ RECOMMENDED"
            recommended.append((label, dim, layers, mlp_mult, quant, total, mb))
        print(f"  {label:<35} {dim:>5} {layers:>3} {mlp_mult:>4.1f} {quant:>8} "
              f"{tie_str:>5} {total:>10,} {mb:>8.2f}MB  {status}{flag}")

    print("-" * (len(header) + 4))
    print()
    print("★ RECOMMENDED FOR MORNING H100 RUN:")
    print()
    for label, dim, layers, mlp_mult, quant, total, mb in recommended:
        headroom = 16.0 - mb
        print(f"  • {label}")
        print(f"    dim={dim}, layers={layers}, mlp_mult={mlp_mult}x, quant={quant}")
        print(f"    params={total:,}  size={mb:.2f}MB  headroom={headroom:.2f}MB")
        print()

    print("Notes:")
    print(f"  vocab_size = {VOCAB_SIZE:,} (GPT-NeoX BPE)")
    print(f"  All configs assume tied input/output embeddings unless marked 'untied'")
    print(f"  'mixed'   = ternary MLP + int4 attn + int6 embed, effective ~3.1b/weight")
    print(f"  'ternary' = ~2.0b/weight (includes practical packing overhead)")
    print(f"  'bitnet'  = 1.58b/weight (BitNet b1.58 ternary {{-1,0,+1}})")
    print(f"  Compression ratio {COMPRESSION:.4f} accounts for codebook/metadata overhead")


if __name__ == "__main__":
    main()
