## Record: 11L Eval-Only XSA + TrigramHash + ValueResidual + GradQuant + AdamW TTT (Full Stack v5.1)

**Target: <1.0829 val_bpb** | 8xH100 SXM, 600s train + 600s eval

### Key Addition Over PR #486 (SOTA 1.0887)

Built on PR #486's base (TrigramHash + ValueResidual + GradQuant + Cosine TTT 30ep), this submission adds:

**XSA (Exclusive Self-Attention) at eval time only on all 11 layers.** Removes self-position contribution from attention output, forcing context-only prediction. Applied ONLY during eval/TTT — training proceeds identically to PR #486.

XSA implementation: projects attention output onto normalized value subspace and subtracts, GQA-aware. No additional parameters. The model trains normally, then XSA improves eval predictions by forcing context-dependent outputs.

### Stress Test Findings (pre-run)

1. **SWA_EVERY is dead code** when EMA is enabled (line 1681 guard: `not args.ema_enabled`). Reverted to match PR #486 defaults. EMA (decay=0.997) is the active weight averaging method.
2. **XSA moved to eval-only** to avoid training regression risk. Training model uses xsa_last_n=0, eval model uses xsa_last_n=11.
3. **Flash Attention + GQA + XSA**: Verified compatible. XSA's _xsa_efficient handles GQA grouping correctly.
4. **ValueResidual + XSA interaction**: v0 blending happens before attention; XSA subtracts post-attention. No conflict.
5. **Artifact size**: Unchanged from PR #486 (~15.34MB). XSA adds zero parameters.
6. **Time budget**: XSA adds <1s to eval (~0.1ms × 11 layers × ~500 batches). Safe within 600s.

### Changes from PR #486 (actual diff, 3 lines)

```diff
- num_layers = int(os.environ.get("NUM_LAYERS", 9))
+ num_layers = int(os.environ.get("NUM_LAYERS", 11))

- xsa_last_n = int(os.environ.get("XSA_LAST_N", 0))
+ xsa_last_n = int(os.environ.get("XSA_LAST_N", 11))

  # Training model: xsa_last_n=0 (eval-only XSA)
- xsa_last_n=args.xsa_last_n,
+ xsa_last_n=0,  # Train WITHOUT XSA
```

### Full Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- 3x MLP relu-squared + SmearGate + BigramHash(4096) + TrigramHash(4096)
- Value Residual (ResFormer) across all layers
- XSA on all 11 layers (eval only)
- EMA (decay=0.997), OrthoInit
- Partial RoPE (16/64 dims), LN Scale (1/sqrt(layer+1))
- GradQuant: adaptive Int5/6/7 + zstd-22

### TTT Configuration

- AdamW (lr=0.0005, weight_decay=0.0), 30 epochs, cosine decay
- Per-layer LR: MLP output 3x, MLP input 0.5x, rest 1x
- All blocks unfrozen (freeze_blocks=0)
- Time: ~465s of 600s eval budget

### Hypothesis

XSA at eval time forces the model to rely on context tokens rather than self-position when making predictions. This should improve perplexity without any training cost, because:
- The model already learned to use context during training (via attention to other positions)
- Removing the self-position "shortcut" during eval forces higher-quality contextual predictions
- Complementary to TTT (which adapts weights to the specific validation context)

Expected improvement: -0.002 to -0.005 bpb from eval-only XSA.

### Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are set as defaults in train_gpt.py.

### Results

_Pending — needs 8xH100 validation runs._

### Ablation Plan

1. PR #486 base (XSA_LAST_N=0): reproduces 1.0887
2. + Eval-only XSA (XSA_LAST_N=11, training xsa=0): our submission
3. + Training XSA (both train and eval xsa=11): compare
