# v10 Moonshot Plan — 2026-03-26

## Objective
Beat our current best (1.0705 BPB, submitted 2026-03-25 as PR #771) by at least 0.005 nats.
Target: ≤1.065 BPB. Stretch: ≤1.060 BPB.

## Base
`train_gpt_v9a_11gram_no_ttt.py` — 11L × 512 dim, 11-gram entropy-adaptive, no TTT, score-first n-gram.

## v10 Additions

### 1. GradQuant (Priority: HIGH, Expected Δ: -0.001 to -0.003 BPB)
**What:** Gradient-guided adaptive per-layer bit allocation replacing uniform Int6.
- Compute gradient norms on the LAST training batch (training data only — no legality concern)
- Sort layers by gradient sensitivity (low → fewer bits, high → more bits)
- Int5 (clip_range=15) for 35% least-sensitive layers: better zstd compression
- Int7 (clip_range=63) for 15% most-sensitive layers: better post-quant quality
- Int6 (clip_range=31) for remaining 50%: unchanged from v9a

**Why it helps:** Current uniform Int6 wastes bits on insensitive layers and under-serves critical ones. GradQuant trades compression space from "easy" layers to boost "hard" layers. Expected effect: slightly smaller artifact (more compression headroom) AND slightly better post-quant quality.

**Risk:** Low. Uses same quantize_int6_per_row / dequantize_mixed_int6 infrastructure. Only the per-layer clip_range changes. If GradQuant hurts (negative result), set `GRADQUANT_ENABLED=0` to fall back to uniform Int6.

**Legality:** LEGAL. Uses training data only for gradient measurement. No val data accessed.

### 2. Hedge Mixer (Priority: HIGH, Expected Δ: -0.001 to -0.004 BPB)
**What:** Online multiplicative-weights expert ensemble between neural and n-gram-enhanced predictions.
- Expert 1: pure neural model probability (before n-gram mixing)
- Expert 2: n-gram-enhanced probability (v9a's entropy-adaptive mixing)
- Hedge weights adapt during eval: if n-gram consistently helps → weight 2 grows; if neural is better → weight 1 grows
- Per-segment batch update with hedge_beta=2.0 learning rate
- Score-first protocol fully maintained (weights updated AFTER scoring)

**Why it helps:** The current entropy-adaptive alpha in v9a uses a fixed formula (sigmoid of model entropy). The hedge mixer LEARNS the optimal weighting online across the eval run. Different documents have different repetition levels — hedge adapts to this automatically.

**Risk:** Low. The hedge starts with uniform weights (w=0.5 each), so at worst it matches v9a. The multiplicative update is numerically stable (we clip log-weights to [-20, 0]).

**Legality:** LEGAL. Score-first protocol preserved. Uses only already-evaluated tokens for weight updates.

### 3. No TTT (by design in v10a)
**Rationale:** TTT takes ~200-300s of the 600s eval budget. v9a's n-gram + hedge should recoup most of TTT's benefit within the eval budget. If hedge + n-gram together give -0.005 BPB, that matches TTT's contribution at this base quality level (per Lesson #6: TTT gains diminish on stronger bases).

**If we need TTT:** Set `TTT_ENABLED=1 TTT_EPOCHS=20 TTT_LR=0.0005` — the TTT code is already included in v10 (inherited from v9a).

## Experiment Plan

### Run 1: v10 baseline (3 seeds)
```bash
SEED=1337 bash submit.sh
SEED=42   bash submit.sh
SEED=9999 bash submit.sh
```
Expected: ~1.068-1.072 BPB (v9a was 1.0705, we add GradQuant + hedge).

### Run 2: Hedge ablation
```bash
HEDGE_ENABLED=0 SEED=1337 bash submit.sh
```
Measures hedge contribution alone (vs GradQuant-only).

### Run 3: GradQuant ablation
```bash
GRADQUANT_ENABLED=0 SEED=1337 bash submit.sh
```
Measures GradQuant contribution alone.

### Run 4 (if headroom): Add TTT
```bash
TTT_ENABLED=1 TTT_EPOCHS=20 SEED=1337 bash submit.sh
```
Only if Run 1 beats v9a. TTT + hedge together might be the best combo.

## Model Scaling Notes (NOT in v10a)
For future versions:
- `MODEL_DIM=640 NUM_LAYERS=11`: ~41M params. At uniform Int6 → ~23MB compressed. EXCEEDS 16MB.
  - Needs ternary QAT for MLP layers (1.58 bits) + Int4 attention to fit.
  - Ternary QAT requires training from scratch with STE; risky without validation.
  - Do NOT attempt until v10a validates GradQuant + hedge and establishes baseline.
- `MODEL_DIM=640 NUM_LAYERS=13`: Even larger. Would need aggressive ternary.

## Key Constraints Check
| Constraint | v10 Status |
|-----------|------------|
| Artifact < 16MB | Expected ~14.8-15.2MB (GradQuant may save ~100-200KB vs v9a) |
| Train time < 10min | No change from v9a (TTT disabled) |
| Eval time < 10min | Hedge adds ~5-10s; n-gram adds ~60-90s. Total ~90-120s eval. ✓ |
| No prefill | ✓ (v9c's prefill excluded — gray area legality) |
| Score-first | ✓ (all n-gram and hedge updates happen after scoring) |
| 3 seeds for SOTA claim | Required before PR submission |

## Success Criteria
- v10 must beat v9a (1.0705) by ≥0.005 nats (val_bpb ≤ 1.0655) across 3 seeds at p<0.01
- Artifact must be < 16,000,000 bytes (code + compressed model)
- Eval time must be < 10 minutes on 8×H100

## Failure Modes and Mitigations
1. **GradQuant makes things worse**: `GRADQUANT_ENABLED=0` reverts to uniform Int6
2. **Hedge mixer hurts**: `HEDGE_ENABLED=0` reverts to fixed entropy-adaptive alpha
3. **Artifact over 16MB**: Reduce `GRADQUANT_INT7_FRAC` (fewer int7 = better compression)
4. **Eval time over 10min**: Reduce `NGRAM_BUCKETS` from 4M to 2M, or reduce `NGRAM_ORDER` to 9

## Timeline
- 2026-03-26: Submit Run 1 (3 seeds), ablations
- 2026-03-27: Review results, tune hedge_beta / GradQuant fracs
- 2026-03-28+: Add TTT if v10a is validated
