#!/usr/bin/env bash
# submit.sh — Package and launch v10 moonshot for 8×H100 submission.
#
# Usage:
#   bash submit.sh [--seed N] [--dry-run]
#
# Environment variables (override defaults):
#   SEED=1337 (default) — or pass --seed N
#   DRY_RUN=1           — print command only, don't run
#
# Best config (update from auto_experiment.py --best-config):
#   python3 auto_experiment.py --best-config

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train_gpt_v10_moonshot.py"

# --- Defaults (proven from v9a validation) ---
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-0}"

# Parse args
for arg in "$@"; do
    case $arg in
        --seed=*) SEED="${arg#*=}" ;;
        --seed)   shift; SEED="$1" ;;
        --dry-run) DRY_RUN=1 ;;
    esac
done

# --- Config (best proven settings from v9a + v10 additions) ---
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE=1024
export SEED="$SEED"
export RUN_ID="v10_moonshot_seed${SEED}"

# Architecture
export NUM_LAYERS=11
export MODEL_DIM=512
export NUM_HEADS=8
export NUM_KV_HEADS=4
export MLP_MULT=3.0

# Training
export ITERATIONS=20000
export WARMDOWN_ITERS=3500
export WARMUP_STEPS=20
export TRAIN_BATCH_TOKENS=786432
export TRAIN_SEQ_LEN=2048
export MAX_WALLCLOCK_SECONDS=600

# Optimizer
export MATRIX_LR=0.025
export MUON_WD=0.04
export ADAM_WD=0.04
export GRAD_CLIP_NORM=0.3

# Model features
export XSA_LAST_N=11
export GATED_ATTENTION=1
export ROPE_DIMS=16
export LN_SCALE=1
export VE_ENABLED=1
export VE_DIM=128
export VE_LAYERS="7,8,9,10"
export BIGRAM_VOCAB_SIZE=6144
export BIGRAM_DIM=128
export TIE_EMBEDDINGS=1

# Quantization
export LATE_QAT_THRESHOLD=0.15
export QAT_ENABLED=0

# v10 GradQuant
export GRADQUANT_ENABLED=1
export GRADQUANT_INT5_FRAC=0.35
export GRADQUANT_INT7_FRAC=0.15

# v10 Hedge Mixer
export HEDGE_ENABLED=1
export HEDGE_BETA=2.0

# N-gram (11-gram, entropy-adaptive, 4M buckets, score-first protocol)
export NGRAM_CACHE=1
export NGRAM_ORDER=11
export NGRAM_MIN_ORDER=2
export NGRAM_BUCKETS=4194304
export NGRAM_ENTROPY=1
export NGRAM_ALPHA=0.40
export NGRAM_ENT_BASE=0.05
export NGRAM_ENT_RANGE=0.55
export EVAL_STRIDE=64

# SWA / EMA
export SWA_ENABLED=1
export SWA_EVERY=50

# TTT disabled (v10a: all eval budget for n-gram + hedge)
export TTT_ENABLED=0

echo "=== v10 Moonshot Submission ==="
echo "Script:    $TRAIN_SCRIPT"
echo "Run ID:    $RUN_ID"
echo "Seed:      $SEED"
echo "GradQuant: enabled (int5_frac=${GRADQUANT_INT5_FRAC}, int7_frac=${GRADQUANT_INT7_FRAC})"
echo "Hedge:     enabled (beta=${HEDGE_BETA})"
echo "N-gram:    ${NGRAM_ORDER}-gram, ${NGRAM_BUCKETS} buckets, entropy-adaptive"
echo ""

CMD="nohup torchrun --standalone --nproc_per_node=8 $TRAIN_SCRIPT > /workspace/${RUN_ID}.log 2>&1 &"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would execute:"
    echo "$CMD"
    echo ""
    echo "After completion, check: tail -f /workspace/${RUN_ID}.log"
else
    echo "Launching on 8×H100..."
    eval "$CMD"
    echo "PID: $!"
    echo "Log: /workspace/${RUN_ID}.log"
    echo ""
    echo "Monitor: tail -f /workspace/${RUN_ID}.log"
    echo "Check size: ls -lh final_model.gq.ptz"
fi
