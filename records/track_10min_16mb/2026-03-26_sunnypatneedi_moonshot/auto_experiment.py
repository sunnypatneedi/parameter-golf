#!/usr/bin/env python3
"""auto_experiment.py — Hyperparameter search runner for v10 moonshot.

Defines a search space, generates experiment configs, and runs them
sequentially (or logs configs for H100 submission). Uses simple random
search with top-k selection after each round.

Usage:
    # Dry-run: generate configs and validate script syntax only
    python3 auto_experiment.py --dry-run

    # Run local smoke test (CPU, 100 steps, checks script executes)
    python3 auto_experiment.py --local-smoke --n-configs 8

    # Print best config for H100 submission
    python3 auto_experiment.py --best-config
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_gpt_v10_moonshot.py"
RESULTS_FILE = SCRIPT_DIR / "experiment_results.jsonl"

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
SEARCH_SPACE: dict[str, Any] = {
    # Architecture
    "NUM_LAYERS": [11],           # 13 needs ternary quant to fit
    "MODEL_DIM": [512],           # 640 needs ternary quant; keep 512 as base
    "MLP_MULT": [3.0, 3.5],
    "NUM_HEADS": [8],
    "NUM_KV_HEADS": [4],
    # N-gram
    "NGRAM_ORDER": [11],
    "NGRAM_MIN_ORDER": [2],
    "NGRAM_BUCKETS": [4194304, 8388608],   # 4M vs 8M buckets
    "NGRAM_ALPHA": [0.35, 0.40, 0.45],
    "NGRAM_ENT_BASE": [0.04, 0.05, 0.06],
    "NGRAM_ENT_RANGE": [0.50, 0.55, 0.60],
    # Hedge Mixer
    "HEDGE_ENABLED": [1],
    "HEDGE_BETA": [1.5, 2.0, 3.0],
    # GradQuant
    "GRADQUANT_ENABLED": [1],
    "GRADQUANT_INT5_FRAC": [0.25, 0.35, 0.45],
    "GRADQUANT_INT7_FRAC": [0.10, 0.15, 0.20],
    # Training
    "TRAIN_SEQ_LEN": [2048],
    "EVAL_SEQ_LEN": [2048],
    "EVAL_STRIDE": [64],
    "WARMDOWN_ITERS": [3500],
    "MATRIX_LR": [0.020, 0.025, 0.030],
    "MUON_WD": [0.04],
    "LATE_QAT_THRESHOLD": [0.10, 0.15],
    # XSA / architecture flags
    "XSA_LAST_N": [11],
    "GATED_ATTENTION": [1],
    "ROPE_DIMS": [16],
    "LN_SCALE": [1],
    "VE_ENABLED": [1],
    "VE_DIM": [128],
    "VE_LAYERS": ["7,8,9,10"],
    "BIGRAM_VOCAB_SIZE": [6144],
    "BIGRAM_DIM": [128],
    # SWA / EMA
    "SWA_ENABLED": [1],
    "SWA_EVERY": [50],
}

# Fixed settings for all experiments
FIXED_SETTINGS: dict[str, Any] = {
    "VOCAB_SIZE": 1024,
    "TIE_EMBEDDINGS": 1,
    "NGRAM_CACHE": 1,
    "NGRAM_ENTROPY": 1,
    "TTT_ENABLED": 0,    # v10a: no TTT (trade TTT budget for n-gram + hedge)
    "QAT_ENABLED": 0,    # handled by late QAT threshold
    "ITERATIONS": 20000,
    "TRAIN_BATCH_TOKENS": 786432,
    "MAX_WALLCLOCK_SECONDS": 600,
    "VAL_LOSS_EVERY": 0,
    "TRAIN_LOG_EVERY": 500,
}

# Smoke test settings (very fast, just checks script runs)
SMOKE_SETTINGS: dict[str, Any] = {
    "ITERATIONS": 100,
    "MAX_WALLCLOCK_SECONDS": 120,
    "VAL_LOSS_EVERY": 0,
    "TRAIN_BATCH_TOKENS": 8192,
    "TRAIN_SEQ_LEN": 512,
    "EVAL_SEQ_LEN": 512,
    "NGRAM_BUCKETS": 65536,
    "NGRAM_ORDER": 5,
    "VAL_BATCH_SIZE": 16384,
}


def sample_config(seed: int | None = None) -> dict[str, Any]:
    rng = random.Random(seed)
    cfg: dict[str, Any] = {}
    for key, choices in SEARCH_SPACE.items():
        cfg[key] = rng.choice(choices)
    cfg.update(FIXED_SETTINGS)
    return cfg


def config_to_env(cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    for k, v in cfg.items():
        env[k] = str(v)
    return env


def run_smoke_test(cfg: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    """Run a 100-step smoke test on CPU to validate the config/script."""
    smoke_cfg = {**cfg, **SMOKE_SETTINGS, "RUN_ID": experiment_id}
    env = config_to_env(smoke_cfg)
    # Override for CPU/local run
    env.pop("CUDA_VISIBLE_DEVICES", None)

    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    elapsed = time.perf_counter() - t0

    return {
        "experiment_id": experiment_id,
        "config": cfg,
        "returncode": result.returncode,
        "elapsed_s": elapsed,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
        "stderr_tail": result.stderr[-1000:] if result.stderr else "",
        "success": result.returncode == 0,
    }


def parse_val_bpb(stdout: str) -> float | None:
    """Extract val_bpb from training log output."""
    for line in reversed(stdout.splitlines()):
        for prefix in ("final_gq_sliding_window_exact", "final_gq_roundtrip_exact"):
            if prefix in line:
                for token in line.split():
                    if token.startswith("val_bpb:"):
                        try:
                            return float(token.split(":")[1])
                        except ValueError:
                            pass
    return None


def load_results() -> list[dict]:
    if not RESULTS_FILE.exists():
        return []
    results = []
    for line in RESULTS_FILE.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return results


def save_result(result: dict) -> None:
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(result) + "\n")


def best_config_from_results() -> dict | None:
    results = load_results()
    scored = [r for r in results if r.get("val_bpb") is not None]
    if not scored:
        return None
    return min(scored, key=lambda r: r["val_bpb"])


def main() -> None:
    parser = argparse.ArgumentParser(description="v10 hyperparameter search")
    parser.add_argument("--dry-run", action="store_true", help="Generate configs only, no execution")
    parser.add_argument("--local-smoke", action="store_true", help="Run local smoke tests (100 steps)")
    parser.add_argument("--best-config", action="store_true", help="Print best config from past results")
    parser.add_argument("--n-configs", type=int, default=16, help="Number of configs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for config sampling")
    args = parser.parse_args()

    if args.best_config:
        best = best_config_from_results()
        if best is None:
            print("No results found. Run some experiments first.")
        else:
            print(f"Best val_bpb: {best['val_bpb']}")
            print(f"Config: {json.dumps(best['config'], indent=2)}")
        return

    rng = random.Random(args.seed)
    configs = [sample_config(seed=rng.randint(0, 2**32)) for _ in range(args.n_configs)]

    if args.dry_run:
        print(f"Generated {len(configs)} configs:")
        for i, cfg in enumerate(configs):
            print(f"\n--- Config {i+1} ---")
            for k, v in sorted(cfg.items()):
                if k not in FIXED_SETTINGS:
                    print(f"  {k}={v}")
        return

    if args.local_smoke:
        print(f"Running {len(configs)} smoke tests (100 steps each)...")
        passed = 0
        failed = 0
        for i, cfg in enumerate(configs):
            exp_id = f"smoke_{i+1:03d}_{int(time.time())}"
            print(f"\n[{i+1}/{len(configs)}] Running {exp_id}...")
            result = run_smoke_test(cfg, exp_id)
            if result["success"]:
                passed += 1
                val_bpb = parse_val_bpb(result["stdout_tail"])
                result["val_bpb"] = val_bpb
                print(f"  PASS ({result['elapsed_s']:.1f}s)"
                      + (f" val_bpb={val_bpb:.4f}" if val_bpb else ""))
            else:
                failed += 1
                print(f"  FAIL (rc={result['returncode']}) stderr: {result['stderr_tail'][-200:]}")
            save_result(result)
        print(f"\nResults: {passed} passed, {failed} failed. Saved to {RESULTS_FILE}")
        return

    # Default: print H100 configs for manual submission
    print(f"Printing {len(configs)} configs for H100 submission:")
    for i, cfg in enumerate(configs):
        exp_id = f"v10_h100_{i+1:03d}"
        print(f"\n--- Experiment {exp_id} ---")
        env_str = " ".join(f"{k}={v}" for k, v in sorted(cfg.items()) if k not in FIXED_SETTINGS)
        print(f"RUN_ID={exp_id} {env_str} \\")
        print(f"  torchrun --standalone --nproc_per_node=8 {TRAIN_SCRIPT.name}")


if __name__ == "__main__":
    main()
