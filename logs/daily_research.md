# Parameter Golf Daily Research - 2026-04-17

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed)

valerio-oai comment (2026-03-27): "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." Train-then-score ordering is permanently disqualified. No appeal path.

---

## N-GRAM PR STATUS

| PR | Claimed BPB | Status | Notes |
|----|-------------|--------|-------|
| #727 | 0.9674 | **CLOSED (ILLEGAL)** | valerio-oai: target token in hash key = leaks eval tokens |
| #741 | — | **CLOSED (ILLEGAL)** | Same n-gram hash violation |
| #758 | 1.0465 | **OPEN (effectively dead)** | MatoTeziTanka (Apr 12): XOR hash key includes target token — same ruling as #727. Author hasn't responded. |
| #731 | 1.0400 | **OPEN — 1 seed only** | Reviewer "LOOKS CLEAN" (score-first-per-chunk, dense count + Laplace, no hashing). Seeds 1337 and 2024 still pending. If 3rd seed confirms ~1.04, expect merge. |

---

## Leaderboard

| | Score | Author | Date |
|--|-------|--------|------|
| **Merged SOTA** | **1.0810** | bigbag (PR #1493) | 2026-04-09 |
| Best open (no casefold) | **1.00995** | arsenis-cmd (PR #1698) — ⚠️ ARTIFACT SIZE CONCERN | |
| Best open (casefold pending) | 1.05733 | dexhunter (PR #1693) | |
| Best open (SLOT risk) | 1.0616 | powerpratik (PR #1647) | |
| Best open (clean, no reviews) | 1.07139 | MarioPaerle (PR #1667) | |
| Our PR #771 | 1.0705 | sunnypatneedi | CLOSED/ILLEGAL |

**Day 8 plateau** — no new merges since PR #1493 on Apr 9 (longest plateau in competition history).

---

## What Changed (GitHub — Apr 15–17, 2026)

### Critical New PR: #1698 — GatedDeltaNet (FLA) + Legal Score-First TTT

**Author**: arsenis-cmd  
**Claimed BPB**: 1.00995 (3-seed mean, std 0.0012)  
- Seed 42: 1.01130 | Seed 314: 1.00896 | Seed 999: 1.00959  
**Architecture**: GatedDeltaNet linear attention (O(n) recurrence via FLA library) + K_KVShare_Wider (kv_sharing_stride=2, buy width not depth), 10 layers, 544d  
**Quant**: int6 matrix + zstd-22  
**TTT**: Legal score-first, 3ep, SGD lr=0.005 momentum=0.9, 32K-token chunks, freeze first 2 blocks  
**Legality flags**: None from reviewers (no reviews yet)  

**⚠️ ARTIFACT SIZE VIOLATION RISK**: Reported artifact sizes are 16,600,916 / 16,548,775 / 16,474,250 bytes (seeds 42/314/999). The competition limit is **< 16,000,000 bytes** (decimal, not MiB). PR #1698's artifacts are ~600K bytes OVER the decimal MB limit. The author claims "under 16 MiB" (16,777,216 bytes) — a different standard. **Organizer review pending.** If rejected on artifact size, this result is void. If ruled compliant (organizers accept 16 MiB as the limit), GatedDeltaNet becomes our primary target architecture.

### PR #1693 — dexhunter, 1.05733 BPB (new best casefold PR)
Builds on PR #1670 (Casefold V4 + Multi-Phase TTT). Adds Attention Output Gate (1,069 params, same as PR #1667) and SmearGate. -0.00237 vs PR #1670's 1.05970. Still contingent on casefold ruling from @valerio-oai (Issue #1604).

### PR #1687 — resouer, K_KVShare_Wider FLA: CLOSED (BPB bug)
@bigbag found SP byte-counting bug (leading-space double-count, same pattern as PR #1545/PR #1576). Actual score ~1.22 BPB, not 1.04090. Author confirmed and closed. **Do not track.**

### Other new PRs of note (Apr 15–17)
| PR | Author | BPB | Technique | Status |
|----|--------|-----|-----------|--------|
| #1695 | X-Abhishek-X | 1.0759 | Stage 3 + SpinQuant V1 + MP-SGD-TTT | Open |
| #1676 | aazizyan | 1.0788 | Trajectory-State Readout + Muon 0.98 + Legal TTT | Open |
| #1688 | Buld1n | 1.0809 | SP8192 qkramp05 + par-residual L6 + legal TTT | Open |
| #1689 | chris-colinsky | 1.0822 | SP8192 + Adaptive Hessian-Sensitivity GPTQ | Open |

### Existing priority PRs — status unchanged
| PR | Author | BPB | Status |
|----|--------|-----|--------|
| #1586 | dexhunter | 1.07493 | **OPEN, no reviews** — per-layer GPTQ + int7 emb. IMPLEMENT IMMEDIATELY. |
| #1667 | MarioPaerle | 1.07139 | **OPEN, no reviews** — Attn Output Gate + SmearGate. Stack on #1586. |
| #1647 | powerpratik | 1.0616 | **OPEN, no reviews** — SLOT-4 (standard SLOT, high risk). |

---

## New Research Papers

### GatedDeltaNet — arXiv:2412.06464 (ICLR 2025)
"Gated Delta Networks: Improving Mamba2 with Delta Rule" (NVLabs). O(n) recurrence combining key-value matrix memory, delta rule updates, and gating. Hybrid variants (GDN-H1/H2) mix recurrent layers with SWA. **Directly relevant** — PR #1698 implements this via the FLA library (github.com/fla-org/flash-linear-attention). If PR #1698 is ruled compliant (artifact size), GatedDeltaNet becomes the base architecture to adopt. Implementation complexity: high (requires FLA library, full architecture rewrite ~500+ lines).

### SECL: Self-Calibrating LMs via TTDD — arXiv:2604.09624 (Apr 2026)
Adapts only when distribution shifts; trains on 6–26% of test stream. Uses discriminative vs generative error gap as self-supervision. Could improve legal score-first TTT by only updating on high-uncertainty tokens. Low priority — no direct BPB numbers shown.

### LieQ: Layer-wise PTQ for Small LMs — arXiv:2508.03332
Geometry-driven sensitivity proxy for automatic bit-width allocation. Reduces accuracy gap at sub-2-bit. May inform per-layer GPTQ sigma tuning beyond PR #1586's approach. Watch after PR #1586 is implemented.

---

## HuggingFace / Community Discoveries

- **GatedDeltaNet in Qwen 3.5**: GDN deployed in production (3:1 to 6:1 ratio GDN:attn). Confirms architecture viability at scale.
- **BPB bug pattern is recurring**: PR #1687 is 4th instance (after #1545, #1576, #1687) of SP leading-space double-count. Any new PR claiming >0.06 bpb improvement vs stack should be BPB-verified immediately.

---

## Recommended Actions (priority order)

1. **VERIFY PR #1698 artifact limit** — Check challenge rules text for exact byte limit (16,000,000 bytes decimal vs 16,777,216 bytes MiB). If organizers rule PR #1698 compliant: GDN architecture rewrite becomes primary target. Read PR #1698 code for FLA integration pattern.

2. **IMPLEMENT PR #1586 now** — Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 emb (15σ) + MLR=0.026. Config-level, zero legality risk. -0.013 nats vs merged SOTA. This is valid regardless of GDN outcome.

3. **STACK PR #1667 on #1586** — Attention Output Gate (1,069 params, init zero) + SmearGate (width=12). Expected combined: ~-0.019 nats total.

4. **DO NOT IMPLEMENT** casefold (await Issue #1604 ruling), SLOT/PR #1647 (without explicit risk decision), pre-quant TTT (illegal).

5. **AWAIT PR #731** — If Hedge Mixer 3-seed confirms ~1.04 and merges, legal n-gram mixer blueprint available.

---

_Updated: 2026-04-17 (v13.0 — PR #1698 GatedDeltaNet FLA 1.00995 BPB flagged with artifact size concern; PR #1687 CLOSED BPB bug; PR #1693 dexhunter 1.05733 new casefold leader; merged SOTA 1.0810 Day 8 plateau; 13 days to deadline)_
