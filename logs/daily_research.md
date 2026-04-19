# Parameter Golf Daily Research - 2026-04-19

## PR #771 STATUS: CLOSED (ILLEGAL — confirmed)

valerio-oai comment (2026-03-27): "you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." Train-then-score ordering is permanently disqualified. No appeal path.

---

## N-GRAM PR STATUS

| PR | Claimed BPB | Status | Notes |
|----|-------------|--------|-------|
| #727 | 0.9674 | **CLOSED (ILLEGAL)** | valerio-oai: target token in hash key = leaks eval tokens |
| #741 | — | **CLOSED (ILLEGAL)** | Same n-gram hash violation |
| #758 | 1.0465 | **OPEN (effectively dead)** | MatoTeziTanka (Apr 12): XOR hash key includes target token — same ruling as #727. Author hasn't responded. |
| #731 | 1.0400 | **OPEN — 1 seed only** | Reviewer "LOOKS CLEAN" (score-first-per-chunk, dense count + Laplace, no hashing). Seeds 1337 and 2024 still pending. No movement since Apr 17. |

---

## Leaderboard

| | Score | Author | Date |
|--|-------|--------|------|
| **Merged SOTA** | **1.0810** | bigbag (PR #1493) | 2026-04-09 |
| Best open (no CaseOps/SLOT) | **1.00995** | arsenis-cmd (PR #1698) — ⚠️ ARTIFACT BUG + BPB BUG: effectively dead | |
| Best open (CaseOps pending ruling) | **1.03540** | alertcat (PR #1738) — ⚠️ also builds on pre-quant TTT from PR #1735 (illegal) | |
| Best open (CaseOps, dexhunter) | **1.06549** | dexhunter (PR #1736) — clean, no legality flags | |
| Best open (CaseOps + casefold pending) | 1.05733 | dexhunter (PR #1693) — casefold ruling still pending | |
| Best open (SLOT risk) | 1.0616 | powerpratik (PR #1647) | |
| Best open (clean, no CaseOps) | **1.07139** | MarioPaerle (PR #1667) — Attn Output Gate + SmearGate | |
| Our PR #771 | 1.0705 | sunnypatneedi | CLOSED/ILLEGAL |

**Day 10 plateau** — no new merges since PR #1493 on Apr 9 (new record for longest plateau).

---

## What Changed (GitHub — Apr 17–19, 2026)

### New PRs filed since Apr 17

| PR | Author | BPB | Technique | Status | Notes |
|----|--------|-----|-----------|--------|-------|
| #1743 | OleStan | — | FreqGPTQ + GatedDeltaNet + Adaptive Quant | Open WIP | Watch |
| #1738 | alertcat | **1.03540** | CaseOps V15 + PR #1735 Pre-Quant TTT | Open | ⚠️ Builds on illegal PR #1735; no reviews yet |
| #1736 | dexhunter | **1.06549** | CaseOps + GatedAttn + QuantGate + SP8192 | Open | **CLEANEST new PR** — no legality flags, await CaseOps ruling |
| #1735 | AjAnubolu | 1.0429 | Pre-Quant AdamW TTT 21ep | Open | **⚠️ LIKELY ILLEGAL** — dexhunter flagged adapt-then-score, same pattern as PR #1351/#1416 |
| #1732 | Victory963 | 1.0785 | Hadamard Rotation + AWQ + Parallel Residuals | Open | New quant approach, no reviews |
| #1729 | romeerp | 1.0678 | CaseOps (bijective) + Tapered WD | Open | Bijective/reversible; no legality flags |
| #1727 | yahya010 | **1.07217** | MP-SGD TTT 4 phases + QK-Gain 5.25 | Open | **Appears legal** — score-first each phase, explicit compliance notes |
| #1739 | DevelopedByAnurag | 1.1497 | SP8192 + Depth Recurrence + Muon 0.99 | Open | Low interest |

### PR #1698 (GDN FLA, arsenis-cmd): Now effectively dead
- **BPB bug confirmed** by @dexhunter: `build_sentencepiece_luts` double-count inflates byte denominator ~17.7%. Corrected actual score: ~**1.189 BPB**, not 1.00995.
- **Artifact size violation confirmed**: Seeds range 16,474,250–16,600,916 bytes vs 16,000,000-byte decimal limit.
- No organizer response. Author lost GPU access. Do NOT track this PR.

### Key new technique: CaseOps Tokenizer
Three PRs (#1729 romeerp, #1736 dexhunter, #1738 alertcat) use a **bijective case-factoring transform** distinct from casefold:
- Encodes capitalization as control tokens: `TITLE`, `ALLCAPS`, `CAPNEXT`, `ESC`
- Fully reversible — original text reconstructs exactly
- BPB is scored on **original UTF-8 bytes** via a byte sidecar file, not on transformed tokens
- Stronger legality argument than casefold (which destroys information)
- **Issue #1604 ruling still pending** from @valerio-oai — do NOT implement until ruled legal

### PR #1698 successor: PR #1736 (dexhunter) is the cleanest GDN-free path
At 1.06549 with CaseOps + GatedAttn + QuantGate, if CaseOps gets ruled legal, dexhunter's stack is the primary implementation target.

### Existing priority PRs — status unchanged since Apr 17
| PR | Author | BPB | Status |
|----|--------|-----|--------|
| #1586 | dexhunter | 1.07493 | **OPEN, no reviews** — IMPLEMENT IMMEDIATELY |
| #1667 | MarioPaerle | 1.07139 | **OPEN, no reviews** — stack on #1586 |
| #1693 | dexhunter | 1.05733 | **OPEN** — casefold ruling pending (Issue #1604) |
| #1647 | powerpratik | 1.0616 | **OPEN** — SLOT-4, high risk |

---

## New Research Papers

Nothing transformative found for April 2026. Best candidates:

### arXiv:2604.13552 — Training-Free TTT Contrastive Learning (Apr 2026)
Adapts LLMs at test time via contrastive loss (positive/negative token pairs from context), no labeled data. Not directly applicable to score-first per-token BPB — our TTT is tightly constrained by the score-first rule. Low priority.

### arXiv:2604.09624 — SECL: Self-Calibrating LMs (Apr 2026)
Updates only on high-uncertainty tokens (6–26% of stream). Could reduce our TTT compute budget and allow more epochs within 10-min eval budget. Worth reading if MP-SGD TTT 4-phase approach (PR #1727) doesn't fit in time budget.

### arXiv:2412.06464 — GatedDeltaNet (ICLR 2025) — remains key reference
O(n) recurrence combining delta rule + gating. PR #1743 (WIP) is pursuing this. BPB bug pattern has killed every open GDN PR so far (#1576, #1687, #1698) — verify byte-counting before any GDN investment.

---

## HuggingFace / Community Discoveries

- **CaseOps bijective transform** is a new community pattern: three independent PRs from different authors all converged on bijective case-factoring in the last 2 days. Strong signal this is a real technique.
- **MP-SGD TTT 4 phases (PR #1727)** extends earlier 3-phase approach (PR #1700) — yahya010 found a way to fit a 4th phase in eval budget. Stack-compatible.
- **Pre-quant TTT pattern continues to be submitted** — PR #1735 is the latest. Pattern is clear: authors keep trying, dexhunter keeps flagging it. Do not implement.

---

## Recommended Actions (priority order)

1. **IMPLEMENT PR #1586 immediately** — Per-layer GPTQ (MLP=12σ, Attn=13σ) + int7 emb (15σ) + MLR=0.026. Config-level change, zero legality risk, -0.013 nats. 11 days to deadline. This is overdue.

2. **STACK PR #1667 on #1586** — Attention Output Gate (1,056 params, init to zero) + SmearGate (width=12). Expected combined delta: ~-0.019 nats total.

3. **ADD PR #1727 technique (MP-SGD TTT 4 phases)** — Appears legal (score-first per phase), +1 phase vs PR #1693 base. yahya010 reaches 1.07217 BPB. Stack with #1586+#1667.

4. **AWAIT Issue #1604 ruling on CaseOps** — If bijective CaseOps ruled legal (stronger case than casefold), dexhunter's PR #1736 at 1.06549 becomes primary target. CaseOps is reversible and BPB-accurate via sidecar; may pass where casefold fails.

5. **DO NOT IMPLEMENT**: Pre-quant TTT (#1735/#1738), casefold (#1693 component) without ruling, SLOT (#1647) without explicit risk acceptance, anything from GDN PRs until BPB bug pattern is resolved.

6. **AWAIT PR #731** — 3-seed Hedge Mixer still pending seeds 1337/2024. If merged, legal n-gram mixer available.

---

_Updated: 2026-04-19 (v14.0 — PR #1698 GDN effectively dead (BPB bug ~1.189 actual + artifact violation); CaseOps bijective tokenizer emerging as new community technique (#1729, #1736, #1738); PR #1735 pre-quant TTT flagged illegal; PR #1727 MP-SGD TTT 4-phase appears legal; merged SOTA 1.0810 Day 10 plateau; 11 days to deadline)_
