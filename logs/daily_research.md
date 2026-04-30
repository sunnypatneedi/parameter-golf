# Parameter Golf Daily Research - 2026-04-30 (FINAL DAY)

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27)

@valerio-oai: "around line 1500 you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on, so this is not an allowable submission." Train-then-score violation. No appeal path.

---

## N-GRAM PR STATUS

- **PR #727** (0.9674): **CLOSED** — rejected for hash key including target token (XOR leakage). Same normalization violation as n-gram hash cache family.
- **PR #758** (1.0465): **OPEN (effectively dead)** — reviewer MatoTeziTanka (Apr 12) flagged target token in XOR hash key. No new comments. Redesign required.
- **PR #731** (1.0400): **OPEN** — still awaiting seeds 1337 and 2024. Reviewer said "LOOKS CLEAN." Dense count tables + Laplace smoothing approach. No movement since Apr 12.

---

## Leaderboard — MAJOR CHANGE

**Organizer pending branches merged. 12+ new records since yesterday.**

| Rank | Score | Author | PR | Techniques |
|------|-------|--------|-----|------------|
| 1 | **1.0611** | codemath3000 | #1855 | SP8192 + LQER Asym + SparseAttnGate + BOS-Fixed SmearGate + 9-hparam greedy + lrzip |
| 2 | 1.0613 | aquariouseworkman | #1851/#1868 | SmearGate BOS Fix + PR#1787 base + LQER Asym + Phased TTT |
| 3 | 1.0634 | nprime06 | #1787 | CaseOps + Polar Express NS + MIN_LR=0.10 + SparseAttnGate + FusedCE + Warm-A TTT |
| 4 | 1.0645 | dexhunter | #1769 | CaseOps + MLPClip12 + SmearGate + LoRA-TTT |
| 5 | 1.0655 | dexhunter | #1736 | CaseOps + GatedAttn + QuantGate + Loop45 + Phased TTT |
| 6 | 1.0678 | romeerp | #1729 | CaseOps + Tapered WD + Phased TTT |
| 7 | 1.0714 | MarioPaerle | #1667 | SmearGate + Attention Output Gate + Legal TTT |
| 8 | 1.0719 | dexhunter | #1626 | VarLen Attn + Fused MLP + Multi-Phase Global SGD TTT |

**Previous merged SOTA was 1.0810 (PR #1493).** Dropped to **1.0611** — 0.0199 bpb improvement from 12 new merges. Confirms the organizer's pending branches fully landed.

**New target to beat SOTA by required 0.005 nats: ≤ 1.0561**

**Our PR #771**: 1.0705 — CLOSED/REJECTED. No current active submission.

---

## What Changed (GitHub) — April 30 Filings

New PRs opened on the final deadline day:

| PR | Author | Score | Technique | Legality |
|----|--------|-------|-----------|----------|
| #1991 | joshuaswanson | **0.94290** | Byte-PPM Mixer, order-5, tuned PPM_T/H/L gate | Score-first documented. **No organizer ruling yet.** Issue #1872 open. |
| #1992 | jamesEmerson112 | 1.0511 | SP8192 + Headwise Gated Attn + PreQuantTTT 21ep | **ILLEGAL** — 21ep pre-quant TTT flagged by reviewer. Same as PR #1735/#1423. |
| #1987 | TimS-ml | 1.06184 | MHA (8 KV heads) + PR #1855 9-hparam stack + LeakyReLU 0.3 | Appears clean. 15.84MB, eval ~591s. No objections raised. |
| #1972 | BharathSShankar | 1.03983 | SP10240 + SimCTG + PreQuantTTT | **Likely ILLEGAL** — PreQuantTTT pattern matches rejected PRs. |
| #1967 | ndokutovich | 1.05851 | V21 + N-gram Tilt + LeakyReLU 0.3, PR #1945 base | 172s hint-precompute vs 600s eval budget — Issue #677 ruling pending. |

**PR #1854** (ndokutovich, PPM-D, 0.90236): Still open, no @valerio-oai ruling. Issue #1872 confirms legality unresolved. Do NOT implement.

---

## New Research Papers

- **Polar Express NS** (arXiv:2505.16932, ICLR 2026): Already in merged SOTA (PR #1787). Adaptive polynomial Newton-Schulz with cubic convergence, ~2× faster than fixed NS. **Already implemented in best legal stack.**

- **ByteFlow** (arXiv:2603.03583, 2026): Byte-level LM without tokenizer; learns compression-driven segmentation. Not applicable within 10-min/16MB budget.

- **zip2zip** (arXiv:2506.01084): Inference-time adaptive tokenization. Interesting for future work; no overlap with current competition stack.

- **End-to-End TTT for Long Context** (arXiv:2512.23675): Compresses context into weights via next-token prediction. Conceptually aligned with Doc-TTT; no new actionable insight for today.

No new papers found beyond techniques already tracked in CLAUDE.md.

---

## Status Summary

| Item | Status |
|------|--------|
| Competition deadline | **TODAY — April 30, 2026** |
| Merged SOTA | **1.0611** (codemath3000, PR #1855) — down from 1.0810 |
| Required score to file new SOTA | ≤ **1.0561** |
| Our active submissions | **NONE** (PR #771 rejected) |
| Best clean legal open PR | PR #1967 (1.05851) — timing ruling pending |
| Best unruled-but-extraordinary | PR #1991 (0.94290, PPM-D) — no organizer ruling |
| PR #731 (n-gram Hedge Mixer) | Open, seeds 1337/2024 pending, "LOOKS CLEAN" |

---

## Recommended Action

**Competition ends today. Three scenarios:**

1. **GPU run already complete** (CaseOps + PR#1855 base + Polar Express NS + LQER Asym + LoRA-TTT alpha=144 + SmearGate BOS fix + lrzip): File PR immediately if result ≤ 1.0561. Estimated range ~1.052–1.058 based on additive deltas.

2. **No GPU run complete**: Competition is effectively over for new SOTA submissions. The 10-minute training budget means any new run still needs to be kicked off, validated, and filed within today.

3. **PPM-D (PR #1854/#1991, 0.902)**: Do NOT implement. @valerio-oai raised two explicit concerns on PR #1835 and Issue #1872 is open. Zero safe window before deadline.

**Note on PR #1987** (1.06184, clean): Would not beat SOTA by 0.005 nats (gap is only 0.0007). Not viable as a SOTA claim even if filed today.
