# Parameter Golf Daily Research - 2026-05-05 (POST-COMPETITION DAY 5)

## Competition Status: CLOSED (Apr 30, 2026)
Audit COMPLETE — PR #2146 merged May 1. Final results official.

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27) — Final

@valerio-oai verdict stands: "around line 1500 you're first adapting your model to the eval tokens with TTT for multiple epochs, and then reporting val numbers on those tokens you've already trained on." No appeal path. Score 1.0705 void.

## N-gram PR Status

All n-gram PRs from the competition window are now resolved:
- **PR #727**: CLOSED (illegal — unnormalized distribution)
- **PR #741**: CLOSED (illegal — same pattern)
- **PR #758**: Effectively DEAD (XOR hash key includes target token, flagged by MatoTeziTanka)
- **PR #731** (Hedge Mixer, dense count tables + Laplace smoothing): Still OPEN, seeds 1337/2024 never filed before deadline. Technique confirmed "LOOKS CLEAN" but competition closed without merge. Blueprint is sound for future use.

## Audit: PR #2146 MERGED (May 1, 2026)

Organizer cocohearts merged PR #2146 on May 1, finalizing the grace-policy audit. 192 PRs reviewed via parallel Codex shard graders + chronological reconciliation.

**Accepted (grace policy):**
| PR | Author | Score | Technique |
|----|--------|-------|-----------|
| #1945 | alertcat | 1.05943 | PR #1855 + AWQ-lite GPTQ + AsymLogit Rescale |
| #1953 | andrewbaggio1 | 1.05855 | V21 + 2560 context + no-Q/V TTT mask + QK-Gain 5.25 |
| #2014 | simonbissonnette | 1.05759 | CaseOps stack + progressive context growth to 3k + short-doc TTT |
| #2135 | codemath3000 | **1.05651** | PR #2130 base + GPTQ_CALIBRATION_BATCHES=32 ← **NEW OFFICIAL SOTA** |

**Rejected:**
| PR | Reason |
|----|--------|
| #2130 | Data overlap — docs 10,000–49,999 shared between train and val split |
| #2018/#2039/#2041/#2076/#2080/#2083/#2098/#2103 | Various: data overlap, byte-PPM audit failures |
| #2140 | Technically valid but superseded by PR #2135 (non-frontier) |

## Official Final Leaderboard (post-audit)

| Rank | Score | Author | PR |
|------|-------|--------|----|
| 1 | **1.05651** | codemath3000 | #2135 (grace policy) |
| 2 | 1.05759 | simonbissonnette | #2014 (grace policy) |
| 3 | 1.05855 | andrewbaggio1 | #1953 (grace policy) |
| 4 | 1.05943 | alertcat | #1945 (grace policy) |
| 5 | 1.0611 | codemath3000 | #1855 |
| 6 | 1.0614 | aquariouseworkman | #1851/#1868 |

**Our PR #771**: REJECTED. No placement.

## What Changed (GitHub) — Post-Competition Activity (May 1–5)

Post-deadline PRs filed for archival/non-record purposes only:

| PR | Author | Score | Status | Notes |
|----|--------|-------|--------|-------|
| #2157 | vimeto | 1.06043 | Draft | PR #1797 + AWQ-lite top3 + LQER 60k. Post-deadline, non-record. |
| #2155/#2154 | divagr18 | — | #2154 Closed | Mamba3 SSM hybrid SP8192, non-record. |
| #2153 | rixhavraj | 0.9627 | Open | **Likely BPB bug** — "36-hour optimization cycle," no artifact, informal commits, no methodology. Pattern matches prior BPB bug submissions. |
| #2149 | YaseenHQ | — | Open | RandProj384 tied embeddings + Pairwise-QK Muon, non-record. |
| #2143 | upascal | 1.07134 | Open | CaseOps + SparseAttnGate, post-deadline non-record. |
| #2144 | simonbissonnette | 0.9697 | Open | **Non-record** (pre-quant, not 16MB artifact). Explicitly marked non-record by author. |
| #2140 | simon-marcus | 1.0570 | Open | PR #2014 stack + LeakyReLU + n-gram TTT. Audit deemed non-frontier (superseded by #2135). |
| #2158 | izlley | — | Open | PR #2135 + MP3 marker-pair fusion. No score yet. |
| #2139 | varunneal | 1.05749 | Closed | TTT Peer-LoRA Ensemble on PR #2014. Self-closed. |

## New Research Papers

No new actionable papers found for our stack since May 4 entry:
- **LaCT** (arXiv:2505.23884): Already tracked. Large-chunk TTT. Used in Doc-TTT implementations in competition.
- **Test-Time Learning for LLMs** (arXiv:2505.20633): Input perplexity minimization. No direct applicability to score-first constraint.
- **End-to-End TTT for Long Context** (arXiv:2512.23675): Compresses context into weights via NTP loss. Conceptually aligned with Doc-TTT; no new actionable implementation detail.
- No new n-gram interpolation or QAT papers found beyond techniques already tracked.

## Recommended Action

Competition is closed. All SOTA standings are final pending any further organizer decisions on PPM-D (Issue #1872 — still unresolved). Key takeaways for any future competition:

1. **GPTQ_CALIBRATION_BATCHES=32** (vs 16): free ~0.001 bpb — add this to submission checklist.
2. **AsymLogit Rescale**: 2 trainable scalars replace fixed logit_softcap. ~5 lines, zero risk, ~0.001–0.002 bpb.
3. **Data overlap bug pattern**: Always verify train/val split with fingerprint check before filing. PR #2130 lost ~0.002 bpb improvement due to this bug being caught post-submission.
4. **BPB denominator**: Always verify byte count uses raw UTF-8 sidecar, not CaseOps-transformed bytes (PR #2138 bug #7).

---

# Parameter Golf Daily Research - 2026-05-04 (POST-COMPETITION DAY 4)

## Competition Status: CLOSED (Apr 30, 2026)
Post-competition audit in progress via Draft PR #2146. Competition results being finalized.

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27) — Final

No change. @valerio-oai ruling stands: train-then-score TTT violation (30 epochs AdamW on val tokens before scoring). No appeal path. Invalidated score: 1.0705 val_bpb.

## Audit PRs (DRAFT — grace policy)

- **PR #2146** (cocohearts, organizer — OPEN/DRAFT): "Update leaderboard with May 1 audited rows." Grace policy defined: code/scaffold filed pre-cutoff; results/logs filed post-deadline allowed. 22 reactions. PR #2135 included; PR #2130 excluded (data overlap). Status as of May 4: still DRAFT, not merged.
- **PR #2135** (codemath3000 — OPEN): val_bpb 1.05651 (3-seed mean). Change vs PR #2130: GPTQ_CALIBRATION_BATCHES 16→32. Improvement vs merged SOTA (PR #1855): -0.00457 bpb / -0.01000 nats (~2× threshold). Cocohearts confirmed grace-policy inclusion. Not yet merged into main.
- **PR #2130** (TanishGudise — CLOSED): val_bpb 1.05670. EXCLUDED by audit — train/val data overlap (docs 10,000–49,999 overlap with 50k-doc validation split, matching fingerprint from prepare_caseops_data.py with val-docs=10000). Placed in same category as PR #2018. Score void.

## PPM-D Issue #1872 Status

No @valerio-oai ruling found as of May 4. Issue remains open with only the initial post from andrewbaggio1 (Apr 27, 2026) visible. No organizer response. Competition ended unresolved. PPM-D legality remains undetermined.

## Leaderboard (as of 2026-05-04)

### Merged SOTA (upstream/main) — UNCHANGED since Apr 29
| Rank | Score | Author | PR | Key Stack |
|------|-------|--------|----|-----------|
| 1 | **1.0611** | codemath3000 | #1855 | BOS-Fixed SmearGate + LQER Asym + SparseAttnGate + 9-hparam + lrzip |
| 2 | 1.0614 | aquariouseworkman | #1851/#1868 | SmearGate BOS Fix + PR#1787 + LQER Asym + Phased TTT |
| 3 | 1.0634 | nprime06 | #1787 | CaseOps + Polar Express NS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT |
| 4 | 1.0645 | dexhunter | #1769 | CaseOps + MLPClip12 + SmearGate + LoRA-TTT |
| 5 | 1.0655 | dexhunter | #1736 | CaseOps + GatedAttn + QuantGate + PhasedTTT |

- Pending audit SOTA: **1.05651** (PR #2135, if PR #2146 grace policy merges)
- Our PR #771: 1.0705 — REJECTED

### Upstream commits since May 3
git log shows most recent commits are non-record/notable submissions (Mamba3-SSM hybrids, MHALM V2, adapter MLPs, XNOR-net, MDLM diffusion, etc.). No new leaderboard record merges since Apr 29.

## What Changed (GitHub — since May 3)

### New PRs (post-competition filing)
| PR | Author | Score | Technique | Notes |
|----|--------|-------|-----------|-------|
| #2155 | divagr18 | N/A (non-record) | SP8192 + Mamba3 SSM hybrid | Non-record submission, May 4 |
| #2153 | rixhavraj | claimed ~0.9627 | "Balanced Peak Architecture" 12L 768-dim ~7.2M params, 36-hour optimization | No reviews yet; claimed score looks extraordinary — high BPB-bug risk per competition history. Do NOT track until community-verified. |
| #2149 | YaseenHQ | negative result | SP8192 + RandProj384 tied embeddings + Pairwise-QK Muon | Self-labeled negative result, May 3 |
| #2146 | cocohearts | N/A | Leaderboard audit update (draft) | Organizer PR, see Audit section |
| #2145 | aquemy | 1.3477 | MHALM V2 | Non-record (above baseline) |

**PR #2153 (0.9627 claim) assessment**: No reviews, no community verification, "balanced architecture" with 7.2M params — the claimed jump from 1.0611 to 0.9627 (-0.098 bpb) with no novel technique description matches BPB-bug pattern seen 7 times this competition. Treat as likely BPB bug until independently verified.

## New Research Papers (May 1–4)

| Paper | arXiv | Date | Key Technique | Relevance to Future Competition |
|-------|-------|------|---------------|--------------------------------|
| **How Much Is One Recurrence Worth? Iso-Depth Scaling Laws for Looped LMs** | 2604.21106 | Apr 27, 2026 | 116-run iso-depth sweep; at 4 recurrences, 410M looped = 580M non-looped quality but costs 1B training compute; hyperconnections between loop states substantially improve loops; truncated BPTT weakens loop gradient quality | HIGH — direct quantitative guidance for tuning our Triple Loop depth and activation point. Hyperconnections are new and not yet in any competition PR. |
| **Hyperloop Transformers** | 2604.21254 | Apr 23, 2026 | Begin+Middle+End block organization; hyper-connections expand residual stream to matrix-valued streams applied only after each loop; outperforms depth-matched Transformer with ~50% fewer parameters; improvement persists post-quantization | HIGH — "50% fewer params, same quality" is directly applicable. Hyper-connections add minimal params. Post-quant robustness is key for 16MB target. |
| **Test-Time Training Done Right (LaCT)** | 2505.23884 | May 29, 2025 | Large Chunk TTT: 2K–1M token chunks; dramatically improves GPU utilization (up to 70% on A100 vs <5% for existing TTT); scales nonlinear state size to 40% of model params; no custom kernels required | HIGH — dexhunter's Doc-TTT (PR #1560) is likely LaCT-inspired. Full LaCT at our scale could improve TTT quality vs ≤3ep score-first limit. Code: github.com/a1600012888/LaCT |
| **Test-Time Learning for Large Language Models** | 2505.20633 | May 2025 | TTT framework for LLMs; separate training objective for fast-weight adaptation; reduces perplexity across diverse domains | MEDIUM — survey-style; less directly actionable than LaCT |
| **Intra-Layer Recurrence in Transformers** | 2505.01855 | May 2025 | ILR applies recurrence selectively to individual layers within a single forward pass; allocating more iterations to earlier layers is optimal; accepted at Canadian AI 2025 | MEDIUM — alternative to whole-model loop; earlier-layer recurrence matches what Triple Loop does (layers 4-5). Implementation guidance available on GitHub. |
| **Stability and Generalization in Looped Transformers** | 2604.15259 | Apr 2026 | Proves outer normalization (LayerNorm/RMSNorm at loop output) produces stable looped regime; enables deeper loops without residual explosion | HIGH — ~1-3 lines of code, may enable depth 4 or earlier activation in Triple Loop. Already in CLAUDE.md as watch item but now confirmed by this paper. |

### Papers from prior scan still pending reading
| Paper | arXiv | Priority |
|-------|-------|----------|
| In-Place TTT (NTP-aligned loss) | 2604.06169 | HIGH |
| Parcae (stable looped LMs via spectral norm) | 2604.12946 | HIGH |
| Decoupling Tokenization Effects | 2604.27263 | MEDIUM |

## Status Summary

| Item | Status |
|------|--------|
| Competition | **CLOSED** (April 30, 2026) |
| Final Merged SOTA | **1.0611** (codemath3000, PR #1855) |
| Pending Audit SOTA | **1.05651** (PR #2135, pending PR #2146 merge) |
| Our submission | **REJECTED** (PR #771) |
| PR #2146 audit | DRAFT — not merged |
| Issue #1872 (PPM-D) | No ruling — ended unresolved |
| Upstream commits since May 3 | Only non-record/notable submissions |

## Recommended Action

1. **Monitor PR #2146** — still DRAFT as of May 4. If it merges: (a) V22 stack (AsymLogit Rescale + AWQ-lite) is confirmed as the winning unreleased technique; (b) GPTQ calibration 16→32 batches is confirmed as +0.001 bpb; (c) new target for any future competition becomes ≤1.04651 (1.05651 - 0.010 threshold).

2. **Read arXiv:2604.21106 and 2604.21254** — both give quantitative guidance on looped transformer depth optimization that was not available during the competition. Hyperconnections (2604.21254) are particularly novel and appear to have zero competition-PR precedent.

3. **Flag PR #2153 (claimed 0.9627)** as unverified — no community review, extraordinary claim, no novel technique described. Consistent with BPB-bug pattern (7 prior bugs this competition).

4. **Document for future reference**: The grace policy established in PR #2146 (code pre-cutoff, results post-deadline) creates a clear precedent. For any future competition, file code PRs early even if results aren't ready.

---

# Parameter Golf Daily Research - 2026-05-03 (POST-COMPETITION DAY 3)

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27) — Final

No change. Train-then-score TTT violation per @valerio-oai. No appeal path.

## N-GRAM PR STATUS (Final)
- **PR #727**: CLOSED — hash key includes target token (eval leakage). Final.
- **PR #731**: OPEN, dormant — seeds 1337/2024 never filed. Competition closed. Dead.
- **PR #758**: OPEN, dead — same XOR target-token violation as #727.

## Leaderboard

### Current Merged (upstream/main)
| Rank | Score | Author | PR | Key Stack |
|------|-------|--------|----|-----------|
| 1 | **1.0611** | codemath3000 | #1855 | BOS-Fixed SmearGate + LQER Asym + SparseAttnGate + 9-hparam + lrzip |
| 2 | 1.0614 | aquariouseworkman | #1851/#1868 | SmearGate BOS Fix + PR#1787 + LQER Asym + Phased TTT |
| 3 | 1.0634 | nprime06 | #1787 | CaseOps + Polar Express NS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT |
| 4 | 1.0645 | dexhunter | #1769 | CaseOps + MLPClip12 + SmearGate + LoRA-TTT |
| 5 | 1.0655 | dexhunter | #1736 | CaseOps + GatedAttn + QuantGate + PhasedTTT |

No upstream/main commits since Apr 29. Leaderboard frozen at SOTA 1.0611.

### Pending Audit (Draft PR #2146 — NOT merged yet)
Organizer grace policy: code filed pre-cutoff, results filed post-deadline. Four rows pending:
| PR | Score | Techniques | Note |
|----|-------|------------|------|
| #1945 (V22) | 1.05877–1.05943 | AWQ-lite mixed-precision + AsymLogit Rescale + no_qv TTT masking + seq_len=2816 | 3-seed, all <600s |
| #1953 | 1.05855 | PR#1945 base + delta unknown | Under audit |
| #2014 | 1.05759 | PR#1953 base + delta unknown | Under audit |
| **#2135** | **1.05651** | PR#2130 base + GPTQ_CALIBRATION_BATCHES 16→32 | New top if merged |

If PR #2146 merges, effective SOTA drops to **1.05651** and new target becomes **≤1.05151**.

## What Changed (May 2–3, 2026)

### New Open PRs
| PR | Author | Score | Technique | Legality |
|----|--------|-------|-----------|----------|
| #2149 | YaseenHQ | unknown | SP8192 + RandProj384 tied embeddings + Pairwise-QK Muon | Non-record filing, May 3 |
| #2130 | TanishGudise | **1.05670** | Token-only n-gram tilt + AsymLogit Rescale + 3 hyperparams (MATRIX_LR=0.028, LQER_ASYM_GROUP=32, TTT_LORA_LR=8e-5) + NUM_PHASES=1 | ⚠️ Reviewer flagged train/val data overlap (docs 10,000–49,999). Excluded by audit. |
| #2124 | vaibhavmishra1 | **1.05933** | CaseOps + Gated XSA + NgramTilt + LQER g32/top4 + Phased TTT | ⚠️ 3-seed config inconsistency: headline uses third seed from different config. "Not record-ready as submitted." |
| #2138 | anmarhindi | ~~0.979556~~ → **1.067219** | Lock-In Byte Mixer (PPM-D gate, λ activates only at PPM_conf≥0.9999) | **CONFIRMED BPB BUG** (7th in competition): divides by CaseOps bytes not raw-text sidecar bytes. Corrected score 1.067219 = below SOTA. Do NOT track. |

### Key Technique: AsymLogit Rescale (PR #1923 / #2130)
- Replace single `logit_softcap=30.0` with two trainable scalars `softcap_pos`, `softcap_neg`
- Parameters adapt via TTT global prefix pass
- Implementation: ~5 lines, zero legality risk
- Used in V22 stack (PR #1945) and post-deadline leader PR #2135

### BPB Bug Tally: 7 confirmed this competition
Bugs in: PR #1545, #1576, #1687, #1698, #1848 (risk), #1858 (partial data), #2138.

## New Research Papers (May 3 scan)

No new highly relevant papers since May 2 scan. Prior high-priority items still pending:

| Paper | arXiv | Priority |
|-------|-------|----------|
| In-Place TTT (NTP-aligned loss) | 2604.06169 | High — read before next competition TTT design |
| Bell Box Quantization (BBQ) | 2603.01599 | High — ITO quantization; could replace GPTQ/LQER |
| EntroLLM entropy coding | 2505.02380 | High — additive to lrzip artifact compression |
| Decoupling Tokenization Effects | 2604.27263 | Medium — theoretical backing for CaseOps BPB debate |

**No new May 2026 competition-relevant papers found in this scan.**

## Status Summary

| Item | Status |
|------|--------|
| Competition | **CLOSED** (April 30, 2026) |
| Final Merged SOTA | **1.0611** (codemath3000, PR #1855) |
| Pending Audit SOTA | **1.05651** (PR #2135, DRAFT PR #2146, not merged) |
| Our submission | **REJECTED** (PR #771, train-then-score violation) |
| Upstream commits since close | 5 — all non-record/notable submissions |
| Issue #1872 (PPM-D legality) | No ruling — competition ended unresolved |

## Recommended Action

Competition is over. Three actionable items:

1. **Monitor PR #2146** — if the grace-policy audit merges, it reveals: (a) V22 lineage (AWQ-lite + AsymLogit Rescale) is the actual winning stack; (b) AsymLogit Rescale delivers ~0.003 bpb standalone; (c) GPTQ calibration batch count matters at the margin (0.001 bpb).
2. **Read arXiv:2604.06169** (In-Place NTP-aligned TTT) — directly applicable to future competition legal TTT design.
3. **Document lesson**: Data overlap audit (docs 10,000–49,999 train/val overlap) invalidated PR #2130 despite otherwise clean technique. Any future competition needs explicit validation-set isolation check before filing.

---

# Parameter Golf Daily Research - 2026-05-02 (POST-COMPETITION DAY 2)

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27) — Final

No change. @valerio-oai ruled train-then-score TTT violation. No appeal path.

## N-GRAM PR STATUS (Final)
- **PR #727**: CLOSED — @valerio-oai: hash key includes target token via XOR. Eval leakage. Final.
- **PR #731**: OPEN — seeds 1337/2024 never filed. Competition ended. Technique sound (dense Hedge Mixer + Laplace), reviewer said "LOOKS CLEAN", but never merged. Dormant.
- **PR #758**: OPEN but dead — same XOR target-token violation as #727. No organizer action pending.

## Leaderboard (FINAL)
| Rank | Score | Author | PR | Techniques |
|------|-------|--------|-----|------------|
| 1 | **1.0611** | codemath3000 | #1855 | BOS-Fixed SmearGate + LQER Asym + SparseAttnGate + 9-hparam greedy + lrzip |
| 2 | 1.0614 | aquariouseworkman | #1851/#1868 | SmearGate BOS Fix + PR#1787 + LQER Asym + Phased TTT |
| 3 | 1.0634 | nprime06 | #1787 | CaseOps + Polar Express NS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT |
| 4 | 1.0645 | dexhunter | #1769 | CaseOps + MLPClip12 + SmearGate + LoRA-TTT |
| 5 | 1.0655 | dexhunter | #1736 | CaseOps + GatedAttn + QuantGate + PhasedTTT |
| 6 | 1.0678 | romeerp | #1729 | CaseOps + Tapered WD + Phased TTT |
| 7 | 1.0714 | MarioPaerle | #1667 | SmearGate + Attention Output Gate + Legal TTT |
| 8 | 1.0719 | dexhunter | #1626 | VarLen Attn + Fused MLP + Multi-Phase Global SGD TTT |

No upstream/main commits since Apr 29. Leaderboard frozen.

## What Changed (Post-Competition, May 1–2 2026)

### New post-deadline PRs filed (no official record eligibility):

| PR | Author | Score | Technique | Notes |
|----|--------|-------|-----------|-------|
| #2130 | (anonymous) | **1.05670** | Token-Only N-gram Tilt + AsymLogit Rescale + 3 hyperparams from PR#2060 (MATRIX_LR=0.028, LQER_ASYM_GROUP=32, TTT_LORA_LR=8e-5) + NUM_PHASES=1 | Beats SOTA by only 0.00438 (below 0.005 threshold). Artifact 15.95MB. WITHIN_TAU=99.0/WORD_TAU=99.0 disables non-causal channels. AsymLogit Rescale from open PR#1923. |
| #2135 | codemath3000 | **1.05651** (3-seed) | PR#2130 base + GPTQ_CALIBRATION_BATCHES=32 (vs 16) | Paired t-test verified. −0.00457 vs SOTA — just misses 0.005 threshold. Filed post-deadline. Otherwise clean. |
| #2138 | (anonymous) | ~~0.979556~~ → **~1.0671** | Lock-In Byte Mixer (PPM-D gate λ=1−sigmoid(25·(PPM_conf−0.9999))) | **CONFIRMED BPB BUG** (@codemath3000): divides by CaseOps-transformed bytes (164,594,398) not raw-text sidecar (151,074,309). Corrected score ~1.0671 — worse than SOTA. Do NOT track. |
| #2139 | (anonymous) | **1.05749** | TTT Peer-LoRA Ensemble: blend peer docs' trained LoRAs for uncertain tokens (entropy≥0.5 threshold, ~75% activation) | Single seed, author filed "for fun." Novel technique. −0.00106 vs PR#2014 base. |
| #2140 | (anonymous) | **1.05601** | PR#2014 + LeakyReLU 0.3 + n-gram tilt (in-timer, strict causal) | Flagged by @codemath3000: within-word/word-start n-gram channels gate on `boundary_lut[tok]` (target-token-dependent). Same Rule 1 violation as PR#1420. Post-deadline regardless. |
| #2141–#2145 | various | mixed | Non-record or post-deadline exploration (MHALM V2 1.3477, CaseOps 1.07134, JEPA ablation, etc.) | Research filings, no competitive relevance. |

### BPB bug pattern note
PR #2138 is the 7th confirmed BPB bug in this competition (after #1545, #1576, #1687, #1698, PR#1848 risk, PR#1858 partial data). All involve extraordinary score claims later corrected by community review. Pattern: byte denominator manipulation or double-counting.

## New Research Papers (May 2 scan)

### High relevance (future competition)

| Paper | arXiv ID | Date | Key Technique | Impact |
|-------|----------|------|---------------|--------|
| Bell Box Quantization (BBQ) | 2603.01599 | ICLR 2026 | First ITO (information-theoretically optimal) + compute-efficient quantization. Hadamard + probability integral transform + uniform quantize. Up to 18 PPL improvement vs SOTA at 1-bit. | High — could replace or supplement GPTQ/LQER pipeline in future challenge. |
| EntroLLM | 2505.02380 | May 2025 | Entropy coding of quantized weights for edge models. 30% storage savings over uint8, 65% over uint4. | High — additive to lrzip artifact compression; directly relevant to 16MB budget. |
| In-Place TTT (NTP-aligned) | 2604.06169 | Apr 2026 | NTP-aligned objective for TTT instead of reconstruction loss; chunk-wise score-first updates; outperforms standard LoRA TTT on long contexts. | High — would improve legal TTT quality without legality risk. |
| Decoupling Tokenization Effects | 2604.27263 | Apr 2026 | Isolates "tokenization bias" — shows different tokenizers produce structurally different BPB distributions. | Medium — theoretical backing for CaseOps/casefold BPB debate. |

### Already tracked / not actionable
- arXiv:2505.16932 (Polar Express NS): Already in merged SOTA (PR #1787). ✓
- arXiv:2604.13552 (TF-TTCL): Training-free TTT via contrastive distillation — large-model focused, not applicable
- arXiv:2505.22857 (NGPU-LM): GPU n-gram LM for ASR context biasing — wrong domain
- arXiv:2504.04718 (T1): Self-verification for reasoning tasks — not compression-focused

## New Techniques for Future Reference

**AsymLogit Rescale** (PR #2130, open PR #1923):
- Replace single `logit_softcap=30.0` with two trainable scalars: `softcap_pos`, `softcap_neg`
- Parameters adapt via TTT global prefix pass
- Implementation: ~5 lines. Zero legality risk.
- Estimated gain: unknown standalone; super-additive with n-gram tilt in PR #2130

**TTT Peer-LoRA Ensemble** (PR #2139):
- After per-document LoRA training, run k−1 extra forwards with peer docs' LoRAs
- Blend `p = w·p_own + (1−w)·mean(p_peers)` only when predictive entropy ≥ threshold (0.5)
- ~75% of tokens activate ensemble; confident tokens use own prediction
- No cross-document information leak (each LoRA trained only on its own doc before scoring)
- Estimated gain: −0.00106 bpb standalone (small but could stack)

## Status Summary

| Item | Status |
|------|--------|
| Competition | **CLOSED** (April 30, 2026) |
| Final Merged SOTA | **1.0611** (codemath3000, PR #1855) |
| Our submission | **REJECTED** (PR #771, train-then-score violation) |
| Upstream commits since close | **0** — no activity |
| Post-deadline PRs | 10+ filed (non-record); no new techniques that beat SOTA legally |
| Issue #1872 (PPM-D legality) | No ruling — competition ended unresolved |
| PR #731 (Hedge Mixer) | Open, dormant — seeds never filed |

## Recommended Action

Competition is over. Priorities for any future challenge:

1. **Study PR #1855 code** — extract full CaseOps + LQER Asym + SparseAttnGate + SmearGate BOS-fix + lrzip stack as the canonical winning template.
2. **Implement AsymLogit Rescale** (PR #1923/2130) as a cheap addition to any future TTT stack — ~5 lines, no legality risk.
3. **Read arXiv:2604.06169** (In-Place TTT, NTP-aligned loss) — for improved legal TTT objective.
4. **Monitor Issue #1872** — if @valerio-oai ever rules on PPM-D, it determines whether the 0.9x BPB scores (PRs #1850, #1854, #1991) were legal paths and whether the technique should be a first-move in the next competition.
5. **Consider TTT Peer-LoRA Ensemble** (PR #2139) — novel direction with causal soundness; worth a GPU ablation if competing again.

---

# Parameter Golf Daily Research - 2026-05-01 (POST-COMPETITION)

## PR #771 STATUS: CLOSED (REJECTED 2026-03-27)

Same as prior days. No change. Final.

## N-GRAM PR STATUS (Final)
- **PR #727**: CLOSED — rejected by @valerio-oai (hash key includes target token = eval leakage). Final.
- **PR #731**: OPEN — competition ended without merge. Seeds 1337/2024 never filed. "LOOKS CLEAN" from reviewer but no organizer action. Technique (dense Hedge Mixer + Laplace) documented as sound.
- **PR #758**: OPEN but dead — same normalization violation as #727 (XOR key includes target). No organizer action needed; community ruled it out.

## Leaderboard (FINAL — competition closed April 30, 2026)
| Rank | Score | Author | PR |
|------|-------|--------|----|
| 1 | **1.0611** | codemath3000 | #1855 — BOS-Fixed SmearGate + LQER + SparseAttnGate + 9-Hparam + lrzip |
| 2 | 1.0614 | aquariouseworkman | #1851/#1868 — SmearGate BOS Fix + PR#1787 + LQER Asym + Phased TTT |
| 3 | 1.0634 | nprime06 | #1787 — CaseOps + Polar Express NS + MIN_LR + SparseAttnGate + FusedCE + Warm-A TTT |
| 4 | 1.0645 | dexhunter | #1769 — CaseOps + MLPClip12 + SmearGate/LoRA-TTT |
| 5 | 1.0655 | dexhunter | #1736 — CaseOps + GatedAttn + QuantGate + PhasedTTT |
| 6 | 1.0678 | romeerp | #1729 — CaseOps + Tapered WD + Phased TTT |
| 7 | 1.0714 | MarioPaerle | #1667 — SmearGate + Attention Output Gate + Legal TTT |
| 8 | 1.0719 | dexhunter | #1626 — VarLen Attn + Fused MLP + Multi-Phase Global SGD TTT |
| 9 | 1.0810 | bigbag | #1493 — SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT |

**Our submission (PR #771): REJECTED.** Final standing: none.

## What Changed (Post-Competition, May 1 2026)

Competition closed April 30. Multiple PRs filed on May 1 — likely for non-record track, research credit, or future reference:

- **PR #2118** (aquariouseworkman, **1.0435**): "Gated XSA + token-only n-gram tilt + LQER + AWQ-lite + asymmetric logit rescale + LeakyReLU 0.3 + no_qv TTT mask + 1-phase score-first TTT." ⚠️ **LEGALITY QUESTIONED**: Reviewer andrewbaggio1 flagged that full n-gram paths (WITHIN_BOOST, WORD_BOOST, AGREE_ADD_BOOST) were active in logs despite "token-only" claim (word_gate=2,891,588 non-zero). Author acknowledged this. Score may void if full n-gram paths are illegal. Post-deadline anyway.
- **PR #2124** (vaibhavmishra1, **1.05933**): CaseOps + Gated XSA + N-gram Tilt + LQER + AWQ-lite + g32/top4 retune. Combinatorial stack of public techniques. Post-deadline.
- **PR #2101** (OnlyJundong, **1.05845**): AWQ-lite + AsymLogit + GradCentral. Post-deadline.
- **PR #2100** (someone114514, **1.05807**): LongCtx No-QV Prefix3500. Post-deadline.
- **PR #2121** (Kbediako, **1.06099**): StageB v2 CaseOps TTT. Post-deadline.
- **PR #2119** (dexhunter, non-record): PR #1953 K+O-only TTT + QK_GAIN_INIT=5.35 — dexhunter's own ablation/research filing.

**Issue #1872 (PPM-D legality)**: No @valerio-oai ruling. Competition closed without resolution. PPM-D technique remains unruled for legality.

**Winning techniques stack (final analysis)**:
CaseOps bijective tokenizer + LQER Asymmetric + SparseAttnGate + SmearGate with BOS fix + Polar Express Newton-Schulz + MIN_LR=0.10 + lrzip compression + LoRA-TTT warm-start A + alpha=144

## New Research Papers

- **arXiv:2505.20633** — "Test-Time Learning for Large Language Models" (May 2026). TTL framework minimizing input perplexity on unlabeled test data for self-supervised domain adaptation. Could refine score-first TTT objective alignment. Complexity: medium (new loss function).
- **arXiv:2604.06169** — "In-Place Test-Time Training" (Apr 7, 2026). NTP-aligned objective for TTT (not reconstruction loss). 4B-param model outperforms standard TTT approaches on long contexts up to 128k. Distinguishes from Session 3's failed in-place attempt (which used reconstruction loss on MLP projections). Complexity: medium.
- **arXiv:2505.23884** — "LaCT: Test-Time Training Done Right" (2025). Large-chunk TTT (2K–1M tokens) for hardware utilization. Our Doc-TTT (PR #1560, chunk=48) is a smaller-chunk variant. Potential: larger chunks may improve TTT quality.
- **arXiv:2601.02875** — "Revisiting Data Compression with Language Modeling" (Jan 2026). Shows 3-bit representation achieves only slight compression-rate drop vs higher precision. Validates aggressive quantization direction.
- **arXiv:2402.02446** — LQER (Low-Rank Quantization Error Reconstruction). Confirmed ICML 2024 publication. Asymmetric variant used in competition-winning PR #1855 and #1797. Well-established technique.

## HuggingFace / Community Discoveries

No notable HuggingFace blog posts or model releases directly relevant to Parameter Golf post-competition. Community activity is now concentrated in the GitHub PR thread itself.

## Status Summary (Post-Competition)

| Item | Status |
|------|--------|
| Competition | **CLOSED** (April 30, 2026) |
| Final Merged SOTA | **1.0611** (codemath3000, PR #1855) |
| Our best submission | **REJECTED** (PR #771, train-then-score) |
| PR #731 (Hedge Mixer) | Open, seeds never filed, effectively dormant |
| Issue #1872 (PPM-D) | No ruling — competition ended without resolution |
| Post-deadline filings | 6+ PRs filed May 1 (non-record or late research) |

## Recommended Action

**Competition is over.** No SOTA-chasing actions are needed or possible within the official window.

Post-competition learning priorities:
1. **Study PR #1855 code** (winning submission) — extract the full CaseOps + LQER Asym + SparseAttnGate + SmearGate BOS-fix + lrzip stack for reference in future challenges.
2. **Monitor PR #2118** (aquariouseworkman, 1.0435) — if organizers rule the n-gram paths legal post-competition, "Gated XSA + N-gram Tilt + LQER + AWQ-lite" stack represents a ~0.018 bpb improvement over the winning submission. Worth understanding for future competitions.
3. **Read arXiv:2604.06169** (In-Place TTT with NTP-aligned objective) — cleanest TTT improvement not yet in competition stack.
4. **Monitor Issue #1872** — if @valerio-oai ever rules on PPM-D byte mixture, it would define whether the 0.9x BPB scores (PRs #1850, #1854, #1991) were legal paths or not.

---

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
