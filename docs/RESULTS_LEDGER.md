# Results ledger — findings established beyond the ERM-only draft

`docs/DISCUSSION.md` covers what the ERM-only measurement already establishes.
`docs/DIRECTIONS.md` lists claims the results point at but do not yet establish.
This file is for findings that cross that line **after** the ERM-only draft —
each entry names the hypothesis, the evidence, the verdict, and what scope it
does and does not cover. Newest first within each finding is not the model;
findings are added as they're settled, oldest first.

Source data: `processed/FINAL_ERM_ResNet_3300_T3.json`,
`processed/FINAL_MMD_ResNet_1800_T3.json`,
`processed/FINAL_DANN_ResNet_5000_T3.json`. Run history:
`docs/RUNLOG.md` 2026-08-26/27 entry. Raw comparison output:
`results/W1_backbone_comparison.json`.

---

## F1. Explicit invariance training does not reduce harmful-invariant concepts (D1 closed)

**What was tested.** `docs/DIRECTIONS.md` D1 / `docs/ABLATIONS_AND_EXPERIMENTS.md`
T3.1 — the paper's own named falsification test (§7): *"our argument predicts
that a model trained with an explicit alignment objective should retain
comparably many invariant-and-harmful concepts... testing it is the most
direct way to falsify the paper's central claim."* Extended beyond D1's
original MMD-only scope to a second alignment objective, DANN, at the user's
request.

**Method.** Same SAE geometry, same thresholds (τ_D=1e-4, τ_H=τ_R=0.7, support
floor=30) as the ERM headline numbers — reused rather than re-derived per
backbone, so any difference is attributable to the model, not to re-tuned
cutoffs. Non-oracle top-1 checkpoint per backbone (MMD step 1800, DANN step
5000), each with its own independently-trained, standalone SAE (not a tied
USAE — see the scope note in F1c). Deciding evidence pre-committed before
running: a Fisher exact test on the harmful-invariant retention rate, ERM vs.
each other backbone.

**Result.**

| comparison | ERM rate | other rate | odds ratio | p | verdict |
|---|---|---|---|---|---|
| harmful-invariant, ERM vs MMD | 36.0% (32/89) | 51.7% (45/87) | 0.524 | 0.048 | MMD significantly *higher* — comparably many |
| harmful-invariant, ERM vs DANN | 36.0% (32/89) | 28.6% (16/56) | 1.4 | 0.372 | not significant — comparably many |

Raw bucket sizes: `harmful_invariant` = 32 (ERM), 45 (MMD), 16 (DANN).

**Verdict.** Neither alignment objective significantly reduces the
harmful-invariant bucket. MMD retains **significantly more** of it. DANN's
lower raw count is not statistically distinguishable from ERM's rate given
DANN's smaller harmful population (n=56 vs. ERM's n=89) — "comparably many,"
not "fewer," is the honest call. Per D1's own pre-registered outcome branches,
this is the strong-claim branch: *"the paper's central argument is
demonstrated, not merely asserted."* §7's hedge — *"that prediction is untested
here"** — converts to a measured result.

**Why this holds up mechanistically, not just statistically.** MMD and DANN
both align feature *statistics* during backbone training, before any SAE
exists. The harmful-invariant concepts this measures only become visible after
decomposing the frozen features with a sparse dictionary trained *afterward*.
There is no mechanism by which the alignment loss could have targeted these
specific directions — it operates on the feature space in aggregate; harm is a
property of individual directions within it, invisible to an aggregate
statistic. This generalizes past MMD/DANN specifically to any objective that
matches activation distributions without reference to the downstream task loss
along individual directions — it does **not** extend to invariance methods
that tie their penalty to prediction risk instead (e.g. IRM), which is a
mechanistically different approach the paper doesn't test.

---

## F1a. The §5.3 support/harm asymmetry itself does not replicate uniformly

**What was tested.** A secondary Fisher test (not the deciding one for F1) on
whether the supportive-side asymmetry — invariant support tends to distribute
across domains — holds at the same strength in MMD and DANN.

**Result.**

| comparison | ERM rate | other rate | p | direction |
|---|---|---|---|---|
| supportive-invariant, ERM vs MMD | 72.4% (76/105) | 96.4% (428/444) | 2.4e-12 | MMD stronger |
| supportive-invariant, ERM vs DANN | 72.4% (76/105) | 53.6% (60/112) | 0.005 | DANN weaker |

**Verdict.** Mixed, and real in both directions (both p-values are decisive,
not noise). MMD's supportive concepts distribute across domains even more
uniformly than ERM's. DANN's distribute *less* uniformly — a genuine
attenuation of the asymmetry, not present in MMD. If this goes into the paper,
state both directions rather than only the confirming one.

---

## F1b. MMD's supportive population is far larger, individually more decisive, and its removal breaks source-domain accuracy too

**Observation.** At the identical τ_D=1e-4, `all_supportive` is 135 pairs
(ERM), 561 (MMD, 4.2×), 148 (DANN). `S+_hi` is 81 (ERM), 491 (MMD, 6×), 74
(DANN). Masking MMD's `all_supportive`/`S+_hi` collapses not only sketch but
**source-domain accuracy too** (e.g. `all_supportive`: art_painting 99%→18%,
cartoon 99%→14%, photo 99%→25%) — a real departure from the property
`docs/DISCUSSION.md` §7 establishes for ERM, that every intervention leaves
source domains within about one point. DANN shows the same pattern even more
sharply on a *smaller* bucket (`S+_lo`, only 74 pairs, still drops every source
domain by 20+ points).

**Is this a threshold-scale artifact or a real property of the model?**
Partial evidence for "real": `all_harmful` bucket sizes are nearly identical
across backbones at the same τ_D (138 / 142 / 91) — if the fixed τ_D were
simply landing at a very different percentile of MMD's D-distribution (a
uniform scale-shift artifact), harmful would have inflated symmetrically with
supportive, and it didn't. Mean |D| per pair in the bucket is also larger for
MMD/DANN, not smaller, on both signs:

| | ERM | MMD | DANN |
|---|---|---|---|
| harmful, mean \|D\|/pair | 0.00181 | 0.00323 (1.8×) | 0.00208 (1.2×) |
| supportive, mean \|D\|/pair | 0.00194 | 0.00467 (2.4×) | 0.00388 (2.0×) |

A pure dictionary-granularity artifact (more, smaller atoms splitting the same
total effect) predicts smaller per-pair magnitude as pair count grows;
instead both go up together.

**Verdict: suggestive, not settled.** This is *not* a finding to state as
established fact in the paper yet. The decisive test — training a second,
differently-seeded SAE on the same MMD checkpoint (the D8 method, scoped to
MMD only) and checking whether `all_supportive` size and the
source-domain-fragility result replicate — has not been run. Track as a
`docs/DIRECTIONS.md`-style open item before this claim is written up as more
than "worth investigating."

---

## F1c. Scope limit: this is an aggregate-level result, not a latent-level one

D1's original spec preferred a tied USAE spanning ERM+MMD checkpoints (which
`sae.py`'s `SparseAEs` supports, and which the existing
`USAE_ERM_Multi_test_3300_2100.pt` demonstrates for two ERM checkpoints),
specifically so individual latents would be comparable across backbones, not
just aggregate bucket counts. What was actually run uses three independently-
trained, untied SAEs (the MMD and DANN ones trained standalone). F1's
conclusion holds at the aggregate level — bucket sizes, rates, the Fisher
test — but **not** at the level of "the same specific concept that was
harmful-invariant in ERM is still harmful-invariant in MMD." That is the same
category of caveat D8 exists to address (SAE non-identifiability across
seeds/dictionaries), encountered here across backbones instead of across
seeds on one backbone. State this scope limit alongside F1 if it goes in the
paper.

---

## F2. R carries information beyond the sign and magnitude of D (D7, mostly settled)

**What was tested.** `docs/DIRECTIONS.md` D7 — is the R-vs-D conflation the
paper discloses four times (§10, and repeatedly as the "most-repeated caveat")
real, or does consistency just track effect magnitude? Per D7's own design
caveat (not the sum-matched E3/E4 spec as originally written — see
`docs/DISCUSSION.md` §5's point that aggregate mass is a poor predictor of
consequence), the comparison matches the **per-pair |D| distribution**, the
same way `stratified_random_control` already does for random controls.

**Method.** New `clean_lib/masks.py:distribution_matched_subset()`: for a
target bucket and a named candidate pool (not "everything else," a specific
opposite-R bucket), draw a same-size, same-|D|-shape subset. Compare masking
the true bucket against masking its matched-but-opposite-R counterpart, 3
seeds. Run on all three backbones (ERM, MMD, DANN), both directions:

- **Harmful (E3):** target=`S-_hi` (true, high-R), pool=`S-_lo` (draw matched
  subset from here — harm is enriched at low R, so this pool is larger).
- **Supportive (E4):** target=`S+_lo` (true, low-R), pool=`S+_hi` (direction
  reversed — `S+_lo`'s mass is below `S+_hi`'s in every class).

**Result — harmful side, clean and consistent, 3/3 backbones:**

| backbone | S-_hi (true, high-R) Δmacro | matched low-R subset Δmacro (mean of 3 seeds) | ratio |
|---|---|---|---|
| ERM | +3.68 (35 pairs) | +1.32 (35 matched) | 2.8x |
| MMD | +2.51 (48 pairs) | +1.07 (42 matched) | 2.3x |
| DANN | +2.46 (16 pairs) | +0.45 (16 matched) | 5.5x |

The true high-R bucket recovers 2.3–5.5x more accuracy than a magnitude-matched
low-R subset of the same size, same direction in every backbone tested. **This
directly contradicts the paper's own §10 hedge** ("we do not claim R carries
information independent of D's sign and magnitude") on the harmful side.

**Result — supportive side, 2/3 clean, DANN unresolved:**

| backbone | S+_lo (true, low-R) Δmacro | matched high-R-pool subset Δmacro | note |
|---|---|---|---|
| ERM | -0.65 (54 pairs) | -5.39 (42 matched) | matched costs 8x more |
| MMD | -0.15 (70 pairs) | -1.54 (70 matched) | matched costs 10x more |
| DANN | **-18.77** (74 pairs) | -1.15 (**30 of 74 matched**) | reversed, undercovered |

ERM and MMD agree with the harmful-side story: drawing from the high-R pool,
even magnitude-matched down, is far more costly to remove than the real
low-R bucket. DANN's comparison inverted, but the matcher only found 30 of
74 needed pairs (DANN's `S+_hi`/`S+_lo` are equal-sized overall but very
differently distributed per class, so several classes ran out of candidates)
— and DANN's `S+_lo` was already the anomalous bucket in F1b (it collapses
*source*-domain accuracy too, unlike any other backbone's supportive bucket).
**Read this as DANN's existing peculiarity resurfacing under a small,
compromised sample, not as a clean counter-finding.**

**Verdict.** D7 is settled on the harmful side: R does independent causal
work beyond D's magnitude, replicated cleanly across three backbones. On the
supportive side it's settled in ERM and MMD, unresolved in DANN — worth a
larger/better-covered rerun if DANN's supportive concepts get investigated
further (see F1b), but not urgent on its own.

---

## Open items this ledger creates

- **F1b's seed-robustness check** — train a second MMD SAE, different seed,
  compare `all_supportive` size and source-domain fragility. Settles whether
  F1b is a property of MMD or of one dictionary.
- **τ_D sensitivity for MMD/DANN** — the ERM τ_D sweep (`CLAUDE.md` §9.6)
  showing τ_D is a "nuisance parameter" has not been repeated for MMD/DANN.
  Cheap (bucket-size-only sweep is read-only); would directly test whether
  F1b's bucket-size gap survives away from τ_D=1e-4.
- **DANN's supportive side keeps behaving anomalously** (F1b's source-domain
  collapse, now F2's inverted/undercovered E3E4 result). Two independent
  symptoms of the same underlying thing is more than coincidence — worth a
  closer look at DANN's `S+_lo`/`S+_hi` structure specifically before either
  result goes in the paper as-is.
