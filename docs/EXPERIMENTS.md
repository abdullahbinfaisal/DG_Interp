# EXPERIMENTS.md — the MVP plan for the Results section

Goal: a complete, defensible Results section for
`docs/TMLR_Journal_Submissions__1_.md` with the minimum number of experiments.

Companion to `CLAUDE.md` (blueprint, conventions, code-running policy). Together
those two files are the complete live spec; there is no third document.

This file absorbed and replaced the original handoff brief (`docs/context.md`,
deleted 2026-07-31). That brief was written against the superseded signed-R score
file, a τ_R of 0.8, and an 80%-split evaluation protocol, and its results tables
carried inverted mask labels — see §11 for the record of what it contained and why
none of its numbers should be reused. §10 preserves the method specs for the
experiments it defined that we chose not to run.

All numbers below are defined against **`processed/FINAL_ERM_ResNet_3300_T3.json`**
— source-domains-only, magnitude-R, full-domain deterministic scoring.

Status: **the MVP programme is complete.** Group A and stages B1–B5 have all run;
results are written up in `docs/RESULTS_DRAFT.md` with a patch list for the paper.
What follows is the plan as executed, kept because it records why each experiment
exists and what its expected outcome was before we saw the data.

---

## 1. Decisions that shape this plan

| Decision | Consequence |
|---|---|
| §5.3 is **descriptive**, not intervention-backed | No mass-matched E3/E4. R is presented as a structural property; the sign-enrichment confound is disclosed with its number. |
| §5.6 conflict computed from **D alone** | No new GPU passes, no `M.py`. `max_k D > τ_D` and `min_k D < -τ_D` over the class axis. |
| Table 4 = **plain per-class counts** | RSM/HIM/DCM/conflict-mass formalism deferred; §3.8 becomes future work. |
| Figures **reused** from `analysis/` where possible | Scores re-looked-up from the clean file. New renders only if a bucket lacks an example. |
| Scope: **ERM ResNet-50, checkpoint 3300, one SAE seed** | §5.7 (model/checkpoint profiles) is cut. Becomes a limitation. |
| τ_H = τ_R = **0.7**, τ_D = **1e-4** | Only value where the `log2/log3 = 0.63` bound argument holds, and the only one leaving the central harmful-invariant bucket viable (32 pairs; 6 at τ=0.9). |

---

## 2. Target structure of the Results section

Eight subsections collapse to five. The two merges are the substantive editorial
calls: old §5.4 (typology counts) and §5.8 (oracle interventions) are the same
result told twice, and old §5.5 is repurposed from a qualitative aside into the
paper's mechanism.

| New § | Heading | Result | Fed by |
|---|---|---|---|
| 5.1 | The sparse basis preserves classifier behaviour | Per-domain original vs SAE-reconstruction accuracy, micro and macro. Drops ≤0.1pp everywhere including sketch. Licenses ablation-based scores. | **B1** |
| 5.2 | Activation invariance does not imply discriminative usefulness | Of 1542 pairs above the support floor, 1269 (82.3%) are neutral, 135 (8.8%) supportive, 138 (8.9%) harmful; the high-H region spans all three regimes. Falsifies the premise invariance-based DG objectives rest on. | **A1** |
| 5.3 | Consistency of effect is a second, asymmetric axis | (a) only ~50% of high-H pairs clear τ_R; (b) 8/273 non-neutral pairs are low-H/high-R, so R filters *within* the invariant population; (c) **invariant support distributes (72.4%), invariant harm concentrates (36.0%)**, monotone in τ (2.0× → 2.4× → 3.2×); (d) 38.4% of *neutral* pairs are high-H/high-R, hence the magnitude gate. | **A1** |
| 5.4 | The typology has functional consequences | Per-bucket ablation with controls. Each row: pairs masked, Σ\|D\|, micro+macro per domain, margin over a size-matched \|D\|-stratified random control. Oracle labelling and the sign tautology disclosed inline. | **B2, B3, B4** |
| 5.5 | Domain-contingent effects concentrate in specific source domains | For low-R pairs, `argmax_d \|D_d\|` by sign and class, plus 1–2 concept figures. The paper's mechanism: if harmful concentrated concepts peak in `photo`, the model learned photorealistic cues that survive into sketch as interference. | **A2** |
| 5.6 | Concepts are not class-isolated | Count of concepts supporting one class while harming another; per-class conflict counts; the dog/horse chest example. The third leg of the abstract, and the reason all scores are class-conditional. | **A3, A4** |

Arc: the lens is faithful → invariance isn't enough → here is the missing axis →
the axis has functional teeth → here is the mechanism → and here is why
class-conditionality was necessary all along.

---

## 3. Set definitions and populations

At τ_D = 1e-4, τ_H = τ_R = 0.7, support floor 30 → 1542 pairs, 273 non-neutral:

```
S+_lo = { D > +τ_D , R <  τ_R }   concentrated support      54
S+_hi = { D > +τ_D , R >= τ_R }   distributed support       81
S-_lo = { D < -τ_D , R <  τ_R }   concentrated harm        103
S-_hi = { D < -τ_D , R >= τ_R }   distributed harm          35
```

Split further by H (rows H<τ / H≥τ, cols R<τ / R≥τ):

| | supportive (135) | harmful (138) |
|---|---|---|
| H<τ | 25 / 5 | 46 / 3 |
| H≥τ | 29 / **76** | 57 / **32** |

Bold = the two headline categories: **robust support** (76) and
**harmful invariant** (32).

**Support floor is not the leaking count filter.** `CLAUDE.md` §3 warns that
masking thin-support pairs leaks labels and buys ~2.7pp. Here the floor does the
opposite — thin-support pairs are *excluded from the mask*, i.e. left intact —
because R is an entropy over three per-domain estimates and a cell with four
images gives a meaningless `D_d`. Every bucket reports its pair count so the
restriction stays visible.

---

## 4. Code

| File | Role |
|---|---|
| `clean_lib/eval.py` | `MaskedAccuracyEvaluator`: micro **and** macro, predicted-label histogram, full-domain deterministic loading. Takes backbone + SAE directly, so no empty score template is needed to describe eval domains. |
| `clean_lib/masks.py` | `load_scores`, `build_buckets`, `stratified_random_control`, `conflict_analysis`, `domain_attribution`. **True = mask out** everywhere. |
| `scripts/analyze_scores.py` | Group A. Zero GPU. |
| `scripts/run_experiments.py` | Group B. `--stage B1..B4`, resumable, writes `results/E_*.json` + `summary.csv`. |

---

## 5. Group A — zero GPU

One command, ~1 minute, no crash exposure. Run this first: it changes how the
Group B results are read, and it produces two of the paper's figures.

```powershell
& $PYEXE scripts\analyze_scores.py --scores processed\FINAL_ERM_ResNet_3300_T3.json
```

| # | What | Output | Feeds |
|---|---|---|---|
| A1 | H×R contingency per D regime at τ ∈ {0.7,0.8,0.9}; the asymmetry table; Fig 1 (H vs D); Fig 5 (H×R heatmap) | `typology_counts.csv`, `asymmetry.csv`, `fig_h_vs_d.png`, `fig_hr_heatmap.png` | §5.2, §5.3 |
| A2 | `argmax_d \|D_d\|` for low-R pairs, by sign and class | `domain_attribution.{json,csv}` | §5.5 |
| A3 | Class conflict from D; per-class counts; concepts ranked by `max−min` contrast | `conflict.{json,csv}` | §5.6 |
| A4 | Per-class support / supportive / harmful / conflicted, optionally joined with sketch accuracy | `per_class_counts.csv` | Table 4 |
| A5 | Pair counts and Σ\|D\| per bucket, plus the E4 mass-match feasibility check | `bucket_masses.csv` | control inputs |

**Expected.** A1 should reproduce the asymmetry already observed (72.4% / 36.0%
at τ=0.7, strengthening to 49.4% / 15.4% at τ=0.9). A2 is the genuine unknown and
the most interesting single output in the plan. A3's conflict count is unknown;
the original brief never measured it.

---

## 6. Group B — GPU, ~20 minutes, 34 configurations

Run **one stage at a time**. Each is independently resumable; existing results are
skipped unless `--force`, so a crash costs only the configuration in flight.

Smoke-test the path once before the first real stage:

```powershell
& $PYEXE scripts\run_experiments.py --stage B2 --limit-batches 2 --out-dir results\_smoke
```

### B1 — baseline and reconstruction fidelity  [1 min, 2 configs]

```powershell
& $PYEXE scripts\run_experiments.py --stage B1
```

**What.** Original backbone, then SAE reconstruction with nothing masked. Micro
and macro, per class, all four full domains.

**Why.** Table 1, and the denominator for every Δ in the paper. **Must run before
B2/B3/B4** — those stages read `E_B1_sae_reconstruction.json` to compute deltas.

**Expected.** Sketch micro ≈ 80.2–80.3 and macro ≈ 83.4–83.5; sources 99.0–99.8;
reconstruction drop ≤ 0.1pp per domain. Micro should reproduce Table 2's 80.29 —
verified analytically already (weighting per-class sketch accuracies by PACS
counts gives 80.30).

**Watch for.** Full-domain evaluation now includes the 20% of source images
previously excluded from scoring, so a shift of a few tenths from the notebook's
83.4x is expected. A larger shift needs explaining before proceeding.

### B2 — bucket interventions  [4 min, 8 configs]

```powershell
& $PYEXE scripts\run_experiments.py --stage B2
```

**What.** Mask each of: `all_harmful`, `all_supportive`, `S-_lo`, `S-_hi`,
`S+_lo`, `S+_hi`, `harmful_invariant`, and `keep_robust_support` (keep-only).

**Why.** Gives every typology row a functional signature and fills Table 6,
including the "test robust sufficiency" cell that is currently empty.

**Expected.** `all_supportive` collapses toward 14.29% with predictions piling
onto `person`. `all_harmful` improves sketch substantially. `keep_robust_support`
is the row to watch: the archived (contaminated, four-domain) run gave sketch
46.6% macro with elephant/guitar/house at exactly 0.00%, which suggests robust
support alone is **not** sufficient to classify — a genuinely important negative
result if it holds. Several single-bucket masks will sit near baseline simply
because the buckets are small (`S-_hi` = 35, `harmful_invariant` = 32); report
pair counts alongside so small-set nulls are not read as substantive.

### B3 — stratified random controls  [12 min, 24 configs]

```powershell
& $PYEXE scripts\run_experiments.py --stage B3
```

**What.** For each B2 bucket except the keep-only one, a size-matched,
|D|-histogram-matched random set drawn from the support-floored pairs the target
did *not* select. Three seeds.

**Why.** **This is the experiment that makes B2 publishable.** Without it a
reviewer can say that ablating any comparable set of concepts helps a weak
domain, and they would be right to. Stratification matters: 1269 of 1542
support-floored pairs are neutral, so a uniformly-sampled control is trivially
easy to beat and proves nothing.

**Expected.** Uniform-random would be ≈ no change; the stratified control should
capture a real fraction of each bucket's effect. **The honest claim in §5.4 is
whatever margin survives this comparison, not the raw Δ against baseline.**

**If the control matches the target**, §5.4 shrinks to a much more modest claim
and §5.3 carries the paper alone. That is survivable — §5.3 is descriptive by
design — but it changes the section's shape, which is why B3 runs immediately
after B2 rather than last.

### B4 — inert-pair denoising control  [1 min, 1 config]

```powershell
& $PYEXE scripts\run_experiments.py --stage B4
```

**What.** Mask only the pairs with `R == 0` — 111,530 of 114,688 — i.e. those with
no measured effect in any source domain, or that never fire for that class.

**Why.** The largest unexamined threat to the low-R story, and flagged as a TODO
in the draft's §5.3. The old R-sweep gained +3.35pp by τ=0.1, where nearly
everything removed is inert. If masking the inert set alone reproduces that, a
large share of the effect is **dictionary denoising**, not a claim about
concentrated concepts.

**Expected.** Near-neutral if these pairs are genuinely inert. If not, the low-R
framing needs requalification and any sweep must be reported gated on |D| > τ_D.

---

## 7. Deliberately excluded

| Excluded | Why it is safe to skip |
|---|---|
| Mass-matched E3/E4 | Only needed to claim R is *causally* independent of D's sign. §5.3 is descriptive, so the claim is never made. Revisit if a reviewer pushes. |
| R-threshold sweep (E11) | Cheap in GPU but drags in the label-free variants and gated sweep to honour three open TODOs. B2+B4 cover the same ground with fewer claims. |
| Label-free variants (E7/E8) | Both known to be null. Reported as a stated finding in §5.4 rather than a new experiment. |
| τ_D = 1e-3 sensitivity | A1 already emits τ ∈ {0.7,0.8,0.9}; the τ_D axis can be added later with one flag. |
| RSM/HIM/DCM profiles (E10) | Experimental per decision 7, and n=7 cannot support the correlation. `dog` fits the story (50 harmful pairs, worst accuracy 48.71%); `person` contradicts it (32 harmful, 94.62%). |
| Additional SAE seeds (E13) | Out of scope. Named as a limitation; the tied USAE gives *cross-checkpoint* alignment, which is a partial substitute worth documenting in §4.2. |
| §5.7 checkpoint profiles | Single checkpoint. Cut. |

---

## 8. Risk register

**8.1 The sign tautology.** Masking negative-D pairs must raise `p(y)` — that is
how D is defined. No row showing "masking harmful concepts helps" is informative
alone. Only three things carry information: the **magnitude**, the **source→target
asymmetry**, and the **margin over B3's controls**. This must be stated in §5.4,
not buried.

**8.2 R as a proxy for the sign of D.** Quantified: the harmful population is 2.0×
enriched in low-R at τ=0.7. Handled by framing §5.3 descriptively and disclosing
the number. Not resolved — resolving it needs E3.

**8.3 Denoising rather than structure.** 111,530 of 114,688 pairs have R exactly 0.
B4 measures this directly.

**8.4 Oracle masks throughout.** Every 2-D mask uses the true label. Disclosed
inline in §5.4 and in the limitations; both label-free variants tried so far are
null, and the paper should say so plainly rather than omit it.

**8.5 Small cells.** `harmful_invariant` is 32 pairs at τ=0.7 and 6 at τ=0.9;
two typology cells hold 3 and 5. Every per-cell claim needs its n inline.

**8.6 Stale numbers.** Every masking figure in the deleted brief and in old `CLAUDE.md` §8
came from signed-R on the 80% split. Superseded. Do not copy into the paper.

**8.7 Single model / checkpoint / seed / dataset.** Scope decision, not a defect —
but state the cap rather than letting a reviewer find it.

---

## 9. Order of operations

```
A   analyze_scores.py                 zero GPU, ~1 min, run first
      -> figures + all score-derived tables + A5 control inputs
B1  baseline                          must precede B2/B3/B4 (delta denominator)
B2  bucket interventions
B3  stratified controls               decides how strong §5.4 can be
B4  inert control
      -> hand results/summary.csv + results/analysis/ over for the write-up
```

Write-up order once results are in: 5.1, 5.2, 5.3 (all independent of Group B
except 5.1), then 5.5 and 5.6 (Group A only), then 5.4 last — because §5.4's
strength depends on B3, and its framing determines how much weight §5.3 has to
carry.

---

## 10. Deferred experiments — method specs

Kept so these are buildable without re-deriving them. None is needed for the MVP
draft; each is the answer to a specific reviewer objection.

Notation: sets as defined in §3, scores from source domains only.

### E3 / E4 — mass-matched bucket comparison

**Answers:** "is R doing anything beyond the sign and magnitude of D?" — the
conflation acknowledged in `RESULTS_DRAFT.md` §5.4.

1. Compute `M_hi = Σ|D|` over `S−_hi`, **per class**.
2. Select `S−_lo,matched ⊆ S−_lo` whose per-class `Σ|D|` equals `M_hi`. Two
   selection rules: (a) descending `|D|` until the target is met, (b) random to
   target, 5 seeds.
3. Mask `S−_lo,matched`; compare against masking all of `S−_hi`.
4. Report the masses actually achieved — exact matching is impossible.

Interpretation, to be committed to before running: **equal performance** → R adds
nothing beyond `|D|` and sign, and §5.3 must stay strictly correlational;
**low-R subset wins** → concentration carries independent information.

E4 is the same construction on `S+_lo` vs `S+_hi`. **Direction matters:** A5
established that `S+_lo`'s mass is below `S+_hi`'s in *every* class
(e.g. dog 5.87e-3 vs 6.25e-2), so the match must subset `S+_hi` **down** to
`S+_lo`'s mass. Matching the other way is infeasible and would silently produce an
unmatched comparison.

### E7 — predicted-label mask

**Answers:** "these are all oracle masks; what happens without the true label?"

1. Forward pass with no mask, record predicted label `ŷ_i`.
2. Build the mask using `R(ŷ_i, ·) < τ_R` instead of `R(y_i, ·) < τ_R`.
3. Re-evaluate, reporting overall accuracy **and** accuracy split by whether
   `ŷ_i == y_i` on the first pass. The split is the informative part.

Under a leakage account this gains little (correct predictions reinforced, wrong
ones entrenched); under a real-structure account it recovers a meaningful fraction,
since `ŷ = y` on ~80% of sketch images.

### E8 — argmax-|D| label-free mask

For each concept `c`, let `k* = argmax_k |D(k,c)|`. If `R(k*,c) < τ_R`, mask `c`
globally for every image regardless of label. Report pairs masked, for comparison
against the ~15k of the all-classes intersection rule and the ~114k of the oracle
rule. Less conservative than the intersection rule, which was null.

### E11 — R-threshold sweep

Sweep the masking threshold on R from 0.0 to 0.9 in steps of 0.1, three variants:
ungated (`R < τ`), gated (`R < τ ∧ |D| > τ_D`), harmful-only
(`R < τ ∧ D < −τ_D`). Per-domain micro and macro plus pairs masked per threshold
per class. The *shape* matters as much as the peak — a monotone rise would suggest
ablation simply helps a weak domain, whereas a turnover suggests the mask selects a
specific population.

Note: B4 already showed inert-pair masking is null (+0.08), so the low-τ end of the
old sweep is explained without rerunning it.

### E12 — threshold sensitivity

`τ_D ∈ {1e-4, 1e-3}`, `τ_H`, `τ_R ∈ {0.7, 0.8, 0.9}`, support floor on/off.
`scripts/analyze_scores.py --tau-d 1e-3 --taus 0.7 0.8 0.9` regenerates every
§5.2/5.3 number in about a minute. The τ ∈ {0.7,0.8,0.9} axis is already reported
(the asymmetry is monotone: 2.01× → 2.39× → 3.21×); only the τ_D axis is open.

### E13 — additional SAE seeds

Dictionaries are not comparable across independently trained SAEs, so compare only
aggregate quantities: heatmap proportions, typology masses, the E3 outcome. Out of
scope; named as a limitation. Worth noting in the paper's §4.2 that the tied
multi-checkpoint USAE aligns dictionaries across steps 2100 and 3300 *by
construction*, which is a partial substitute and is currently undocumented.

### Result file schema

`scripts/run_experiments.py` writes one `results/E_<id>.json` per configuration
with keys: `experiment_id`, `stage`, `description`, `mask_definition`, `oracle`,
`oracle_note`, `tau_D`, `tau_H`, `tau_R`, `support_floor_stats`,
`count_filter_in_forward_pass`, `pairs_masked{total,per_class}`,
`masked_D_mass{total,per_class}`, `accuracy{domain:{micro,macro,per_class}}`,
`counts`, `predicted_label_histogram`, `seed`, `smoke`, `notes`, and `bucket`
(bucket size vs masked size, which differ for keep-only masks). Plus one flat row
per configuration in `results/summary.csv`. The code is the spec.

---

## 11. Record of the deleted handoff brief

`docs/context.md` was the original E0–E13 brief. Deleted 2026-07-31 after its
content was absorbed here. Recoverable from git history if ever needed. It is
recorded rather than silently dropped because **its results tables are dangerous**:
the numbers were correct but the labels inverted, so it reads as a plausible set of
findings that mean something different from what they say.

**The inversion.** Its tables described masks like "Mask `R < 0.8 ∧ D < 0`". The
notebook cells behind them chained `Analyzer.filter(...)`, which **keeps** matching
pairs, then `get_mask()`, which returns the **complement**. So every such row was a
keep-only intervention over ~114k pairs, not a targeted mask:

| Label in the brief | What actually ran | pairs masked |
|---|---|---|
| "Mask `R < 0.8 ∧ D < 0` → 93.47 (+10.03)" | keep only `{R ≥ 0.7 ∧ D > 1e-4}` | 114,609 |
| "Mask `R < 0.8 ∧ D > 0` → 13.41 (collapse)" | keep only `{R ≥ 0.7 ∧ D < −1e-4}` | 114,654 |
| "Mask `R < 0.8` → +6.13 / +7.61" | keep only `{R ≥ 0.7}` | 113,953 |
| "Mask all `D < 0` → 96.66 (+13.20)" | mask all `D < 0`, ungated, no support floor | ~tens of thousands |

Read correctly, the headline row reproduces: 93.47 then, **94.02** now
(`keep_robust_support`, `RESULTS_DRAFT.md` Table 4). The gated, support-floored
buckets in this plan mask 32–138 pairs and are **not comparable** to any of the
above.

**Its two other defects.** Scores came from
`processed/NEW_ERM_ResNet_3300_T3.json`, which holds legacy signed R with 349
`R = −1` sentinels (one per mixed-sign pair) folded into every "low-R" population;
and evaluation used the 80% shuffled split with `drop_last=True`, dropping a
different ~7 sketch images per call for ±0.08pp of jitter.

**Its secondary hypothesis was refuted.** It proposed that low-R concepts are "not
junk" and that the model's class evidence "substantially consists of
domain-contingent concepts". The opposite holds — see `RESULTS_DRAFT.md` §5.4 and
`CLAUDE.md` §1.

**What it got right and is preserved:** the micro/macro diagnosis (confirmed,
80.25 micro vs 83.56 macro on sketch); the warning that the class-conditional count
filter leaks labels and must never touch a forward pass (`CLAUDE.md` §3); the
insistence on reporting pairs masked and Σ|D| mass on every row; the requirement for
predicted-label histograms near chance; and the instruction never to average the
four domains together. All are now protocol invariants in §0.
