# ABLATIONS_AND_EXPERIMENTS.md — closing the gap to acceptance

## 1. Rating the current draft

**Overall: a well-executed single-setting paper with one unmeasured causal claim
sitting at its center.** The measurement layer (H/D/R, the typology, the
interventions with matched controls) is careful and the writing is honest about
its own limits — probably more so than the median TMLR submission. The risk is not
that anything in the draft is wrong; three independent passes (this repo's own
audit trail in `CLAUDE.md` §1/§5, and `docs/critique.md`'s lookups) already caught
and fixed the errors that existed. The risk is **scope**: every result is one
model, one checkpoint, one SAE seed, one dataset, oracle-only, and the paper's
most quotable sentence (the MMD prediction in §7) is stated as a hypothesis the
paper does not test. A TMLR action editor will send this to reviewers who ask "but
does this hold up anywhere else," and right now the honest answer is "we don't
know" for every axis at once.

| Dimension | Read | Why |
|---|---|---|
| Novelty / framing | Strong | Class-conditional, three-axis scoring of SAE concepts against DG invariance is a genuinely new angle; §2 positions it correctly against IRM/CORAL/MMD theory and against concept-interpretability work. |
| Internal rigor | Strong | Matched random controls (§5.4), the tautology-of-D disclosure, the R = 0 denoising control, the below-floor mass check — this is more adversarial to its own claims than most interpretability papers. |
| Statistical backing | Good | Fisher exact tests on the two headline claims (p = 0.002, p < 1e-6); sensitivity swept on all three thresholds. Weak point: small-cell claims (the harmful-invariant bucket at n = 32, the domain-attribution table at n = 35 / n = 81) are flagged but not bootstrapped. |
| Generality / external validity | **Weak — the central risk** | One backbone, one training objective (ERM), one checkpoint, one SAE seed, one dataset (PACS), oracle labels throughout. §7 names all five gaps; none is closed. |
| Causal vs. correlational claims | **Mixed** | The interventions are genuinely causal (ablate, measure). But the paper's *framing* claim — invariance training doesn't reduce harmful-invariant concepts — is argued, not measured (D1 below). The domain-attribution "style" mechanism (§5.6) is explicitly labeled correlational. The redundancy-collapse conjecture in the Conclusion is asserted from one comparison (source vs. sketch sensitivity to the same ablation), not measured directly. |
| Writing / structure | Strong | Already passed a structural revision (`docs/critique.md`); vocabulary is consistent, bucket predicates are stated once, tables carry their own denominators. |
| Reproducibility | Strong | `manifest.json` per run, deterministic full-domain evaluation, resumable stages, score-file provenance documented in `CLAUDE.md` §5. |

**The single highest-leverage fact about this draft:** its own §7 already predicts
the falsifying experiment (D1 below) and calls it "the most direct way to falsify
the paper's central claim," then does not run it. A reviewer who reads carefully
will ask for exactly that experiment. Doing it before submission, rather than in
response to a review, converts a hypothesis into a result and is worth more than
any other item on this list.

### Concrete weaknesses, numbered for reference below

1. **W1 — The central causal claim is a prediction, not a measurement.** §7: *"our
   argument predicts that a model trained with an explicit alignment objective
   should retain comparably many invariant-and-harmful concepts... That prediction
   is untested here, and testing it is the most direct way to falsify the paper's
   central claim."* `PACS_ResNet_Sketch_Test_Only/MMD_ResNet_T3` is already on disk.
2. **W2 — Every intervention is oracle (true-label) and the paper says so, but
   offers no positive label-free result.** §7 ¶2: *"the label-free variants we
   examined... are null, but they also discard the class-conditionality... so they
   are not a decisive test."* A reviewer's first question about a diagnostics paper
   is "is this deployable," and the current answer is "we didn't find one, but our
   attempts weren't fair tests."
3. **W3 — R vs. |D| magnitude conflation is disclosed four times and resolved
   zero times.** §7 ¶4 (final limitation): *"we do not fully separate consistency
   from effect magnitude... a mass-matched intervention would be required to settle
   it."* The mass-*normalized* comparison in §5.4 (Finding 2) is a partial answer
   the paper itself calls insufficient.
4. **W4 — Single SAE seed.** §3.2 and §7 ¶3 both note SAE dictionaries are not
   identifiable across seeds/init/hyperparameters — a standard, expected objection
   for any SAE-based result — and the paper does not empirically address it, only
   argues from prior work that aggregate quantities should be more stable than
   individual latents.
5. **W5 — Single dataset, checkpoint, architecture.** Named in §7 ¶3 as scope, not
   defect, but combined with W1 and W4 this is three "single X" limitations stacked
   on the same section, which reads as under-tested rather than well-scoped once a
   reviewer starts listing them.
6. **W6 — The §5.3 asymmetry is support-floor-dependent in magnitude (2.49 → 1.31
   across floors 10 → 100, per `CLAUDE.md` §9.6/E12), and the stratified control
   that would convert this disclosure into a defense (E14) has not been run.** This
   is the cheapest open item in the entire project — read-only, no GPU — and it
   directly defuses a "your asymmetry is a support artifact" review comment before
   it's made.
7. **W7 — The Conclusion's strongest sentence is underdetermined.** *"Redundancy,
   rather than invariance, may be what degrades under shift"* rests on one
   comparison (mean |D_d| across three source domains happens to rank-order with
   domain difficulty, per `docs/DISCUSSION.md` §4) that the paper does not even
   report in the main text — it is stated more strongly in the Conclusion than the
   evidence for it appears anywhere in Results.
8. **W8 — The domain-attribution mechanism (§5.6) is explicitly hedged** as *"a
   coherent reading rather than a demonstrated mechanism"* and the enrichment
   ratios it rests on partially reflect exposure/magnitude differences across
   domains rather than pure concentration (§5.6 already half-controls for this with
   a null row, but not fully).
9. **W9 — No answer to "is the residual error fixable."** After `keep_robust_support`
   the model still misses ~6pp of sketch. The Discussion (§6.2) claims the framework
   "distinguishes" a coverage problem from a selection problem but never actually
   performs that diagnosis on the residual.
10. **W10 — Per-class conflict burden is acknowledged non-monotone** (dog worst
    accuracy *and* most harmful pairs; person second-best *and* second-most harmful)
    with n = 7 classes. Correctly left uncorrelated rather than forced into a
    profile score (`DIRECTIONS.md`'s deprioritized RSM/HIM/DCM) — not a gap to close,
    but worth a one-line preemptive framing so it doesn't read as an unexplained
    loose end.

---

## 2. Experiments, easy → hard

Every entry names which weakness (W1–W10) it closes. Costs are wall-clock on this
machine, GPU vs. not is called out. "Code" means whether it's a read-only pass over
`processed/FINAL_ERM_ResNet_3300_T3.json`, a new evaluation config, or new training.

### Tier 0 — read-only, no GPU, minutes

#### T0.1 — E14: support-stratified asymmetry — closes **W6**

**The single cheapest, highest-value item available.** Bin the invariant
non-neutral population by support `n(k,c)` (quartiles of the pooled population, or
fixed bands matching the existing R = 0-at-low-support measurement in §4.3), then
within each bin recompute the fraction of supportive vs. harmful pairs clearing
τ_R. If the gap survives within bins at comparable magnitude, §5.3/§4.4's
disclosure paragraph upgrades from a concession to a **control**, and the paper can
drop the hedge entirely. If it vanishes or inverts within a bin, that must be
reported and §5.3 reweighted around the stratified estimate — better to find this
before a reviewer does.

**Spec.** Restrict to invariant non-neutral pairs from `FINAL_ERM_ResNet_3300_T3.json`.
Report per-bin counts (the top bin will be thin) and a pooled bin-weighted estimate.
**Cost:** ~1s of code, no GPU. **Falsifier:** within-bin gap vanishes/inverts.

#### T0.2 — Margin-redundancy correlation (D3, leak-free variant) — closes **W7**

The Conclusion's redundancy-collapse conjecture currently rests on an unreported
three-point comparison across source domains. Compute, per source domain, the
model's mean prediction margin (top-1 minus top-2 logit) on the unablated model,
and correlate it against that domain's mean |D_d| (already computed for §5.6:
1.15e-3 photo / 1.92e-3 cartoon / 2.13e-3 art). If margin predicts
concept-sensitivity **within** sources, report sketch's own margin as the
leak-free extrapolation supporting the Conclusion's claim — no sketch ablation
needed, so the source-only protocol is never violated.

**Spec.** One forward pass per domain (already-computed logits may suffice —
check whether `eval.py` already retains them). **Cost:** ~30s, no new GPU pass if
logits are cached; ~2 min otherwise. **Outcome:** confirmed → promote this from a
Conclusion aside to a reported §5-adjacent measurement with a number behind it;
refuted → soften the Conclusion's claim to match what's actually shown.

#### T0.3 — Within-domain standardized domain-attribution (D6b) — closes part of **W8**

Recompute the §5.6 attribution using |D_d| divided by that domain's own mean |D|,
removing the global per-domain scale difference that the current null-row
comparison only partially controls. Zero GPU, read-only.

**Cost:** ~30 min of code (new column in the existing analysis script), no GPU.
**Outcome:** if the enrichment ratios survive standardization, §5.6 gains a second,
independent piece of evidence for the same conclusion at no experimental cost.

---

### Tier 1 — existing checkpoints/SAE, no new training, minutes of GPU

#### T1.1 — D4/E7: predicted-label mask — closes **W2**

**The highest-value cheap GPU experiment.** Every current label-free attempt
collapsed the class-conditional mask across classes (intersection-over-classes, or
global argmax-|D|), which is a structurally different and strictly harder
condition than what the paper's own diagnostic measures. The untried, fairer test:
forward-pass unmasked, record the predicted label `ŷ_i`, then rebuild the
**class-conditional** mask as `S(ŷ_i, ·)` instead of `S(y_i, ·)` and re-evaluate.
This preserves exactly the class-conditionality the paper argues is where the
signal lives (§1, §3.6) — it only swaps which label indexes the mask.

**Spec.** Run for `all_harmful` and `keep_robust_support`, the two headline
buckets. Report overall accuracy **and** accuracy split by whether `ŷ_i = y_i` on
the unmasked pass — the split is what's informative, since `ŷ = y` on ~80% of
sketch images already. **Cost:** 2–4 configurations, ~2 min GPU, using the
existing evaluator. **Outcome:** recovers a substantial fraction of the oracle
gain → a genuinely deployable variant exists, which raises the paper's ceiling
considerably and directly answers §7 ¶2's open question; gains little → report the
split plainly (the leakage account holds for keep-only rows specifically) and note
the diagnostic value of the mask-only rows is unaffected either way, since their
claim was never about deployability.

#### T1.2 — D2: checkpoint 2100 vs. 3300 via the tied USAE — closes part of **W5**

`SAEs/normalization_testing/USAE_ERM_Multi_test_3300_2100.pt` is a multi-checkpoint
SAE trained by rotating the encoder across both checkpoints and decoding each code
with *every* SAE — the two dictionaries are **aligned by construction**, so latent
`k` means the same thing at both steps. This capability exists, is unused, and is
undocumented in the draft (`CLAUDE.md` §2 flags this explicitly). Score H/D/R at
step 2100 (the oracle-selected checkpoint) and compare bucket sizes and the §5.3
asymmetry ratio against step 3300 (non-oracle, the paper's main result). Because
dictionaries are aligned, also check whether individual harmful-invariant pairs at
3300 already exist at 2100.

**Spec.** `build_scores.py` pointed at the 2100 checkpoint, then the existing
analysis pipeline. **Cost:** ~10 min scoring + ~5 min interventions, no new
training. **Outcome:** bucket grows 2100 → 3300 → non-robust material accumulates
during training, a second finding the accuracy curve alone would not show; bucket
stable → report it as evidence the phenomenon is architectural/data-driven rather
than a training-duration artifact — either result adds a genuinely new, cheap
finding and partially answers "why only one checkpoint."

#### T1.3 — D5: residual-error ceiling analysis — closes **W9**

For every sketch image still wrong under `keep_robust_support` (§5.7), record how
many of its true class's supportive concepts were active on it, and cross-tabulate
against whether it was already wrong at baseline. This converts §6.2's claim that
the framework "distinguishes" coverage problems from selection problems into an
actual diagnosis rather than an assertion of capability.

**Spec.** Small extension to `eval.py` to dump per-image active-concept counts
under a mask (no new forward passes beyond what `keep_robust_support` already
runs). **Cost:** ~1h of code, one evaluation pass, zero additional GPU risk.
**Outcome:** near-zero active concepts on residual errors → a representational
ceiling claim the paper can state plainly; concepts active but still wrong → the
class-conflict story of §5.5 extends to explain the residual, which is a second use
for material already in the paper.

---

### Tier 2 — new selection/analysis code, no training, minutes of GPU

#### T2.1 — E3/E4: mass-matched bucket comparison — closes **W3**

The one experiment the paper's own Limitations section says would be needed and is
not done. Per class, compute `M_hi = Σ|D|` over `S-_hi` (distributed harm), then
select `S-_lo,matched ⊆ S-_lo` (concentrated harm) whose per-class `Σ|D|` equals
`M_hi` — two selection rules, descending-|D|-until-target and random-to-target (5
seeds) — and compare masking that matched subset against masking all of `S-_hi`.
Mirror on the supportive side (`S+_lo` vs. a mass-matched subset of `S+_hi`);
**direction matters** here, since `S+_lo`'s mass is below `S+_hi`'s in every class
(`CLAUDE.md` §1 records this), so the match must subset `S+_hi` **down**, not the
reverse.

**Interpretation, commit before running:** equal performance at matched mass → R
adds nothing beyond sign and magnitude of D, and §5.4/§7 must say so plainly
rather than hedge; low-R subset wins → consistency is earned as an independent
axis, which converts the paper's most-repeated caveat into a resolved result.

**Cost:** ~60 lines for the matched-subset selector (per `docs/DIRECTIONS.md` D7's
note that matching the per-pair magnitude *distribution*, as
`stratified_random_control` already does, is more meaningful than matching the raw
sum — reuse that machinery rather than rebuilding it), then ~12 configurations,
~5–10 min GPU.

#### T2.2 — D6a: style-mechanism interventions — closes the rest of **W8**

Mask only the low-R pairs whose effect peaks in art_painting, then only those
peaking in cartoon, then only photo — each size- and mass-matched — and compare
sketch effect. If the style hypothesis (§5.6) holds, art/cartoon-peaked masks
should matter more for sketch than photo-peaked ones, beyond what size alone
predicts.

**Cost:** 6–9 configurations, ~5 min GPU. **Outcome:** confirmed → §5.6 becomes a
mechanism with a causal test behind it rather than a hedged correlational reading,
arguably the paper's most concrete causal story if it lands; not confirmed → §5.6
stays exactly as hedged as it is now, which costs nothing since it's already
labeled a conjecture.

#### T2.3 — E8: argmax-|D| label-free global mask — supplementary to **W2**

For each concept `c`, let `k* = argmax_k |D(k,c)|`; if `R(k*,c) < τ_R`, mask `c`
globally for every image regardless of label. Less conservative than the
all-classes-intersection rule that was null, and a useful second label-free data
point alongside T1.1. Report pairs masked for scale comparison against the ~15k
intersection rule and the ~114k oracle rule.

**Cost:** ~30 min code, ~2 min GPU.

---

### Tier 3 — new SAE training required (~2–3h GPU each)

#### T3.1 — D1: MMD backbone, tied USAE — closes **W1**

**Do this one if only one Tier-3 item gets done.** Train an SAE on the MMD
checkpoint (`PACS_ResNet_Sketch_Test_Only/MMD_ResNet_T3`, already on disk) — ideally
as a *tied* USAE spanning the ERM and MMD checkpoints, which `sae.py`'s
`SparseAEs` already supports, so individual latents become comparable across the
two models by construction rather than only in aggregate. Score H/D/R on source
domains and compare: harmful-invariant bucket size, the §5.3 asymmetry ratio, and
error recovery from masking `S-_hi`.

**This is the paper's own falsification test, stated in its own §7.** Running it
before submission is strictly better than having a reviewer request it, because
either outcome is publishable and the paper currently oversells its confidence in
one of them:

- **MMD retains comparably many harmful-invariant pairs** → the central argument
  (*"alignment has no gradient to apply to already-aligned concepts"*) is
  demonstrated rather than merely asserted, and this becomes the paper's headline
  experiment, likely warranting a new subsection rather than a limitations fix.
- **MMD has materially fewer** → the argument needs to soften from "invariance
  training can't fix this" to "this ERM model has this problem, unclear if
  alignment training helps" — a real weakening, but far better to know and reframe
  §1/§2/§7 accordingly than to have a reviewer discover it.

**Cost:** one SAE training run (250 epochs, ~2–3h) + one scoring run (~10 min) +
Group-B-style interventions (~10 min). No new code beyond pointing existing scripts
at the MMD checkpoint.

#### T3.2 — D8: second SAE seed, aggregates only — closes **W4**

Train a second SAE on the **same** ERM checkpoint with a different seed (and
optionally a different dictionary size), then compare only aggregate quantities
across the two SAEs: the H×R contingency proportions (Table 1), the asymmetry
ratio at each τ, the fraction of target error recovered by masking all harmful
pairs, and the keep-only sufficiency figure (Table 5/§5.7). Per `CLAUDE.md` §2 and
§3.2, do **not** attempt to match individual latent indices across the two SAEs —
only aggregate distributions are meaningful, since dictionaries are not
identifiable across independent training runs.

**Cost:** one SAE training run (~2–3h) + a full re-run of the existing Group A/B
harness (~25 min) — mechanically trivial given the harness already exists, just
slow. **Outcome:** aggregates stable → the claims are about the model, not the
dictionary, which is exactly the standard SAE-robustness objection answered
directly; aggregates move materially → report it prominently as a limitation on
proportions rather than silently keep one seed's numbers as if they were exact.

---

### Tier 4 — out of scope for this submission cycle, name explicitly rather than omit

These would strengthen the paper further but represent a scope increase beyond
what a single revision cycle should attempt. List them in Limitations as *named*
future work rather than silently excluding them — a reviewer distinguishes
"we know and are choosing not to" from "we didn't think of this."

| Item | Why deferred |
|---|---|
| Additional datasets (OfficeHome, DomainNet — `analysis/results/` already has some DomainBed logs) | `data.py` is PACS-only; each new dataset needs a loader, a backbone, and an SAE. Large scope increase for external validity that is more honestly scoped as a limitation than half-attempted. |
| Additional DG objectives beyond ERM/MMD (CORAL, IRM, DANN) | If T3.1 (MMD) lands, a third objective would generalize the claim further, but T3.1 alone already converts the central claim from assertion to measurement — diminishing returns per hour versus Tier 0–3 items. |
| Full SAE hyperparameter grid (dictionary size × sparsity × seed) | D8 (one extra seed) already answers the standard objection; a full grid is a different, larger paper. |
| RSM/HIM/DCM per-class profile scores | Already deprioritized in `DIRECTIONS.md` with the correct reasoning: `DISCUSSION.md` shows per-class vulnerability is at least two-dimensional (burden × tie-winning), so a one-dimensional profile is likely to mislead rather than clarify. Cut, not deferred. |
| `processors/B.py` | Unexplained stub, nothing depends on it, not a paper gap. |

---

## 3. Recommended order if time is scarce

If only a subset of this list is feasible before submission, run in this order —
each item is chosen so that an early "no" doesn't waste a later item's setup cost,
and the list front-loads the cheapest, most reviewer-preempting items:

```
T0.1  E14 support-stratified asymmetry        1 s,  no GPU   closes W6
T0.2  margin-redundancy correlation           2 min, no GPU  closes W7
T1.1  D4/E7 predicted-label mask              2 min GPU      closes W2 (the likely #1 review comment)
T1.3  D5 residual-error ceiling               1 h code       closes W9
T2.1  E3/E4 mass-matched comparison           ~1 h code      closes W3 (the most-repeated caveat)
T1.2  D2 checkpoint 2100 vs 3300              15 min GPU     partial W5, free extra finding
T3.1  D1 MMD tied-USAE comparison             ~3 h GPU       closes W1 (the paper's own falsifier)
T3.2  D8 second SAE seed                      ~3 h GPU       closes W4
```

T0.1–T2.1 are all achievable in an afternoon with no training runs and would let
the paper state, truthfully, that every self-identified caveat in its own
Limitations section (W2, W3, W6, W7, W9) has either been resolved or been given a
number instead of a hedge. T3.1 (D1) is the one item worth the multi-hour cost even
under time pressure, because it is the paper's own named falsification test and
currently sits unrun in a Limitations paragraph that all but invites a reviewer to
ask for it.

---

## 4. Reviewer-objection → experiment lookup

| Likely reviewer comment | Experiment | Tier |
|---|---|---|
| "Is this deployable, or only an oracle diagnostic?" | T1.1 (D4/E7), T2.3 (E8) | 1–2 |
| "Does invariance training actually fix this, or are you assuming it doesn't?" | T3.1 (D1) | 3 |
| "Isn't R just measuring the sign/size of D?" | T2.1 (E3/E4) | 2 |
| "SAE dictionaries aren't identifiable — how do I know this isn't seed noise?" | T3.2 (D8) | 3 |
| "Your asymmetry could just be a support-count artifact." | T0.1 (E14) | 0 |
| "You only ran one checkpoint — is this a training-time effect?" | T1.2 (D2) | 1 |
| "The redundancy-collapse claim in your conclusion isn't really shown." | T0.2 (margin) | 0 |
| "Is the style-attribution finding causal or just a correlation?" | T2.2 (D6a) | 2 |
| "Can the remaining errors ever be fixed, or is this a hard ceiling?" | T1.3 (D5) | 1 |
| "Why does `dog` fail and `person` doesn't, despite similar harmful-pair counts?" | Not an experiment — state as an intentional scope limit (W10), n = 7 classes cannot support a profile score. |

---

## Appendix — method specs carried over from the old `EXPERIMENTS.md` §10

Kept verbatim in substance so these remain buildable without re-deriving them.
Notation: bucket symbols as defined in `tmlr.tex` §3.7 / `CLAUDE.md` §1; all scores
from source domains only.

### E3 / E4 (= T2.1) — mass-matched bucket comparison, full spec

1. Compute `M_hi = Σ|D|` over `S-_hi`, **per class**.
2. Select `S-_lo,matched ⊆ S-_lo` whose per-class `Σ|D|` equals `M_hi`. Two
   selection rules: (a) descending `|D|` until the target is met, (b) random to
   target, 5 seeds.
3. Mask `S-_lo,matched`; compare against masking all of `S-_hi`.
4. Report the masses actually achieved — exact matching is impossible.
5. Mirror for `S+_lo` vs. `S+_hi` (E4). Direction constraint: `S+_lo`'s mass is
   below `S+_hi`'s in every class (e.g. dog 5.87e-3 vs. 6.25e-2), so the match
   must subset `S+_hi` **down** to `S+_lo`'s mass — matching the other way is
   infeasible.

### E7 (= T1.1) — predicted-label mask, full spec

1. Forward pass with no mask, record predicted label `ŷ_i`.
2. Build the mask using `S(ŷ_i, ·)` in place of `S(y_i, ·)`.
3. Re-evaluate, reporting overall accuracy **and** accuracy split by whether
   `ŷ_i = y_i` on the first pass.

### E8 (= T2.3) — argmax-|D| label-free mask, full spec

For each concept `c`, `k* = argmax_k |D(k,c)|`. If `R(k*,c) < τ_R`, mask `c`
globally for every image regardless of label. Report pairs masked against the
intersection-rule and oracle-rule counts for scale.

### E11 — R-threshold sweep (not prioritized above; folded into existing Fig. 6)

Sweep the masking threshold on R from 0.0 to 0.9 in steps of 0.1, three variants:
ungated (`R < τ`), gated (`R < τ ∧ |D| > τ_D`), harmful-only
(`R < τ ∧ D < -τ_D`). B4 already showed inert-pair masking is null (+0.08), which
explains the low-τ end without rerunning it; not included as a standalone tier
item above because the existing τ-sensitivity figure already carries the same
argument more cheaply.

### E14 (= T0.1) — support-stratified asymmetry, full spec

Read-only over the score file, no GPU, ~1s. Restrict to invariant non-neutral
pairs. Bin by support `n(k,c)` — quartiles of the pooled non-neutral population, or
fixed bands matching the existing low-support R = 0 measurement. Within each bin
compute the fraction of supportive and of harmful pairs clearing τ_R, report
per-bin plus a pooled stratified estimate weighting bins equally, and report
per-bin counts since the top bin will be thin.

### Result file schema (unchanged)

Any new configuration from Tiers 1–3 should write to `results/E_<id>.json` with the
same schema the existing harness uses: `experiment_id`, `stage`, `description`,
`mask_definition`, `oracle`, `oracle_note`, `tau_D`, `tau_H`, `tau_R`,
`support_floor_stats`, `count_filter_in_forward_pass`,
`pairs_masked{total,per_class}`, `masked_D_mass{total,per_class}`,
`accuracy{domain:{micro,macro,per_class}}`, `counts`,
`predicted_label_histogram`, `seed`, `smoke`, `notes`, `bucket`. Plus one flat row
in `results/summary.csv`.
